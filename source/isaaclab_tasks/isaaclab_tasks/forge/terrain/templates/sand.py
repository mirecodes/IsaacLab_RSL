# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sand field: rough hardpan with lowered sand patches (sinking, lift-off failure).

The patches form a patchwork: the tile is cut into an irregular grid (random column widths and row heights), and a
random share of the cells holds a rectangle inset from the cell by a random margin, so the patches vary in size and
position yet stay axis aligned, like the box slabs that fill them. The importer fills each patch with a compliant
slab and the runtime event adds drag in it. With ``crust_prob > 0`` some patches are crusts that collapse under load
(the ``crust_collapse`` terrain).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, height_field_template, lerp, noise, smoothstep, upto

RIM = 0.35
"""Reach [m] of the flattened hardpan around a patch (0.05 m footprint margin plus a 0.3 m blend)."""


def partition(rng: np.random.Generator, lo: float, hi: float, cell: Range) -> np.ndarray:
    """Cut ``[lo, hi]`` into consecutive cells of width ``U(cell)`` (the last one is cut short)."""
    widths = rng.uniform(*cell, int((hi - lo) / cell[0]) + 2)
    edges = lo + np.concatenate([[0.0], np.cumsum(widths)])
    return np.append(edges[edges < hi], hi)


@height_field_template
def sand_field_terrain(difficulty: float, cfg: SandFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Around each patch the hardpan is flattened to the patch's surface level ``s`` (so the slab top meets the
    hardpan without a step); inside it is lowered by ``sand_depth`` keeping half its roughness, so the sinkage
    depends on where the foot lands. Patches keep ``2 x RIM`` apart, so no flattened rim reaches another patch."""
    g = Grid(cfg, difficulty)
    rough = noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    h = g.fade * rough
    lim = g.half - cfg.edge_width
    xs, ys = partition(g.rng, -lim[0], lim[0], cfg.cell_size), partition(g.rng, -lim[1], lim[1], cfg.cell_size)
    keep = cfg.platform_width / 2 + 0.3
    prob = lerp(cfg.sand_prob, difficulty)
    patches = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if g.rng.uniform() >= prob:
                continue
            inset = RIM + 0.05 + g.rng.uniform(0.0, cfg.inset_jitter, 4)
            x0, x1, y0, y1 = xs[i] + inset[0], xs[i + 1] - inset[1], ys[j] + inset[2], ys[j + 1] - inset[3]
            if min(x1 - x0, y1 - y0) < cfg.min_patch or (x0 < keep and x1 > -keep and y0 < keep and y1 > -keep):
                continue
            x, y, size = (x0 + x1) / 2, (y0 + y1) / 2, np.array([x1 - x0, y1 - y0])
            depth = upto(g.rng, cfg.sand_depth, difficulty)
            win = g.window(x, y, size.max() / 2 + 0.5)
            # Chebyshev coordinate: < 1 inside the patch
            q = np.maximum(np.abs(g.X[win] - x) / (size[0] / 2), np.abs(g.Y[win] - y) / (size[1] / 2))
            surface = float(h[win][q <= 1.0].mean())
            # flatten the hardpan to the surface level within the slab footprint (+0.05 m), blend out over 0.3 m
            dist_box = np.maximum(np.abs(g.X[win] - x) - size[0] / 2, np.abs(g.Y[win] - y) - size[1] / 2) - 0.05
            flat = 1.0 - smoothstep(dist_box / (RIM - 0.05))
            h[win] = flat * surface + (1.0 - flat) * h[win]
            # lower the patch interior by depth with a 0.1 m ramp at its outline; keep the hardpan roughness below
            ramp = smoothstep((1.0 - q) * size.min() / 2 / 0.1)
            bottom = np.minimum(surface - depth + 0.5 * rough[win], surface - 0.03)
            h[win] = ramp * bottom + (1.0 - ramp) * h[win]
            patches.append({
                "x": x, "y": y, "size": size.tolist(), "surface": surface, "bottom": float(h[win][q <= 1.0].min()),
                "depth": depth, "stiffness": float(g.rng.uniform(*cfg.stiffness)),
                "crust": bool(g.rng.uniform() < cfg.crust_prob),
            })  # fmt: skip
    cfg.features = {"patches": patches}
    return finish(g, h, cfg)


@configclass
class SandFieldCfg(TerrainTemplateCfg):
    function: Callable = sand_field_terrain
    zone: str = "sand"
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.02, 0.06)
    cell_size: Range = (1.2, 4.5)
    """Side [m] of the patchwork cells, uniform per column and per row (not interpolated)."""
    sand_prob: Range = (0.25, 0.45)
    """Share of the cells that hold a patch."""
    inset_jitter: float = 0.6
    """Extra random inset [m] of each patch side from its cell, on top of the ``RIM`` clearance."""
    min_patch: float = 0.3
    """Patches narrower than this [m] (about one foot print) are dropped."""
    sand_depth: Range = (0.08, 0.20)
    stiffness: Range = (800.0, 3000.0)
    """Slab contact stiffness [N/m], uniform per patch."""
    crust_prob: float = 0.0
    """Probability that a patch is a collapsing crust (the ``crust_collapse`` terrain)."""
