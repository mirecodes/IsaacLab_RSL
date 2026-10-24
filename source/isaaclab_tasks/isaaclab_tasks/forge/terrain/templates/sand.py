# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sand field: rough hardpan with lowered sand patches (sinking, lift-off failure).

The importer fills each patch with a compliant slab and the runtime event adds drag in it. With ``crust_prob > 0``
some patches are crusts that collapse under load (the ``crust_collapse`` terrain).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, lerp, noise, smoothstep, upto


@height_field_to_mesh
def sand_field_terrain(difficulty: float, cfg: SandFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Around each patch the hardpan is flattened to the patch's surface level ``s`` (so the slab top meets the
    hardpan without a step); inside it is lowered by ``sand_depth`` keeping half its roughness, so the sinkage
    depends on where the foot lands."""
    g = Grid(cfg, difficulty)
    rough = noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    h = g.fade * rough
    classes = list(cfg.patch_size_classes.items())
    weights = np.array([cfg.patch_class_weights[name] for name, _ in classes], dtype=float)
    placed, patches = [], []
    for _ in range(round(lerp(cfg.patch_count, difficulty))):
        name, (lo, hi) = classes[g.rng.choice(len(classes), p=weights / weights.sum())]
        for _attempt in range(50):
            size = g.rng.uniform(lo, hi, 2)
            spot = g.sample_xy(1, cfg, margin=size.max() / 2 + 0.3)
            if not len(spot):
                continue
            (x, y) = spot[0]
            box = np.array([x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2])
            gap = 0.4  # flattened rim of 0.3 m plus clearance
            if all(box[0] > b[2] + gap or box[2] < b[0] - gap or box[1] > b[3] + gap or box[3] < b[1] - gap
                   for b in placed):  # fmt: skip
                break
        else:
            continue
        placed.append(box)
        ellipse = bool(g.rng.uniform() < 0.5)
        depth = upto(g.rng, cfg.sand_depth, difficulty)
        win = g.window(x, y, size.max() / 2 + 0.5)
        dx, dy = (g.X[win] - x) / (size[0] / 2), (g.Y[win] - y) / (size[1] / 2)
        # signed distance-like coordinate: < 1 inside the patch outline
        q = np.hypot(dx, dy) if ellipse else np.maximum(np.abs(dx), np.abs(dy))
        surface = float(h[win][q <= 1.0].mean())
        # flatten the hardpan to the surface level within the slab footprint (+0.05 m), blend out over 0.3 m
        dist_box = np.maximum(np.abs(g.X[win] - x) - size[0] / 2, np.abs(g.Y[win] - y) - size[1] / 2) - 0.05
        flat = 1.0 - smoothstep(dist_box / 0.3)
        h[win] = flat * surface + (1.0 - flat) * h[win]
        # lower the patch interior by depth with a 0.1 m ramp at its outline; keep the hardpan roughness below
        ramp = smoothstep((1.0 - q) * size.min() / 2 / 0.1)
        bottom = np.minimum(surface - depth + 0.5 * rough[win], surface - 0.03)
        h[win] = ramp * bottom + (1.0 - ramp) * h[win]
        crust = bool(g.rng.uniform() < cfg.crust_prob)
        patches.append({
            "x": x, "y": y, "size": size.tolist(), "shape": "ellipse" if ellipse else "rect", "class": name,
            "surface": surface, "bottom": float(h[win][q <= 1.0].min()), "depth": depth,
            "stiffness": float(g.rng.uniform(*cfg.stiffness)), "crust": crust,
        })  # fmt: skip
    cfg.features = {"patches": patches}
    return finish(g, h, cfg)


@configclass
class SandFieldCfg(TerrainTemplateCfg):
    function: Callable = sand_field_terrain
    zone: str = "sand"
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.02, 0.06)
    patch_count: Range = (6.0, 14.0)
    patch_size_classes: dict = {"foot": (0.3, 0.5), "pair": (0.8, 1.5), "body": (2.5, 5.0)}
    """Side ranges [m] of each patch class (uniform per side)."""
    patch_class_weights: dict = {"foot": 3.0, "pair": 2.0, "body": 2.0}
    sand_depth: Range = (0.08, 0.20)
    stiffness: Range = (800.0, 3000.0)
    """Slab contact stiffness [N/m], uniform per patch."""
    crust_prob: float = 0.0
    """Probability that a patch is a collapsing crust (the ``crust_collapse`` terrain)."""
