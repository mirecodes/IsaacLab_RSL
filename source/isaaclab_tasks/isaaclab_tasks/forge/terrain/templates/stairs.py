# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough stairs: a one-direction flight of stairs at ``slope_deg``, clean or with uneven treads and eroded edges
(missed steps, edge catches).

Same layout as ``slope.py``: the flight runs along x through a central landing, and the rows chain into one
continuous staircase that steepens with difficulty.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, finish, height_field_template, lerp, noise
from .slope import RoughSlopeCfg, is_smooth


@height_field_template
def stairs_terrain(difficulty: float, cfg: RoughStairsCfg) -> tuple[np.ndarray, np.ndarray]:
    """Treads of ``step_width`` (riser ``step_width x tan(slope_deg)``) up along +x and down along -x from the
    landing. Rough tiles: tread widths ``U(1 - j, 1 + j)`` (rescaled so the flight ends where a clean one does),
    step edges offset by a smooth field of amplitude ``edge_jitter`` before quantizing, and fine noise on the treads.
    """
    g = Grid(cfg, difficulty)
    smooth = is_smooth(cfg)
    tread = lerp(cfg.step_width, difficulty)
    riser = tread * np.tan(np.radians(lerp(cfg.slope_deg, difficulty)))
    jitter = 0.0 if smooth else lerp(cfg.edge_jitter, difficulty)
    # the same number of steps on every tile of the row, ending before the edge band, so the row edges match
    count = int((g.half[0] - cfg.edge_width - cfg.platform_width / 2 - jitter) // tread)
    u = np.abs(g.X) - cfg.platform_width / 2
    eroded = u + g.fade * noise(g, 0.5, jitter, 3) if jitter > 0 else u
    base, h = np.zeros_like(u), np.zeros_like(u)
    for side in (1.0, -1.0):
        widths = np.full(count, tread)
        if not smooth:
            widths *= g.rng.uniform(1 - cfg.width_jitter, 1 + cfg.width_jitter, count)
            widths *= count * tread / widths.sum()
        edges = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
        mask = side * g.X > 0
        base[mask] = side * riser * np.searchsorted(edges, u[mask], side="right")
        h[mask] = side * riser * np.searchsorted(edges, eroded[mask], side="right")
    if not smooth:
        h += g.fade * noise(g, 0.2, lerp(cfg.amplitude, difficulty), 1)
    cfg.features = {"tread": tread, "riser": riser, "steps": count, "smooth": smooth}
    return finish(g, h, cfg, base)


@configclass
class RoughStairsCfg(RoughSlopeCfg):
    function: Callable = stairs_terrain
    step_width: Range = (0.35, 0.25)
    """Tread [m] (narrower at higher difficulty); 45 deg gives 0.25 m risers."""
    width_jitter: float = 0.2
    """Relative spread of the tread widths on rough tiles, ``U(1 - j, 1 + j) * step_width``."""
    edge_jitter: Range = (0.0, 0.08)
    """Amplitude [m] of the step-edge erosion on rough tiles."""
    amplitude: Range = (0.01, 0.03)
    """Fine noise [m] on the treads of rough tiles."""
