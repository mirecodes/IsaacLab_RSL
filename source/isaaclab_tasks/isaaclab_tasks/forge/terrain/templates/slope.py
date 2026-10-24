# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough slope: a pyramid (or bowl) with gravel noise and flat-topped rock ledges (climb and descent failures).

With ``zone="slip"`` and gentler knobs it is the ``slip_slope`` terrain (random foot slips).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, gravel, lerp, pyramid, upto


@height_field_to_mesh
def slope_terrain(difficulty: float, cfg: RoughSlopeCfg) -> tuple[np.ndarray, np.ndarray]:
    """Pyramid ``h = s min(fx, fy)`` to a flat top, gravel noise, and ledges raised above the local ground."""
    g = Grid(cfg, difficulty)
    h = pyramid(g, cfg, lerp(cfg.slope, difficulty), cfg.inverted) + gravel(g, cfg, difficulty)
    ledges = []
    for x, y in g.sample_xy(round(lerp(cfg.ledge_count, difficulty)), cfg, margin=1.0):
        sx, sy = upto(g.rng, cfg.ledge_size, difficulty, 2) / 2
        win = g.window(x, y, max(sx, sy))
        inside = (np.abs(g.X[win] - x) < sx) & (np.abs(g.Y[win] - y) < sy)
        top = g.height_at(h, np.array([[x, y]]))[0] + upto(g.rng, cfg.ledge_height, difficulty)
        h[win] = np.where(inside, np.maximum(h[win], top), h[win])
        ledges.append({"x": x, "y": y, "size": [2 * sx, 2 * sy], "top": top})
    cfg.features = {"ledges": ledges}
    return finish(g, h, cfg)


@configclass
class RoughSlopeCfg(TerrainTemplateCfg):
    function: Callable = slope_terrain
    slope: Range = (0.45, 1.0)
    """Rise over run of every pyramid face."""
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.03, 0.10)
    ledge_count: Range = (0.0, 8.0)
    ledge_size: Range = (0.3, 0.8)
    ledge_height: Range = (0.05, 0.20)
    inverted: bool = False
