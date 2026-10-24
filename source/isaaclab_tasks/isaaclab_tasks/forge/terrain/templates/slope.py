# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough slope: a one-direction incline ``z = tan(theta) x``, plain or with fractal roughness (climb and descent
failures).

The incline runs along the map's x axis, the difficulty axis. Every tile of a row has the same base profile and the
generator lifts each row onto the previous one's top edge, so the rows form one continuous hillside that steepens
with difficulty. A flat landing strip across the tile center (``platform_width`` wide) holds the spawns; robots walk
up (+x) or down (-x) from it, or along the contour (y). With ``zone="slip"`` and gentler knobs it is the
``slip_slope`` terrain (random foot slips).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, height_field_template, lerp, rough, smoothstep


def incline(grid: Grid, cfg: RoughSlopeCfg, slope: float) -> np.ndarray:
    """Base profile ``b(x)`` [m] on the tile grid: gradient ``slope`` (rise over run) everywhere except the landing
    ``|x| < platform_width / 2`` (where ``b = 0``), blended in over ``blend_width``."""
    s = slope * smoothstep((np.abs(grid.x) - cfg.platform_width / 2) / cfg.blend_width)
    b = np.concatenate([[0.0], np.cumsum((s[1:] + s[:-1]) / 2 * grid.hs)])
    b -= b[np.argmin(np.abs(grid.x))]
    return np.broadcast_to(b[:, None], grid.X.shape)


def is_smooth(cfg: RoughSlopeCfg) -> bool:
    return cfg.tile in cfg.smooth_tiles


@height_field_template
def slope_terrain(difficulty: float, cfg: RoughSlopeCfg) -> tuple[np.ndarray, np.ndarray]:
    """Incline of ``slope_deg``, plus the fractal roughness on the tiles not in ``smooth_tiles``."""
    g = Grid(cfg, difficulty)
    slope_deg = lerp(cfg.slope_deg, difficulty)
    base = incline(g, cfg, np.tan(np.radians(slope_deg)))
    h = base if is_smooth(cfg) else base + rough(g, cfg, difficulty)
    cfg.features = {"slope_deg": slope_deg, "smooth": is_smooth(cfg)}
    return finish(g, h, cfg, base)


@configclass
class RoughSlopeCfg(TerrainTemplateCfg):
    function: Callable = slope_terrain
    difficulty_range: Range | None = (0.0, 1.0)
    """The rows span the full ``slope_deg`` range: 0 deg at the first row, 45 deg at the last."""
    slope_deg: Range = (0.0, 45.0)
    """Incline angle [deg]."""
    smooth_tiles: list = [0, 1]
    """Tile columns with a plain incline; the other columns carry the fractal roughness below."""
    unit_size: Range = (0.1, 0.25)
    amplitude: Range = (0.02, 0.06)
    fold_size: Range = (6.0, 4.0)
    fold_amplitude: Range = (0.05, 0.25)
