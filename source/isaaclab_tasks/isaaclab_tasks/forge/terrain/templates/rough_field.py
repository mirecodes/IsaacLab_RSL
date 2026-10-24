# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough field: fractal random ground, fine bumps on rolling folds (generic uneven terrain, misplaced feet).

Isaac Lab's ``HfRandomUniformTerrainCfg`` (``random_rough``) is one octave of uniform noise on a flat plane; here
``octaves`` layers run from meter-scale folds down to the same fine bumps, as in fractal (Minecraft-style) terrain.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, height_field_template, rough


@height_field_template
def rough_field_terrain(difficulty: float, cfg: RoughFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Sum of value-noise octaves (see :func:`~.common.rough`)."""
    g = Grid(cfg, difficulty)
    h = rough(g, cfg, difficulty)
    return finish(g, h, cfg)


@configclass
class RoughFieldCfg(TerrainTemplateCfg):
    function: Callable = rough_field_terrain
    unit_size: Range = (0.1, 0.2)
    """Fine bumps, as ``random_rough`` (0.1 m noise step at 0.1 m resolution)."""
    amplitude: Range = (0.02, 0.05)
    fold_size: Range = (8.0, 4.0)
    fold_amplitude: Range = (0.1, 0.4)
