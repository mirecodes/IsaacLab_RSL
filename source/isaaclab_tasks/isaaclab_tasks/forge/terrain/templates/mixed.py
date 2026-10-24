# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed: a gentle pyramid combining gravel, stubs, pits, and trenches at halved densities (compound causes)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, add_pits, add_stubs, add_trenches, finish, gravel, lerp, pyramid


@height_field_to_mesh
def mixed_terrain(difficulty: float, cfg: MixedCfg) -> tuple[np.ndarray, np.ndarray]:
    g = Grid(cfg, difficulty)
    h = pyramid(g, cfg, lerp(cfg.slope, difficulty), False) + gravel(g, cfg, difficulty)
    cfg.features = {
        "stubs": add_stubs(g, h, cfg, difficulty, density_scale=0.5),
        "pits": add_pits(g, h, cfg, difficulty, density_scale=0.5),
        "trenches": add_trenches(g, h, cfg, difficulty),
    }
    return finish(g, h, cfg)


@configclass
class MixedCfg(TerrainTemplateCfg):
    function: Callable = mixed_terrain
    slope: Range = (0.1, 0.3)
    unit_size: Range = (0.15, 0.4)
    amplitude: Range = (0.03, 0.12)
    stub_density: Range = (0.5, 3.0)
    stub_radius: Range = (0.02, 0.05)
    stub_height: Range = (0.02, 0.06)
    root_count: Range = (2.0, 8.0)
    root_width: Range = (0.03, 0.05)
    root_height: Range = (0.02, 0.04)
    root_length: Range = (0.5, 3.0)
    pit_density: Range = (0.0, 1.5)
    pit_radius: Range = (0.06, 0.14)
    pit_depth: Range = (0.08, 0.25)
    pit_rim: Range = (0.0, 0.03)
    trench_count: Range = (0.0, 2.0)
    trench_length: Range = (2.0, 6.0)
    trench_width: Range = (0.08, 0.16)
    trench_depth: Range = (0.10, 0.25)
