# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gravel: faceted rough ground with conical pits and V-trenches (foot traps, trips)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, add_pits, add_trenches, finish, gravel


@height_field_to_mesh
def gravel_terrain(difficulty: float, cfg: GravelCfg) -> tuple[np.ndarray, np.ndarray]:
    """Uniform heights on a grid of ``unit_size``, linearly interpolated (facets), plus pits and trenches."""
    g = Grid(cfg, difficulty)
    h = gravel(g, cfg, difficulty)
    cfg.features = {"pits": add_pits(g, h, cfg, difficulty), "trenches": add_trenches(g, h, cfg, difficulty)}
    return finish(g, h, cfg)


@configclass
class GravelCfg(TerrainTemplateCfg):
    function: Callable = gravel_terrain
    pit_density: Range = (0.0, 1.5)
    """Pits per m^2."""
    pit_radius: Range = (0.06, 0.14)
    pit_depth: Range = (0.08, 0.25)
    pit_rim: Range = (0.0, 0.03)
    trench_count: Range = (0.0, 4.0)
    trench_length: Range = (2.0, 6.0)
    """Length [m] of each trench, uniform (not interpolated)."""
    trench_width: Range = (0.08, 0.16)
    trench_depth: Range = (0.10, 0.25)
