# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pit field: faceted rough ground with conical pits and V-trenches (foot traps, trips)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, add_pits, add_trenches, finish, height_field_template, rough


@height_field_template
def pit_field_terrain(difficulty: float, cfg: PitFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Uniform heights on a grid of ``unit_size``, linearly interpolated (facets), plus pits and trenches."""
    g = Grid(cfg, difficulty)
    h = rough(g, cfg, difficulty)
    cfg.features = {"pits": add_pits(g, h, cfg, difficulty), "trenches": add_trenches(g, h, cfg, difficulty)}
    return finish(g, h, cfg)


@configclass
class PitFieldCfg(TerrainTemplateCfg):
    function: Callable = pit_field_terrain
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
