# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stub field: near-flat ground with stones and roots below the height scanner's resolution (toe stubbing)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, add_stubs, finish, height_field_template, lerp, noise


@height_field_template
def stub_field_terrain(difficulty: float, cfg: StubFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """±1 cm smooth ground, spherical domes ``H sqrt(1 - (rho/r)^2)`` (r <= 5 cm) and half-cylinder roots."""
    g = Grid(cfg, difficulty)
    h = g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), 3)
    cfg.features = {"stubs": add_stubs(g, h, cfg, difficulty)}
    return finish(g, h, cfg)


@configclass
class StubFieldCfg(TerrainTemplateCfg):
    function: Callable = stub_field_terrain
    unit_size: Range = (0.3, 0.3)
    amplitude: Range = (0.01, 0.01)
    stub_density: Range = (0.5, 3.0)
    """Stones per m^2."""
    stub_radius: Range = (0.02, 0.05)
    stub_height: Range = (0.02, 0.06)
    root_count: Range = (2.0, 8.0)
    root_width: Range = (0.03, 0.05)
    root_height: Range = (0.02, 0.04)
    root_length: Range = (0.5, 3.0)
