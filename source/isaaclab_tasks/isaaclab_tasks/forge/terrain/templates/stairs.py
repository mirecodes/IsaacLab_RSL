# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rough stairs: pyramid stairs with uneven step widths and eroded edges (missed steps, edge catches)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, lerp, noise, smoothstep


@height_field_to_mesh
def stairs_terrain(difficulty: float, cfg: RoughStairsCfg) -> tuple[np.ndarray, np.ndarray]:
    """Step widths ``w U(1 - j, 1 + j)``, step edges offset by a smooth field of amplitude ``edge_jitter`` before
    quantizing, plus fine noise on the treads."""
    g = Grid(cfg, difficulty)
    step_height, step_width = lerp(cfg.step_height, difficulty), lerp(cfg.step_width, difficulty)
    run = g.half.min() - cfg.edge_width - cfg.platform_width / 2
    widths = step_width * g.rng.uniform(1 - cfg.width_jitter, 1 + cfg.width_jitter, int(2 * run / step_width) + 2)
    edges = np.cumsum(widths)
    edges = edges[edges < run]
    # erosion: offset the distance with a smooth field before quantizing into steps
    r = g.r + noise(g, 0.5, lerp(cfg.edge_jitter, difficulty), 3)
    h = step_height * np.searchsorted(edges, np.clip(r, 0.0, None), side="right")
    h = np.where(g.r <= 0.0, 0.0, h)
    h += np.where(g.p > 0, noise(g, 0.2, lerp(cfg.amplitude, difficulty), 1), 0.0) * smoothstep(g.r / 0.3)
    h = -h if cfg.inverted else h
    cfg.features = {"step_edges": edges.tolist(), "step_height": step_height}
    return finish(g, h, cfg)


@configclass
class RoughStairsCfg(TerrainTemplateCfg):
    function: Callable = stairs_terrain
    step_height: Range = (0.10, 0.25)
    step_width: Range = (0.3, 0.6)
    width_jitter: float = 0.3
    """Relative spread of individual step widths, ``U(1 - j, 1 + j) * step_width``."""
    edge_jitter: Range = (0.0, 0.1)
    amplitude: Range = (0.01, 0.05)
    inverted: bool = False
