# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mesa drop: a raised mesa with terraces down to the edge band (falls from terraces, hard landings)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, lerp, noise, smoothstep


@height_field_to_mesh
def mesa_terrain(difficulty: float, cfg: MesaDropCfg) -> tuple[np.ndarray, np.ndarray]:
    """Terraces of ``drop = mesa_height / ceil(mesa_height / drop_height)`` and ``tread_width``; each of the four
    sides is a cliff, a 45 deg scree ramp, or rubble."""
    g = Grid(cfg, difficulty)
    mesa_height = lerp(cfg.mesa_height, difficulty)
    count = int(np.ceil(mesa_height / lerp(cfg.drop_height, difficulty)))
    drop = mesa_height / count
    run = g.half.min() - cfg.edge_width - cfg.platform_width / 2 - 0.5
    tread = min(lerp(cfg.tread_width, difficulty), run / count)
    walls = tread * np.arange(1, count + 1)
    # side of each pixel (the nearest border) and its wall type: 0 cliff, 1 scree, 2 rubble
    sides = np.argmin(np.stack([g.half[0] - g.X, g.half[0] + g.X, g.half[1] - g.Y, g.half[1] + g.Y]), axis=0)
    u = g.rng.uniform(size=4)
    types = np.where(u < cfg.scree_prob, 1, np.where(u < cfg.scree_prob + cfg.rubble_prob, 2, 0))
    kind = types[sides]
    r = g.r[..., None]
    cliff = drop * (r >= walls).sum(-1)
    # scree/rubble: the drop spread over a 45 deg ramp ending at the wall
    scree = drop * np.clip((r - walls + drop) / drop, 0.0, 1.0).sum(-1)
    rubble = scree + noise(g, 0.3, 0.3 * drop, 1) * np.clip(np.abs(scree - cliff) / drop * 4.0, 0.0, 1.0)
    h = np.choose(kind, [cliff, scree, rubble])
    h = np.where(g.r <= 0.0, 0.0, h)
    h += noise(g, 0.3, lerp(cfg.amplitude, difficulty), 1) * smoothstep(g.r / 0.3) * (np.abs(h - cliff) < 1e-6)
    h = np.where(g.p <= 0.0, mesa_height, h)
    cfg.features = {"drop": drop, "tread": tread, "walls": walls.tolist(), "wall_types": types.tolist()}
    return finish(g, h, cfg)


@configclass
class MesaDropCfg(TerrainTemplateCfg):
    function: Callable = mesa_terrain
    mesa_height: Range = (3.0, 4.0)
    drop_height: Range = (1.5, 2.0)
    tread_width: Range = (4.0, 4.0)
    """Spec (0.6, 2.0), (0.3, 1.0), (1.0, 2.5): no flight (all feet off > 0.3 s) in 60 s. The policy steps down a
    drop of up to 1 m (front feet land before the hind feet leave), and on narrow treads the robots spent most of
    the time on a 12 m wide top. Now two terraces of 1.5-2.0 m, 4 m deep (see ``REPORT.md``)."""
    scree_prob: float = 0.2
    rubble_prob: float = 0.1
    """Spec 0.4 (and 0.2 rubble): the robots step down screes, so flights came from the cliff sides only; 70 % cliffs
    make a flight-free minute unlikely."""
    amplitude: Range = (0.02, 0.06)
