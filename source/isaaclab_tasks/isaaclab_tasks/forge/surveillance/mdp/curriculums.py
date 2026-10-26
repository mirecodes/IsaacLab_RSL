# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn-tile draw. A curriculum term, because the curriculum runs before the reset events (``reset_base`` places the
robot at its env origin)."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def random_tile(env, env_ids: Sequence[int]) -> torch.Tensor:
    """Move each resetting robot's origin to a tile (row, column) of the terrain grid drawn uniformly; returns the mean
    row."""
    terrain = env.scene.terrain
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    rows, cols = terrain.terrain_origins.shape[:2]
    terrain.terrain_levels[ids] = torch.randint(rows, (len(ids),), device=env.device)
    terrain.terrain_types[ids] = torch.randint(cols, (len(ids),), device=env.device)
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return terrain.terrain_levels.float().mean()
