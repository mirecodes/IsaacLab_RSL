# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination on a low base: ends the lying-down optimum of ``soft_s0`` (belly down, feet unloaded)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_below_terrain_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """True where the base is less than ``minimum_height`` above the terrain under it (mean of the finite ray hits
    of a height scanner, as ``base_height_l2``; all rays missing counts as terrain at z = 0)."""
    hits_z = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    finite = torch.isfinite(hits_z)
    terrain = torch.where(finite, hits_z, 0.0).sum(1) / finite.sum(1).clamp(min=1)
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - terrain < minimum_height
