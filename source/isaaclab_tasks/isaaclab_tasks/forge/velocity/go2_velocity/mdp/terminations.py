# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the Fu et al. 2021 energy environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def base_height_below_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base is less than ``minimum_height`` [m] above the mean terrain height under it.

    Rays that miss the terrain are ignored; an environment whose rays all miss uses terrain height 0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w.torch[..., 2]
    finite = torch.isfinite(hits_z)
    terrain_height = torch.where(finite, hits_z, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1)
    return asset.data.root_pos_w.torch[:, 2] - terrain_height < minimum_height


def bad_roll_pitch(
    env: ManagerBasedRLEnv, max_roll: float, max_pitch: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the base roll or pitch magnitude [rad] exceeds its limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return (roll.abs() > max_roll) | (pitch.abs() > max_pitch)
