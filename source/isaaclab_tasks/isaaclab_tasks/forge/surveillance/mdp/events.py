# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Root reset facing up or down an incline."""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def reset_root_up_or_down(
    env,
    env_ids: torch.Tensor | slice,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """``isaaclab.envs.mdp.reset_root_state_uniform`` (copied) with the sampled yaw turned by pi for a random half of
    the robots: with a yaw range around 0, a robot faces +x (up the incline) or -x (down it)."""
    asset = env.scene[asset_cfg.name]
    default_root_pose = asset.data.default_root_pose.torch[env_ids]
    default_root_vel = asset.data.default_root_vel.torch[env_ids]
    num = default_root_pose.shape[0]

    rand_samples = math_utils.sample_uniform_from_ranges(
        pose_range, ("x", "y", "z", "roll", "pitch", "yaw"), num, device=asset.device
    )
    rand_samples[:, 5] += math.pi * torch.randint(2, (num,), device=asset.device)
    positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)

    rand_samples = math_utils.sample_uniform_from_ranges(
        velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"), num, device=asset.device
    )
    velocities = default_root_vel + rand_samples
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
