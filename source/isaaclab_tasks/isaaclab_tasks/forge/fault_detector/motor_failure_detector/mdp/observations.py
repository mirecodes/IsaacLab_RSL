# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recorded signals that Isaac Lab's ``mdp`` lacks: privileged base state and analysis-only quantities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Base height above the terrain: z_base - mean z of the finite height-scanner hits (as ``mdp.base_height_l2``)."""
    hits = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    finite = torch.isfinite(hits)
    ground = torch.where(finite, hits, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1)
    return (env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - ground).unsqueeze(1)


def imu_lin_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Specific force of the base in the base frame, a - g (what an accelerometer at the base CoM reads)."""
    data = env.scene[asset_cfg.name].data
    acc = data.body_com_acc_w.torch[:, 0, :3].clone()
    acc[:, 2] += 9.81
    return quat_apply_inverse(data.root_quat_w.torch, acc)


def joint_pos_target_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Joint position targets sent to the actuators, relative to the default pose."""
    data = env.scene[asset_cfg.name].data
    return data.joint_pos_target.torch - data.default_joint_pos.torch


def applied_torque(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint torques after the actuator model (incl. the fault's health)."""
    return env.scene[asset_cfg.name].data.applied_torque.torch


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """1 where the body's net contact force exceeds ``threshold`` [N]."""
    forces = env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > threshold).float()


def foot_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame net contact forces of the bodies, flattened (B x 3)."""
    return env.scene[sensor_cfg.name].data.net_forces_w.torch[:, sensor_cfg.body_ids].flatten(1)
