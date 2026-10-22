# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms from Miki et al. 2022 (Sec. S7).

Penalties return non-negative magnitudes; the sign lives in the reward weight, as elsewhere in Isaac Lab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, RayCaster

    from .actions import CPGJointPositionAction


def _saturated_tracking(projected: torch.Tensor, desired: torch.Tensor, std: float) -> torch.Tensor:
    """1 once ``projected`` reaches ``desired`` (>= 0), otherwise ``exp(-((projected - desired) / std)^2)``."""
    return torch.where(projected > desired, 1.0, torch.exp(-torch.square((projected - desired) / std)))


def track_lin_vel_saturated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
    zero_command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """r_lv: reward the base velocity projected on the commanded direction, saturating at the commanded speed.

    For a zero command the reward is ``exp(-|v / std|^2)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b.torch[:, :2]
    v_des = env.command_manager.get_command(command_name)[:, :2]
    speed_des = v_des.norm(dim=-1)
    direction = v_des / speed_des.clamp(min=1e-6).unsqueeze(-1)
    moving = _saturated_tracking((v * direction).sum(dim=-1), speed_des, std)
    standing = torch.exp(-torch.square(v / std).sum(dim=-1))
    return torch.where(speed_des > zero_command_threshold, moving, standing)


def track_ang_vel_saturated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
    zero_command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """r_av: same form as :func:`track_lin_vel_saturated` for the yaw rate."""
    asset: Articulation = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b.torch[:, 2]
    w_des = env.command_manager.get_command(command_name)[:, 2]
    moving = _saturated_tracking(w * torch.sign(w_des), w_des.abs(), std)
    standing = torch.exp(-torch.square(w / std))
    return torch.where(w_des.abs() > zero_command_threshold, moving, standing)


def lin_vel_orthogonal_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """r_lvo: ``exp(-3 |v_o / std|^2)`` for the planar velocity orthogonal to the commanded direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b.torch[:, :2]
    v_des = env.command_manager.get_command(command_name)[:, :2]
    direction = v_des / v_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_orthogonal = v - (v * direction).sum(dim=-1, keepdim=True) * direction
    return torch.exp(-3.0 * torch.square(v_orthogonal / std).sum(dim=-1))


def body_motion(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """r_b magnitude: ``1.25 v_z^2 + 0.4 |w_x| + 0.4 |w_y|``."""
    asset: Articulation = env.scene[asset_cfg.name]
    v_z = asset.data.root_lin_vel_b.torch[:, 2]
    w_xy = asset.data.root_ang_vel_b.torch[:, :2]
    return 1.25 * torch.square(v_z) + 0.4 * w_xy.abs().sum(dim=-1)


def foot_clearance(env: ManagerBasedRLEnv, sensor_names: list[str], threshold: float) -> torch.Tensor:
    """r_fc magnitude: number of feet whose highest surrounding terrain sample lies more than ``threshold`` below."""
    count = torch.zeros(env.num_envs, device=env.device)
    for name in sensor_names:
        sensor: RayCaster = env.scene.sensors[name]
        heights = sensor.data.ray_hits_w.torch[..., 2] - sensor.data.pos_w.torch[:, 2:3]
        heights = torch.nan_to_num(heights, nan=0.0, posinf=0.0, neginf=-1.0e3)
        count += (heights.max(dim=-1).values < -threshold).float()
    return count


def joint_upper_threshold_l2(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """r_jc magnitude: ``sum (q - q_th)^2`` over the selected joints where ``q > q_th``."""
    asset: Articulation = env.scene[asset_cfg.name]
    excess = (asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - threshold).clamp(min=0.0)
    return torch.square(excess).sum(dim=-1)


def joint_target_smoothness(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """r_s magnitude: squared first and second differences of the joint position targets [rad^2]."""
    term: CPGJointPositionAction = env.action_manager.get_term(action_name)
    first = term.processed_actions - term.prev_joint_targets
    second = term.processed_actions - 2.0 * term.prev_joint_targets + term.prev_prev_joint_targets
    return torch.square(first).sum(dim=-1) + torch.square(second).sum(dim=-1)


def feet_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """r_slip magnitude: ``sum |v_foot|^2`` over feet in contact."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1) > threshold
    foot_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids]
    return (torch.square(foot_vel).sum(dim=-1) * in_contact).sum(dim=-1)


"""
Fu et al. 2021 energy reward.
"""


def lin_vel_x_error_l1(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """``|v_x - v_x^target|`` [m/s]: forward speed error in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return torch.abs(asset.data.root_lin_vel_b.torch[:, 0] - command[:, 0])


def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """``v_y^2`` [m^2/s^2]: squared lateral velocity in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 1])


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """``w_yaw^2`` [rad^2/s^2]: squared yaw rate in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b.torch[:, 2])


def mechanical_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """``tau^T q_dot`` [W]: summed joint mechanical power, signed as in Fu et al. 2021 (Eq. 5)."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return torch.sum(torque * asset.data.joint_vel.torch[:, asset_cfg.joint_ids], dim=1)
