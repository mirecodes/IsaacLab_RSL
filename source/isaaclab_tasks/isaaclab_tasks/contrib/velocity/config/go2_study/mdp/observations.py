# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Miki et al. 2022 teacher (Table S3) and the Fu et al. 2021 energy policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensor, RayCaster

    from .actions import CPGJointPositionAction
    from .events import randomize_foot_friction, randomize_pd_gains


"""
Proprioception.
"""


def cpg_phase(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Phase offsets, cos/sin of the leg phases, and the base frequency [Hz] (13 values for 4 legs)."""
    term: CPGJointPositionAction = env.action_manager.get_term(action_name)
    base_frequency = torch.full((env.num_envs, 1), term.base_frequency, device=env.device)
    return torch.cat((term.delta_phase, torch.cos(term.phase), torch.sin(term.phase), base_frequency), dim=-1)


def joint_pos_error(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Joint target minus measured joint position [rad]."""
    term: CPGJointPositionAction = env.action_manager.get_term(action_name)
    asset: Articulation = env.scene[term.cfg.asset_name]
    return term.processed_actions - asset.data.joint_pos.torch[:, term.joint_ids]


def joint_target_rel(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Joint position target relative to the default joint positions [rad]."""
    term: CPGJointPositionAction = env.action_manager.get_term(action_name)
    return term.processed_actions - term.default_joint_pos


def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll and pitch [rad] of the base in [-pi, pi]."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return torch.stack((roll, pitch), dim=-1)


"""
Exteroception.
"""


def foot_height_scan(env: ManagerBasedEnv, sensor_names: list[str]) -> torch.Tensor:
    """Terrain height [m] relative to each foot at the foot-centered sample points, concatenated over feet.

    Negative values mean the terrain is below the foot.
    """
    scans = []
    for name in sensor_names:
        sensor: RayCaster = env.scene.sensors[name]
        heights = sensor.data.ray_hits_w.torch[..., 2] - sensor.data.pos_w.torch[:, 2:3]
        scans.append(torch.nan_to_num(heights, nan=0.0, posinf=1.0, neginf=-1.0))
    return torch.cat(scans, dim=-1)


"""
Privileged information.
"""


def contact_states(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Binary contact state of the selected bodies."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > threshold).float()


def contact_forces(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact forces [N] on the selected bodies in the world frame, flattened."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].flatten(start_dim=1)


def contact_normals(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Unit direction of the net normal contact force per body (zero without contact), flattened."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    normal_forces = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids]
    magnitude = normal_forces.norm(dim=-1, keepdim=True)
    normals = torch.where(magnitude > threshold, normal_forces / magnitude.clamp(min=1e-6), 0.0)
    return normals.flatten(start_dim=1)


def foot_friction(env: ManagerBasedEnv, event_name: str = "foot_friction") -> torch.Tensor:
    """Friction coefficient of each foot as sampled by :class:`randomize_foot_friction`."""
    term: randomize_foot_friction = env.event_manager.get_term_cfg(event_name).func
    return term.foot_friction


def external_wrench_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Persistent external force [N] and torque [N*m] on the selected body, in its body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    composer = asset.permanent_wrench_composer
    force = composer.out_force_b.torch[:, asset_cfg.body_ids].flatten(start_dim=1)
    torque = composer.out_torque_b.torch[:, asset_cfg.body_ids].flatten(start_dim=1)
    return torch.cat((force, torque), dim=-1)


def air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Time [s] since each selected body left the ground (zero while in contact)."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]


def planar_base_velocity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base forward and lateral velocity [m/s] and yaw rate [rad/s] in the base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_lin_vel_b.torch[:, :2], asset.data.root_ang_vel_b.torch[:, 2:3]), dim=-1)


def ground_friction(env: ManagerBasedEnv, event_name: str = "foot_friction") -> torch.Tensor:
    """Friction coefficient shared by all feet, as sampled by :class:`randomize_foot_friction` with ``shared``."""
    term: randomize_foot_friction = env.event_manager.get_term_cfg(event_name).func
    return term.foot_friction[:, :1]


def motor_strength(env: ManagerBasedEnv, event_name: str = "pd_gains") -> torch.Tensor:
    """Per-joint motor strength factors sampled by :class:`randomize_pd_gains`."""
    term: randomize_pd_gains = env.event_manager.get_term_cfg(event_name).func
    return term.motor_strength


def payload_and_com(
    env: ManagerBasedEnv,
    mass_event_name: str = "add_payload",
    com_event_name: str = "base_com",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Added mass [kg] and planar center-of-mass offset [m] of one body (3 values).

    Offsets are relative to the defaults recorded by the stock mass and CoM randomization terms;
    they read zero until those terms have run once.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    mass_term = env.event_manager.get_term_cfg(mass_event_name).func
    com_term = env.event_manager.get_term_cfg(com_event_name).func
    payload = torch.zeros(env.num_envs, 1, device=env.device)
    offset = torch.zeros(env.num_envs, 2, device=env.device)
    if mass_term.default_mass is not None:
        payload = (asset.data.body_mass.torch[:, [body_id]] - mass_term.default_mass[:, [body_id]]).to(env.device)
    if com_term.default_com is not None:
        offset = (asset.data.body_com_pos_b.torch[:, body_id, :2] - com_term.default_com[:, body_id, :2]).to(env.device)
    return torch.cat((payload, offset), dim=-1)
