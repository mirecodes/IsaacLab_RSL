# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust A2 policy: the Go2 robust teacher of :mod:`..go2_study.robust_env_cfg`, rescaled to Unitree A2.

The method is unchanged: CPG action space, observations, Miki et al. 2022 reward terms, terrain families,
and curricula. Only robot-dependent quantities change. A2 is treated as Go2 scaled by
:data:`~isaaclab_assets.robots.unitree.A2_LENGTH_SCALE` ``L`` (~1.29, 0.275 m vs. 0.213 m leg segments) in
length and :data:`~isaaclab_assets.robots.unitree.A2_MASS_SCALE` ``M`` (~2.67, 40.1 kg vs. 15 kg) in mass,
and every Go2 value is mapped by Froude (dynamic) similarity:

* Lengths and heights x L: terrain obstacles, CPG swing height, foot-clearance threshold, foot-scan radii,
  CoM offsets, base-height target.
* Linear velocities x sqrt(L): commands, pushes, reset velocities, tracking std. Angular velocities and the
  CPG base frequency x 1/sqrt(L).
* Forces x M and torques x M L: external wrench, and the torque used to normalize the torque penalty.
* Penalties on squared dimensional quantities are divided by the square of that quantity's scale, so each
  term weighs as much as on Go2: joint velocity x L, joint acceleration x L^2, foot slip / L, base height / L^2,
  torque / (M L)^2. Terms on joint angles or counts (knee, target smoothness, clearance, collisions) and the
  body-motion term, which mixes v_z^2 and |w|, keep Go2's weights.
* Force observations / M, so they keep Go2's magnitude: the PPO config does not normalize observations.

Deviations from a pure rescaling, forced by the A2 asset:

* The importer nests the links (``Robot/Geometry/base_link/FL_hip/...``) and names the base ``base_link``,
  so prim paths and body names differ from ``go2.usd``.
* The obstacle heights reach 0.34 m (grid and large steps) and 0.19 m (stairs) at the top difficulty
  level, above A2's rated 0.30 m step height for the large steps. They are kept as the curriculum ceiling,
  as for Go2 and the paper.
"""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import A2_LENGTH_SCALE, A2_MASS_SCALE, UNITREE_A2_CFG

from ..go2_study.robust_env_cfg import (
    BASE_FREQUENCY,
    FEET,
    GO2_SCALE,
    STEP_DT,
    Go2RobustTeacherEnvCfg,
    TeacherSceneCfg,
    foot_scanner,
    teacher_terrain_generator,
)

L = A2_LENGTH_SCALE
M = A2_MASS_SCALE

BASE = "base_link"
"""Base body name in the imported A2 asset (``base`` on Go2)."""

FOOT_PATHS = {foot: f"Geometry/{BASE}/{foot[:2]}_hip/{foot[:2]}_thigh/{foot[:2]}_calf/{foot}" for foot in FEET}
"""Foot prim paths below the robot prim."""

A2_BASE_FREQUENCY = BASE_FREQUENCY / math.sqrt(L)
"""CPG base frequency [Hz]: Go2's, scaled by Froude similarity."""


@configclass
class A2TeacherSceneCfg(TeacherSceneCfg):
    """Go2 teacher scene with the foot scanners on A2's nested feet and radii scaled to A2."""

    foot_scanner_FL = foot_scanner(FOOT_PATHS["FL_foot"], length_scale=L)
    foot_scanner_FR = foot_scanner(FOOT_PATHS["FR_foot"], length_scale=L)
    foot_scanner_RL = foot_scanner(FOOT_PATHS["RL_foot"], length_scale=L)
    foot_scanner_RR = foot_scanner(FOOT_PATHS["RR_foot"], length_scale=L)


@configclass
class A2RobustTeacherEnvCfg(Go2RobustTeacherEnvCfg):
    """A2 teacher policy after Miki et al. 2022, rescaled from the Go2 teacher."""

    scene: A2TeacherSceneCfg = A2TeacherSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # -- robot and sensors on the imported asset's prim layout
        self.scene.robot = replace(UNITREE_A2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{BASE}"
        # body-level sensing only matches rigid bodies, so this skips the collision and visual prims
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
        for name in ("foot_scanner_FL", "foot_scanner_FR", "foot_scanner_RL", "foot_scanner_RR"):
            getattr(self.scene, name).offset.pos = (0.0, 0.0, 0.5 * L)
        for event in ("add_base_mass", "base_com", "base_external_force_torque"):
            getattr(self.events, event).params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)
        self.observations.policy.external_wrench.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE

        # -- commands: Froude-scaled velocity ranges
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (-math.sqrt(L), math.sqrt(L))
        ranges.lin_vel_y = (-math.sqrt(L), math.sqrt(L))
        ranges.ang_vel_z = (-1.0 / math.sqrt(L), 1.0 / math.sqrt(L))

        # -- actions: A2 leg geometry, swing height and base frequency scaled from Go2
        cpg = self.actions.joint_pos
        cpg.thigh_length = 0.275
        cpg.calf_length = 0.275
        cpg.base_frequency = A2_BASE_FREQUENCY
        cpg.swing_height = 0.2 * GO2_SCALE * L
        cpg.phase_offset_scale = 0.5 * 2.0 * math.pi * A2_BASE_FREQUENCY * STEP_DT

        # -- observations: forces in Go2 units, heights clipped at the scaled range
        policy = self.observations.policy
        policy.foot_height_scan.clip = (-L, L)
        policy.foot_contact_forces.scale = 0.01 / M
        policy.external_wrench.scale = 0.05 / M

        # -- terrain: Go2's families with obstacle heights and widths scaled by L
        self.scene.terrain.terrain_generator = teacher_terrain_generator(length_scale=L)

        # -- domain randomization
        self.events.base_com.params["com_range"] = {
            "x": (-0.05 * L, 0.05 * L),
            "y": (-0.05 * L, 0.05 * L),
            "z": (-0.01 * L, 0.01 * L),
        }
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5 * math.sqrt(L), 0.5 * math.sqrt(L)),
            "y": (-0.5 * math.sqrt(L), 0.5 * math.sqrt(L)),
        }
        v, w = 0.5 * math.sqrt(L), 0.5 / math.sqrt(L)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-v, v),
            "y": (-v, v),
            "z": (-v, v),
            "roll": (-w, w),
            "pitch": (-w, w),
            "yaw": (-w, w),
        }
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0 / math.sqrt(L), 1.0 / math.sqrt(L))
        # Go2's 30 N (about 0.2 x its weight) and 3 N*m, scaled by the weight and the lever arm
        self.events.base_external_force_torque.params["force_range"] = (-30.0 * M, 30.0 * M)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0 * M * L, 3.0 * M * L)

        # -- rewards: Go2's weights, rescaled where the penalized quantity has a dimension (module docstring)
        rewards = self.rewards
        lin_std, ang_std = math.sqrt(GO2_SCALE * L), math.sqrt(GO2_SCALE / L)
        rewards.track_lin_vel.params["std"] = lin_std
        rewards.track_ang_vel.params["std"] = ang_std
        rewards.lin_vel_orthogonal.params["std"] = lin_std
        rewards.foot_clearance.params["threshold"] = 0.2 * GO2_SCALE * L
        rewards.joint_vel_l2.weight *= L
        rewards.dof_acc_l2.weight *= L**2
        rewards.foot_slip.weight /= L
        # Go2's ~30 N*m reference torque, scaled by the weight and the lever arm
        rewards.dof_torques_l2.weight = -2 * 1.0e-6 * (80.0 / (30.0 * M * L)) ** 2
        rewards.base_height_l2.params["target_height"] = 0.4 * L
        rewards.base_height_l2.weight /= L**2
