# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base configuration for the Go2 robust vs. energy-efficient locomotion study.

Every value below restates the effective default of ``Isaac-Velocity-Rough-UnitreeGo2``
(shared velocity defaults + Go2 overrides), except the flat-orientation penalty, which takes the stock
Go2 flat-terrain weight instead of 0.
It exists to collect the study's tuning knobs in one place: the robust (Miki et al. 2022) and
energy-efficient variants subclass it and change only what they need.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class Go2StudyBaseEnvCfg(UnitreeGo2RoughEnvCfg):
    """Stock Go2 rough-terrain velocity task with all study-relevant defaults spelled out."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (-1.0, 1.0)
        ranges.lin_vel_y = (-1.0, 1.0)
        ranges.ang_vel_z = (-1.0, 1.0)

        # -- actions
        self.actions.joint_pos.scale = 0.25

        # -- domain randomization
        material = self.events.physics_material.params
        material["static_friction_range"] = (0.8, 0.8)
        material["dynamic_friction_range"] = (0.6, 0.6)
        material["restitution_range"] = (0.0, 0.0)
        self.events.add_base_mass.params["mass_distribution_params"] = (1 / 1.25, 1.25)
        self.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        # -- terrain (scaled down for Go2 by the parent config)
        self.scene.terrain.max_init_terrain_level = 5
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.12)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.12)
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)

        # -- rewards
        # Source: no paper. "shared" is Isaac Lab's velocity-task default (velocity_env_cfg.RewardsCfg, descended
        # from legged_gym, Rudin et al. 2022, tuned for ANYmal); "Go2" is the stock Go2 rough-task override
        # (config/go2/rough_env_cfg.py), hand-tuned for Go2, not scaled from ANYmal by a rule.
        # Isaac Lab multiplies every weight by the 0.02 s step, so the values are per second.
        rewards = self.rewards
        # Go2 (shared 1.0); exp(-|v_xy^cmd - v_xy|^2 / 0.25)
        rewards.track_lin_vel_xy_exp.weight = 1.5
        # Go2 (shared 0.5); exp(-(w_z^cmd - w_z)^2 / 0.25)
        rewards.track_ang_vel_z_exp.weight = 0.75
        # shared; v_z^2
        rewards.lin_vel_z_l2.weight = -2.0
        # shared; w_x^2 + w_y^2
        rewards.ang_vel_xy_l2.weight = -0.05
        # Go2 (shared -1e-5); sum tau^2
        rewards.dof_torques_l2.weight = -0.0002
        # shared; sum q_ddot^2
        rewards.dof_acc_l2.weight = -2.5e-7
        # Go2: half the shared -0.01, which locked a hind foot into a dragging gait under Newton; sum (a_t - a_{t-1})^2
        rewards.action_rate_l2.weight = -0.005
        # Go2 rough (shared 0.125, Go2 flat 0.25); swing time beyond 0.5 s at touchdown
        rewards.feet_air_time.weight = 0.01
        # stock rough Go2 uses 0; -2.5 is the stock flat Go2 weight; |g_xy|^2 of the projected gravity
        rewards.flat_orientation_l2.weight = -2.5
        # shared (off)
        rewards.dof_pos_limits.weight = 0.0
        # Go2-only term: (h - 0.4 m)^2 against Go2's spawn height, h measured above the terrain by the height scanner
        rewards.base_height_l2.weight = -30.0
        # Go2: removed (shared -1.0 on ANYmal's THIGH contacts; Go2 has no such body names)
        rewards.undesired_contacts = None

        # -- terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
