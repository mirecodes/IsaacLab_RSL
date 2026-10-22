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
        rewards = self.rewards
        rewards.track_lin_vel_xy_exp.weight = 1.5
        rewards.track_ang_vel_z_exp.weight = 0.75
        rewards.lin_vel_z_l2.weight = -2.0
        rewards.ang_vel_xy_l2.weight = -0.05
        rewards.dof_torques_l2.weight = -0.0002
        rewards.dof_acc_l2.weight = -2.5e-7
        rewards.action_rate_l2.weight = -0.005
        rewards.feet_air_time.weight = 0.01
        # stock rough Go2 uses 0; -2.5 is the stock flat Go2 weight
        rewards.flat_orientation_l2.weight = -2.5
        rewards.dof_pos_limits.weight = 0.0
        rewards.base_height_l2.weight = -30.0
        rewards.undesired_contacts = None

        # -- terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
