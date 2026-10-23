# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base configuration of the A2 study: the Go2 study's base task (``go2_velocity``) on Unitree A2.

The robot's own values (gains, limits, armature, default pose) are Unitree's, see
:obj:`~isaaclab_assets.robots.unitree.UNITREE_A2_CFG`. Values the Go2 study chose are scaled to A2:

* Terrain heights the Go2 study chose for Go2 x :data:`GO2_TO_A2`; slopes are angles and kept.
* The torque weight by the squared ratio of peak joint torques (Go2 ~30 N*m, A2 ~140 N*m).
* Commands, actions, pushes, and the remaining randomization keep the shared velocity-task defaults, as for Go2.

Deviations forced by the A2 asset: the importer nests the links (``Robot/Geometry/base_link/FL_hip/...``) and
names the base ``base_link``, so prim paths and body names differ from ``go2.usd``; the base-height target is
the base height of Unitree's default pose instead of Go2's spawn height.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from isaaclab_assets.robots.unitree import UNITREE_A2_CFG

A2_SCALE = (0.275 + 0.275) / (0.285 + 0.349)
"""A2 / ANYmal length scale, from the leg lengths (hip-flexion joint to foot) of A2 and ANYmal C.

The same ratio for Go2 (0.426 m) is 0.672, which matches the Go2 study's 0.667 from base heights.
"""

GO2_TO_A2 = A2_SCALE / (0.4 / 0.6)
"""Length scale from values the Go2 study chose for Go2 to A2 (Go2 / ANYmal = 0.4 m / 0.6 m base heights)."""

A2_MASS = 40.07
"""Total link mass [kg] of the A2 URDF."""

A2_STANDING_HEIGHT = 0.385
"""Base height [m] of Unitree's default pose (hip -/+0.1, thigh 0.9, calf -1.8 rad) from the URDF kinematics,
including the 0.032 m foot radius."""

BASE = "base_link"
"""Base body name in the imported A2 asset."""


@configclass
class BaseRewardsCfg(RewardsCfg):
    """Shared velocity rewards plus a base-height term.

    Without it, a robot with leg colliders can rest its weight on the legs and collect the full alive bonus
    from a permanent crouch: torso-contact termination never fires and flat-orientation reward can't see it.
    """

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        # rough terrain: the target is measured above the terrain by the height-scanner rays
        params={"target_height": A2_STANDING_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class A2StudyBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go2 study base task on A2."""

    rewards: BaseRewardsCfg = BaseRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- simulation: execute the DC motor actuators through the backend-native path
        self.sim.use_newton_actuators = True

        # -- robot and sensors on the imported asset's prim layout
        self.scene.robot = replace(UNITREE_A2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{BASE}"
        # body-level sensing only matches rigid bodies, so this skips the collision and visual prims
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"

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
        for event in ("add_base_mass", "base_com", "base_external_force_torque"):
            getattr(self.events, event).params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)
        self.events.add_base_mass.params["mass_distribution_params"] = (1 / 1.25, 1.25)
        self.events.base_com.params["com_range"] = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}
        self.events.base_external_force_torque.params["force_range"] = (0.0, 0.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        # -- terrain: the Go2 study's heights scaled to A2
        s = GO2_TO_A2
        self.scene.terrain.max_init_terrain_level = 5
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["pyramid_stairs"].step_height_range = (0.025 * s, 0.12 * s)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025 * s, 0.12 * s)
        terrains["boxes"].grid_height_range = (0.025 * s, 0.1 * s)
        terrains["random_rough"].noise_range = (0.01 * s, 0.06 * s)
        terrains["random_rough"].noise_step = 0.01
        terrains["hf_pyramid_slope"].slope_range = (0.0, 0.4)
        terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.4)

        # -- rewards
        # Source: the Go2 study's weights ("shared" = Isaac Lab's velocity-task default, descended from legged_gym,
        # Rudin et al. 2022, tuned for ANYmal; "Go2" = the stock Go2 rough-task override, hand-tuned for Go2).
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
        # Go2's -0.0002 x (Go2's ~30 N*m / A2's ~140 N*m peak torque)^2, so the same fraction of the peak torque
        # costs the same as on Go2; sum tau^2
        rewards.dof_torques_l2.weight = -0.0002 * (30.0 / 140.0) ** 2
        # shared; sum q_ddot^2
        rewards.dof_acc_l2.weight = -2.5e-7
        # Go2: half the shared -0.01, which locked a hind foot into a dragging gait under Newton; sum (a_t - a_{t-1})^2
        rewards.action_rate_l2.weight = -0.005
        # Go2 rough (shared 0.125, Go2 flat 0.25); swing time beyond 0.5 s at touchdown
        rewards.feet_air_time.weight = 0.01
        rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # stock flat Go2 weight; |g_xy|^2 of the projected gravity
        rewards.flat_orientation_l2.weight = -2.5
        # shared (off)
        rewards.dof_pos_limits.weight = 0.0
        # Go2 study term: (h - A2_STANDING_HEIGHT)^2, h measured above the terrain by the height scanner
        rewards.base_height_l2.weight = -30.0
        # removed as for Go2 (shared -1.0 on ANYmal's THIGH contacts)
        rewards.undesired_contacts = None

        # -- terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = BASE
