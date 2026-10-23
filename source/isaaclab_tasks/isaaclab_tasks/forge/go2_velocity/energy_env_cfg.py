# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Energy-efficient Go2 policy: the base policy of Fu et al. 2021, "Minimizing Energy Consumption Leads to
the Emergence of Gaits in Legged Robots" (CoRL), adapted to Go2.

Follows the paper's base policy (RMA phase 1, trained with the true extrinsics): the energy reward
``r = r_forward + 0.04 * r_energy + r_alive`` at a fixed target speed, a 30-dim state plus the previous
action and the 19-dim extrinsics vector as input, joint-position deltas with the paper's clipping, fractal
terrain, the "normal perturbation" column of Table 1, and the paper's early-termination rules. Like the
robust teacher, the policy sees noiseless state.

Deviations, either forced by Isaac Lab or not stated in the paper:

* A1 and Go2 share the leg layout (0.20 m vs. 0.213 m links), so lengths are scaled by
  :data:`LENGTH_SCALE` (~1.07) and payload by :data:`MASS_SCALE`; the target speed is kept.
* 50 Hz control (sim dt 0.005 s x decimation 4, as in the robust teacher) instead of 100 Hz. The
  1000-step episode is kept as 10 s, and the parameter re-sample probability of 0.02 per 100 Hz step
  (one re-sample per 0.5 s on average) as a uniform interval of 0-1 s.
* Reward weights equal the paper's per-step coefficients. Isaac Lab multiplies them by the step time,
  which preserves every ratio between the terms.
* Kp/Kd ranges are the paper's ranges relative to its nominal gains (55, 0.6), applied to Go2's 25 / 0.5,
  so both study policies share one actuator model.
* Height limit: the paper's 0.28 m is ~90% of A1's nominal standing height. Go2 with its softer gains
  stands at ~0.26 m at zero action, so the limit is :data:`MIN_BASE_HEIGHT` = 90% of that. A length-scaled
  0.30 m limit sits above Go2's natural stance and kept episodes at ~8 steps for 40 iterations.
* Payload and center of mass are re-sampled at reset and startup, not mid-episode: Newton changes
  inertial properties through CPU tensors, and runtime CoM changes can destabilize MuJoCo-Warp.
  The CoM range follows RMA's Table 1 (+-0.15 cm), where Fu et al.'s Table 1 is taken from.
* Actions are scaled by 0.25 before the paper's clipping, as in the stock Go2 task.
* Fractal terrain: RaiSim's frequency is relative to its unstated terrain size, so the octave wavelength
  is chosen here (see :func:`study_mdp.fractal_terrain`) to keep slopes inside the 0.2 rad pitch limit.
* Stock Go2 PPO hyperparameters and network, matching the robust teacher, instead of the paper's 3 x 128 MLP.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.velocity_env_cfg import EventsCfg, TerminationsCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp as study_mdp
from .base_env_cfg import BaseRewardsCfg, Go2StudyBaseEnvCfg
from .robust_env_cfg import FEET, FEET_CFG, teacher_terrain_generator

TARGET_SPEED = 0.9
"""Target forward speed [m/s] of the paper's trotting expert."""

LENGTH_SCALE = 0.213 / 0.2
"""Go2 / A1 length scale, from the thigh and calf lengths."""

MASS_SCALE = 15.0 / 12.0
"""Go2 / A1 mass scale."""

MIN_BASE_HEIGHT = 0.9 * 0.26
"""Termination height [m] of the base above the terrain: 90% of Go2's zero-action standing height."""

JOINT_DELTA_LIMITS = {"hip": 0.15, "thigh": 0.4, "calf": 0.4}
"""Paper's limits [rad] on the joint target relative to the standing pose, for HAA, HFE, and KFE."""


@configclass
class EnergyObservationsCfg:
    """Base-policy input: state x_t (30), previous action (12), and extrinsics e_t (19)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- state x_t
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        base_roll_pitch = ObsTerm(func=study_mdp.base_roll_pitch)
        foot_contact = ObsTerm(func=study_mdp.contact_states, params={"sensor_cfg": FEET_CFG})
        # -- previous action a_{t-1}
        actions = ObsTerm(func=mdp.last_action)
        # -- extrinsics e_t
        payload_and_com = ObsTerm(func=study_mdp.payload_and_com)
        motor_strength = ObsTerm(func=study_mdp.motor_strength)
        friction = ObsTerm(func=study_mdp.ground_friction)
        base_velocity = ObsTerm(func=study_mdp.planar_base_velocity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EnergyEventsCfg(EventsCfg):
    """Stock events plus the paper's friction, PD-gain, motor-strength, and payload randomization."""

    # re-sampled during the episode
    foot_friction = EventTerm(
        func=study_mdp.randomize_foot_friction,
        mode="interval",
        interval_range_s=(0.0, 1.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "friction_range": (0.6, 1.2),
            "low_friction_range": (0.6, 1.2),
            "low_friction_prob": 0.0,
            "shared": True,
        },
    )

    pd_gains = EventTerm(
        func=study_mdp.randomize_pd_gains,
        mode="interval",
        interval_range_s=(0.0, 1.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_scale_range": (50.0 / 55.0, 60.0 / 55.0),
            "damping_scale_range": (0.4 / 0.6, 0.8 / 0.6),
            "motor_strength_range": (0.95, 1.05),
        },
    )

    # re-sampled at reset
    add_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 0.5 * MASS_SCALE),
            "operation": "add",
        },
    )


@configclass
class EnergyRewardsCfg(BaseRewardsCfg):
    """Fu et al. 2021 Eq. 3-5. Weights are set in :class:`Go2EnergyTeacherEnvCfg`."""

    # r_forward = -20 |v_x - v_x^target| - v_y^2 - w_yaw^2
    forward_vel_error = RewTerm(func=study_mdp.lin_vel_x_error_l1, weight=0.0, params={"command_name": "base_velocity"})
    lateral_vel = RewTerm(func=study_mdp.lin_vel_y_l2, weight=0.0)
    yaw_rate = RewTerm(func=study_mdp.ang_vel_z_l2, weight=0.0)
    # r_energy = -tau^T q_dot
    energy = RewTerm(func=study_mdp.mechanical_power, weight=0.0)
    # r_alive = 20 * v_x^target
    alive = RewTerm(func=mdp.is_alive, weight=0.0)


@configclass
class EnergyTerminationsCfg(TerminationsCfg):
    """Time-out plus the paper's height, roll, and pitch limits. Base contact is removed in the env config."""

    base_height = DoneTerm(
        func=study_mdp.base_height_below_terrain,
        params={"minimum_height": MIN_BASE_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    bad_orientation = DoneTerm(func=study_mdp.bad_roll_pitch, params={"max_roll": 0.4, "max_pitch": 0.2})


@configclass
class Go2EnergyTeacherEnvCfg(Go2StudyBaseEnvCfg):
    """Go2 energy-efficient base policy after Fu et al. 2021."""

    observations: EnergyObservationsCfg = EnergyObservationsCfg()
    events: EnergyEventsCfg = EnergyEventsCfg()
    rewards: EnergyRewardsCfg = EnergyRewardsCfg()
    terminations: EnergyTerminationsCfg = EnergyTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # -- commands: fixed forward target speed (the paper trains one expert per speed)
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 0.0
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (TARGET_SPEED, TARGET_SPEED)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)

        # -- actions: joint target deltas around the standing pose, clipped per joint type
        self.actions.joint_pos.clip = {
            name: (pos - JOINT_DELTA_LIMITS[joint], pos + JOINT_DELTA_LIMITS[joint])
            for name, pos in UNITREE_GO2_CFG.init_state.joint_pos.items()
            for joint in JOINT_DELTA_LIMITS
            if joint in name
        }

        # -- terrain: "flat" fractal terrain of the structured-gait setting, without curriculum
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,
            sub_terrains={
                "fractal": study_mdp.FractalTerrainCfg(
                    proportion=1.0, amplitude=0.23 * LENGTH_SCALE, base_wavelength=2.0 * LENGTH_SCALE
                ),
            },
        )
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # -- domain randomization
        # MuJoCo-Warp uses max(mu_foot, mu_ground); a low ground value lets foot_friction set the contact friction
        self.scene.terrain.physics_material.static_friction = 0.05
        self.scene.terrain.physics_material.dynamic_friction = 0.05
        self.sim.physics_material = self.scene.terrain.physics_material
        self.events.add_base_mass = None
        # RMA Table 1: +-0.15 cm
        self.events.base_com.params["com_range"] = {"x": (-0.0015, 0.0015), "y": (-0.0015, 0.0015), "z": (0.0, 0.0)}
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # not stated in the paper: start near the standing pose just above the ground. The stock 0.5-1.5 x
        # joint scaling and 0.4 m spawn height make Go2 land below the height limit on its first steps.
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.35)

        # -- terminations: only the paper's rules
        self.terminations.base_contact = None

        # -- rewards: the paper's coefficients (see module docstring)
        # Source: Fu et al. 2021, Eq. 3-5 (A1), r = r_forward + 0.04 r_energy + r_alive. All weights are the paper's
        # per-step coefficients, unscaled for Go2 (Go2/A1 mass 1.25, length 1.07). Isaac Lab multiplies them by the
        # 0.02 s step, which keeps every ratio between the terms.
        rewards = self.rewards
        # r_forward = -20 |v_x - v_x^target| - v_y^2 - w_yaw^2: paper -20, -1, -1
        rewards.forward_vel_error.weight = -20.0
        rewards.lateral_vel.weight = -1.0
        rewards.yaw_rate.weight = -1.0
        # 0.04 r_energy, r_energy = -tau^T q_dot [W] (signed, so regenerative motion is rewarded): paper 0.04
        rewards.energy.weight = -0.04
        # r_alive = 20 v_x^target: paper 20 x the 0.9 m/s target, so standing still scores 0 with r_forward
        rewards.alive.weight = 20.0 * TARGET_SPEED
        # the paper uses no other terms
        for name in (
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "lin_vel_z_l2",
            "ang_vel_xy_l2",
            "dof_torques_l2",
            "dof_acc_l2",
            "action_rate_l2",
            "feet_air_time",
            "undesired_contacts",
            "flat_orientation_l2",
            "dof_pos_limits",
            "base_height_l2",
        ):
            setattr(rewards, name, None)


@configclass
class Go2EnergyTeacherRoughEnvCfg(Go2EnergyTeacherEnvCfg):
    """Fu et al. 2021 energy policy trained on the robust teacher's terrains and terrain curriculum.

    Not in the paper: only the terrain, its curriculum, and the terminations differ from
    :class:`Go2EnergyTeacherEnvCfg`, so the energy and robust policies see the same terrain distribution.
    The paper's height, roll, and pitch limits assume near-flat ground (a stair flight alone pitches the
    base past 0.2 rad), so the robust teacher's base-contact and 1 rad tilt terminations replace them.
    """

    def __post_init__(self):
        super().__post_init__()

        # -- terrain: the robust teacher's families, sorted into difficulty levels
        self.scene.terrain.terrain_generator = teacher_terrain_generator()
        self.scene.terrain.max_init_terrain_level = 5
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # -- terminations: as in the robust teacher
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
        )
        self.terminations.base_height = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
