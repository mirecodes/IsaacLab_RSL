# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-step A2 walk: the robust teacher's task with a four-beat walk that loads each new foot slowly, and a leg
signal that makes the robot unload that foot and stand on the other three (``REPORT.md``)."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.velocity.a2_velocity.base_env_cfg import A2_MASS, A2_SCALE, A2_STANDING_HEIGHT
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import (
    FEET,
    LEG_JOINTS,
    A2RobustTeacherEnvCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
)

from . import mdp

FEET_CFG = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
NOMINAL_FOOT_LOAD = A2_MASS * 9.81 / 4
"""A2's weight on four feet [N]: the full load of a foot, F_0."""
LOAD_S = 0.25
"""Loading time [s] of a new foot (user's choice): about the detector's median latency (0.16-0.22 s) plus margin."""
SWING_S = 0.25
"""Swing time [s], chosen here: the teacher's trot swings for 0.37 s at ~3x the speed."""
P_FAULT = 0.25
P_STUCK = 0.25
"""Episode probabilities of a motor fault and a stuck foot, chosen here: below a2_fault_aware's 0.4, since a permanent
fault keeps the robot standing to the end of the episode."""
FOOT_STUCK_SEED = 7919
"""As ``a2_fault_aware``: with the env seed both injectors would draw the same numbers."""
SPEED_SCALE = 0.2
"""Command range / the teacher's (+-1 m/s): the walk's cycle is 2 s, so a 0.4 m stance stroke allows ~0.2 m/s."""


@configclass
class SoftStepPolicyCfg(TeacherObservationsCfg.PolicyCfg):
    """The teacher's input (391; ``cpg_phase`` now reads this task's walk clock, its phase-action part is zero) plus
    the signal (4)."""

    leg_flag = ObsTerm(func=mdp.leg_flag)


@configclass
class SoftStepObservationsCfg(TeacherObservationsCfg):
    policy: SoftStepPolicyCfg = SoftStepPolicyCfg()


@configclass
class SoftStepRewardsCfg(TeacherRewardsCfg):
    load_ramp = RewTerm(
        func=mdp.load_ramp,
        # chosen here: a new foot at full load from touchdown costs 2 x (1 - tau / T) per second over T, as
        # a2_leg_guard's load term
        weight=-2.0,
        params={"sensor_cfg": FEET_CFG, "load_s": LOAD_S, "nominal_load": NOMINAL_FOOT_LOAD},
    )
    flagged_foot_contact = RewTerm(
        func=mdp.flagged_foot_contact,
        # the user's discrete penalty: 1 + F_z / F_0 from 5 N on; 1.5-4.5 per second, about what the three tracking
        # terms pay at best (3 x 1.5)
        weight=-1.5,
        params={"sensor_cfg": FEET_CFG, "nominal_load": NOMINAL_FOOT_LOAD, "threshold": 5.0},
    )
    # chosen here: a fall costs about one second of the best return (-200 x 0.02 s), so falling never beats a touch
    termination = RewTerm(func=base_mdp.is_terminated, weight=-200.0)


@configclass
class A2SoftStepEnvCfg(A2RobustTeacherEnvCfg):
    """Robust teacher with the soft-step walk, the leg signal, faults and a stop command while signalled."""

    observations: SoftStepObservationsCfg = SoftStepObservationsCfg()
    rewards: SoftStepRewardsCfg = SoftStepRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # -- actions: the walk gait; leg geometry and swing height as the teacher
        teacher = self.actions.joint_pos
        self.actions.joint_pos = mdp.SoftStepCPGActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINTS,
            thigh_length=teacher.thigh_length,
            calf_length=teacher.calf_length,
            swing_s=SWING_S,
            load_s=LOAD_S,
            swing_height=teacher.swing_height,
            residual_scale=teacher.residual_scale,
        )
        # -- commands: zero while signalled; slower, matching the walk
        base = self.commands.base_velocity
        self.commands.base_velocity = mdp.StopOnFlagVelocityCommandCfg(
            **{k: getattr(base, k) for k in vars(base) if k != "class_type"}
        )
        ranges = self.commands.base_velocity.ranges
        ranges.lin_vel_x = (-SPEED_SCALE, SPEED_SCALE)
        ranges.lin_vel_y = (-0.5 * SPEED_SCALE, 0.5 * SPEED_SCALE)
        # chosen here: 0.4 rad/s, twice the linear scale, so a turn in place is still a few steps
        ranges.ang_vel_z = (-0.4, 0.4)
        # tracking std scaled with the command range, else standing still earns ~90 % of the moving reward
        lin_std = (A2_SCALE**0.5) * SPEED_SCALE
        self.rewards.track_lin_vel.params["std"] = lin_std
        self.rewards.lin_vel_orthogonal.params["std"] = lin_std
        self.rewards.track_ang_vel.params["std"] = (A2_SCALE**0.5) * 0.4
        # -- faults, then the signal that reads them (events run in this order)
        add_motor_faults(self, MotorFaultInjectorCfg(p_fault=P_FAULT))
        add_foot_stuck(self, FootStuckInjectorCfg(p_stuck=P_STUCK, seed=FOOT_STUCK_SEED))
        self.events.leg_signal = EventTerm(
            func=mdp.leg_signal,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={
                "sensor_cfg": FEET_CFG,
                # chosen here: ~40 touchdowns per 20 s episode, so ~0.8 random signals per episode
                "p_touchdown": 0.02,
                "load_s": LOAD_S,
                # chosen here: long enough to settle on three legs, short enough to practice resuming
                "hold_s": (1.0, 3.0),
                # as a2_leg_guard: the detection study's latency (median 0.16-0.22 s, p90 0.66 s) and a miss share
                "latency_s": (0.1, 0.5),
                "p_miss": 0.15,
            },
        )
        # -- terrain: at ~0.2 m/s a 20 s episode cannot cross half a tile (4 m), the distance-based level curriculum
        # would never promote; robots keep their initial levels 0-5 of 10
        self.curriculum.terrain_levels = None


@configclass
class A2SoftStepBodyContactEnvCfg(A2SoftStepEnvCfg):
    """:class:`A2SoftStepEnvCfg` (CPG walk) that ends the episode when any link but a foot touches anything."""

    def __post_init__(self):
        super().__post_init__()
        # -- terminations: the user's rule after soft_s0 lay belly down: any non-foot link in contact ends the
        # episode; the teacher's 1 N threshold; replaces base_contact, which covered base_link only
        self.terminations.base_contact = None
        self.terminations.body_contact = DoneTerm(
            func=base_mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot$).*"), "threshold": 1.0},
        )


@configclass
class NoCpgPolicyCfg(SoftStepPolicyCfg):
    """The soft-step input without the CPG terms, plus the last action (12)."""

    cpg_phase = None
    joint_target = None
    # the teacher's target-error term reads CPG-only attributes; target = q_0 + 0.25 a follows from the last action
    joint_pos_error = None
    last_action = ObsTerm(func=base_mdp.last_action)


@configclass
class NoCpgObservationsCfg(TeacherObservationsCfg):
    policy: NoCpgPolicyCfg = NoCpgPolicyCfg()


@configclass
class A2SoftStepNoCpgEnvCfg(A2SoftStepEnvCfg):
    """Soft step without the CPG: joint position offsets around the default pose, no gait clock. Rewards, signal,
    faults and commands as :class:`A2SoftStepEnvCfg`, plus a low-base termination."""

    observations: NoCpgObservationsCfg = NoCpgObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # -- actions: the A2 base task's joint offsets around the default pose
        self.actions.joint_pos = base_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
        )
        # -- rewards: the teacher's target smoothness reads CPG-only attributes; the A2 base task's action rate
        # (sum (a_t - a_{t-1})^2, weight -0.005) instead, under the same name so the curriculum still scales it
        self.rewards.target_smoothness = RewTerm(func=base_mdp.action_rate_l2, weight=-0.005)
        # -- terminations: soft_s0 learned to lie belly down with the feet unloaded (no foot penalty, base level, no
        # base_link contact reported); chosen here: end the episode below half the standing height above the terrain
        self.terminations.base_low = DoneTerm(
            func=mdp.base_below_terrain_height,
            params={"minimum_height": 0.5 * A2_STANDING_HEIGHT, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
