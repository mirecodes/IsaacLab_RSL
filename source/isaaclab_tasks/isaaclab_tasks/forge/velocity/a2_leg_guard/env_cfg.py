# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robust teacher's task with faults and a 4-d leg flag: the policy learns to stop relying on the flagged leg
(``REPORT.md``)."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import (
    FEET,
    LEG_JOINTS,
    A2RobustTeacherEnvCfg,
    TeacherObservationsCfg,
    TeacherRewardsCfg,
)

from . import mdp

P_FAULT = 0.4
P_STUCK = 0.4
"""Episode probabilities, as ``a2_fault_aware`` (the practiced policy this task starts from)."""
FOOT_STUCK_SEED = 7919
"""As ``a2_fault_aware``: with the env seed both injectors would draw the same numbers."""
NOMINAL_FOOT_LOAD = 40.07 * 9.81 / 4
"""A2's weight on four feet [N]."""


@configclass
class LegGuardPolicyCfg(TeacherObservationsCfg.PolicyCfg):
    """The teacher's input (391) plus the flag (4), last so the teacher's weights keep their columns."""

    leg_flag = ObsTerm(
        func=mdp.leg_flag,
        params={
            # the detection study's latency: median 0.16-0.22 s, p90 0.66 s
            "latency_s": (0.1, 0.5),
            # chosen here: the flag-free branch stays practiced, and the policy cannot count on the flag
            "p_miss": 0.15,
            # chosen here: a phantom in 30 % of the episodes, 1-3 s; also the practice for terrain causes
            "p_phantom": 0.3,
            "phantom_start_s": (1.0, 15.0),
            "phantom_hold_s": (1.0, 3.0),
        },
    )


@configclass
class LegGuardObservationsCfg(TeacherObservationsCfg):
    policy: LegGuardPolicyCfg = LegGuardPolicyCfg()


@configclass
class LegGuardRewardsCfg(TeacherRewardsCfg):
    flagged_foot_load = RewTerm(
        func=mdp.flagged_foot_load,
        # chosen here: a flagged stance foot costs as much as the tracking terms pay (3 x 1.5 per second at best)
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "nominal_load": NOMINAL_FOOT_LOAD,
        },
    )


@configclass
class A2LegGuardSoftEnvCfg(A2RobustTeacherEnvCfg):
    """The flagged foot costs reward: the policy finds its own way to unload the leg."""

    observations: LegGuardObservationsCfg = LegGuardObservationsCfg()
    rewards: LegGuardRewardsCfg = LegGuardRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        add_motor_faults(self, MotorFaultInjectorCfg(p_fault=P_FAULT))
        add_foot_stuck(self, FootStuckInjectorCfg(p_stuck=P_STUCK, seed=FOOT_STUCK_SEED))
        # fine-tuning from a trained policy: the teacher's final reward weights and disturbances
        self.curriculum.curriculum_factor.params["initial_factor"] = 1.0


@configclass
class A2LegGuardTuckEnvCfg(A2LegGuardSoftEnvCfg):
    """The flagged foot is pulled back by the action term: the policy learns to balance without it."""

    def __post_init__(self):
        super().__post_init__()
        base = self.actions.joint_pos
        self.actions.joint_pos = mdp.GuardCPGActionCfg(**{k: getattr(base, k) for k in vars(base) if k != "class_type"})


@configclass
class StandPolicyCfg(TeacherObservationsCfg.PolicyCfg):
    """The teacher's input without the CPG terms, plus the last action (12) and the flag (4)."""

    cpg_phase = None
    joint_target = None
    # the teacher's target-error term reads CPG-only attributes; target = q_0 + 0.25 a follows from the last action
    joint_pos_error = None
    last_action = ObsTerm(func=base_mdp.last_action)
    leg_flag = ObsTerm(
        func=mdp.leg_flag,
        params={
            # no injectors: every flag is a scheduled one, without latency or misses
            "latency_s": (0.0, 0.0),
            "p_miss": 0.0,
            # chosen here: 80 % of the episodes flag one leg; 20 % stay on four legs
            "p_phantom": 0.8,
            # chosen here: on after 0.5-8 s, off after 3-15 s (past the 20 s episode = until the end), so the policy
            # practices lifting, holding and putting the foot back down
            "phantom_start_s": (0.5, 8.0),
            "phantom_hold_s": (3.0, 15.0),
        },
    )


@configclass
class StandObservationsCfg(TeacherObservationsCfg):
    policy: StandPolicyCfg = StandPolicyCfg()


@configclass
class StandRewardsCfg(TeacherRewardsCfg):
    flagged_foot_contact = RewTerm(
        func=mdp.flagged_foot_contact,
        # chosen here: 1.5-4.5 per second while the flagged foot touches (F_z > 5 N), about what the three tracking
        # terms pay at best (3 x 1.5), so a flagged foot on the ground halves the return or worse
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            "nominal_load": NOMINAL_FOOT_LOAD,
            "threshold": 5.0,
        },
    )
    # chosen here: a fall costs about one second of the best return (-200 x 0.02 s), so falling never beats a touch
    termination = RewTerm(func=base_mdp.is_terminated, weight=-200.0)


@configclass
class A2LegGuardStandEnvCfg(A2RobustTeacherEnvCfg):
    """Stand still; while a leg is flagged, hold its foot off the ground and balance on the other three.

    Trained from scratch with joint-position actions: the CPG keeps every leg stepping, which a stance cannot use.
    """

    observations: StandObservationsCfg = StandObservationsCfg()
    rewards: StandRewardsCfg = StandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # -- commands: stand still
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_standing_envs = 1.0
        command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0.0, 0.0)
        # -- actions: the base task's joint offsets around the default pose
        self.actions.joint_pos = base_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINTS, scale=0.25, use_default_offset=True, preserve_order=True
        )
        # -- rewards: the teacher's target smoothness reads CPG-only attributes; the A2 base task's action rate
        # (sum (a_t - a_{t-1})^2, weight -0.005) instead, under the same name so the curriculum still scales it
        self.rewards.target_smoothness = RewTerm(func=base_mdp.action_rate_l2, weight=-0.005)
        # -- terrain: no distance-based levels when standing; robots keep their initial levels 0-5 of 10
        self.curriculum.terrain_levels = None
