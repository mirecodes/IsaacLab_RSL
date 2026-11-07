# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The robust teacher's task with faults and a 4-d leg flag: the policy learns to stop relying on the flagged leg
(``REPORT.md``)."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import (
    FEET,
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
