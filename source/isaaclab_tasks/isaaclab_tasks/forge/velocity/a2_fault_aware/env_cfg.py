# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H6 of the mechanism-detection study: does a failure-mode input help the robust teacher? The teacher's task with
motor faults and stuck feet injected and a 12-d mode observation appended to its input (``REPORT.md``)."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import A2RobustTeacherEnvCfg, TeacherObservationsCfg

from . import mdp

P_FAULT = 0.4
P_STUCK = 0.4
"""Episode probabilities of a motor fault and of a stuck foot, drawn independently (as the surveillance ``-Faults``
collection the detector was trained on): 36 % nominal, 24 % each alone, 16 % both."""
FOOT_STUCK_SEED = 7919
"""The stuck-foot schedule's own seed: with the env seed both injectors draw the same numbers (always together)."""


@configclass
class FaultAwarePolicyCfg(TeacherObservationsCfg.PolicyCfg):
    """The teacher's input (391) plus the mode observation (12), last so the teacher's weights keep their columns."""

    fault_mode = ObsTerm(func=mdp.fault_mode, params={"source": "blind"})


@configclass
class FaultAwareObservationsCfg(TeacherObservationsCfg):
    policy: FaultAwarePolicyCfg = FaultAwarePolicyCfg()


@configclass
class A2FaultAwareEnvCfg(A2RobustTeacherEnvCfg):
    """Robust teacher task with faults; ``mode_source`` fills the mode observation (blind / oracle / detector)."""

    observations: FaultAwareObservationsCfg = FaultAwareObservationsCfg()
    mode_source: str = "blind"

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.fault_mode.params["source"] = self.mode_source
        add_motor_faults(self, MotorFaultInjectorCfg(p_fault=P_FAULT))
        add_foot_stuck(self, FootStuckInjectorCfg(p_stuck=P_STUCK, seed=FOOT_STUCK_SEED))
        # fine-tuning from the trained teacher: its final reward weights and disturbances (curriculum factor 1)
        self.curriculum.curriculum_factor.params["initial_factor"] = 1.0


@configclass
class A2FaultAwareOracleEnvCfg(A2FaultAwareEnvCfg):
    mode_source: str = "oracle"


@configclass
class A2FaultAwareDetectorEnvCfg(A2FaultAwareEnvCfg):
    mode_source: str = "detector"
