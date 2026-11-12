# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.forge.velocity.a2_velocity.agents.rsl_rl_ppo_cfg import A2RobustTeacherPPORunnerCfg


@configclass
class A2SoftStepPPORunnerCfg(A2RobustTeacherPPORunnerCfg):
    """The teacher's network, PPO settings and budget (3000 iterations), from scratch: the action space is new."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_soft_step"


@configclass
class A2SoftStepBodyContactPPORunnerCfg(A2SoftStepPPORunnerCfg):
    """As :class:`A2SoftStepPPORunnerCfg`, for the body-contact termination."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "a2_soft_step_bodycontact"


@configclass
class A2SoftStepNoCpgPPORunnerCfg(A2SoftStepPPORunnerCfg):
    """As :class:`A2SoftStepPPORunnerCfg`, for the joint-position action."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "a2_soft_step_nocpg"
