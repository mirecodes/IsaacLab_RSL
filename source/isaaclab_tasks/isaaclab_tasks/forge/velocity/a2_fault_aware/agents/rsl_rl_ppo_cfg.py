# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.forge.velocity.a2_velocity.agents.rsl_rl_ppo_cfg import A2RobustTeacherPPORunnerCfg


@configclass
class A2FaultAwarePPORunnerCfg(A2RobustTeacherPPORunnerCfg):
    """The teacher's network and PPO settings; fine-tuned from its checkpoint (input padded with zero weights,
    ``scripts/forge/fault_aware/pad_checkpoint.py``)."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_fault_aware"
        # iterations on top of the teacher's 3000 (chosen here: about 25 min on one GPU)
        self.max_iterations = 1000
