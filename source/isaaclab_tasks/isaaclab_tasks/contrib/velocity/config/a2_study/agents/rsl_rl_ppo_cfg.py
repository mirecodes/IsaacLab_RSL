# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...go2_study.agents.rsl_rl_ppo_cfg import Go2RobustTeacherPPORunnerCfg


@configclass
class A2RobustTeacherPPORunnerCfg(Go2RobustTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # same network, PPO hyperparameters and budget as the Go2 teacher
        self.experiment_name = "a2_study_robust_teacher"
        # used only with --logger wandb
        self.wandb_project = "a2_study"
