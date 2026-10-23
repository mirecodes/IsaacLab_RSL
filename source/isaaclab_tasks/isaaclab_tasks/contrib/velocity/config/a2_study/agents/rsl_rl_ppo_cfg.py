# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from ...go2_study.agents.rsl_rl_ppo_cfg import (
    Go2EnergyTeacherPPORunnerCfg,
    Go2EnergyTeacherRoughPPORunnerCfg,
    Go2RobustTeacherPPORunnerCfg,
    Go2StudyBasePPORunnerCfg,
)

# same network, PPO hyperparameters and budgets as the Go2 study


@configclass
class A2StudyBasePPORunnerCfg(Go2StudyBasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_base"
        # used only with --logger wandb
        self.wandb_project = "a2_study"


@configclass
class A2RobustTeacherPPORunnerCfg(Go2RobustTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_robust_teacher"
        self.wandb_project = "a2_study"


@configclass
class A2EnergyTeacherPPORunnerCfg(Go2EnergyTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_energy_teacher"
        self.wandb_project = "a2_study"


@configclass
class A2EnergyTeacherRoughPPORunnerCfg(Go2EnergyTeacherRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_energy_teacher_rough"
        self.wandb_project = "a2_study"
