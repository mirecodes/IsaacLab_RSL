# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg


@configclass
class Go2StudyBasePPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_study_base"
        # used only with --logger wandb
        self.wandb_project = "go2_study"


@configclass
class Go2RobustTeacherPPORunnerCfg(Go2StudyBasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_study_robust_teacher"
        # harder terrain and the CPG action space need longer training than the stock task
        self.max_iterations = 3000


@configclass
class Go2EnergyTeacherPPORunnerCfg(Go2StudyBasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_study_energy_teacher"
        # same budget as the robust teacher; the paper trains 15000 iterations of ~100k transitions
        self.max_iterations = 3000


@configclass
class Go2EnergyTeacherRoughPPORunnerCfg(Go2EnergyTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_study_energy_teacher_rough"
