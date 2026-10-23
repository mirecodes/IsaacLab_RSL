# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class A2StudyBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same network, PPO hyperparameters and budgets as the Go2 study (the stock Go2 rough task's)."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "a2_study_base"
    # used only with --logger wandb
    wandb_project = "a2_study"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class A2RobustTeacherPPORunnerCfg(A2StudyBasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_robust_teacher"
        # harder terrain and the CPG action space need longer training than the stock task
        self.max_iterations = 3000


@configclass
class A2EnergyTeacherPPORunnerCfg(A2StudyBasePPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_energy_teacher"
        # same budget as the robust teacher; the paper trains 15000 iterations of ~100k transitions
        self.max_iterations = 3000


@configclass
class A2EnergyTeacherRoughPPORunnerCfg(A2EnergyTeacherPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "a2_study_energy_teacher_rough"
