# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A2 counterpart of the Go2 study: the base, robust, and energy-efficient tasks rescaled to A2."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-A2Study-Base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:A2StudyBaseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2StudyBasePPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2Study-Steering",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.steering_env_cfg:A2SteeringEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2SteeringPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2Study-RobustTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_env_cfg:A2RobustTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2RobustTeacherPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2Study-EnergyTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.energy_env_cfg:A2EnergyTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2EnergyTeacherPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2Study-EnergyTeacherRough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.energy_env_cfg:A2EnergyTeacherRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2EnergyTeacherRoughPPORunnerCfg",
    },
)
