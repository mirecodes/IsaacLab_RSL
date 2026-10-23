# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree A2 counterpart of the Go2 study: the robust teacher rescaled to A2."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-A2Study-RobustTeacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.robust_env_cfg:A2RobustTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2RobustTeacherPPORunnerCfg",
    },
)
