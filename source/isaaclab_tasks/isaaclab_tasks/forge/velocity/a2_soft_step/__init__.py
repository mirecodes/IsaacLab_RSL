# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A2 walk that loads each new foot slowly and stands on three legs while a leg is signalled:
``IsaacContrib-Velocity-A2SoftStep``. See ``REPORT.md``."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Velocity-A2SoftStep",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:A2SoftStepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2SoftStepPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2SoftStep-BodyContact",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:A2SoftStepBodyContactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2SoftStepBodyContactPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-A2SoftStep-NoCPG",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:A2SoftStepNoCpgEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2SoftStepNoCpgPPORunnerCfg",
    },
)
