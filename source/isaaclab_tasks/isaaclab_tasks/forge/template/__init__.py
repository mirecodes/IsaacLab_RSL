# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template forge task: copy this folder to ``forge/<domain>/<task>`` (see ``forge/AGENT.md``).

The template itself registers nothing, since its empty environment cannot run. In the copy, uncomment and
rename the registration below.
"""

# import gymnasium as gym
#
# from . import agents
#
# gym.register(
#     id="IsaacContrib-<Domain>-<Task>-<Variant>",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.env_cfg:TemplateEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TemplatePPORunnerCfg",
#     },
# )
