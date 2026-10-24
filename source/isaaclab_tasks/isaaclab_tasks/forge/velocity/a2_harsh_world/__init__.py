# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harsh World: trained A2 policies on one large grid of hard-terrain cells, with zone labels for fault analysis.

Play-only tasks. The runner configs are the trained tasks' own, so their checkpoints load unchanged.
"""

import gymnasium as gym

AGENTS = "isaaclab_tasks.forge.velocity.a2_velocity.agents.rsl_rl_ppo_cfg"

for variant, env_cfg, runner_cfg in (
    ("Robust", "A2HarshWorldRobustEnvCfg", "A2RobustTeacherPPORunnerCfg"),
    ("Energy", "A2HarshWorldEnergyEnvCfg", "A2EnergyTeacherRoughPPORunnerCfg"),
    ("Robust-Full", "A2HarshWorldRobustFullEnvCfg", "A2RobustTeacherPPORunnerCfg"),
    ("Energy-Full", "A2HarshWorldEnergyFullEnvCfg", "A2EnergyTeacherRoughPPORunnerCfg"),
):
    gym.register(
        id=f"IsaacContrib-Velocity-A2HarshWorld-{variant}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{AGENTS}:{runner_cfg}",
        },
    )
