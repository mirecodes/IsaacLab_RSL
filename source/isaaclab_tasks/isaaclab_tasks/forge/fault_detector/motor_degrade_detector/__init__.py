# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor-degradation detector: data collection task with the frozen A2 robust teacher (see ``REPORT.md``).

The detector itself is trained offline by ``scripts/forge/fault_detector``. Play-only: the runner config is the
trained task's, so its checkpoint loads unchanged.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-FaultDetector-MotorDegrade-A2Robust",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MotorDegradeA2RobustEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.forge.velocity.a2_velocity.agents.rsl_rl_ppo_cfg:A2RobustTeacherPPORunnerCfg"
        ),
    },
)
