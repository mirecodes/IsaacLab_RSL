# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust A2 teacher with motor faults and stuck feet and a failure-mode input: ``IsaacContrib-Velocity-A2FaultAware-
{Blind,Oracle,Detector}`` (H6 of the mechanism-detection study). See ``REPORT.md``."""

import gymnasium as gym

from . import agents

for _variant in ("Blind", "Oracle", "Detector"):
    gym.register(
        id=f"IsaacContrib-Velocity-A2FaultAware-{_variant}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:A2FaultAware{'' if _variant == 'Blind' else _variant}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2FaultAwarePPORunnerCfg",
        },
    )
