# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust A2 policy that stops relying on a flagged leg: ``IsaacContrib-Velocity-A2LegGuard-{Soft,Tuck}``. See
``REPORT.md``."""

import gymnasium as gym

from . import agents

for _variant in ("Soft", "Tuck"):
    gym.register(
        id=f"IsaacContrib-Velocity-A2LegGuard-{_variant}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:A2LegGuard{_variant}EnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:A2LegGuardPPORunnerCfg",
        },
    )
