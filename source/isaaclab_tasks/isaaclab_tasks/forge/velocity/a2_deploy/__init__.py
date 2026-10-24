# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment play tasks ``IsaacContrib-Velocity-A2Deploy-<Type>-{Robust,Energy}`` on the terrain maps.

Play-only: the runner configs are the trained tasks' own, so their checkpoints load unchanged. See ``REPORT.md``.
"""

import gymnasium as gym

from isaaclab_tasks.forge.terrain import AGENTS, RUNNERS, TERRAINS


def task_id(terrain: str, policy: str) -> str:
    """``rough_field``, ``Robust`` -> ``IsaacContrib-Velocity-A2Deploy-RoughField-Robust``."""
    return f"IsaacContrib-Velocity-A2Deploy-{terrain.title().replace('_', '')}-{policy}"


def _env_cfg_entry_point(terrain: str, policy: str):
    def make():
        from .env_cfg import make_env_cfg

        return make_env_cfg(terrain, policy)

    return make


for _terrain in TERRAINS:
    for _policy in ("Robust", "Energy"):
        gym.register(
            id=task_id(_terrain, _policy),
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _env_cfg_entry_point(_terrain, _policy),
                "rsl_rl_cfg_entry_point": f"{AGENTS}:{RUNNERS['A2' + _policy]}",
            },
        )
