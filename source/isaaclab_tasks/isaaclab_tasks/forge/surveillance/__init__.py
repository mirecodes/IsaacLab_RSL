# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment play tasks ``IsaacContrib-Surveillance-<Type>-{Robust,Energy}`` on the terrain maps: the 9 types of
``forge/terrain`` plus :data:`EXTRA_TERRAINS`, 20 tasks.

Play-only: the runner configs are the trained tasks' own, so their checkpoints load unchanged. See ``REPORT.md``.
"""

import gymnasium as gym

from isaaclab_tasks.forge.terrain import AGENTS, RUNNERS, TERRAINS

EXTRA_TERRAINS = ("slip_field",)
"""Terrain types this task adds to ``forge/terrain``'s catalog (``env_cfg.extra_templates``), listed here so that
registration imports nothing."""


def task_id(terrain: str, policy: str) -> str:
    """``rough_field``, ``Robust`` -> ``IsaacContrib-Surveillance-RoughField-Robust``."""
    return f"IsaacContrib-Surveillance-{terrain.title().replace('_', '')}-{policy}"


def _env_cfg_entry_point(terrain: str, policy: str):
    def make():
        from .env_cfg import make_env_cfg

        return make_env_cfg(terrain, policy)

    return make


for _terrain in (*TERRAINS, *EXTRA_TERRAINS):
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
