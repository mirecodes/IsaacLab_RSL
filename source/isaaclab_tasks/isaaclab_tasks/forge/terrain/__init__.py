# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain templates and one play task per (terrain type, trained policy), ``IsaacContrib-Terrain-<Type>-<Policy>``.

Each task's map holds one terrain type: 5 difficulty rows x 5 tiles by default. Play-only: the runner configs are
the trained tasks' own, so their checkpoints load unchanged. See ``REPORT.md`` and ``templates/__init__.py``.
"""

import gymnasium as gym

TERRAINS = (
    "rough_field",
    "pit_field",
    "stub_field",
    "rough_slope",
    "rough_stairs",
    "slip_slope",
    "sand_field",
    "gravel_field",
    "crust_collapse",
)
"""Terrain types: the keys of ``world_cfg.default_terrains``, listed here so that registration imports nothing."""

AGENTS = "isaaclab_tasks.forge.velocity.a2_velocity.agents.rsl_rl_ppo_cfg"
RUNNERS = {"A2Robust": "A2RobustTeacherPPORunnerCfg", "A2Energy": "A2EnergyTeacherRoughPPORunnerCfg"}


def task_id(terrain: str, policy: str) -> str:
    """``rough_field``, ``A2Robust`` -> ``IsaacContrib-Terrain-RoughField-A2Robust``."""
    return f"IsaacContrib-Terrain-{terrain.title().replace('_', '')}-{policy}"


def _env_cfg_entry_point(terrain: str, policy: str):
    def make():
        from .env_cfg import make_env_cfg

        return make_env_cfg(terrain, policy)

    return make


for _terrain in TERRAINS:
    for _policy, _runner in RUNNERS.items():
        gym.register(
            id=task_id(_terrain, _policy),
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _env_cfg_entry_point(_terrain, _policy),
                "rsl_rl_cfg_entry_point": f"{AGENTS}:{_runner}",
            },
        )
