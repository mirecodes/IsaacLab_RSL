# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the Miki et al. 2022 teacher environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class exponential_curriculum_factor(ManagerTermBase):
    """Scale selected reward weights and event ranges by ``c_k = c_0^(d^k)`` (Miki et al. 2022, Sec. S3).

    As in the paper, ``k`` counts training episodes: the number of completed episodes per environment,
    averaged over environments. ``c_k`` rises from ``c_0`` towards 1. Reward weights and event ranges
    are read from the config at construction and treated as the values at ``c = 1``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._reward_weights = {
            name: env.reward_manager.get_term_cfg(name).weight for name in cfg.params["reward_terms"]
        }
        self._event_ranges = {
            (event, param): tuple(env.event_manager.get_term_cfg(event).params[param])
            for event, params in cfg.params["event_ranges"].items()
            for param in params
        }
        self._episodes_per_env = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        initial_factor: float,
        decay: float,
        reward_terms: list[str],
        event_ranges: dict[str, list[str]],
    ) -> float:
        # the curriculum manager runs on reset, once per finished episode of each env in env_ids
        if env.common_step_counter > 0:
            num_reset = env.num_envs if env_ids is None else len(env_ids)
            self._episodes_per_env += num_reset / env.num_envs
        factor = initial_factor ** (decay**self._episodes_per_env)
        for name, weight in self._reward_weights.items():
            term_cfg = env.reward_manager.get_term_cfg(name)
            term_cfg.weight = weight * factor
            env.reward_manager.set_term_cfg(name, term_cfg)
        for (event, param), (low, high) in self._event_ranges.items():
            env.event_manager.get_term_cfg(event).params[param] = (low * factor, high * factor)
        return factor
