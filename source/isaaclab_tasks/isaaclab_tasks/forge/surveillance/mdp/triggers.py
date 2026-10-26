# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event triggers: per robot, whether its terrain event is acting on it this step (bool [num_envs]).

Read by ``scripts/forge/surveillance/collect.py``, which records a window around each rising edge. They read the
state of ``forge/terrain``'s terrain event term (``terrain_step``), which is one policy step old at that point.
"""

from __future__ import annotations

import torch

from isaaclab_tasks.forge.terrain.mdp.events import EVENT


def _terrain(env):
    return env.event_manager.get_term_cfg(EVENT).func


def crust_collapse(env) -> torch.Tensor:
    """The crust patch under one of the robot's feet collapsed."""
    return _terrain(env).collapse_event.any(dim=1)


def sand_sink(env) -> torch.Tensor:
    """A foot is in soft sand: below the surface of an ordinary or collapsed patch (the terrain's sand-drag rule)."""
    return _terrain(env).in_sand.any(dim=1)


def foot_slip(env, threshold: float = 1.0) -> torch.Tensor:
    """A foot that touched down with a low-friction sample is loaded: vertical contact force above ``threshold`` [N].
    The rising edge is the slippery touchdown."""
    term = _terrain(env)
    loaded = term.contact.data.net_forces_w.torch[:, term.contact_foot_ids, 2].abs() > threshold
    return (term.low_mu & loaded).any(dim=1)


def stone_step(env, threshold: float = 1.0, clearance: float = 0.02) -> torch.Tensor:
    """A loaded foot stands on one of the robot's stones: vertical contact force above ``threshold`` [N], within the
    stone's circle bound (half its length, as the terrain's stone recycling uses) and ``clearance`` [m] above the
    ground under it."""
    term = _terrain(env)
    feet = term.robot.data.body_link_pos_w.torch[:, term.foot_ids]  # (N, F, 3)
    loaded = term.contact.data.net_forces_w.torch[:, term.contact_foot_ids, 2].abs() > threshold
    stones = term.rocks.data.body_link_pose_w.torch[..., :2]  # (N, K, 2)
    # private access: the stone lengths of the terrain event (K,)
    near = ((feet[:, :, None, :2] - stones[:, None]).norm(dim=-1) < term._rock_sizes / 2).any(dim=-1)
    raised = feet[..., 2] > term.map.height(feet[..., :2]) + term.world.sand.foot_radius + clearance
    return (loaded & near & raised).any(dim=1)


def random_walk(env, period_s: float = 5.0) -> torch.Tensor:
    """No event: fires at random, on average once per ``period_s`` per robot, to sample normal walking."""
    return torch.rand(env.num_envs, device=env.device) < env.step_dt / period_s
