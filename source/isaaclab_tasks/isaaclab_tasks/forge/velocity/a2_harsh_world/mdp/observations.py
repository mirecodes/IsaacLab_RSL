# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zone labels, one observation term per field of the spec's ``zone_logger`` (group ``zone``, not concatenated).

The recorder samples observations right after the physics step and before the step's events, so position-based
labels are looked up here from the current state; the applied friction and the collapse events come from the
:class:`~.events.harsh_world_step` term (the values in effect during this step's physics).

:func:`base_cell`, the group's first term, computes every position-based label at once (:func:`refresh`); the
other terms read them. The lookups are many small kernels, and repeating them per term cost about 3 ms per step.

Condition fields: ``foot_in_sand``, ``foot_mu``, ``foot_low_mu``. Event fields: ``foot_on_rock``, ``rock_moving``,
``collapse_event``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .events import harsh_world_step

EVENT = "harsh_world"


def _term(env: ManagerBasedEnv) -> harsh_world_step:
    return env.event_manager.get_term_cfg(EVENT).func


def refresh(env: ManagerBasedEnv, boundary_distance: float = 0.5, contact_threshold: float = 1.0) -> dict:
    """Compute the position-based labels from the current state into ``harsh_world_step.labels``."""
    term = _term(env)
    zones = env.scene.terrain.zone_map
    base = env.scene["robot"].data.root_link_pos_w.torch[:, :2]
    feet = term.robot.data.body_link_pos_w.torch[:, term.foot_ids]
    at_base, at_feet = zones.lookup(base), zones.lookup(feet[..., :2])
    patch, surface = zones.patch(feet[..., :2])
    r = term.hw.sand.foot_radius
    local = torch.remainder(base - zones.corner, zones.size)
    labels = {
        "base_cell": at_base["cell"],
        "base_property": at_base["property"],
        "base_zone": at_base["zone"],
        "base_difficulty": at_base["difficulty"],
        "boundary": (torch.minimum(local, zones.size - local) < boundary_distance).any(dim=-1),
        "foot_cell": at_feet["cell"],
        "foot_property": at_feet["property"],
        "foot_zone": at_feet["zone"],
        "foot_sand_patch": patch,
        "terrain_height_under_foot": torch.where(patch >= 0, surface, zones.height(feet[..., :2])),
        "foot_in_sand": (at_feet["zone"] == 2)
        & (patch >= 0)
        & (~term.crust[patch] | term.collapsed[patch])
        & (feet[..., 2] < surface + r),
    }
    # rocks: geometric attribution of the feet in contact
    contact = term.contact.data.net_forces_w.torch[:, term.contact_foot_ids].norm(dim=-1) > contact_threshold
    on = torch.zeros_like(contact)
    if len(zones.rocks_static):
        rock = zones.rocks_static
        near = (feet[..., None, :2] - rock[:, :2]).norm(dim=-1) < rock[:, 4] + r
        above = (feet[..., None, 2] - r > rock[:, 2] + 0.01) & (feet[..., None, 2] - r < rock[:, 3] + 0.02)
        on |= (near & above).any(dim=-1)
    moving = torch.zeros_like(contact)
    if term.rocks is not None:
        data = term.rocks.data
        # body origin at the rock's bottom; its center is roughly a quarter size above
        center = data.body_link_pos_w.torch.clone()
        center[..., 2] += term.rock_sizes / 4
        touch = (feet[:, :, None] - center[:, None]).norm(dim=-1) < term.rock_sizes / 2 + r + 0.02  # (N, 4, K)
        speed = data.body_com_lin_vel_w.torch.norm(dim=-1)  # (N, K)
        on |= touch.any(dim=-1)
        moving = (touch & (speed[:, None] > term.hw.rocks.moving_speed)).any(dim=-1)
    labels["foot_on_rock"] = contact & on
    labels["rock_moving"] = contact & moving
    term.labels = labels
    return labels


def base_cell(env: ManagerBasedEnv) -> torch.Tensor:
    """(row, col) under the base, -1 outside the grid. (N, 2) long. Refreshes all position-based labels, so it
    must stay the group's first term."""
    return refresh(env)["base_cell"]


def base_property(env: ManagerBasedEnv) -> torch.Tensor:
    """Property id under the base (names in ``world_meta.yaml:properties``), -1 outside. (N,) long."""
    return _term(env).labels["base_property"]


def base_zone(env: ManagerBasedEnv) -> torch.Tensor:
    """0 rigid, 1 slip, 2 sand. (N,) long."""
    return _term(env).labels["base_zone"]


def base_difficulty(env: ManagerBasedEnv) -> torch.Tensor:
    return _term(env).labels["base_difficulty"]


def boundary(env: ManagerBasedEnv) -> torch.Tensor:
    """Base within 0.5 m of a cell border, where the property label is ambiguous. (N,) bool."""
    return _term(env).labels["boundary"]


def foot_cell(env: ManagerBasedEnv) -> torch.Tensor:
    """(N, 4, 2) long."""
    return _term(env).labels["foot_cell"]


def foot_property(env: ManagerBasedEnv) -> torch.Tensor:
    return _term(env).labels["foot_property"]


def foot_zone(env: ManagerBasedEnv) -> torch.Tensor:
    return _term(env).labels["foot_zone"]


def foot_sand_patch(env: ManagerBasedEnv) -> torch.Tensor:
    """Sand patch id under each foot (``world_meta.yaml:sand_patches``), -1 if none. (N, 4) long."""
    return _term(env).labels["foot_sand_patch"]


def foot_in_sand(env: ManagerBasedEnv) -> torch.Tensor:
    """Foot inside a soft patch, below its surface + foot radius. (N, 4) bool."""
    return _term(env).labels["foot_in_sand"]


def foot_mu(env: ManagerBasedEnv) -> torch.Tensor:
    """Friction applied to each foot. (N, 4) float."""
    return _term(env).foot_mu


def foot_low_mu(env: ManagerBasedEnv) -> torch.Tensor:
    """A low-friction slip sample is active on the foot. (N, 4) bool."""
    return _term(env).low_mu


def spawn_cell(env: ManagerBasedEnv) -> torch.Tensor:
    """(row, col) the robot was spawned in this episode. (N, 2) long."""
    terrain = env.scene.terrain
    return torch.stack([terrain.terrain_levels, terrain.terrain_types], dim=-1)


def terrain_height_under_foot(env: ManagerBasedEnv) -> torch.Tensor:
    """Hardpan height under each foot, or the sand surface over a patch. (N, 4) float."""
    return _term(env).labels["terrain_height_under_foot"]


def foot_on_rock(env: ManagerBasedEnv) -> torch.Tensor:
    """(N, 4) bool."""
    return _term(env).labels["foot_on_rock"]


def rock_moving(env: ManagerBasedEnv) -> torch.Tensor:
    """Foot in contact with a dynamic rock moving faster than ``RockCfg.moving_speed``. (N, 4) bool."""
    return _term(env).labels["rock_moving"]


def collapse_event(env: ManagerBasedEnv) -> torch.Tensor:
    """The crust patch under the foot collapsed in the last step. (N, 4) bool."""
    return _term(env).collapse_event
