# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Soft-step walk: a four-beat CPG walk with a four-foot loading pause after every touchdown, and its load reward.

Copied from ``velocity/a2_velocity/mdp/actions.py`` (CPG action of Miki et al. 2022) and changed (``REPORT.md`` §3.1):

* Gait clock: each leg swings for ``swing_s``, then all four feet stay down for ``load_s`` while the new foot takes
  its load; then the next leg swings, in the lateral sequence RL, FL, RR, FR. Cycle ``4 (swing_s + load_s)``.
* No phase action: the action is the 12 residual joint targets only, so the legs keep their spacing.
* While a leg is flagged (``env.leg_flag``) the clock stops: legs in stance hold, a leg in swing finishes it (unless it
  is the flagged one). When the flag clears, the phases restart from the walk pattern with the flagged leg's swing,
  so it is put down again and reloaded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTermCfg, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .signal import _t, get_flag

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

WALK_ORDER = (2, 0, 3, 1)
"""Swing order as leg indices in (FL, FR, RL, RR): RL, FL, RR, FR (lateral-sequence walk)."""


class SoftStepCPGAction(ActionTerm):
    """Walk-gait phase generator with IK foot trajectories plus residual joint position targets (12)."""

    cfg: SoftStepCPGActionCfg
    _asset: Articulation

    def __init__(self, cfg: SoftStepCPGActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        self._num_legs = len(self._joint_names) // 3
        if self._num_legs != 4 or len(self._joint_names) != 12:
            raise ValueError(f"Expected (hip, thigh, calf) for FL, FR, RL, RR, got {self._joint_names}.")

        self._default_joint_pos = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        default = self._default_joint_pos.view(self.num_envs, self._num_legs, 3)
        self._default_hip = default[..., 0]
        # nominal foot position in each thigh joint's sagittal plane (x forward, z up)
        l1, l2 = cfg.thigh_length, cfg.calf_length
        thigh, calf = default[..., 1], default[..., 2]
        self._foot_x = -l1 * torch.sin(thigh) - l2 * torch.sin(thigh + calf)
        self._foot_z = -l1 * torch.cos(thigh) - l2 * torch.cos(thigh + calf)

        slot = cfg.swing_s + cfg.load_s
        self._base_frequency = 1.0 / (4.0 * slot)
        self._swing_end = 2.0 * math.pi * cfg.swing_s / (4.0 * slot)
        self._base_delta_phase = 2.0 * math.pi * self._base_frequency * env.step_dt
        # the k-th leg in WALK_ORDER starts its swing k slots after the first
        pattern = torch.zeros(self._num_legs, device=self.device)
        for k, leg in enumerate(WALK_ORDER):
            pattern[leg] = (-k * 0.5 * math.pi) % (2.0 * math.pi)
        self._pattern = pattern
        self._phase = pattern.repeat(self.num_envs, 1)
        self._delta_phase = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        self._paused_leg = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._joint_targets = self._default_joint_pos.clone()
        self._prev_joint_targets = self._default_joint_pos.clone()
        self._prev_prev_joint_targets = self._default_joint_pos.clone()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3 * self._num_legs

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint position targets [rad], ordered as :attr:`SoftStepCPGActionCfg.joint_names`."""
        return self._joint_targets

    @property
    def joint_ids(self) -> torch.Tensor:
        return self._joint_ids

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._default_joint_pos

    @property
    def phase(self) -> torch.Tensor:
        """Leg phases [rad] in ``[0, 2*pi)``; swing while below ``2 pi swing_s / cycle``. Shape (num_envs, 4)."""
        return self._phase

    @property
    def delta_phase(self) -> torch.Tensor:
        """Always zero (no phase action); kept for the teacher's ``cpg_phase`` observation."""
        return self._delta_phase

    @property
    def base_frequency(self) -> float:
        return self._base_frequency

    @property
    def prev_joint_targets(self) -> torch.Tensor:
        return self._prev_joint_targets

    @property
    def prev_prev_joint_targets(self) -> torch.Tensor:
        return self._prev_prev_joint_targets

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        flag = get_flag(self._env) > 0
        paused = flag.any(1)
        # flag cleared: restart the walk pattern at the flagged leg's swing
        cleared = (self._paused_leg >= 0) & ~paused
        if cleared.any():
            leg = self._paused_leg[cleared]
            self._phase[cleared] = torch.remainder(self._pattern - self._pattern[leg].unsqueeze(1), 2.0 * math.pi)
        self._paused_leg = torch.where(paused, flag.float().argmax(1), -1)
        # advance the clock; while paused only a swing of another leg runs on to its touchdown
        in_swing = self._phase < self._swing_end
        advance = ~paused.unsqueeze(1) | (in_swing & ~flag)
        self._phase = torch.remainder(self._phase + self._base_delta_phase * advance, 2.0 * math.pi)
        # nominal joint targets from the foot trajectory
        foot_z = self._foot_z + self.cfg.swing_height * self._swing_profile(self._phase)
        thigh, calf = self._sagittal_ik(self._foot_x, foot_z)
        nominal = torch.stack((self._default_hip, thigh, calf), dim=-1).view(self.num_envs, -1)
        # shift the target history used by the smoothness reward
        self._prev_prev_joint_targets[:] = self._prev_joint_targets
        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets = nominal + actions * self.cfg.residual_scale

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._phase[env_ids] = self._pattern
        self._paused_leg[env_ids] = -1
        self._joint_targets[env_ids] = self._default_joint_pos[env_ids]
        self._prev_joint_targets[env_ids] = self._default_joint_pos[env_ids]
        self._prev_prev_joint_targets[env_ids] = self._default_joint_pos[env_ids]

    """
    Helpers.
    """

    def _swing_profile(self, phase: torch.Tensor) -> torch.Tensor:
        """Normalized foot lift in ``[0, 1]``: Hermite spline up over the first half of the swing, down over the
        second."""
        u = (phase / (0.5 * self._swing_end)).clamp(max=2.0)
        s = torch.where(u <= 1.0, u, 2.0 - u)
        lift = 3.0 * s**2 - 2.0 * s**3
        return torch.where(phase < self._swing_end, lift, torch.zeros_like(lift))

    def _sagittal_ik(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Thigh and calf angles [rad] placing the foot at ``(x, z)`` from the thigh joint, knee bent backward."""
        l1, l2 = self.cfg.thigh_length, self.cfg.calf_length
        dist_sq = (x**2 + z**2).clamp(max=(0.999 * (l1 + l2)) ** 2)
        cos_calf = ((dist_sq - l1**2 - l2**2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0)
        calf = -torch.acos(cos_calf)
        thigh = torch.atan2(-x, -z) - torch.atan2(l2 * torch.sin(calf), l1 + l2 * torch.cos(calf))
        return thigh, calf


@configclass
class SoftStepCPGActionCfg(ActionTermCfg):
    """Configuration for :class:`SoftStepCPGAction`."""

    class_type: type[ActionTerm] = SoftStepCPGAction

    joint_names: list[str] = MISSING
    """Joint names ordered leg by leg (FL, FR, RL, RR) as (hip, thigh, calf)."""

    thigh_length: float = MISSING
    """Distance [m] from the thigh joint to the calf joint."""

    calf_length: float = MISSING
    """Distance [m] from the calf joint to the foot center."""

    swing_s: float = 0.25
    """Swing time [s] of one leg."""

    load_s: float = 0.25
    """Four-foot loading time [s] after each touchdown, before the next leg swings."""

    swing_height: float = 0.2
    """Peak foot lift [m] above the nominal stance position."""

    residual_scale: float = 0.25
    """Residual joint target [rad] per unit action."""


def load_ramp(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, load_s: float, nominal_load: float) -> torch.Tensor:
    """Load taken too fast by a new foot: ``sum_l min(max(F_z - (tau / T) F_0, 0) / F_0, 2)`` over unflagged feet in
    their first ``T = load_s`` of contact (``tau``: contact time)."""
    data = env.scene.sensors[sensor_cfg.name].data
    fz = _t(data.net_forces_w)[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    tau = _t(data.current_contact_time)[:, sensor_cfg.body_ids]
    allowed = (tau / load_s).clamp(max=1.0) * nominal_load
    excess = ((fz - allowed).clamp(min=0.0) / nominal_load).clamp(max=2.0)
    return (excess * (tau < load_s) * (1.0 - get_flag(env))).sum(1)
