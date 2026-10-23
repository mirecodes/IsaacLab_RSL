# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPG-based joint position action from Miki et al. 2022 (Science Robotics, Sec. S4).

Each leg ``l`` keeps a phase ``phi_l`` in ``[0, 2*pi)``. The policy outputs a per-leg phase offset
``dphi_l`` and a residual joint target ``dq``; the joint target is

    q = IK(p_l(phi_l)) + dq,  with  phi_l <- phi_l + dphi_0 + dphi_l  every policy step,

where ``dphi_0 = 2*pi*f_0*dt`` is the base frequency. The nominal foot trajectory ``p_l`` lifts the
foot by ``swing_height`` with cubic Hermite splines during swing (``0 <= phi <= pi``) and keeps the
nominal stance position otherwise (``pi < phi < 2*pi``).

The IK is solved in the sagittal plane of each thigh joint with the hip (abduction) joint held at its
default angle, which is exact for a purely vertical foot trajectory when the hip angle is zero and a
close approximation for Go2's small default abduction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class CPGJointPositionAction(ActionTerm):
    """Leg phase generator with IK foot trajectories plus residual joint position targets.

    Action layout: ``[dphi (num_legs), dq (3 * num_legs)]``.
    """

    cfg: CPGJointPositionActionCfg
    _asset: Articulation

    def __init__(self, cfg: CPGJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True, as_proxy=True)
        self._joint_ids = joint_ids.torch
        if len(self._joint_names) % 3 != 0:
            raise ValueError(f"Expected (hip, thigh, calf) per leg, got {len(self._joint_names)} joints.")
        self._num_legs = len(self._joint_names) // 3
        if len(cfg.initial_phase) != self._num_legs:
            raise ValueError(f"initial_phase has {len(cfg.initial_phase)} entries for {self._num_legs} legs.")

        self._default_joint_pos = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        default = self._default_joint_pos.view(self.num_envs, self._num_legs, 3)
        self._default_hip = default[..., 0]
        # nominal foot position in each thigh joint's sagittal plane (x forward, z up)
        l1, l2 = cfg.thigh_length, cfg.calf_length
        thigh, calf = default[..., 1], default[..., 2]
        self._foot_x = -l1 * torch.sin(thigh) - l2 * torch.sin(thigh + calf)
        self._foot_z = -l1 * torch.cos(thigh) - l2 * torch.cos(thigh + calf)

        self._base_delta_phase = 2.0 * math.pi * cfg.base_frequency * env.step_dt
        self._initial_phase = torch.tensor(cfg.initial_phase, device=self.device).repeat(self.num_envs, 1)
        self._phase = self._initial_phase.clone()
        self._delta_phase = torch.zeros(self.num_envs, self._num_legs, device=self.device)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._joint_targets = self._default_joint_pos.clone()
        self._prev_joint_targets = self._default_joint_pos.clone()
        self._prev_prev_joint_targets = self._default_joint_pos.clone()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 4 * self._num_legs

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Joint position targets [rad], ordered as :attr:`CPGJointPositionActionCfg.joint_names`."""
        return self._joint_targets

    @property
    def joint_ids(self) -> torch.Tensor:
        return self._joint_ids

    @property
    def default_joint_pos(self) -> torch.Tensor:
        return self._default_joint_pos

    @property
    def phase(self) -> torch.Tensor:
        """Leg phases [rad] in ``[0, 2*pi)``. Shape is (num_envs, num_legs)."""
        return self._phase

    @property
    def delta_phase(self) -> torch.Tensor:
        """Policy phase offsets [rad] applied at the last step. Shape is (num_envs, num_legs)."""
        return self._delta_phase

    @property
    def base_frequency(self) -> float:
        return self.cfg.base_frequency

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
        # advance the leg phases; clamping keeps each leg frequency within [0, 2 * f_0] for unit scale
        self._delta_phase = actions[:, : self._num_legs].clamp(-1.0, 1.0) * self.cfg.phase_offset_scale
        self._phase = torch.remainder(self._phase + self._base_delta_phase + self._delta_phase, 2.0 * math.pi)
        # nominal joint targets from the foot trajectory
        foot_z = self._foot_z + self.cfg.swing_height * self._swing_profile(self._phase)
        thigh, calf = self._sagittal_ik(self._foot_x, foot_z)
        nominal = torch.stack((self._default_hip, thigh, calf), dim=-1).view(self.num_envs, -1)
        # shift the target history used by the smoothness reward
        self._prev_prev_joint_targets[:] = self._prev_joint_targets
        self._prev_joint_targets[:] = self._joint_targets
        self._joint_targets = nominal + actions[:, self._num_legs :] * self.cfg.residual_scale

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._delta_phase[env_ids] = 0.0
        self._phase[env_ids] = self._initial_phase[env_ids]
        self._joint_targets[env_ids] = self._default_joint_pos[env_ids]
        self._prev_joint_targets[env_ids] = self._default_joint_pos[env_ids]
        self._prev_prev_joint_targets[env_ids] = self._default_joint_pos[env_ids]

    """
    Helpers.
    """

    @staticmethod
    def _swing_profile(phase: torch.Tensor) -> torch.Tensor:
        """Normalized foot lift in ``[0, 1]``: Hermite spline up over ``[0, pi/2]``, down over ``(pi/2, pi]``."""
        u = (phase / (0.5 * math.pi)).clamp(max=2.0)
        s = torch.where(u <= 1.0, u, 2.0 - u)
        lift = 3.0 * s**2 - 2.0 * s**3
        return torch.where(phase <= math.pi, lift, torch.zeros_like(lift))

    def _sagittal_ik(self, x: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Thigh and calf angles [rad] placing the foot at ``(x, z)`` from the thigh joint, knee bent backward."""
        l1, l2 = self.cfg.thigh_length, self.cfg.calf_length
        dist_sq = (x**2 + z**2).clamp(max=(0.999 * (l1 + l2)) ** 2)
        cos_calf = ((dist_sq - l1**2 - l2**2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0)
        calf = -torch.acos(cos_calf)
        thigh = torch.atan2(-x, -z) - torch.atan2(l2 * torch.sin(calf), l1 + l2 * torch.cos(calf))
        return thigh, calf


@configclass
class CPGJointPositionActionCfg(ActionTermCfg):
    """Configuration for :class:`CPGJointPositionAction`."""

    class_type: type[ActionTerm] = CPGJointPositionAction

    joint_names: list[str] = MISSING
    """Joint names ordered leg by leg as (hip, thigh, calf). The leg order must match :attr:`initial_phase`."""

    thigh_length: float = MISSING
    """Distance [m] from the thigh joint to the calf joint."""

    calf_length: float = MISSING
    """Distance [m] from the calf joint to the foot center."""

    base_frequency: float = 1.25
    """Base stepping frequency f_0 [Hz]."""

    swing_height: float = 0.2
    """Peak foot lift [m] above the nominal stance position."""

    initial_phase: tuple[float, ...] = (0.0, math.pi, math.pi, 0.0)
    """Leg phases [rad] after reset. Defaults to a trot for (FL, FR, RL, RR)."""

    phase_offset_scale: float = 1.0
    """Phase offset [rad] per unit action. Actions are clamped to [-1, 1] before scaling."""

    residual_scale: float = 0.25
    """Residual joint target [rad] per unit action."""
