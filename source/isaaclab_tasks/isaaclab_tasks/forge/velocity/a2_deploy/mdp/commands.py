# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Steering command: forward walking with smooth turns by a random angle."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class SteeringCommand(UniformVelocityCommand):
    """Forward speed ``v_x ~ U(lin_vel_x)``, ``v_y = 0``, and a turn by ``dpsi ~ U(-max_turn, max_turn)`` per segment.

    The yaw-rate command steers the remaining angle to zero and then stops (see ``REPORT.md`` §3.1)::

        dpsi_rem <- dpsi_rem - w_z dt                                    (measured world yaw rate)
        w_z^des  = clip(k dpsi_rem, ang_vel_z) while turning, else 0    (turning ends at |dpsi_rem| < tol)
        w_z^cmd <- w_z^cmd + dt / (tau + dt) (w_z^des - w_z^cmd)        (first-order smoothing)
    """

    cfg: SteeringCommandCfg

    def __init__(self, cfg: SteeringCommandCfg, env):
        super().__init__(cfg, env)
        self.turn_left = torch.zeros(self.num_envs, device=self.device)
        """Yaw [rad] still to turn in the current segment (signed)."""
        self.turning = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the segment's turn is still in progress."""

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # a new episode starts without a turn in progress
        self.vel_command_b[slice(None) if env_ids is None else env_ids, 2] = 0.0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        r = torch.empty(num_envs, device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = 0.0
        self.turn_left[env_ids] = r.uniform_(-self.cfg.max_turn, self.cfg.max_turn)
        self.turning[env_ids] = True

    def _update_command(self):
        dt = self._env.step_dt
        self.turn_left -= self.robot.data.root_link_ang_vel_w.torch[:, 2] * dt
        self.turning &= self.turn_left.abs() >= self.cfg.turn_tolerance
        desired = (self.cfg.turn_stiffness * self.turn_left).clamp(*self.cfg.ranges.ang_vel_z)
        desired = torch.where(self.turning, desired, torch.zeros_like(desired))
        self.vel_command_b[:, 2] += dt / (self.cfg.smoothing_s + dt) * (desired - self.vel_command_b[:, 2])


@configclass
class SteeringCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`SteeringCommand`; ``ranges.lin_vel_y`` and the heading options are unused."""

    class_type: type = SteeringCommand
    max_turn: float = math.pi / 4
    """Largest turn [rad] per command segment, either side."""
    turn_stiffness: float = 1.0
    """Gain k [1/s] of the yaw-rate command on the remaining angle."""
    turn_tolerance: float = math.radians(2.0)
    """Remaining angle [rad] below which the turn ends and the yaw-rate command returns to zero."""
    smoothing_s: float = 0.5
    """Time constant tau [s] of the yaw-rate command's first-order smoothing."""
