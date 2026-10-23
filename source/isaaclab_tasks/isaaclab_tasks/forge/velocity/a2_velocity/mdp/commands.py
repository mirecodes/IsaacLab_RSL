# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity command for the steering environment."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class SteeringVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command whose planar speed is capped while turning.

    Walking an arc of radius ``|v| / |w_z|`` needs the centripetal acceleration ``|v_xy| |w_z|``. Commands above
    ``max_centripetal_acc`` are scaled down in ``v_xy`` (keeping ``w_z``), so fast turns come with slow walking.
    """

    cfg: SteeringVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        cmd = self.vel_command_b[env_ids]
        # scale = min(1, a_max / (|v_xy| |w_z|))
        demand = cmd[:, :2].norm(dim=1) * cmd[:, 2].abs()
        scale = (self.cfg.max_centripetal_acc / demand.clamp(min=1e-6)).clamp(max=1.0)
        cmd[:, :2] *= scale.unsqueeze(1)
        self.vel_command_b[env_ids] = cmd


@configclass
class SteeringVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`SteeringVelocityCommand`."""

    class_type: type = SteeringVelocityCommand

    max_centripetal_acc: float = 0.5
    """Upper bound on ``|v_xy| |w_z|`` [m/s^2] of a sampled command."""
