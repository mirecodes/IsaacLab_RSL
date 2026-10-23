# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Steering A2 policy: the base task with a directly commanded yaw rate instead of a heading target.

The base task samples a heading target and turns it into the yaw-rate command with a P controller,
``w_z^cmd = clip(0.5 * wrap(psi^cmd - psi), w_z range)``, so w_z^cmd decays to zero once the heading is reached
and the policy mostly sees straight-line commands. Here ``(v_x, v_y, w_z)`` are all sampled uniformly and held,
so the policy learns to walk arcs (turn while moving) and to follow a joystick-style yaw-rate input.

Deviations from the base task (values chosen here, not from a source):

* Commands are resampled every 4-8 s instead of every 10 s, so each 20 s episode has several transitions
  between turning and straight walking.
* v_y^cmd is limited to (-0.3, 0.3) m/s, so the robot changes direction by yawing instead of crab-walking.
* While turning, v_xy^cmd is scaled down so that ``|v_xy^cmd| |w_z^cmd| <= 0.5 m/s^2``
  (:class:`~.mdp.SteeringVelocityCommand`): no fast-walk-and-fast-turn commands the robot cannot track.
"""

from isaaclab.utils import configclass

from . import mdp as study_mdp
from .base_env_cfg import A2StudyBaseEnvCfg


@configclass
class A2SteeringEnvCfg(A2StudyBaseEnvCfg):
    """A2 base task with a direct yaw-rate command."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands: sample w_z^cmd uniformly instead of from a heading target
        base = self.commands.base_velocity
        self.commands.base_velocity = study_mdp.SteeringVelocityCommandCfg(
            asset_name=base.asset_name,
            # chosen here: more command changes per episode to learn turning transitions
            resampling_time_range=(4.0, 8.0),
            rel_standing_envs=base.rel_standing_envs,
            heading_command=False,
            debug_vis=base.debug_vis,
            ranges=study_mdp.SteeringVelocityCommandCfg.Ranges(
                # chosen here: little lateral velocity, so direction changes come from yawing
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-1.0, 1.0),
            ),
            # chosen here: |v_xy| <= 0.5 m/s at the full 1 rad/s turn; unscaled below ~0.48 rad/s
            max_centripetal_acc=0.5,
        )
