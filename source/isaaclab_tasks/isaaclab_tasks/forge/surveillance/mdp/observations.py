# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the ``-Faults`` variant."""

from __future__ import annotations

import torch


def nominal_motor_strength(env, num_joints: int = 12) -> torch.Tensor:
    """Motor strength 1 for every joint: stands in for the energy policy's ``motor_strength`` extrinsics when its PD
    gain randomization (Newton actuators only) is off, so the policy is not told about an injected motor fault."""
    return torch.ones(env.num_envs, num_joints, device=env.device)
