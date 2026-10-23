# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray-cast pattern for the foot-centered height samples of Miki et al. 2022 (Table S3)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch

from isaaclab.sensors.ray_caster.patterns import PatternBaseCfg
from isaaclab.utils import configclass


def concentric_circles_pattern(cfg: ConcentricCirclesPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Downward rays on concentric circles, evenly spaced in angle on each circle."""
    starts = []
    for radius, num_points in zip(cfg.radii, cfg.num_points):
        angles = torch.arange(num_points, device=device) * (2.0 * math.pi / num_points)
        starts.append(
            torch.stack((radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros_like(angles)), -1)
        )
    ray_starts = torch.cat(starts, dim=0)
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0
    return ray_starts, ray_directions


@configclass
class ConcentricCirclesPatternCfg(PatternBaseCfg):
    """Configuration for :func:`concentric_circles_pattern`."""

    func: Callable = concentric_circles_pattern

    radii: Sequence[float] = (0.08, 0.16, 0.26, 0.36, 0.48)
    """Circle radii [m]."""

    num_points: Sequence[int] = (6, 8, 10, 12, 16)
    """Number of rays on each circle."""
