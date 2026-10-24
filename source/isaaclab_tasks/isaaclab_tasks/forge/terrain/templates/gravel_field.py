# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gravel field: the rough field with loose, mostly flat stones, as on a mountain trail (support tilting under the
foot, kicked stones, ankle twists).

The height field is the rough field's (``rough_field.py``). Every stone is a dynamic body: each robot has its own
stones, which the terrain event keeps around it (``RockCfg`` in ``world_cfg.py``). Most stones are flat (height
15-40 % of their length, flat top and bottom) and rest on the ground's bumps: stepped on near the middle they hold,
stepped on near an edge they tilt and rock. This module holds the stone shapes.
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.utils import configclass

from .common import Range
from .rough_field import RoughFieldCfg


def rock_mesh(seed: int, template: int, size: float, axis_ratio, yaw: float) -> trimesh.Trimesh:
    """Random convex stone: hull of 12-30 noisy points on an ellipsoid, largest extent ``size``, with a flat top and
    bottom face (the highest and lowest quarter of the points, at least three each, moved to 70 % of the half height;
    stones split and weather along planes).

    The template index fixes the point cloud (16 templates per seed); ``axis_ratio`` (y/x, z/x) and ``yaw`` shape
    the instance. The bottom is at ``z = 0``.
    """
    rng = np.random.default_rng([seed, 7, template])
    points = rng.normal(size=(int(rng.integers(12, 31)), 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.8, 1.0, (len(points), 1))
    half = np.array([1.0, axis_ratio[0], axis_ratio[1]]) * size / 2
    points *= half
    order, n = np.argsort(points[:, 2]), max(3, len(points) // 4)
    points[order[:n], 2], points[order[-n:], 2] = -0.7 * half[2], 0.7 * half[2]
    c, s = np.cos(yaw), np.sin(yaw)
    points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
    hull = trimesh.convex.convex_hull(points)
    hull.apply_translation([0.0, 0.0, -hull.bounds[0, 2]])
    return hull


def stone_shape(rng: np.random.Generator, cfg: GravelFieldCfg) -> list[float]:
    """``axis_ratio`` (width / length, height / length) of one stone: flat with probability ``flat_frac``."""
    flat = rng.uniform() < cfg.flat_frac
    return [rng.uniform(*cfg.width_ratio), rng.uniform(*(cfg.flat_height if flat else cfg.round_height))]


@configclass
class GravelFieldCfg(RoughFieldCfg):
    """The rough field's ground and generator, plus the shape knobs of the stones."""

    flat_frac: float = 0.8
    """Share of flat stones; the others are round cobbles."""
    width_ratio: Range = (0.5, 0.9)
    """Width / length of a stone, uniform."""
    flat_height: Range = (0.15, 0.4)
    """Height / length of a flat stone, uniform."""
    round_height: Range = (0.5, 1.0)
    """Height / length of a round stone, uniform."""
    template_count: int = 16
    """Point clouds the stone hulls are drawn from."""
