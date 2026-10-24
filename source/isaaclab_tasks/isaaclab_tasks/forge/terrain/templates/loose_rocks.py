# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Loose rocks: gentle gravel strewn with static and dynamic rocks (support moving under the foot, body jamming).

The height field only carries the gravel; the rocks are listed in ``cfg.features["rocks"]`` and spawned by the
importer (static) and the scene's rock collection (dynamic), all from :func:`rock_mesh`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import trimesh

from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

from .common import Grid, Range, TerrainTemplateCfg, finish, gravel, lerp


def rock_mesh(seed: int, template: int, size: float, axis_ratio, yaw: float, flatten: float = 1.0) -> trimesh.Trimesh:
    """Random convex rock: hull of 12-30 noisy points on an ellipsoid, largest extent ``size``.

    The template index fixes the point cloud (16 templates per seed); ``axis_ratio`` (y/x, z/x), ``flatten``
    (extra z scale of buried rocks), and ``yaw`` shape the instance. The bottom is at ``z = 0``.
    """
    rng = np.random.default_rng([seed, 7, template])
    points = rng.normal(size=(int(rng.integers(12, 31)), 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.8, 1.0, (len(points), 1))
    points *= np.array([1.0, axis_ratio[0], axis_ratio[1] * flatten]) * size / 2
    c, s = np.cos(yaw), np.sin(yaw)
    points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
    hull = trimesh.convex.convex_hull(points)
    hull.apply_translation([0.0, 0.0, -hull.bounds[0, 2]])
    return hull


@height_field_to_mesh
def loose_rocks_terrain(difficulty: float, cfg: LooseRocksCfg) -> tuple[np.ndarray, np.ndarray]:
    """Static rocks are flattened (buried) or whole, plus a dense quadrant of large rocks spaced about one body width
    apart (body jamming). The first ``len(cfg.dynamic_sizes)`` rocks are the dynamic slots on a ring around the
    platform; their sizes are fixed for the whole world so that every environment shares one rock collection."""
    g = Grid(cfg, difficulty)
    h = gravel(g, cfg, difficulty)
    rocks = []

    def place(xy, size, dynamic, buried):
        for (x, y), s, b in zip(xy, size, buried):
            win = g.window(x, y, s / 2)
            rocks.append({
                "x": x, "y": y, "size": s, "ground": float(h[win].min()), "dynamic": dynamic, "buried": bool(b),
                "template": int(g.rng.integers(cfg.template_count)), "yaw": float(g.rng.uniform(0, 2 * np.pi)),
                "axis_ratio": g.rng.uniform(cfg.axis_ratio[0], cfg.axis_ratio[1], 2).tolist(),
            })  # fmt: skip

    # dynamic slots: on a ring just outside the platform, where robots spawned here walk first
    k = len(cfg.dynamic_sizes)
    if k:
        angle = np.linspace(0, 2 * np.pi, k, endpoint=False) + g.rng.uniform(0, 2 * np.pi)
        radius = cfg.platform_width / 2 + 0.6 + g.rng.uniform(0.0, cfg.dynamic_ring, k)
        place(np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1), cfg.dynamic_sizes, True, [False] * k)
    # static rocks: power-law sizes p(s) ~ s^exponent on [lo, hi], by inverse CDF
    count = round(lerp(cfg.rocks_per_tile, difficulty))
    lo, hi, a = cfg.rock_size[0], cfg.rock_size[1], cfg.size_exponent + 1.0
    sizes = (lo**a + g.rng.uniform(size=count) * (hi**a - lo**a)) ** (1.0 / a)
    place(g.sample_xy(count, cfg, margin=0.3), sizes, False, g.rng.uniform(size=count) < cfg.buried_frac)
    # dense quadrant: large rocks on a jittered grid with gaps of spacing_body_frac x body width
    quad = g.rng.choice([-1.0, 1.0], 2)
    gap = lerp(cfg.spacing_body_frac, 1.0 - difficulty) * cfg.body_width
    pitch = cfg.rock_size[1] + gap
    span = np.arange(cfg.platform_width / 2 + 1.0, g.half.min() - cfg.edge_width - 0.5, pitch)
    gx, gy = np.meshgrid(span, span, indexing="ij")
    xy = np.stack([quad[0] * gx.ravel(), quad[1] * gy.ravel()], 1) + g.rng.uniform(-0.05, 0.05, (gx.size, 2))
    place(xy, np.full(len(xy), cfg.rock_size[1]), False, [False] * len(xy))
    cfg.features = {"rocks": rocks, "dense_gap": gap}
    return finish(g, h, cfg)


@configclass
class LooseRocksCfg(TerrainTemplateCfg):
    function: Callable = loose_rocks_terrain
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.01, 0.03)
    rocks_per_tile: Range = (150.0, 400.0)
    """Static rocks per tile (the dynamic ones come on top)."""
    rock_size: Range = (0.05, 0.30)
    """Largest extent [m]; sampled from a power law, not interpolated."""
    size_exponent: float = -2.5
    axis_ratio: Range = (0.5, 1.0)
    buried_frac: float = 0.3
    spacing_body_frac: Range = (0.8, 1.5)
    """Gap between the dense-quadrant rocks, in body widths (narrower at higher difficulty)."""
    body_width: float = 0.35
    """A2 trunk width [m] (hip-to-hip 0.15 m plus hips), for the dense quadrant."""
    template_count: int = 16
    dynamic_sizes: list = []
    """Sizes [m] of the dynamic rocks, set by the world cfg (shared by all tiles)."""
    dynamic_ring: float = 3.0
    """Radial spread [m] of the dynamic rocks, starting 0.6 m outside the platform."""
