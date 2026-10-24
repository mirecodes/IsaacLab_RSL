# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What every terrain template shares: the base cfg, the tile grid, and reusable feature generators.

A template is a function ``(difficulty, cfg) -> (int16 height field, origin)`` decorated with
:func:`~isaaclab.terrains.height_field.utils.height_field_to_mesh`, plus a cfg class deriving from
:class:`TerrainTemplateCfg` whose ``function`` is that template. Conventions (see ``REPORT.md``):

* Coordinates ``(X, Y)`` are relative to the tile center; the outer ``edge_width`` band is flat (``z = 0``) so that
  neighboring tiles meet without a step, and the central ``platform_width`` square is flat for spawning.
* A ``(min, max)`` knob is interpolated by difficulty, ``lo + difficulty * (hi - lo)`` (:func:`lerp`); a scalar
  means ``(v, v)``. A per-instance size (a pit radius, a stub height, ...) is drawn from ``U(lo, lerp)`` (:func:`upto`),
  so the hardest tile spans the full range.
* Randomness comes from ``Grid.rng``, seeded by ``(seed, name, difficulty, tile)``, so a tile does not depend on the
  generation order, and the tiles of one difficulty differ.
* Objects the importer needs (sand patches, rocks) and features the tests check (pits, stubs) go into
  ``cfg.features``, in meters relative to the tile center.
"""

from __future__ import annotations

import zlib

import numpy as np
import scipy.interpolate as interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import to_discrete_units
from isaaclab.utils import configclass

ZONES = ("rigid", "slip", "sand")
"""Runtime interaction of a terrain type (:attr:`TerrainTemplateCfg.zone`); the zone id is the index."""

Range = tuple[float, float]


@configclass
class TerrainTemplateCfg(HfTerrainBaseCfg):
    """Fields shared by every template. Knobs are ``(min, max)`` pairs interpolated by difficulty."""

    name: str = ""
    """Terrain type (the key in :func:`~..world_cfg.default_terrains`, filled in by the world cfg); seeds the tile's
    random generator."""

    seed: int = 0
    tile: int = 0
    """World seed and tile index within the difficulty row, both set by the generator for each tile."""

    horizontal_scale: float | None = None
    vertical_scale: float | None = None
    """Both inherited from the generator."""

    convert_to_heightfield: bool = True
    """Collide against a Newton heightfield of the same grid. Against the triangle mesh, simulation was 12x slower,
    a 30-tile world ran out of memory, and convex rocks fell through (see ``REPORT.md``)."""

    zone: str = "rigid"
    """Runtime interaction: one of :data:`ZONES`."""

    platform_width: float = 3.0
    """Side [m] of the flat spawn square at the center (set by the world cfg)."""

    edge_width: float = 1.0
    """Width [m] of the flat band along the tile border (heights are zero there)."""

    blend_width: float = 1.0
    """Distance [m] over which features fade in from the edge band and the platform."""

    unit_size: Range = (0.15, 0.5)
    """Facet size [m] of the gravel noise."""

    amplitude: Range = (0.05, 0.18)
    """Half range [m] of the gravel noise heights."""

    interp_order: int = 1
    """1: angular facets (linear interpolation); 3: smooth (cubic spline)."""


##
# Knobs.
##


def lerp(value: Range | float, difficulty: float) -> float:
    """Value of a ``(min, max)`` knob at ``difficulty``; a scalar is ``(v, v)``."""
    lo, hi = (value, value) if np.isscalar(value) else value
    return float(lo + difficulty * (hi - lo))


def upto(rng: np.random.Generator, value: Range | float, difficulty: float, size=None):
    """Per-instance size: ``U(min, lerp(value, difficulty))``."""
    lo = value if np.isscalar(value) else value[0]
    return rng.uniform(lo, lerp(value, difficulty), size)


def smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


##
# Tile grid.
##


class Grid:
    """Pixel coordinates of one tile, the masks every template uses, and the tile's random generator."""

    def __init__(self, cfg: TerrainTemplateCfg, difficulty: float):
        self.hs = cfg.horizontal_scale
        n = to_discrete_units(cfg.size[0], self.hs)
        m = to_discrete_units(cfg.size[1], self.hs)
        self.center = np.array([(n - 1) * self.hs / 2, (m - 1) * self.hs / 2])
        x = np.arange(n) * self.hs - self.center[0]
        y = np.arange(m) * self.hs - self.center[1]
        self.x, self.y = x, y
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.half = self.center.copy()
        # distance to the tile border and to the inner side of the flat edge band
        self.d_edge = np.minimum(self.half[0] - np.abs(self.X), self.half[1] - np.abs(self.Y))
        self.r = np.clip(self.d_edge - cfg.edge_width, 0.0, None)
        # Chebyshev distance from the center, beyond the platform
        self.p = np.maximum(np.abs(self.X), np.abs(self.Y)) - cfg.platform_width / 2
        self.fade = smoothstep(self.r / cfg.blend_width) * smoothstep(self.p / cfg.blend_width)
        """1 where features may be placed, 0 on the edge band and platform."""
        key = zlib.crc32(cfg.name.encode())
        self.rng = np.random.default_rng([cfg.seed, key, round(difficulty * 1e6), cfg.tile])

    def sample_xy(self, count: int, cfg: TerrainTemplateCfg, margin: float) -> np.ndarray:
        """Up to ``count`` uniform points with ``margin`` clearance to the edge band and the platform (fewer, or
        none, if the tile is too small for that margin)."""
        lim = self.half - cfg.edge_width - margin
        inner = cfg.platform_width / 2 + margin
        if (lim <= inner).all() or (lim <= 0).any():
            return np.empty((0, 2))
        points = np.empty((0, 2))
        for _ in range(100):
            if len(points) >= count:
                break
            xy = self.rng.uniform(-lim, lim, size=(2 * count + 8, 2))
            points = np.vstack([points, xy[np.abs(xy).max(axis=1) > inner]])
        return points[:count]

    def area(self, cfg: TerrainTemplateCfg) -> float:
        """Area [m^2] of the feature region (tile minus edge band and platform)."""
        side = 2 * (self.half - cfg.edge_width)
        return float(side[0] * side[1] - cfg.platform_width**2)

    def window(self, cx: float, cy: float, radius: float) -> tuple[slice, slice]:
        """Pixel slices of the square of half-width ``radius`` around ``(cx, cy)``."""
        i0 = max(0, int((cx - radius + self.center[0]) / self.hs))
        i1 = min(len(self.x), int((cx + radius + self.center[0]) / self.hs) + 2)
        j0 = max(0, int((cy - radius + self.center[1]) / self.hs))
        j1 = min(len(self.y), int((cy + radius + self.center[1]) / self.hs) + 2)
        return slice(i0, i1), slice(j0, j1)

    def height_at(self, h: np.ndarray, xy: np.ndarray) -> np.ndarray:
        """Nearest-pixel height [m] of ``h`` at points ``xy`` (N, 2)."""
        i = np.clip(np.rint((xy[:, 0] + self.center[0]) / self.hs).astype(int), 0, len(self.x) - 1)
        j = np.clip(np.rint((xy[:, 1] + self.center[1]) / self.hs).astype(int), 0, len(self.y) - 1)
        return h[i, j]


def finish(grid: Grid, h: np.ndarray, cfg: TerrainTemplateCfg) -> tuple[np.ndarray, np.ndarray]:
    """Clear the edge band and level the platform (features may reach into both), quantize, and return the spawn
    origin on the platform center. Every template ends with it."""
    h = np.where(grid.r <= 0.0, 0.0, h)
    h = np.where(grid.p <= 0.0, np.median(h[grid.p <= 0.0]), h)
    hf = np.rint(h / cfg.vertical_scale).astype(np.int16)
    # kept for the terrain map (the mesh merges duplicate vertices, so its vertex grid cannot be read back)
    cfg.heights = hf.astype(np.float32) * cfg.vertical_scale
    c = (np.array(hf.shape) - 1) // 2
    origin = np.array([*grid.center, hf[c[0], c[1]] * cfg.vertical_scale])
    return hf, origin


##
# Feature generators shared by several templates.
##


def noise(grid: Grid, unit_size: float, amplitude: float, order: int) -> np.ndarray:
    """Heights sampled from ``U(-amplitude, amplitude)`` on a grid of spacing ``unit_size``, interpolated.

    ``order=1`` gives flat triangular-ish facets of size ``unit_size`` (the reference gravel), ``order=3`` a smooth
    surface. The coarse grid has a random phase so facet edges do not line up with the tile.
    """
    if amplitude <= 0.0:
        return np.zeros_like(grid.X)
    phase = grid.rng.uniform(0.0, unit_size, 2)
    gx = np.arange(grid.x[0] - unit_size - phase[0], grid.x[-1] + 2 * unit_size, unit_size)
    gy = np.arange(grid.y[0] - unit_size - phase[1], grid.y[-1] + 2 * unit_size, unit_size)
    samples = grid.rng.uniform(-amplitude, amplitude, (len(gx), len(gy)))
    spline = interpolate.RectBivariateSpline(gx, gy, samples, kx=order, ky=order, s=0)
    return spline(grid.x, grid.y)


def gravel(grid: Grid, cfg: TerrainTemplateCfg, difficulty: float) -> np.ndarray:
    """The template's gravel noise (``unit_size``, ``amplitude``, ``interp_order``), faded out at the edge band
    and the platform."""
    return grid.fade * noise(grid, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)


def pyramid(grid: Grid, cfg: TerrainTemplateCfg, slope: float, inverted: bool) -> np.ndarray:
    """True pyramid ``slope * min(fx, fy)`` from the edge band to a flat top of ``platform_width``."""
    top = slope * (grid.half.min() - cfg.edge_width - cfg.platform_width / 2)
    h = np.minimum(slope * grid.r, top)
    return -h if inverted else h


def add_pits(grid: Grid, h: np.ndarray, cfg, difficulty: float, density_scale: float = 1.0) -> list[dict]:
    """Conical pits with an optional raised rim (foot traps). Knobs: ``pit_density`` [1/m^2], ``pit_radius``,
    ``pit_depth``, ``pit_rim``."""
    count = round(lerp(cfg.pit_density, difficulty) * density_scale * grid.area(cfg))
    pits = []
    for x, y in grid.sample_xy(count, cfg, margin=0.3):
        r, depth, rim = upto(grid.rng, cfg.pit_radius, difficulty), upto(grid.rng, cfg.pit_depth, difficulty), 0.0
        if lerp(cfg.pit_rim, difficulty) > 0:
            rim = upto(grid.rng, cfg.pit_rim, difficulty)
        win = grid.window(x, y, 1.5 * r)
        rho = np.hypot(grid.X[win] - x, grid.Y[win] - y)
        h[win] += rim * np.clip(1.0 - np.abs(rho - 1.2 * r) / (0.2 * r), 0.0, 1.0)
        h[win] -= depth * np.clip(1.0 - rho / r, 0.0, 1.0)
        pits.append({"x": x, "y": y, "radius": r, "depth": depth, "rim": rim})
    return pits


def add_trenches(grid: Grid, h: np.ndarray, cfg, difficulty: float) -> list[dict]:
    """Straight V-shaped trenches of random heading. Knobs: ``trench_count``, ``trench_length`` (uniform, not
    interpolated), ``trench_width``, ``trench_depth``."""
    trenches = []
    for x, y in grid.sample_xy(round(lerp(cfg.trench_count, difficulty)), cfg, margin=1.0):
        length = grid.rng.uniform(*cfg.trench_length)
        yaw = grid.rng.uniform(0.0, np.pi)
        width, depth = upto(grid.rng, cfg.trench_width, difficulty), upto(grid.rng, cfg.trench_depth, difficulty)
        win = grid.window(x, y, length / 2 + width)
        # distance to the segment
        u = np.array([np.cos(yaw), np.sin(yaw)])
        dx, dy = grid.X[win] - x, grid.Y[win] - y
        t = np.clip(dx * u[0] + dy * u[1], -length / 2, length / 2)
        dist = np.hypot(dx - t * u[0], dy - t * u[1])
        h[win] -= depth * np.clip(1.0 - dist / (width / 2), 0.0, 1.0)
        trenches.append({"x": x, "y": y, "yaw": yaw, "length": length, "width": width, "depth": depth})
    return trenches


def add_stubs(grid: Grid, h: np.ndarray, cfg, difficulty: float, density_scale: float = 1.0) -> list[dict]:
    """Small spherical domes (stones) and half-cylinder ridges (roots), below the scanner's resolution. Knobs:
    ``stub_density`` [1/m^2], ``stub_radius``, ``stub_height``, ``root_count``, ``root_length``, ``root_width``,
    ``root_height``."""
    stubs = []
    count = round(lerp(cfg.stub_density, difficulty) * density_scale * grid.area(cfg))
    for x, y in grid.sample_xy(count, cfg, margin=0.1):
        r, height = upto(grid.rng, cfg.stub_radius, difficulty), upto(grid.rng, cfg.stub_height, difficulty)
        win = grid.window(x, y, r)
        rho = np.hypot(grid.X[win] - x, grid.Y[win] - y)
        h[win] += height * np.sqrt(np.clip(1.0 - (rho / r) ** 2, 0.0, None))
        stubs.append({"x": x, "y": y, "radius": r, "height": height})
    roots = []
    for x, y in grid.sample_xy(round(lerp(cfg.root_count, difficulty) * density_scale), cfg, margin=1.0):
        length, yaw = upto(grid.rng, cfg.root_length, difficulty), grid.rng.uniform(0.0, np.pi)
        width, height = upto(grid.rng, cfg.root_width, difficulty), upto(grid.rng, cfg.root_height, difficulty)
        win = grid.window(x, y, length / 2 + width)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        dx, dy = grid.X[win] - x, grid.Y[win] - y
        t = np.clip(dx * u[0] + dy * u[1], -length / 2, length / 2)
        dist = np.hypot(dx - t * u[0], dy - t * u[1])
        h[win] = np.maximum(h[win], h[win] + height * np.sqrt(np.clip(1.0 - (2 * dist / width) ** 2, 0.0, None)))
        roots.append({"x": x, "y": y, "yaw": yaw, "length": length, "width": width, "height": height})
    return stubs + [dict(r, root=True) for r in roots]
