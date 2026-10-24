# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Height-field properties of the Harsh World: one function and one cfg per property.

Every function maps ``(difficulty, cfg)`` to an ``int16`` height field and the spawn origin, like Isaac Lab's
own height-field terrains. Shared conventions (see ``REPORT.md``):

* Coordinates ``(X, Y)`` are relative to the cell center; the outer ``edge_width`` band is flat (``z = 0``) so
  that neighboring cells meet without a step, and the central ``platform_width`` square is flat for spawning.
* A scalar knob is the value at ``difficulty``: ``lo + difficulty * (hi - lo)`` (:func:`lerp`). A per-instance
  size (a pit radius, a stub height, ...) is drawn from ``U(lo, lerp)``, so the hardest cell spans the full range.
* Randomness comes from a generator seeded by ``(seed, name, difficulty)``, so a cell does not depend on the
  generation order or the global NumPy state.
* Objects the importer needs (sand patches, rocks) and features the tests check (pits, stubs) are written to
  ``cfg.features`` in meters relative to the returned origin's ``(x, y)``.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np
import scipy.interpolate as interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh, to_discrete_units
from isaaclab.utils import configclass

ZONES = ("rigid", "slip", "sand")
"""Zone names; the zone id is the index."""

Range = tuple[float, float]


##
# Helpers.
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


class Grid:
    """Pixel coordinates of one cell and the masks every property uses."""

    def __init__(self, cfg: HarshTerrainCfg, difficulty: float):
        self.hs = cfg.horizontal_scale
        n = to_discrete_units(cfg.size[0], self.hs)
        m = to_discrete_units(cfg.size[1], self.hs)
        self.center = np.array([(n - 1) * self.hs / 2, (m - 1) * self.hs / 2])
        x = np.arange(n) * self.hs - self.center[0]
        y = np.arange(m) * self.hs - self.center[1]
        self.x, self.y = x, y
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.half = self.center.copy()
        # distance to the cell border and to the inner side of the flat edge band
        self.d_edge = np.minimum(self.half[0] - np.abs(self.X), self.half[1] - np.abs(self.Y))
        self.r = np.clip(self.d_edge - cfg.edge_width, 0.0, None)
        # Chebyshev distance from the center, beyond the platform
        self.p = np.maximum(np.abs(self.X), np.abs(self.Y)) - cfg.platform_width / 2
        self.fade = smoothstep(self.r / cfg.blend_width) * smoothstep(self.p / cfg.blend_width)
        """1 where features may be placed, 0 on the edge band and platform."""
        key = zlib.crc32(cfg.name.encode())
        self.rng = np.random.default_rng([getattr(cfg, "seed", None) or 0, key, round(difficulty * 1e6)])

    def free(self, margin: float) -> tuple[np.ndarray, float]:
        """Bounds ``(lo, hi)`` and area of the region at least ``margin`` inside the feature region."""
        hi = self.half - 0 - margin
        return hi, (2 * hi[0]) * (2 * hi[1])

    def sample_xy(self, count: int, cfg: HarshTerrainCfg, margin: float) -> np.ndarray:
        """Up to ``count`` uniform points with ``margin`` clearance to the edge band and the platform (fewer, or
        none, if the cell is too small for that margin)."""
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

    def area(self, cfg: HarshTerrainCfg) -> float:
        """Area [m^2] of the feature region (cell minus edge band and platform)."""
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


def noise(grid: Grid, unit_size: float, amplitude: float, order: int) -> np.ndarray:
    """Heights sampled from ``U(-amplitude, amplitude)`` on a grid of spacing ``unit_size``, interpolated.

    ``order=1`` gives flat triangular-ish facets of size ``unit_size`` (the reference gravel), ``order=3`` a smooth
    surface. The coarse grid has a random phase so facet edges do not line up with the cell.
    """
    if amplitude <= 0.0:
        return np.zeros_like(grid.X)
    phase = grid.rng.uniform(0.0, unit_size, 2)
    gx = np.arange(grid.x[0] - unit_size - phase[0], grid.x[-1] + 2 * unit_size, unit_size)
    gy = np.arange(grid.y[0] - unit_size - phase[1], grid.y[-1] + 2 * unit_size, unit_size)
    samples = grid.rng.uniform(-amplitude, amplitude, (len(gx), len(gy)))
    spline = interpolate.RectBivariateSpline(gx, gy, samples, kx=order, ky=order, s=0)
    return spline(grid.x, grid.y)


def add_pits(grid: Grid, h: np.ndarray, cfg, difficulty: float, density_scale: float = 1.0) -> list[dict]:
    """Conical pits with an optional raised rim (foot traps)."""
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
    """Straight V-shaped trenches of random length and heading."""
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
    """Small spherical domes (stones) and half-cylinder ridges (roots), below the scanner's resolution."""
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


def pyramid(grid: Grid, cfg, slope: float, inverted: bool) -> np.ndarray:
    """True pyramid ``slope * min(fx, fy)`` from the edge band to a flat top of ``platform_width``."""
    top = slope * (grid.half.min() - cfg.edge_width - cfg.platform_width / 2)
    h = np.minimum(slope * grid.r, top)
    return -h if inverted else h


def finish(grid: Grid, h: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Clear the edge band and level the platform (features may reach into both), quantize, and return the spawn
    origin on the platform center."""
    h = np.where(grid.r <= 0.0, 0.0, h)
    h = np.where(grid.p <= 0.0, np.median(h[grid.p <= 0.0]), h)
    hf = np.rint(h / cfg.vertical_scale).astype(np.int16)
    # kept for the zone map (the mesh merges duplicate vertices, so its vertex grid cannot be read back)
    cfg.heights = hf.astype(np.float32) * cfg.vertical_scale
    c = (np.array(hf.shape) - 1) // 2
    origin = np.array([*grid.center, hf[c[0], c[1]] * cfg.vertical_scale])
    return hf, origin


##
# Properties.
##


@height_field_to_mesh
def gravel_terrain(difficulty: float, cfg: GravelCfg) -> tuple[np.ndarray, np.ndarray]:
    """Faceted gravel (coarse uniform samples, linear interpolation) with optional pits and V-trenches."""
    g = Grid(cfg, difficulty)
    h = g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    cfg.features = {"pits": add_pits(g, h, cfg, difficulty), "trenches": add_trenches(g, h, cfg, difficulty)}
    return finish(g, h, cfg)


@height_field_to_mesh
def stub_field_terrain(difficulty: float, cfg: StubFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Near-flat ground with stones and roots smaller than half the scanner spacing."""
    g = Grid(cfg, difficulty)
    h = g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), 3)
    cfg.features = {"stubs": add_stubs(g, h, cfg, difficulty)}
    return finish(g, h, cfg)


@height_field_to_mesh
def slope_terrain(difficulty: float, cfg: RoughSlopeCfg) -> tuple[np.ndarray, np.ndarray]:
    """Pyramid (or bowl) with gravel noise and flat-topped rock ledges."""
    g = Grid(cfg, difficulty)
    h = pyramid(g, cfg, lerp(cfg.slope, difficulty), cfg.inverted)
    h += g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    ledges = []
    for x, y in g.sample_xy(round(lerp(cfg.ledge_count, difficulty)), cfg, margin=1.0):
        sx, sy = upto(g.rng, cfg.ledge_size, difficulty, 2) / 2
        win = g.window(x, y, max(sx, sy))
        inside = (np.abs(g.X[win] - x) < sx) & (np.abs(g.Y[win] - y) < sy)
        top = g.height_at(h, np.array([[x, y]]))[0] + upto(g.rng, cfg.ledge_height, difficulty)
        h[win] = np.where(inside, np.maximum(h[win], top), h[win])
        ledges.append({"x": x, "y": y, "size": [2 * sx, 2 * sy], "top": top})
    cfg.features = {"ledges": ledges}
    return finish(g, h, cfg)


@height_field_to_mesh
def stairs_terrain(difficulty: float, cfg: RoughStairsCfg) -> tuple[np.ndarray, np.ndarray]:
    """Pyramid stairs with random step widths, eroded (jittered) step edges, and gravel noise."""
    g = Grid(cfg, difficulty)
    step_height, step_width = lerp(cfg.step_height, difficulty), lerp(cfg.step_width, difficulty)
    run = g.half.min() - cfg.edge_width - cfg.platform_width / 2
    widths = step_width * g.rng.uniform(1 - cfg.width_jitter, 1 + cfg.width_jitter, int(2 * run / step_width) + 2)
    edges = np.cumsum(widths)
    edges = edges[edges < run]
    # erosion: offset the distance with a smooth field of amplitude edge_jitter before quantizing into steps
    r = g.r + noise(g, 0.5, lerp(cfg.edge_jitter, difficulty), 3)
    h = step_height * np.searchsorted(edges, np.clip(r, 0.0, None), side="right")
    h = np.where(g.r <= 0.0, 0.0, h)
    h += np.where(g.p > 0, noise(g, 0.2, lerp(cfg.amplitude, difficulty), 1), 0.0) * smoothstep(g.r / 0.3)
    h = -h if cfg.inverted else h
    cfg.features = {"step_edges": edges.tolist(), "step_height": step_height}
    return finish(g, h, cfg)


@height_field_to_mesh
def sand_field_terrain(difficulty: float, cfg: SandFieldCfg) -> tuple[np.ndarray, np.ndarray]:
    """Rough hardpan with lowered sand patches; the importer fills each patch with a compliant slab.

    Around each patch the hardpan is flattened to the patch's surface level ``s`` (so the slab top meets the
    hardpan without a step) and inside it is lowered by ``sand_depth`` while keeping its roughness, so the
    sinkage depends on where the foot lands.
    """
    g = Grid(cfg, difficulty)
    rough = noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    h = g.fade * rough
    classes = list(cfg.patch_size_classes.items())
    weights = np.array([cfg.patch_class_weights[name] for name, _ in classes], dtype=float)
    placed, patches = [], []
    for _ in range(round(lerp(cfg.patch_count, difficulty))):
        name, (lo, hi) = classes[g.rng.choice(len(classes), p=weights / weights.sum())]
        for _attempt in range(50):
            size = g.rng.uniform(lo, hi, 2)
            spot = g.sample_xy(1, cfg, margin=size.max() / 2 + 0.3)
            if not len(spot):
                continue
            (x, y) = spot[0]
            box = np.array([x - size[0] / 2, y - size[1] / 2, x + size[0] / 2, y + size[1] / 2])
            gap = 0.4  # flattened rim of 0.3 m plus clearance
            if all(box[0] > b[2] + gap or box[2] < b[0] - gap or box[1] > b[3] + gap or box[3] < b[1] - gap
                   for b in placed):  # fmt: skip
                break
        else:
            continue
        placed.append(box)
        ellipse = bool(g.rng.uniform() < 0.5)
        depth = upto(g.rng, cfg.sand_depth, difficulty)
        win = g.window(x, y, size.max() / 2 + 0.5)
        dx, dy = (g.X[win] - x) / (size[0] / 2), (g.Y[win] - y) / (size[1] / 2)
        # signed distance-like coordinate: < 1 inside the patch outline
        q = np.hypot(dx, dy) if ellipse else np.maximum(np.abs(dx), np.abs(dy))
        surface = float(h[win][q <= 1.0].mean())
        # flatten the hardpan to the surface level within the slab footprint (+0.05 m), blend out over 0.3 m
        dist_box = np.maximum(np.abs(g.X[win] - x) - size[0] / 2, np.abs(g.Y[win] - y) - size[1] / 2) - 0.05
        flat = 1.0 - smoothstep(dist_box / 0.3)
        h[win] = flat * surface + (1.0 - flat) * h[win]
        # lower the patch interior by depth with a 0.1 m ramp at its outline; keep the hardpan roughness below
        ramp = smoothstep((1.0 - q) * size.min() / 2 / 0.1)
        bottom = np.minimum(surface - depth + 0.5 * rough[win], surface - 0.03)
        h[win] = ramp * bottom + (1.0 - ramp) * h[win]
        crust = bool(g.rng.uniform() < cfg.crust_prob)
        patches.append({
            "x": x, "y": y, "size": size.tolist(), "shape": "ellipse" if ellipse else "rect", "class": name,
            "surface": surface, "bottom": float(h[win][q <= 1.0].min()), "depth": depth,
            "stiffness": float(g.rng.uniform(*cfg.stiffness)), "crust": crust,
        })  # fmt: skip
    cfg.features = {"patches": patches}
    return finish(g, h, cfg)


@height_field_to_mesh
def mixed_terrain(difficulty: float, cfg: MixedCfg) -> tuple[np.ndarray, np.ndarray]:
    """Gentle pyramid with gravel noise, stubs, pits, and trenches; feature densities halved."""
    g = Grid(cfg, difficulty)
    h = pyramid(g, cfg, lerp(cfg.slope, difficulty), False)
    h += g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
    cfg.features = {
        "stubs": add_stubs(g, h, cfg, difficulty, density_scale=0.5),
        "pits": add_pits(g, h, cfg, difficulty, density_scale=0.5),
        "trenches": add_trenches(g, h, cfg, difficulty),
    }
    return finish(g, h, cfg)


@height_field_to_mesh
def mesa_terrain(difficulty: float, cfg: MesaDropCfg) -> tuple[np.ndarray, np.ndarray]:
    """Raised mesa with terraces down to the edge band; each side is a cliff, a 45 deg scree, or rubble."""
    g = Grid(cfg, difficulty)
    mesa_height = lerp(cfg.mesa_height, difficulty)
    count = int(np.ceil(mesa_height / lerp(cfg.drop_height, difficulty)))
    drop = mesa_height / count
    run = g.half.min() - cfg.edge_width - cfg.platform_width / 2 - 0.5
    tread = min(lerp(cfg.tread_width, difficulty), run / count)
    walls = tread * np.arange(1, count + 1)
    # side of each pixel (the nearest border) and its wall type: 0 cliff, 1 scree, 2 rubble
    sides = np.argmin(np.stack([g.half[0] - g.X, g.half[0] + g.X, g.half[1] - g.Y, g.half[1] + g.Y]), axis=0)
    u = g.rng.uniform(size=4)
    types = np.where(u < cfg.scree_prob, 1, np.where(u < cfg.scree_prob + cfg.rubble_prob, 2, 0))
    kind = types[sides]
    r = g.r[..., None]
    cliff = drop * (r >= walls).sum(-1)
    # scree/rubble: the drop spread over a 45 deg ramp ending at the wall
    scree = drop * np.clip((r - walls + drop) / drop, 0.0, 1.0).sum(-1)
    rubble = scree + noise(g, 0.3, 0.3 * drop, 1) * np.clip(np.abs(scree - cliff) / drop * 4.0, 0.0, 1.0)
    h = np.choose(kind, [cliff, scree, rubble])
    h = np.where(g.r <= 0.0, 0.0, h)
    h += noise(g, 0.3, lerp(cfg.amplitude, difficulty), 1) * smoothstep(g.r / 0.3) * (np.abs(h - cliff) < 1e-6)
    h = np.where(g.p <= 0.0, mesa_height, h)
    cfg.features = {"drop": drop, "tread": tread, "walls": walls.tolist(), "wall_types": types.tolist()}
    return finish(g, h, cfg)


@height_field_to_mesh
def loose_rocks_terrain(difficulty: float, cfg: LooseRocksCfg) -> tuple[np.ndarray, np.ndarray]:
    """Gentle gravel; rock instances are only listed here and spawned by the importer and the rock event.

    Static rocks are flattened (buried) or whole, plus a dense quadrant of large rocks spaced about one body
    width apart (body jamming). The first ``len(cfg.dynamic_sizes)`` slots near the platform are the dynamic
    rocks, whose sizes are fixed for the whole world so that every environment shares one rock collection.
    """
    g = Grid(cfg, difficulty)
    h = g.fade * noise(g, lerp(cfg.unit_size, difficulty), lerp(cfg.amplitude, difficulty), cfg.interp_order)
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
    count = round(lerp(cfg.rocks_per_cell, difficulty))
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


##
# Configurations.
##


@configclass
class HarshTerrainCfg(HfTerrainBaseCfg):
    """Shared fields of every Harsh World property. Knobs are ``(min, max)`` pairs interpolated by difficulty."""

    name: str = ""
    """Property name (the key in :attr:`HarshWorldCfg.properties`, filled in by it); seeds the cell's random
    generator."""

    horizontal_scale: float | None = None
    vertical_scale: float | None = None
    """Both inherited from the generator."""

    convert_to_heightfield: bool = True
    """Collide against a Newton heightfield of the same grid. Against the triangle mesh, simulation was 12x slower,
    the full world ran out of memory, and the convex rocks fell through (see ``REPORT.md``)."""

    zone: str = "rigid"
    """Runtime interaction: one of :data:`ZONES`."""

    platform_width: float = 2.0
    """Side [m] of the flat spawn square at the center."""

    edge_width: float = 1.0
    """Width [m] of the flat band along the cell border (heights are zero there)."""

    blend_width: float = 1.0
    """Distance [m] over which features fade in from the edge band and the platform."""

    unit_size: Range = (0.15, 0.5)
    """Facet size [m] of the gravel noise."""

    amplitude: Range = (0.05, 0.18)
    """Half range [m] of the gravel noise heights."""

    interp_order: int = 1
    """1: angular facets (linear interpolation); 3: smooth (cubic spline)."""


@configclass
class GravelCfg(HarshTerrainCfg):
    function: Callable = gravel_terrain
    pit_density: Range = (0.0, 1.5)
    """Pits per m^2."""
    pit_radius: Range = (0.06, 0.14)
    pit_depth: Range = (0.08, 0.25)
    pit_rim: Range = (0.0, 0.03)
    trench_count: Range = (0.0, 4.0)
    trench_length: Range = (2.0, 6.0)
    """Length [m] of each trench, uniform (not interpolated)."""
    trench_width: Range = (0.08, 0.16)
    trench_depth: Range = (0.10, 0.25)


@configclass
class StubFieldCfg(HarshTerrainCfg):
    function: Callable = stub_field_terrain
    unit_size: Range = (0.3, 0.3)
    amplitude: Range = (0.01, 0.01)
    stub_density: Range = (0.5, 3.0)
    stub_radius: Range = (0.02, 0.05)
    stub_height: Range = (0.02, 0.06)
    root_count: Range = (2.0, 8.0)
    root_width: Range = (0.03, 0.05)
    root_height: Range = (0.02, 0.04)
    root_length: Range = (0.5, 3.0)


@configclass
class RoughSlopeCfg(HarshTerrainCfg):
    function: Callable = slope_terrain
    slope: Range = (0.45, 1.0)
    """Rise over run of every pyramid face."""
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.03, 0.10)
    ledge_count: Range = (0.0, 8.0)
    ledge_size: Range = (0.3, 0.8)
    ledge_height: Range = (0.05, 0.20)
    inverted: bool = False


@configclass
class RoughStairsCfg(HarshTerrainCfg):
    function: Callable = stairs_terrain
    step_height: Range = (0.10, 0.25)
    step_width: Range = (0.3, 0.6)
    width_jitter: float = 0.3
    """Relative spread of individual step widths, ``U(1 - j, 1 + j) * step_width``."""
    edge_jitter: Range = (0.0, 0.1)
    amplitude: Range = (0.01, 0.05)
    inverted: bool = False


@configclass
class SandFieldCfg(HarshTerrainCfg):
    function: Callable = sand_field_terrain
    zone: str = "sand"
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.02, 0.06)
    patch_count: Range = (6.0, 14.0)
    patch_size_classes: dict = {"foot": (0.3, 0.5), "pair": (0.8, 1.5), "body": (2.5, 5.0)}
    """Side ranges [m] of each patch class (uniform per side)."""
    patch_class_weights: dict = {"foot": 3.0, "pair": 2.0, "body": 2.0}
    sand_depth: Range = (0.08, 0.20)
    stiffness: Range = (800.0, 3000.0)
    """Slab contact stiffness [N/m], uniform per patch."""
    crust_prob: float = 0.0
    """Probability that a patch is a collapsing crust (property ``crust_collapse``)."""


@configclass
class MixedCfg(HarshTerrainCfg):
    function: Callable = mixed_terrain
    slope: Range = (0.1, 0.3)
    unit_size: Range = (0.15, 0.4)
    amplitude: Range = (0.03, 0.12)
    stub_density: Range = (0.5, 3.0)
    stub_radius: Range = (0.02, 0.05)
    stub_height: Range = (0.02, 0.06)
    root_count: Range = (2.0, 8.0)
    root_width: Range = (0.03, 0.05)
    root_height: Range = (0.02, 0.04)
    root_length: Range = (0.5, 3.0)
    pit_density: Range = (0.0, 1.5)
    pit_radius: Range = (0.06, 0.14)
    pit_depth: Range = (0.08, 0.25)
    pit_rim: Range = (0.0, 0.03)
    trench_count: Range = (0.0, 2.0)
    trench_length: Range = (2.0, 6.0)
    trench_width: Range = (0.08, 0.16)
    trench_depth: Range = (0.10, 0.25)


@configclass
class MesaDropCfg(HarshTerrainCfg):
    function: Callable = mesa_terrain
    mesa_height: Range = (3.0, 4.0)
    drop_height: Range = (1.5, 2.0)
    tread_width: Range = (4.0, 4.0)
    """Spec (0.6, 2.0), (0.3, 1.0), (1.0, 2.5): no flight (all feet off > 0.3 s) in 60 s. The policy steps down a
    drop of up to 1 m (front feet land before the hind feet leave), and on narrow treads the robots spent most of
    the time on a 12 m wide top. Now two terraces of 1.7-2.0 m, 4 m deep, with the top edge 2 m beyond the
    platform (see ``REPORT.md``)."""
    scree_prob: float = 0.2
    rubble_prob: float = 0.1
    """Spec 0.4 (and 0.2 rubble): the robots step down screes, so flights came from the cliff sides only, about
    2.4 per minute for the 12 robots of the default world; 70 % cliffs make a flight-free minute unlikely."""
    amplitude: Range = (0.02, 0.06)


@configclass
class LooseRocksCfg(HarshTerrainCfg):
    function: Callable = loose_rocks_terrain
    unit_size: Range = (0.2, 0.4)
    amplitude: Range = (0.01, 0.03)
    rocks_per_cell: Range = (150.0, 400.0)
    """Static rocks per cell (the dynamic ones come on top)."""
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
    """Sizes [m] of the dynamic rocks, set by :class:`HarshWorldCfg` (shared by all cells)."""
    dynamic_ring: float = 3.0
    """Radial spread [m] of the dynamic rocks, starting 0.6 m outside the platform."""
