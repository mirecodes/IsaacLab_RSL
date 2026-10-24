# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry tests of the terrain templates and the tile grid (no simulator).

    python -m pytest scripts/reinforcement_learning/terrain/test_terrain.py -s

Measured values are printed with a ``[T1]`` / ``[T2]`` / ``[T6]`` prefix (the spec's test ids) so the log keeps them.
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from build import build, plot_map  # noqa: E402

from isaaclab.terrains.height_field.utils import to_discrete_units  # noqa: E402

from isaaclab_tasks.forge.terrain import TERRAINS, task_id  # noqa: E402
from isaaclab_tasks.forge.terrain.mdp.generator import write_meta  # noqa: E402
from isaaclab_tasks.forge.terrain.templates.common import Grid, lerp, noise  # noqa: E402
from isaaclab_tasks.forge.terrain.templates.loose_rocks import rock_mesh  # noqa: E402
from isaaclab_tasks.forge.terrain.world_cfg import TerrainWorldCfg, default_terrains  # noqa: E402

HS, VS = 0.03, 0.005


@pytest.fixture(scope="module")
def gravel(tmp_path_factory):
    """The default gravel map (5 difficulties x 5 tiles)."""
    world, generator, meta, seconds = build("gravel", 42)
    out = tmp_path_factory.mktemp("map")
    write_meta(str(out / "a.yaml"), meta)
    plot_map(meta, str(out / "a.png"), generator.map_heights()[0])
    return {"world": world, "gen": generator, "meta": meta, "seconds": seconds, "out": out}


def tile(name, difficulty=1.0, **kwargs):
    """Generate one tile of a template directly; returns (cfg with features and heights, origin)."""
    cfg = default_terrains()[name].replace(name=name, size=(24.0, 24.0), horizontal_scale=HS, vertical_scale=VS,
                                           slope_threshold=0.75, **kwargs)  # fmt: skip
    _, origin = cfg.function(difficulty, cfg)
    return cfg, origin


def test_catalog_matches_registration():
    assert tuple(default_terrains()) == TERRAINS
    assert task_id("stub_field", "A2Robust") == "IsaacContrib-Terrain-StubField-A2Robust"


def test_layout_and_files(gravel):
    meta, out = gravel["meta"], gravel["out"]
    assert (out / "a.yaml").exists() and (out / "a.png").exists()
    assert meta["num_rows"] == 5 and meta["num_cols"] == 5 and meta["terrain"] == "gravel"
    assert sorted({t["difficulty"] for t in meta["tiles"]}) == [0.4, 0.55, 0.7, 0.85, 1.0]
    print(f"[T1] generation time: {gravel['seconds']:.1f} s for 5x5 gravel tiles")
    assert gravel["seconds"] < 60.0


def test_deterministic(gravel):
    _, _, meta, _ = build("gravel", 42)
    write_meta(str(gravel["out"] / "b.yaml"), meta)
    assert (gravel["out"] / "a.yaml").read_bytes() == (gravel["out"] / "b.yaml").read_bytes()


def test_tiles_of_a_level_differ(gravel):
    heights = gravel["gen"].tile_heights
    for row in range(5):
        first = heights[(row, 0)]
        assert all(not np.array_equal(first, heights[(row, col)]) for col in range(1, 5)), row


@pytest.mark.parametrize("terrain", TERRAINS)
def test_edges_and_platforms(terrain):
    """Every template: flat 1 m edge band (tiles meet without a step) and a flat spawn platform."""
    _, generator, _, seconds = build(terrain, 42, tiles_per_level=2)
    worst_edge, worst_platform = 0.0, 0.0
    band, k = round(1.0 / HS), round(1.0 / HS)
    for h in generator.tile_heights.values():
        edge = np.concatenate([h[:band].ravel(), h[-band:].ravel(), h[:, :band].ravel(), h[:, -band:].ravel()])
        worst_edge = max(worst_edge, np.abs(edge).max())
        c = (np.array(h.shape) - 1) // 2
        worst_platform = max(worst_platform, np.ptp(h[c[0] - k + 1 : c[0] + k, c[1] - k + 1 : c[1] + k]))
    print(f"[T1] {terrain}: max |z| in the edge bands {worst_edge * 100:.2f} cm, platform range {worst_platform}, "
          f"{seconds:.1f} s for 3x2 tiles")  # fmt: skip
    assert worst_edge < 0.02 and worst_platform == 0.0


def test_gravel_correlation_amplitude_and_facets():
    for unit, amp in ((0.15, 0.05), (0.3, 0.1), (0.5, 0.18)):
        cfg = default_terrains()["gravel"].replace(name="gravel", size=(24.0, 24.0), horizontal_scale=HS,
                                                   vertical_scale=VS)  # fmt: skip
        h = noise(Grid(cfg, 1.0), unit, amp, 1)
        z = h - h.mean()
        # e-folding length of the autocorrelation along x
        lags = np.arange(1, round(2 * unit / HS))
        corr = np.array([np.mean(z[:-k] * z[k:]) for k in lags]) / np.mean(z * z)
        length = lags[np.argmax(corr < np.exp(-1))] * HS
        # linear interpolation of iid samples: R(t) = 1 - 1.5 t^2 + 0.75 t^3, e-folding at 0.86 unit
        print(f"[T1] gravel unit {unit}: correlation length {length:.3f} m (0.86 x unit = {0.86 * unit:.3f}), "
              f"max |h| {np.abs(h).max():.3f} m (amplitude {amp})")  # fmt: skip
        assert abs(length - 0.86 * unit) < 0.3 * unit
        assert abs(np.abs(h).max() - amp) < 0.2 * amp
        # facets: the second difference concentrates on the coarse grid lines (the facet edges)
        d2 = np.abs(np.diff(h, 2, axis=0)).sum(axis=1)
        top = np.sort(d2)[::-1][: round(len(d2) * HS / unit * 3)].sum() / d2.sum()
        print(f"[T1] share of |d2h| on the facet edges: {top:.2f}")
        assert top > 0.9


def test_stub_visibility():
    cfg, _ = tile("stub_field")
    stubs = [s for s in cfg.features["stubs"] if not s.get("root")]
    assert all(s["radius"] <= 0.05 for s in stubs)
    rng = np.random.default_rng(0)
    seen = []
    for _ in range(20):
        offset = rng.uniform(0, 0.1, 2)
        for s in stubs:
            # grid rays within the stub footprint that see more than half its height
            lo = (np.array([s["x"], s["y"]]) - s["radius"] - offset) / 0.1
            hi = (np.array([s["x"], s["y"]]) + s["radius"] - offset) / 0.1
            gx, gy = np.arange(np.ceil(lo[0]), np.floor(hi[0]) + 1), np.arange(np.ceil(lo[1]), np.floor(hi[1]) + 1)
            px, py = np.meshgrid(gx * 0.1 + offset[0], gy * 0.1 + offset[1])
            rho = np.hypot(px - s["x"], py - s["y"]) / s["radius"]
            seen.append(bool(np.any(np.sqrt(np.clip(1 - rho**2, 0, None)) > 0.5)))
    ratio = np.mean(seen)
    print(f"[T1] stubs seen by a 0.1 m scanner grid: {ratio * 100:.1f} % of {len(stubs)} stubs")
    assert ratio < 0.35


def test_pits_in_range():
    for name in ("gravel", "mixed"):
        cfg, _ = tile(name)
        pits = cfg.features["pits"]
        assert pits
        lo_r, hi_r = cfg.pit_radius
        lo_d, hi_d = cfg.pit_depth
        assert all(lo_r <= p["radius"] <= hi_r and lo_d <= p["depth"] <= hi_d for p in pits)
        grid = Grid(cfg, 1.0)
        measured = []
        for p in pits:
            win = grid.window(p["x"], p["y"], 2.0 * p["radius"])
            rho = np.hypot(grid.X[win] - p["x"], grid.Y[win] - p["y"])
            h = cfg.heights[win]
            measured.append(np.median(h[(rho > 1.5 * p["radius"])]) - h[rho < 0.5 * p["radius"]].min())
        ratio = np.array(measured) / np.array([p["depth"] for p in pits])
        print(f"[T1] {name}: {len(pits)} pits, measured / set depth median {np.median(ratio):.2f}")
        assert np.median(ratio) > 0.5


@pytest.mark.parametrize("terrain", ["sand_field", "crust_collapse"])
def test_sand_slab_meets_hardpan(terrain):
    """T2 (geometric): the hardpan just outside each slab is within 1 cm of the slab top."""
    _, generator, meta, _ = build(terrain, 42, tiles_per_level=3)
    heights, corner = generator.map_heights()
    worst = 0.0
    for p in meta["sand_patches"]:
        hx, hy = p["size"][0] / 2 + 0.05, p["size"][1] / 2 + 0.05
        ring = []
        for t in np.linspace(-1, 1, 41):
            ring += [(p["x"] + t * hx, p["y"] - hy - 0.04), (p["x"] + t * hx, p["y"] + hy + 0.04),
                     (p["x"] - hx - 0.04, p["y"] + t * hy), (p["x"] + hx + 0.04, p["y"] + t * hy)]  # fmt: skip
        ij = np.rint((np.array(ring) - corner) / HS).astype(int)
        worst = max(worst, np.abs(heights[ij[:, 0], ij[:, 1]] - p["surface"]).max())
    crusts = sum(p["crust"] for p in meta["sand_patches"])
    print(f"[T2] {terrain}: max |hardpan - slab top| around the slabs {worst * 100:.2f} cm over "
          f"{len(meta['sand_patches'])} patches ({crusts} crusts)")  # fmt: skip
    assert worst <= 0.01 + VS
    assert (crusts > 0) == (terrain == "crust_collapse")


def test_mesa():
    """T6: terrace drops within range, zero edge, platform at mesa height."""
    for d in (0.0, 0.5, 1.0):
        cfg, origin = tile("mesa_drop", d)
        mesa = lerp(cfg.mesa_height, d)
        drop = cfg.features["drop"]
        print(f"[T6] mesa d={d}: height {origin[2]:.2f} m (set {mesa:.2f}), drop {drop:.2f} m, "
              f"walls {cfg.features['wall_types']}")  # fmt: skip
        assert abs(origin[2] - mesa) <= VS and drop <= lerp(cfg.drop_height, d) + 1e-9
        assert np.abs(cfg.heights[:30]).max() < 0.02


def test_rock_templates_and_slots():
    """T6: 16 templates are watertight convex hulls with at most 60 faces; every tile has its dynamic rock slots."""
    for template in range(16):
        mesh = rock_mesh(42, template, 0.2, (0.7, 0.6), 0.0)
        assert mesh.is_watertight and mesh.is_convex and len(mesh.faces) <= 60
    world, _, meta, _ = build("loose_rocks", 42, tiles_per_level=2)
    for t in meta["tiles"]:
        slots = [r for r in meta["rocks"] if r["dynamic"] and (r["row"], r["col"]) == (t["row"], t["col"])]
        assert len(slots) == world.rocks.dynamic_rocks


def test_knob_change_from_world_cfg():
    """T5: a template knob and the grid shape, changed from TerrainWorldCfg alone."""
    _, base, _, _ = build("gravel", 1, num_levels=1, tiles_per_level=2)
    terrains = default_terrains()
    terrains["gravel"].unit_size = (0.4, 0.4)
    world, changed, meta, _ = build("gravel", 1, num_levels=1, tiles_per_level=2, terrains=terrains)
    assert meta["num_rows"] == 1 and meta["num_cols"] == 2 and world.num_envs == 2 * world.robots_per_tile
    assert not np.array_equal(changed.tile_heights[(0, 0)], base.tile_heights[(0, 0)])


def test_yaml_loads(gravel):
    with open(gravel["out"] / "a.yaml") as f:
        meta = yaml.safe_load(f)
    assert meta["zone"] == "rigid" and len(meta["tiles"]) == 25
    assert to_discrete_units(24.0, HS) == 800
    assert TerrainWorldCfg().num_envs == 100
