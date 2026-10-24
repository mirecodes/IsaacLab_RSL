# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Acceptance tests T1 (geometry, no simulator) plus the geometric parts of T2, T5, and T6.

    python -m pytest scripts/reinforcement_learning/harsh_world/test_terrain.py -s

Measured values are printed with a ``[T1]`` prefix so the log keeps the numbers.
"""

import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from build_world import build, plot_world  # noqa: E402

from isaaclab.terrains.height_field.utils import to_discrete_units  # noqa: E402

from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp import terrains as T  # noqa: E402
from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp.world import rock_mesh, write_world_meta  # noqa: E402

HS, VS = 0.03, 0.005


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    hw, generator, meta, seconds = build(42, full=True)
    out = tmp_path_factory.mktemp("world")
    write_world_meta(str(out / "a.yaml"), meta)
    heights, _ = generator.world_heights()
    plot_world(meta, str(out / "a.png"), heights)
    return {"hw": hw, "gen": generator, "meta": meta, "seconds": seconds, "out": out, "heights": heights}


def cell(name, difficulty=1.0, **kwargs):
    """Generate one property cell directly; returns (cfg with features and heights, origin)."""
    cfg = T.__dict__[name](name=name, size=(24.0, 24.0), horizontal_scale=HS, vertical_scale=VS,
                           slope_threshold=0.75, **kwargs)  # fmt: skip
    cfg.seed = 0
    _, origin = cfg.function(difficulty, cfg)
    return cfg, origin


def test_layout_and_files(world):
    meta, out = world["meta"], world["out"]
    assert (out / "a.yaml").exists() and (out / "a.png").exists()
    assert meta["num_rows"] == 3 and meta["num_cols"] == 10
    print(f"[T1] generation time: {world['seconds']:.1f} s for 3x10 cells")
    assert world["seconds"] < 60.0


def test_deterministic(world):
    _, _, meta, _ = build(42, full=True)
    write_world_meta(str(world["out"] / "b.yaml"), meta)
    assert (world["out"] / "a.yaml").read_bytes() == (world["out"] / "b.yaml").read_bytes()


def test_columns_match_sub_terrains(world):
    gen, meta = world["gen"], world["meta"]
    names = list(gen.cfg.sub_terrains)
    # Isaac Lab's curriculum column rule with equal proportions
    cumulative = np.cumsum(np.full(len(names), 1.0 / len(names)))
    formula = [names[np.min(np.where(c / gen.cfg.num_cols + 0.001 < cumulative)[0])] for c in range(gen.cfg.num_cols)]
    recorded = [next(c["property"] for c in meta["cells"] if c["col"] == col) for col in range(gen.cfg.num_cols)]
    assert recorded == names == formula


def test_edges_and_platforms(world):
    gen = world["gen"]
    worst_edge, worst_platform = 0.0, 0.0
    for (row, col), h in gen.cell_heights.items():
        band = round(1.0 / HS)
        edge = np.concatenate([h[:band].ravel(), h[-band:].ravel(), h[:, :band].ravel(), h[:, -band:].ravel()])
        worst_edge = max(worst_edge, np.abs(edge).max())
        c, k = (np.array(h.shape) - 1) // 2, round(1.0 / HS)
        worst_platform = max(worst_platform, np.ptp(h[c[0] - k + 1 : c[0] + k, c[1] - k + 1 : c[1] + k]))
        assert np.abs(edge).max() < 0.02, (row, col, gen.cells[row][col]["property"])
    print(f"[T1] max |z| in the 1 m edge bands: {worst_edge * 100:.2f} cm; max platform height range: {worst_platform}")
    assert worst_platform == 0.0


def test_gravel_correlation_amplitude_and_facets():
    for unit, amp in ((0.15, 0.05), (0.3, 0.1), (0.5, 0.18)):
        cfg = T.GravelCfg(name="gravel", size=(24.0, 24.0), horizontal_scale=HS, vertical_scale=VS)
        grid = T.Grid(cfg, 1.0)
        h = T.noise(grid, unit, amp, 1)
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
    cfg, _ = cell("StubFieldCfg")
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
    for name in ("GravelCfg", "MixedCfg"):
        cfg, _ = cell(name)
        pits = cfg.features["pits"]
        assert pits
        lo_r, hi_r = cfg.pit_radius
        lo_d, hi_d = cfg.pit_depth
        assert all(lo_r <= p["radius"] <= hi_r and lo_d <= p["depth"] <= hi_d for p in pits)
        grid = T.Grid(cfg, 1.0)
        measured = []
        for p in pits:
            win = grid.window(p["x"], p["y"], 2.0 * p["radius"])
            rho = np.hypot(grid.X[win] - p["x"], grid.Y[win] - p["y"])
            h = cfg.heights[win]
            measured.append(np.median(h[(rho > 1.5 * p["radius"])]) - h[rho < 0.5 * p["radius"]].min())
        ratio = np.array(measured) / np.array([p["depth"] for p in pits])
        print(f"[T1] {name}: {len(pits)} pits, measured / set depth median {np.median(ratio):.2f}")
        assert np.median(ratio) > 0.5


def test_sand_slab_meets_hardpan(world):
    """T2 (geometric): the hardpan just outside each slab is within 1 cm of the slab top."""
    meta, heights, corner = world["meta"], *world["gen"].world_heights()
    worst = 0.0
    for p in meta["sand_patches"]:
        hx, hy = p["size"][0] / 2 + 0.05, p["size"][1] / 2 + 0.05
        ring = []
        for t in np.linspace(-1, 1, 41):
            ring += [(p["x"] + t * hx, p["y"] - hy - 0.04), (p["x"] + t * hx, p["y"] + hy + 0.04),
                     (p["x"] - hx - 0.04, p["y"] + t * hy), (p["x"] + hx + 0.04, p["y"] + t * hy)]  # fmt: skip
        ij = np.rint((np.array(ring) - corner) / HS).astype(int)
        worst = max(worst, np.abs(heights[ij[:, 0], ij[:, 1]] - p["surface"]).max())
    count = len(meta["sand_patches"])
    print(f"[T2] max |hardpan - slab top| around the slabs: {worst * 100:.2f} cm over {count} patches")
    assert worst <= 0.01 + VS


def test_mesa():
    """T6: terrace drops within range, zero edge, platform at mesa height."""
    for d in (0.0, 0.5, 1.0):
        cfg, origin = cell("MesaDropCfg", d)
        mesa = T.lerp(cfg.mesa_height, d)
        drop = cfg.features["drop"]
        print(f"[T6] mesa d={d}: height {origin[2]:.2f} m (set {mesa:.2f}), drop {drop:.2f} m, "
              f"walls {cfg.features['wall_types']}")  # fmt: skip
        assert abs(origin[2] - mesa) <= VS and drop <= T.lerp(cfg.drop_height, d) + 1e-9
        assert np.abs(cfg.heights[:30]).max() < 0.02


def test_rock_templates():
    """T6: 16 templates are watertight convex hulls with at most 60 faces."""
    for template in range(16):
        mesh = rock_mesh(42, template, 0.2, (0.7, 0.6), 0.0)
        assert mesh.is_watertight and mesh.is_convex and len(mesh.faces) <= 60


def test_single_file_change():
    """T5: one property off and one knob changed, from HarshWorldCfg alone."""
    overrides = {"disabled": ["stub_field", "mesa_drop", "loose_rocks", "crust_collapse"], "num_rows": 1}
    hw, generator, meta, _ = build(1, full=False, overrides=overrides)
    assert meta["num_cols"] == 6 and "stub_field" not in meta["properties"]
    hw.properties["gravel"].unit_size = (0.4, 0.4)
    from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp.world import HarshWorldGenerator

    changed = HarshWorldGenerator(hw.terrain_generator())
    assert not np.array_equal(changed.cell_heights[(0, 0)], generator.cell_heights[(0, 0)])
    assert np.array_equal(changed.cell_heights[(0, 1)], generator.cell_heights[(0, 1)])


def test_yaml_loads(world):
    with open(world["out"] / "a.yaml") as f:
        meta = yaml.safe_load(f)
    assert {c["zone"] for c in meta["cells"]} == {"rigid", "slip", "sand"}
    assert to_discrete_units(24.0, HS) == 800
