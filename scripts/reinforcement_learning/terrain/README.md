# Terrain scripts

Tools for the terrain templates in [`forge/terrain`](../../../source/isaaclab_tasks/isaaclab_tasks/forge/terrain):
one map per terrain type (5 difficulty rows x 5 tiles), played by a trained A2 policy as
`IsaacContrib-Terrain-<Type>-A2{Robust,Energy}`. Design, decisions, and measured results:
[`forge/terrain/REPORT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/forge/terrain/REPORT.md). Every knob lives in
[`world_cfg.py`](../../../source/isaaclab_tasks/isaaclab_tasks/forge/terrain/world_cfg.py).

| File | What it does | Needs |
| --- | --- | --- |
| [build.py](build.py) | Generates maps without a simulator: description yaml, top-down PNG, height map | CPU only |
| [test_terrain.py](test_terrain.py) | Geometry tests of every template and the tile grid (pytest) | CPU only |
| [probe.py](probe.py) | Simulator probes: `ground`, `sand`, `collapse`, `rocks`, `slip`, `smoke`; `view` renders PNGs | GPU (Newton) |

Failure analysis is [`fault_analysis/`](../../forge/surveillance/README.md)'s job: its `collect.py` and `render.py` take
these tasks like any other (tested with `SandField`).

## Terrain types

| Type | Task suffix | Zone | Runtime physics |
| --- | --- | --- | --- |
| `rough_field` | `RoughField` | rigid | – |
| `pit_field` | `PitField` | rigid | – |
| `stub_field` | `StubField` | rigid | – |
| `rough_slope` | `RoughSlope` | rigid | – |
| `rough_stairs` | `RoughStairs` | rigid | – |
| `slip_slope` | `SlipSlope` | slip | low foot friction |
| `sand_field` | `SandField` | sand | compliant slabs, drag |
| `gravel_field` | `GravelField` | rigid | dynamic rocks |
| `crust_collapse` | `CrustCollapse` | sand | slabs, drag, collapsing crusts |

`rough_slope` and `rough_stairs` rows run from 0 deg (first row) to 45 deg (last row) and chain into one hillside
along x; the columns listed in `smooth_tiles` are plain, the others rough.

## Commands

```bash
P=scripts/reinforcement_learning/terrain
CK=/workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt

# 1. maps only (any machine; local container or Euler)
python $P/build.py all --out logs/terrain/build        # or one type: build.py rough_slope
python -m pytest $P/test_terrain.py -s

# 2. physics probes, on Euler
euler enqueue tr_sand -- python $P/probe.py sand
euler enqueue tr_collapse -- python $P/probe.py collapse --collapse_mode disable
euler enqueue tr_rocks -- python $P/probe.py rocks
euler enqueue tr_slip -- python $P/probe.py slip --checkpoint $CK
euler enqueue tr_smoke -- python $P/probe.py smoke --terrain rough_slope --checkpoint $CK

# 3. look at a map: overview_{a,b}.png, tile_level<i>_d<d>.png, robot_level<i>_d<d>.png
euler enqueue tr_view -- python $P/probe.py view /workspace/isaaclab/logs/terrain/view/rough_slope \
    --terrain rough_slope --checkpoint $CK
rsync -a euler:/cluster/project/rsl/hyelee/euler-toolkit/isaaclab3/logs/terrain/view logs/terrain/

# 4. play a trained policy on one terrain type and record failures (fault_analysis, unchanged)
euler enqueue fault_slope -- python scripts/forge/surveillance/collect.py \
    --task IsaacContrib-Terrain-RoughSlope-A2Robust --num_envs 100 --max_steps 15000 --checkpoint $CK
```

`--num_envs` should equal `num_levels x tiles_per_level x robots_per_tile` (100 by default) so every tile gets the
same number of robots; with fewer, the robots of each column cycle through the rows. The importer writes the map
description to `logs/terrain/<type>_seed<seed>.yaml`; `build.py --plot <that yaml>` redraws it.

`--device cpu` runs the probes in the local Mac container, but Newton's CPU contacts are unreliable (robots fall
through the terrain), so CPU runs only check that the code runs.

## Main knobs (`world_cfg.py`)

| Knob | Default | Meaning |
| --- | --- | --- |
| `TerrainWorldCfg.terrain` | per task | terrain type of the map |
| `seed` | 42 | terrain, spawns, commands (`collect.py --seed` overrides it) |
| `num_levels`, `difficulty_range` | 5, (0.4, 1.0) | difficulty rows: 0.4, 0.55, 0.7, 0.85, 1.0 (a template's own `difficulty_range` wins: inclines use (0, 1)) |
| `tiles_per_level`, `tile_size` | 5, 24 m | tiles per row, each with its own random layout (map 120 x 120 m) |
| `horizontal_scale`, `vertical_scale` | 0.03 m, 0.005 m | height-field resolution |
| `robots_per_tile`, `platform_width` | 4, 3 m | robots spawn on the tile's flat center platform |
| `lin_vel_x`, `lin_vel_y`, `ang_vel_z`, `command_resampling_s` | (-0.5, 1.5), (±0.5), (±1), (5, 10) s | random commands (robust policy) |
| `terrains["<type>"].<knob>` | see `templates/<type>.py` | template knobs, `(min, max)` pairs interpolated by difficulty |
| `terrains["rough_slope"].slope_deg`, `.smooth_tiles` | (0, 45), [0, 1] | incline angle; plain columns (the rest are rough) |
| `unit_size`, `amplitude`, `fold_size`, `fold_amplitude`, `octaves` | per template | fractal roughness: fine bumps, rolling folds, layers in between |
| `terrains["sand_field"].cell_size`, `.sand_prob` | (1.2, 4.5) m, (0.25, 0.45) | patchwork cells and the share holding sand |
| `slip.mu_slip_easy`, `slip.mu_slip_hard`, `slip.slip_prob`, `slip.resample_on_touchdown` | (0.05, 0.35), (0.03, 0.25), 0.02, True | low-friction sample per foot and touchdown on slip terrain (a slip lasts one stance) |
| `sand.c_drag`, `sand.c_suction`, `sand.force_limit` | (20, 80), (30, 150) N s/m, 60 N | drag on feet in the sand |
| `collapse.collapse_force`, `collapse_mode`, `recover_time` | (1.5, 3) x 98 N, `soften`, 30 s | crust collapse |
| `rocks.dynamic_rocks`, `rocks.size`, `rocks.friction` | 8 per robot, (0.10, 0.45) m, (0.8, 1.0) | loose stones of `gravel_field`, all dynamic; they follow their robot |
| `rocks.recycle_interval`, `recycle_radius`, `ahead`, `sector` | 1 s, 3 m, (1.2, 3.0) m, 1 rad | stones left behind move to free spots ahead |
| `terrains["gravel_field"].flat_frac`, `.flat_height`, `.width_ratio` | 0.8, (0.15, 0.4), (0.5, 0.9) | stone shapes |
