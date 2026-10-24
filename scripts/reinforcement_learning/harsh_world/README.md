# Harsh World scripts

Tools for the Harsh World tasks (`IsaacContrib-Velocity-A2HarshWorld-{Robust,Energy}[-Full]`). Design, decisions,
knobs, and measured results: [`forge/velocity/a2_harsh_world/REPORT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/forge/velocity/a2_harsh_world/REPORT.md).
Every knob lives in [`harsh_world_cfg.py`](../../../source/isaaclab_tasks/isaaclab_tasks/forge/velocity/a2_harsh_world/harsh_world_cfg.py).

| File | What it does | Needs |
| --- | --- | --- |
| [build_world.py](build_world.py) | Generates the world without a simulator: `world_meta.yaml`, top-down PNG, height map | CPU only |
| [test_terrain.py](test_terrain.py) | Acceptance tests T1 (geometry) and the geometric parts of T2/T5/T6 (pytest) | CPU only |
| [probe.py](probe.py) | Simulator probes: `ground`, `sand`, `collapse`, `rocks`, `slip`, `smoke`, `faults` (T2-T4, T6) | GPU (Newton) |

Failures are recorded with the other agent's [`fault_analysis/collect.py`](../fault_analysis/collect.py), unchanged:
the zone labels are an observation group, so it stores them per frame as `obs_zone_<field>`.

## Commands

```bash
# 1. world only (any machine; local container or Euler)
python scripts/reinforcement_learning/harsh_world/build_world.py --out logs/harsh_world/build [--full] [--seed 42]
python -m pytest scripts/reinforcement_learning/harsh_world/test_terrain.py -s

# 2. sand probe (and the other probes), on Euler
euler enqueue hw_probe_sand -- python scripts/reinforcement_learning/harsh_world/probe.py sand
euler enqueue hw_probe_collapse -- python scripts/reinforcement_learning/harsh_world/probe.py collapse --collapse_mode disable
# T6 rock stability with the spec's 1500 rocks (16 per robot x 94 robots; the default is 4 per robot)
euler enqueue hw_probe_rocks -- python scripts/reinforcement_learning/harsh_world/probe.py rocks \
    --robots_per_cell 94 --set rocks.dynamic_rocks=16
euler enqueue hw_probe_slip -- python scripts/reinforcement_learning/harsh_world/probe.py slip \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt

# 3. play a trained policy in the Harsh World and record failures (5 min of sim time at 84 robots = 15000 steps)
euler enqueue hw_collect_robust -- python scripts/reinforcement_learning/fault_analysis/collect.py \
    --task IsaacContrib-Velocity-A2HarshWorld-Robust --num_envs 84 --max_steps 15000 --max_events 100000 \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt
python scripts/reinforcement_learning/harsh_world/probe.py faults <collection dir>   # faults per property (T4)
```

`collect.py --num_envs` must equal `rows x columns x robots_per_cell` (84 for the 7-property world, 120 for `-Full`),
or the robots are not spread evenly over the cells. The importer writes `logs/harsh_world/world_meta_seed<seed>.yaml`;
`build_world.py --plot <that yaml>` redraws it.

`--device cpu` runs the probes in the local Mac container, but Newton's CPU contacts are unreliable (robots fall
through the terrain, also in the unmodified robust task), so CPU runs only check that the code runs.

`euler exec` runs a probe right away on the running holder's GPU; `euler enqueue` queues it behind other jobs.

**Speed (T4, RTX 4090):** 2.3–2.4 x real time at 84 robots, 1.5–1.6 x at 120 (`-Full`); the unmodified trained
task runs at 3.1 x at 84 robots, so the spec's 5 x is out of reach. The spec's fallback (`horizontal_scale` 0.04,
20 m cells) changed nothing (1.93 vs 1.90). Details in REPORT §5.3.

## Main knobs (`harsh_world_cfg.py`)

| Knob | Default | Meaning |
| --- | --- | --- |
| `HarshWorldCfg.seed` | 42 | terrain, spawns, commands |
| `region_size`, `num_rows`, `difficulty_range` | 24 m, 3, (0.4, 1.0) | cell side, difficulty rows (one difficulty per row) |
| `horizontal_scale`, `vertical_scale` | 0.03 m, 0.005 m | height-field resolution |
| `properties`, `disabled` | 10 properties, 3 disabled | columns, in order; `-Full` tasks enable all 10 |
| `robots_per_cell` | 4 | `num_envs = rows x columns x robots_per_cell` |
| `lin_vel_x`, `lin_vel_y`, `ang_vel_z`, `command_resampling_s` | (-0.5, 1.5), (±0.5), (±1), (5, 10) s | random commands (robust policy) |
| `slip.mu_slip_easy`, `slip.mu_slip_hard`, `slip.slip_prob` | (0.05, 0.35), (0.03, 0.25), 0.5 | low-friction sample on entering a slip cell |
| `sand.c_drag`, `sand.c_suction`, `sand.force_limit` | (20, 80), (30, 150) N s/m, 60 N | drag on feet in the sand |
| `SandFieldCfg.stiffness`, `sand_depth` | per property | slab spring [N/m], sand depth [m] |
| `collapse.collapse_force`, `collapse_mode`, `recover_time` | (1.5, 3) x 98 N, `soften`, 30 s | crust collapse |
| `rocks.dynamic_rocks`, `rocks.size`, `rocks.reset_interval` | 4 per robot, (0.06, 0.25) m, 300 s | dynamic rocks (`max_dynamic_rocks`: 4 x robots; 16 per robot cost 58 % of the real-time factor) |
| per property, e.g. `GravelCfg.unit_size`, `amplitude`, `pit_depth` | see `mdp/terrains.py` | `(min, max)` pairs interpolated by difficulty |
