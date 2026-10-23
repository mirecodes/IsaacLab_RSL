# Fault analysis

These scripts play a trained RSL-RL locomotion policy on thousands of robots in parallel. Each time a robot fails, they save the 5 s of motion that led to the failure. Saved events can then be rendered as videos that follow the robot over the same terrain.

| File | What it does |
| --- | --- |
| [collect.py](collect.py) | Headless parallel play. It saves the last `--buffer_s` seconds before every failure. |
| [render.py](render.py) | Replays saved events on the regenerated terrain and writes one robot-tracking mp4 per event. |

## Usage

On Euler (the container's cwd is the repo root, and `logs/` is the profile's log directory):

```bash
# collect: 4096 robots by default; stops at --max_events failures or --max_steps env steps
euler enqueue fault_a2_robust -- python scripts/reinforcement_learning/fault_analysis/collect.py \
    --task IsaacContrib-Velocity-A2Study-RobustTeacher \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt \
    --max_events 2000

# render selected events (default: all) of one collection
euler enqueue fault_render -- python scripts/reinforcement_learning/fault_analysis/render.py \
    /workspace/isaaclab/logs/fault_analysis/a2_study_robust_teacher/<collection> --events 0 5 12

euler fetch fault_analysis   # -> results/euler/isaaclab3/fault_analysis/... (events, terrain, videos)
```

Replay on the Linux machine, in the local Isaac container. `docker-compose.codes.yaml` mounts `~/hyeslee/results` at `/workspace/results`:

```bash
python scripts/reinforcement_learning/fault_analysis/render.py \
    /workspace/results/euler/isaaclab3/fault_analysis/a2_study_robust_teacher/<collection> \
    --events 12 --viz kit --real_time   # live window at wall-clock speed; the mp4 is written too
```

`collect.py` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--buffer_s` | 5.0 | Seconds kept before each failure. |
| `--max_events` / `--max_steps` | 1000 / none | When to stop. |
| `--foot_bodies` | `.*_foot` | Regex for the foot bodies. Contact on any other body is a failure. |
| `--contact_threshold` | 1.0 | Trigger force [N], the same as the stock `base_contact` termination. |
| `--disturbances` | off | Keep the task's random pushes and external wrench. |
| `--sub_terrains` | all | Keep only these sub-terrains of the task's generator, e.g. `hills rough_discrete` (heightfield rough ground, no blocks or stairs). |
| `--output_dir` | `logs/fault_analysis/<experiment>/<time>_<run>_<model>` | Where the collection is written. |
| `--num_envs`, `--seed`, `--checkpoint`, … | | Same as `play.py`. `--num_envs` defaults to 4096. |

`render.py` takes the collection directory, `--events <ids>`, `--eye dx dy dz` (camera offset from the base, default `-2 -2 1.2` m), `--real_time`, and `--viz` like `play.py`. It reapplies the collection's `--sub_terrains` and seed, so it needs no other arguments.

## Collection layout

Every file of a collection lives in one directory:

```
<collection>/
  meta.json        task, checkpoint, seed, step_dt, joint/body names, masses, limits, field shapes, CLI args
  env.yaml         full env config as played
  terrain.npz      vertices [V,3] and faces [F,3] (world frame), env_origins [N,3], terrain_origins [rows,cols,3]
  index.csv        one row per event: event, env_id, step, sim_time_s, episode_step, num_frames,
                   terrain_level, terrain_type, base_contact, trigger_bodies
  events/<id>.npz  one failure: [T, ...] arrays per field (T <= 250 at 50 Hz), plus env_id, step,
                   env_origin, terrain_level, terrain_type, trigger_bodies
  videos/<id>.mp4  written by render.py
```

Loading an event:

```python
import json, numpy as np

meta = json.load(open(f"{c}/meta.json"))
e = np.load(f"{c}/events/000012.npz")
terrain = np.load(f"{c}/terrain.npz")
e["root_pos_w"]  # [T, 3]; the last frame is the failure step
meta["terrain_columns"][e["terrain_type"]]  # sub-terrain name, e.g. "stairs_up"; level = e["terrain_level"]
```

Frame `t` is sampled at the end of env step `t`, before any reset. It holds the state `s_t`, the observation `o_t = O(s_t)`, the critic value `V(o_t)`, and the action `a_{t-1}` that produced `s_t`. Quaternions are `(x, y, z, w)`. Fields:

| Field | Shape | Source |
| --- | --- | --- |
| `root_pos_w`, `root_quat_w` | 3, 4 | base link pose |
| `root_lin_vel_w`, `root_ang_vel_w`, `root_lin_vel_b`, `root_ang_vel_b` | 3 each | base link twist (world, body) |
| `projected_gravity_b` | 3 | $\bar g$ |
| `root_acc_w` | 6 | base CoM linear + angular acceleration (IMU-like) |
| `joint_pos`, `joint_vel`, `joint_acc` | J | joint state |
| `joint_pos_target` | J | PD target sent to the actuators |
| `computed_torque`, `applied_torque` | J | actuator torque before / after saturation |
| `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_com_pos_w` | B×3, B×4, B×3, B×3 | per body, in `meta.body_names` order |
| `contact_force_w`, `contact_normal_force_w`, `contact_friction_force_w` | C×3 | per contact body, in `meta.contact_body_names` order |
| `contact_air_time`, `contact_time` | C | current air and contact time |
| `base_terrain_height` | – | mean height-scanner hit below the base (tasks with a `height_scanner`) |
| `command_<name>` | e.g. 3 | every command term, e.g. `command_base_velocity` = $(v_x, v_y, \omega_z)$ |
| `action` | A | raw policy output |
| `action_<term>_phase` | e.g. 4 | CPG phase of action terms that have one (robust teacher) |
| `obs_<group>` | per group | every observation group, as the policy sees it |
| `value` | – | critic $V(o_t)$ (PPO runners only) |

## Rule coverage

Only the trigger (non-foot contact) runs online. The other rules are meant to be checked offline on the saved window, and the table shows which fields each one reads.

| Rule | Fields |
| --- | --- |
| A non-foot contact (base vs. hip/thigh/calf) | `trigger_bodies`, `index.csv:base_contact`, `contact_*force_w` |
| A tilt | `projected_gravity_b` |
| A base height above terrain | `root_pos_w`, `base_terrain_height`, or exact via `terrain.npz` |
| B angular-rate spike, $\dot{\bar g}$ | `root_ang_vel_b`, `projected_gravity_b` |
| B impact acceleration | `root_acc_w`, `projected_gravity_b`, gait phase |
| B support polygon / capture point | `body_com_pos_w` + `meta.body_mass` (CoM), foot `body_pos_w`, `contact_*` |
| B critic value drop | `value` |
| B action jump | `action` |
| C trip during swing | foot `contact_force_w` (horizontal) / `contact_friction_force_w`, `action_*_phase` or `contact_air_time` |
| C foot slip | foot `body_lin_vel_w` while in contact |
| C late / early touchdown | `contact_air_time`, `contact_time`, `action_*_phase` |
| C excessive flight | `contact_air_time` |
| C foot sinking | foot `body_pos_w` vs. `terrain.npz` |
| C wedged foot | `joint_pos_target`, foot `body_pos_w`, `contact_force_w` |
| D velocity tracking, no progress, heading | `command_base_velocity`, `root_lin_vel_b`, `root_ang_vel_b`, `root_pos_w` |
| E torque saturation | `applied_torque`, `computed_torque`, `meta.joint_effort_limits` |
| E joint limit | `joint_pos`, `meta.joint_pos_limits` |
| E PD tracking error | `joint_pos_target - joint_pos` |
| E power spike | $\sum$ `applied_torque * joint_vel` |

## Decisions

Choices the request left open, and how they were made here:

- **Trigger = contact on any non-foot body, including calves.** As requested, this is the only online rule, and it replaces the task's own terminations except time-out. The robot is reset at the trigger, so every failure is exactly one event and contact that persists cannot fire twice. Base contact is still separable afterwards through `index.csv:base_contact` and `trigger_bodies`. If a shin scrape on stairs should not count as a failure, pass `--foot_bodies '.*_(foot|calf)'`.
- **Training env, not `play_mode`.** `play_mode` shrinks the terrain to 5×5 tiles without a curriculum layout. Here the training terrain grid is used instead: all levels and types, robots spawned uniformly over levels, and the terrain curriculum off. Each robot therefore stays on its tile, and `terrain_level` / `terrain_type` label every event with a difficulty and a family.
- **Disturbances off by default.** Pushes and the external wrench are removed so that failures reflect the policy and the terrain. `--disturbances` restores them. All curriculum terms are removed, so `--disturbances` applies the full (curriculum factor = 1) ranges.
- **Commands and other randomization are unchanged.** Velocity commands, foot friction, masses, and so on keep the task's training distribution.
- **No video during collection.** It is unknown ahead of time which of the thousands of robots will fail, and recording all of them would be far too costly. Videos are instead produced afterwards from the logged state by `render.py`, which tracks the robot. The replay is kinematic: base pose and joint positions are written each frame and physics is not stepped, so the video shows exactly the logged motion.
- **Same terrain for rendering.** `collect.py` seeds the terrain generator with the env seed, and `render.py` regenerates the terrain from the task and seed in `meta.json`. It then checks the result against `terrain.npz` and refuses to render if the task code has changed since the collection. Hydra overrides given to `collect.py` are recorded in `meta.json:argv` but not re-applied by `render.py`.
- **Window length.** An event shorter than 5 s keeps only the frames since the episode start (`num_frames`).
- **Storage.** The ring buffer stays on the GPU and costs about `num_envs × 250 × fields × 4 B`. For the A2 robust teacher that is about 1000 floats per frame, or roughly 4 GB at 4096 robots. Lower `--num_envs` on smaller GPUs. Each event is written as compressed `.npz` (≈0.3–0.5 MB).

## Changelog

- 2026-10-23: First version: `collect.py` (parallel play, non-foot-contact trigger, 5 s ring buffer, per-event npz, terrain mesh) and `render.py` (kinematic replay with a tracking camera).
