# Fault analysis

These scripts play a trained RSL-RL locomotion policy on many robots in parallel. They save the 5 s of motion before every failure: base contact (the only failure term kept by default), optionally also the onset of a non-foot contact. Every episode's end and length is logged, for success rates per difficulty. Saved events can then be flagged with the fault rules A–E, one rule or group at a time, and replayed as videos that follow the robot over the same terrain. `campaign.py` runs this for every (policy, terrain map) of `forge/terrain`.

| File | What it does |
| --- | --- |
| [collect.py](collect.py) | Headless parallel play. It saves the last `--buffer_s` seconds before every event. |
| [analyze.py](analyze.py) | Offline, numpy only. It flags which fault rules (A–E) each saved window hits, and when. |
| [render.py](render.py) | Replays saved events on the regenerated terrain and writes one robot-tracking mp4 per event. |
| [campaign.py](campaign.py) | Standard library only. Queues one collect + render job per (policy, terrain map) on Euler, and tabulates their `summary.json` files. |
| [test_analyze.py](test_analyze.py), [test_campaign.py](test_campaign.py) | Checks of the geometry helpers of `analyze.py` and of the summary of `campaign.py`. |

## Usage

### Campaign on the terrain maps

```bash
python scripts/forge/surveillance/campaign.py enqueue --name v1    # 2 policies x 10 terrains = 20 jobs
euler status                                                                 # jobs run one after another
euler fetch surveillance/v1
python scripts/forge/surveillance/campaign.py report results/euler/isaaclab3/surveillance/v1
```

Each job plays `IsaacContrib-Surveillance-<Type>-{Robust,Energy}` (`forge/surveillance`; robust teacher
`model_2999`, energy teacher rough `model_2999`): the terrain map's difficulty rows 1–3 (d = 0.4, 0.55, 0.7;
inclines 0, 11.25, 22.5°), 20 robots that start every episode on a random tile, forward-only commands with smooth
turns of at most ±45° (robust policy; the energy policy has no command input). On the inclines (`rough_slope`,
`rough_stairs`, `slip_slope`) the robots face straight up or down the slope (±15°) and get no turn command. The
terrain types are `forge/terrain`'s 9 plus `slip_field`, which is rough_field with random foot slips. On slip
terrain the foot friction is drawn at every touchdown (`forge/terrain` default): low with 2 %, back to 0.8 at the
next touchdown (see `forge/surveillance/REPORT.md` §3.4). It stops at
100 falls or 10 min of sim time.

Besides the falls, it saves up to 100 **event-trigger windows** into `triggers/`, to check that the terrain events
fire. They are 5 s, like the falls, centered on the trigger (2.5 s before, 2.5 s after), and spread evenly over the
10 min (at most `100 · t / 10 min` by sim time t). They do not stop the play. The trigger is the task's
`event_trigger` (`forge/surveillance/mdp/triggers.py`):

| Terrain | Trigger | Fires when |
| --- | --- | --- |
| `crust_collapse` | `crust_collapse` | the crust patch under a foot collapses |
| `sand_field` | `sand_sink` | a foot sinks into soft sand (below the patch surface, the terrain's drag rule) |
| `gravel_field` | `stone_step` | a loaded foot (> 1 N) stands on a stone: within its circle bound and 2 cm above the ground |
| `slip_slope`, `slip_field` | `foot_slip` | a foot that touched down with a low friction (2 % of the touchdowns) is loaded |
| the others | `random_walk` | at random (once per 5 s per robot on average): normal walking, no event |

A trigger window starts on the trigger's rising edge, at most one per robot per 5 s, and only when the robot has
2.5 s of history. A window cut short by a reset is dropped; a fall inside it is already a failure event.

Then it renders 10 falls and 10 trigger windows, each spread evenly over the play (`videos/`, `triggers/videos/`).
`report` writes `report.md` into the campaign directory. Per (policy, terrain, difficulty row) it lists the
episodes, successes, falls, success rate, mean episode length, mean time to fall, the play's sim and wall time, and
the trigger name, trigger windows and their videos. `enqueue` options: `--policies`, `--terrains`, `--failures` 100,
`--minutes` 10, `--triggers` 100 (0: none), `--videos` 10, `--dry_run` (print the `euler` commands only).

### Running campaign `v2` (event triggers; run 2026-10-26, results in REPORT.md §5.2)

The event-trigger code was smoke-tested first (`smoke_trig`, passed). To repeat, from the repo root on the
Mac. The Euler queue is shared and runs FIFO; if the user holds a node, enqueue on its profile.

```bash
# 1. smoke: 4 jobs (slip_field is new), 1 min sim each, 10 trigger windows, 2 + 2 videos
python scripts/forge/surveillance/campaign.py enqueue --name smoke_trig --policies Robust \
    --terrains crust_collapse sand_field gravel_field slip_field --minutes 1 --triggers 10 --videos 2
euler status
euler fetch surveillance/smoke_trig
python scripts/forge/surveillance/campaign.py report results/euler/isaaclab3/surveillance/smoke_trig
```

Check, per terrain:

- `summary.json`: `num_envs` 20; `event_trigger` is `crust_collapse` / `sand_sink` / `stone_step` / `foot_slip` / `random_walk`.
- `triggers/index.csv` has rows with `num_frames` 250.
- In `triggers/videos/`, the event happens at the middle (2.5 s): a foot sinking, a stone underfoot, a crust giving
  way.

If `stone_step`, `sand_sink` or `foot_slip` gives no rows, the trigger (`forge/surveillance/mdp/triggers.py`) needs fixing; the
falls are recorded either way.

```bash
# 2. full campaign: 2 policies x 10 terrains = 20 jobs, each 10 min sim or 100 falls, 100 trigger windows
python scripts/forge/surveillance/campaign.py enqueue --name v2
euler fetch surveillance/v2
python scripts/forge/surveillance/campaign.py report results/euler/isaaclab3/surveillance/v2
# optional: rule flags on the falls of one collection
uv run --no-project --with numpy python scripts/forge/surveillance/analyze.py \
    results/euler/isaaclab3/surveillance/v2/Robust/slip_slope
```

Record the results in `forge/surveillance/REPORT.md` §5 (a new §5.2, set up as §5.1) and in **Collections** below.

### Single collections

On Euler (the container's cwd is the repo root, and `logs/` is the profile's log directory):

```bash
# collect: 4096 robots by default; stops at --max_events events or --max_steps env steps
euler enqueue fault_a2_robust -- python scripts/forge/surveillance/collect.py \
    --task IsaacContrib-Velocity-A2Study-RobustTeacher \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt \
    --max_events 2000

# render selected events (default: all) of one collection
euler enqueue fault_render -- python scripts/forge/surveillance/render.py \
    /workspace/isaaclab/logs/fault_analysis/a2_study_robust_teacher/<collection> --events 0 5 12

euler fetch fault_analysis   # -> results/euler/isaaclab3/fault_analysis/... (events, terrain, videos)
```

Replay on the Linux machine, in the local Isaac container. `docker-compose.codes.yaml` mounts `~/hyeslee/results` at `/workspace/results`:

```bash
python scripts/forge/surveillance/render.py \
    /workspace/results/euler/isaaclab3/fault_analysis/a2_study_robust_teacher/<collection> \
    --events 12 --viz kit --real_time   # live window at wall-clock speed; the mp4 is written too
```

`collect.py` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--buffer_s` | 5.0 | Seconds kept before each event. |
| `--max_events` / `--max_steps` | 1000 / none | When to stop. |
| `--terminations` | `base_contact` | Failure terms kept; the task's others (e.g. `bad_orientation`) are removed. Time-outs stay. |
| `--contact_onsets` | off | Also record the onset of a non-foot contact as an event (the robot keeps walking). |
| `--max_triggers` | 0 | Also save up to this many event-trigger windows into `triggers/` (tasks with `event_trigger`); paced over `--max_steps`. |
| `--foot_bodies` | `.*_foot` | Regex for the foot bodies. With `--contact_onsets`, contact on any other body is an event. |
| `--contact_threshold` | 1.0 | Trigger force [N], the same as the stock `base_contact` termination. |
| `--grace_s` | 1.0 | Contact onsets this soon after a (re)spawn are ignored: the reset pose often puts a calf on the ground. |
| `--disturbances` | off | Keep the task's random pushes and external wrench. |
| `--sub_terrains` | all | Keep only these sub-terrains of the task's generator, e.g. `hills rough_discrete` (heightfield rough ground, no blocks or stairs). |
| `--output_dir` | `logs/surveillance/<experiment>/<time>_<run>_<model>` | Where the collection is written. |
| `--num_envs`, `--seed`, `--checkpoint`, … | | Same as `play.py`. `--num_envs` defaults to the task's. |

`render.py` takes the collection directory, `--triggers` (render the trigger windows instead of the failures), `--events <ids>` or `--pick N` (N full-window failures nearest to N evenly spaced sim times), `--eye dx dy dz` (camera offset from the base, default `-1 -1 0.6` m), `--real_time`, and `--viz` like `play.py`. It reapplies the collection's `--sub_terrains` and seed, so it needs no other arguments.

Run both scripts as queued jobs (`euler enqueue`), not `euler exec`: an `exec` step got OOM-killed on host memory where the same queued command ran fine.

Flag the rules on a fetched collection. Any Python with numpy works (the Isaac container, or `uv run --no-project --with numpy python ...` on the Mac). It takes seconds for 300 events and about a minute for 3400:

```bash
R=results/euler/isaaclab3/fault_analysis
# thresholds from normal walking: frames >= 2 s before the leg-contact onsets of the smoke collection
python scripts/forge/surveillance/analyze.py $R/smoke/rough2
# flag the failures against those thresholds
python scripts/forge/surveillance/analyze.py $R/a2_study_robust_teacher/2026-10-24_rough_terminal \
    --thresholds $R/smoke/rough2/thresholds.json
```

`analyze.py` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--baseline <collections>` | the analyzed one | Collections whose frames make the normal-walking distribution. |
| `--baseline_s` | 2.0 | Only frames at least this long before their event are baseline. |
| `--percentile` | 99.9 | Spike threshold; low-tail rules use `100 - percentile`. |
| `--thresholds` | – | Reuse a `thresholds.json` instead of computing a baseline. |
| `--grace_s` | the collection's | Frames this soon after a spawn are never flagged or used as baseline. |
| `--rules` | all | Flag only the rules starting with these, e.g. `--rules A_tilt` or `--rules B C_slip`, to turn rules on one at a time. |

It writes `flags.csv` and `thresholds.json` into the collection, and prints the share of events that hit each rule in the window, in its last second, and in the first second of full windows (4–5 s before the event, the rule's base rate in normal walking). The masks per frame are available from Python:

```python
import json
import sys

sys.path.insert(0, "scripts/forge/surveillance")
import analyze

meta, rows = analyze.load_collection(c)
thresholds = json.load(open(f"{c}/thresholds.json"))
masks = analyze.rule_masks(analyze.load_event(c, rows[12]), meta, thresholds, grace_s=1.0)
masks["B_value_drop"]  # [T] bool; the last frame is the event step
```

## Collections

Except `terrain_v1`, the collections below were deleted on 2026-10-25 (Euler and local); only the results quoted here remain. They used the old trigger (task terminations including tilt, plus leg-contact onsets).

| Collection (under `logs/fault_analysis/`) | Setup | Content |
| --- | --- | --- |
| `logs/surveillance/v2/<Policy>/<terrain>` | the trigger campaign, 2 policies x 10 terrains, 20 robots, 10 min (2026-10-26) | 25 falls (Robust 6, Energy 19), 2000 trigger windows, 400 videos; results in `forge/surveillance/REPORT.md` §5.2 |
| `logs/surveillance/smoke_trig/Robust/<terrain>` | smoke test of `v2`: crust_collapse, sand_field, gravel_field, rough_field, 1 min | 0 falls, 10 trigger windows each (all 250 frames) |
| `terrain_v1/<Policy>/<terrain>` | the deploy campaign, 2 policies x 9 terrains, 20 robots, 30 min or 200 falls (2026-10-25; kept) | 421 falls in total; Robust: 200 on `slip_slope`, 59 on `gravel_field`, 1 elsewhere; results in `forge/surveillance/REPORT.md` §5.1 |
| `a2_study_robust_teacher/2026-10-24_rough_terminal` | A2 robust teacher `model_2999`, `--sub_terrains hills rough_discrete`, 4096 robots, `--terminal_only` | 300 failures in 460 s sim (523 robot-h, ≈0.57 failures per robot-h, 7 min wall); videos of the first 10 full-window events |
| `harsh_world/v5_robust_full_40` | A2 robust teacher `model_2999`, `IsaacContrib-Velocity-A2HarshWorld-Robust-Full` (10 properties × 3 difficulties), 40 robots (4 per property; row d = 0.4 gets 2, the others 1), 30 min sim | 6136 events: 645 failures (606 tilt, 39 base contact) and 5491 leg-contact onsets, 22 min wall. 95 % of the failures on `slip_slope` 229, `mesa_drop` 216, `rough_slope` 168; none on `stub_field`, `crust_collapse`. Flagged with its own baseline |
| `smoke/rough2` | same terrain, 1024 robots, 30 s, contact onsets too | 3418 leg-contact onsets and 6 failures; videos of events 100, 500 (contact) and 1264, 1291, 1788 (failures) |
| `smoke/collect` | first version (reset on contact), all terrains | superseded; kept only as the evidence for the redesign |

The main collection records failures only. On rough ground, leg-contact onsets come about every 9 s per robot, which at 4096 robots means hundreds of events per simulated second. The leg contacts that preceded each failure are still in its window (`contact_*force_w`).

## Collection layout

Every file of a collection lives in one directory:

```
<collection>/
  meta.json        task, checkpoint, seed, step_dt, joint/body names, masses, limits, field shapes, CLI args
  env.yaml         full env config as played
  terrain.npz      vertices [V,3] and faces [F,3] (world frame), env_origins [N,3], terrain_origins [rows,cols,3]
  summary.json     per difficulty row and in total: episodes, successes, failures, success rate, mean episode and
                   time-to-failure [s], robot time; plus sim / wall time [s] of the play and why it stopped
  episodes.csv     one row per episode: episode, env_id, end_step, end_time_s, duration_s, terrain_level,
                   terrain_type, outcome (first termination hit, or running at the end), failure
  terrain_map.yaml terrain maps only: the map description (tiles, difficulties, patches)
  index.csv        one row per event: event, env_id, step, sim_time_s, episode_step, num_frames,
                   terrain_level, terrain_type, terminal, terminations, base_contact, contact_bodies
  events/<id>.npz  one event: [T, ...] arrays per field (T <= 250 at 50 Hz), plus env_id, step,
                   env_origin, terrain_level, terrain_type, terminations, contact_bodies
  videos/<id>.mp4  written by render.py
  triggers/        the event-trigger windows (--max_triggers): index.csv, events/<id>.npz and videos/ as above;
                   centered windows (the trigger step is frame T/2 - 1 of T = 250), terminal normally False
  flags.csv        written by analyze.py: per event and rule, hit (0/1) and first / last hit time [s, <= 0]
  thresholds.json  written by analyze.py: the fixed parameters and baseline thresholds used for flags.csv
```

Loading an event:

```python
import json, numpy as np

meta = json.load(open(f"{c}/meta.json"))
e = np.load(f"{c}/events/000012.npz")
terrain = np.load(f"{c}/terrain.npz")
e["root_pos_w"]  # [T, 3]; the last frame is the event step
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

## Rule flags

Only the trigger (base contact; with `--contact_onsets` also non-foot contact onsets) runs online. The rules A–E are labels: `analyze.py` gives each a per-frame mask over the saved window, and they never decide what is recorded. A rule is either *fixed* (a physical threshold, `FIXED` in `analyze.py`) or *spike* (a signal past the `--percentile` of normal walking, `SPIKES`). Foot contact is the contact sensor's own state (`contact_time > 0`); a touchdown is the first contact frame after air time.

| Rule | Kind | Hit when |
| --- | --- | --- |
| `A_base_contact` | fixed | base normal force > `contact_threshold` (1 N) |
| `A_leg_contact` | fixed | the same on any hip / thigh / calf (kept apart from the base) |
| `A_tilt` | fixed | $\bar g_z > -\cos 50°$ |
| `A_low_base` | fixed | base height above the height-scanner terrain < 0.5 × baseline median |
| `B_ang_rate` | spike | $\lVert\omega_b\rVert$ |
| `B_gravity_rate` | spike | $\lVert\dot{\bar g}\rVert$ |
| `B_impact` | spike | base CoM acceleration $\lVert a\rVert$, separate thresholds within ±1 frame of a touchdown and elsewhere |
| `B_support` | spike, low | signed distance of the capture point $x + v/\omega_0$ to the stance-feet hull (one or two feet: minus the distance to the point or segment), held 0.1 s |
| `B_value_drop` | spike, low | $V(t) - V(t - 0.5\,\mathrm{s})$ |
| `B_action_jump` | spike | $\lVert a_t - a_{t-1}\rVert$ |
| `C_trip` | spike | horizontal net foot force at touchdown |
| `C_slip` | spike | horizontal speed of a stance foot |
| `C_late_touchdown` | spike | air time before a touchdown |
| `C_early_touchdown` | spike, low | CPG phase at touchdown (swing is $[0, \pi]$), or air time before a touchdown for policies without a CPG |
| `C_flight` | spike | time with all feet in the air |
| `C_sinking` | fixed | a stance foot 2 cm below its touchdown height |
| `C_wedge` | fixed | a foot down and slower than 0.05 m/s for 0.1 s while its CPG phase is in $[0.25\pi, 0.75\pi]$ (foot commanded high); CPG policies only |
| `D_tracking` | fixed | $\lVert v_{cmd,xy} - v_{xy}\rVert > 0.5$ m/s for 1 s |
| `D_heading` | fixed | $\lvert\omega_{cmd} - \omega_z\rvert > 0.5$ rad/s for 1 s |
| `D_no_progress` | fixed | over 2 s with ≥ 0.5 m commanded, the base moved < 25 % of the commanded distance |
| `E_torque_saturation` | fixed | the actuator clipped the torque (`computed - applied` > 1 % of the effort limit) for 0.1 s |
| `E_joint_limit` | fixed + spike, low | a joint within 5 % of its range from a limit and nearer than the baseline's 0.1 percentile |
| `E_pd_error` | spike | $\lvert q_{des} - q\rvert$, per joint |
| `E_power` | spike | $\sum \lvert\tau \dot q\rvert$ |

Results on `2026-10-24_rough_terminal` (300 failures), with thresholds from `smoke/rough2` (357k frames of normal walking on the same terrain). "last 1 s" vs. "first 1 s" (4–5 s before the failure) shows which rules mark the fall rather than normal walking; "onset" is the median first hit within the last 2 s:

| Rule | last 1 s | first 1 s | onset [s] |
| --- | --- | --- | --- |
| `A_tilt` | 89 % | 0 % | −0.04 |
| `B_value_drop` | 89 % | 0.4 % | −0.52 |
| `B_ang_rate` / `B_gravity_rate` | 89 % | 5 / 10 % | −0.50 / −0.56 |
| `C_flight` | 85 % | 5 % | −0.26 |
| `B_impact` | 84 % | 12 % | −0.38 |
| `E_power` | 79 % | 6 % | −0.26 |
| `B_action_jump` | 69 % | 6 % | −0.33 |
| `E_torque_saturation` | 22 % | 0.8 % | −0.10 |
| `C_slip` | 22 % | 7 % | −0.47 |
| `C_trip` | 16 % | 0.8 % | −0.30 |
| `C_wedge` | 15 % | 0.8 % | −0.64 |
| `A_leg_contact`, `E_pd_error`, `E_joint_limit` | 87–89 % | 22–42 % | −1.1 to −0.7 |

The 89 % ceiling is the 267 failures more than 1 s after a spawn: frames inside the spawn grace are never flagged. `A_base_contact`, `A_low_base`, and `D_heading` never hit: these robots tip over (tilt > 1 rad) before the base touches or drops. On `smoke/rough2` itself, the last-second and first-second rates match for every rule except `A_leg_contact` (its trigger), so the rules do not fire just because an event was recorded.

## Decisions

Choices the request left open, and how they were made here:

- **Trigger: base contact only (2026-10-25).** The user's request: a fall is the base touching the ground; tilt no longer ends an episode or triggers a recording, and the rules A–E (tilt among them, `A_tilt`) are evaluated afterwards, one at a time if wanted (`analyze.py --rules`). Leg-contact onsets are recorded only with `--contact_onsets`. The history of the earlier trigger follows.
- **Earlier trigger: non-foot contact onset, plus real failures.** The first version reset a robot at its first non-foot contact. Its smoke test (A2 robust teacher, 512 robots, all terrains) showed this was the wrong design: 45% of events came on the first step after spawn, because the reset pose puts a calf on the ground. The rest were thigh and calf hits during otherwise normal walking (median tilt 7°, base 0.36 m above the terrain, critic value unchanged), and base contact occurred 0 times. Resetting on those events cut every episode short, so real falls never had a chance to happen. The current design:
  - An event is recorded at the onset of a non-foot contact. Onsets inside the `--grace_s` spawn grace are ignored, and each robot records at most one onset per buffer length (5 s) so windows do not overlap. The robot keeps walking.
  - Robots are reset only by the task's own terminations (robust teacher: base contact, tilt > 1 rad) and time-outs. A termination is always recorded, even inside the 5 s, with `terminal=True` and the term names in `terminations`.
  - The data therefore separates "the leg touched but the robot went on" (`terminal=False`) from "the robot failed" (`terminal=True`), and `base_contact` / `contact_bodies` say what touched. `--terminal_only` keeps only failures. If calves should not count at all, pass `--foot_bodies '.*_(foot|calf)'`.
- **Training env, not `play_mode`.** `play_mode` shrinks the terrain to 5×5 tiles without a curriculum layout. Here the training terrain grid is used instead: all levels and types, robots spawned uniformly over levels, and the terrain curriculum off. Each robot therefore stays on its tile, and `terrain_level` / `terrain_type` label every event with a difficulty and a family.
- **Disturbances off by default.** Pushes and the external wrench are removed so that failures reflect the policy and the terrain. `--disturbances` restores them. All curriculum terms are removed, so `--disturbances` applies the full (curriculum factor = 1) ranges.
- **Commands and other randomization are unchanged.** Velocity commands, foot friction, masses, and so on keep the task's training distribution.
- **No video during collection.** It is unknown ahead of time which of the thousands of robots will have an event, and recording all of them would be far too costly. Videos are instead produced afterwards from the logged state by `render.py`, which tracks the robot. The replay is kinematic: base pose and joint positions are written each frame and physics is not stepped, so the video shows exactly the logged motion.
- **Same terrain for rendering.** `collect.py` seeds the terrain generator with the env seed, and `render.py` regenerates the terrain from the task and seed in `meta.json`. It then checks the result against `terrain.npz` and refuses to render if the task code has changed since the collection. Hydra overrides given to `collect.py` are recorded in `meta.json:argv` but not re-applied by `render.py`.
- **Window length.** An event less than 5 s after a spawn keeps only the frames since then (`num_frames`).
- **Replay in Isaac, not on the Mac.** Replay runs `render.py` in an Isaac Lab container: on Euler, or on the Linux machine, whose container mounts the fetched results. A Mac viewer would need extra tools (robot mesh export, a third-party viewer) and would look worse than the Isaac renderer.
- **Rule thresholds.** Rules the request defines by a physical bound keep it fixed; rules it defines against "the normal gait distribution" use the baseline percentile. Changes from the request's examples, each because the example could not work on this data:
  - Tilt 50°, not 60°: the task terminates at 1 rad (57°), so 60° is never logged.
  - The joint-limit rule also needs the joint nearer to a limit than normal walking gets. With the 5 % margin alone it hit 88 % of normal seconds: A2's calves fold to (and slightly past) their −2.77 rad limit in every swing.
  - Trip uses the horizontal foot force at touchdown, not a CPG-phase window. With "contact before 0.8π of the swing", 64 % of normal seconds hit, because the policy shifts its phases; a ratio to the vertical force blows up on first-frame grazes.
  - Early touchdown uses the CPG phase at touchdown. Air time is quantized to 20 ms, and its 0.1 percentile is the minimum, so a low tail on it never hits.
  - Torque saturation is detected as clipping (`computed_torque != applied_torque`), not as $\lvert\tau\rvert \approx \tau_{max}$: the A2 actuator's speed-dependent curve clips well below the effort limit (e.g. 159 of 180 N·m).
  - Sinking is measured against the foot's own touchdown height, not against the terrain mesh. On rigid ground it mostly catches a foot sliding down a slope; on deformable terrain it is the intended sinking.
- **Baseline.** Normal walking is taken from the windows themselves: frames at least 2 s before the event and 1 s after a spawn. `smoke/rough2` (3418 leg-contact onsets, 6 failures) is the better baseline because it does not contain the seconds leading to a fall. Spike rules hit by construction about 0.1 % of normal frames, so read their rate against the "first 1 s" column, not against zero.
- **Campaign layout.** Success = an episode that ends without a failure term: by time-out (60 s) or by leaving its spawn tile (`left_tile`, added by `collect.py` on terrain maps as a time-out). Leaving the tile ends the episode because rows have different difficulties and the robots walk forward (out of a 24 m tile in 12–25 s): without it a robot would fall on a neighboring row and be counted on its spawn row. "Level 1–3" is read as the first three of the map's five rows, at the full map's difficulties. The task (`a2_deploy`) fixes the map, the 20 robots, and their random tiles; `collect.py` keeps a terrain map's curriculum (the tile draw) and seeds its generator with the env seed. 30 min is sim time (`--max_steps 90000`); the wall time is in `summary.json`. Videos are picked evenly over sim time per (policy, terrain), from full 5 s windows.
- **Storage.** The ring buffer stays on the GPU and costs about `num_envs × 250 × fields × 4 B`. For the A2 robust teacher that is about 1000 floats per frame, or roughly 4 GB at 4096 robots. Lower `--num_envs` on smaller GPUs. Each event is written as compressed `.npz` (≈0.3–0.5 MB).

## Changelog

- 2026-10-26: `slip_field` (10 terrain types; `campaign.py` reads `EXTRA_TERRAINS` of `forge/surveillance`). Reason: the user's request.
- 2026-10-26: Event-trigger windows (`--max_triggers`, `triggers/`, `render.py --triggers`); campaign defaults 20 robots, 10 min, 100 falls, 100 trigger windows, 10 + 10 videos. Reason: the user's request (check that the terrain events fire, apart from the falls).
- 2026-10-26: Moved to `scripts/forge/surveillance` with the task (`forge/surveillance`, `IsaacContrib-Surveillance-*`); collections now under `logs/surveillance/`; campaign defaults 100 falls, 10 robots. The campaign `terrain_v1` (20 robots, 200 falls) stays under `logs/fault_analysis/terrain_v1`. Reason: the user's request.
- 2026-10-25: Trigger is base contact only (`--terminations`, default `base_contact`; `--contact_onsets` replaces `--terminal_only`). `episodes.csv`, `summary.json`, `left_tile`, and kept terrain-map curricula in `collect.py` (`--num_envs` now defaults to the task's); `render.py --pick`; `analyze.py --rules`; `campaign.py` on the new deploy tasks (`forge/surveillance`). Scripts moved from `scripts/reinforcement_learning/` to `scripts/forge/`. Reason: the user's request (success rates per difficulty on the terrain maps, rules turned on one at a time during analysis).

- 2026-10-24: `analyze.py` flags the rules A–E on saved windows (`flags.csv`, `thresholds.json`), with baseline percentiles from normal walking; flagged `2026-10-24_rough_terminal` against `smoke/rough2`.
- 2026-10-24: Collected `2026-10-24_rough_terminal`. `render.py` gained `--count`, a closer default camera, and hidden command markers.
- 2026-10-24: Contact onsets no longer reset the robot. The task's terminations do, and are recorded as `terminal`. Added the spawn grace, `--terminal_only`, `--sub_terrains`, the Linux replay mount, and `--real_time` for `render.py`. Reason: in the smoke test, leg contacts were mostly harmless scrapes and spawn artifacts (see Decisions).
- 2026-10-23: First version: `collect.py` (parallel play, non-foot-contact trigger, 5 s ring buffer, per-event npz, terrain mesh) and `render.py` (kinematic replay with a tracking camera).
