# Fault analysis

These scripts play a trained RSL-RL locomotion policy on many robots in parallel. They save the 5 s of motion before every failure: base contact (the only failure term kept by default), optionally also the onset of a non-foot contact. Every episode's end and length is logged, for success rates per difficulty. Saved events can then be flagged with the fault rules A–E, one rule or group at a time, and replayed as videos that follow the robot over the same terrain. `campaign.py` runs this for every (policy, terrain map) of `forge/terrain`.

| File | What it does |
| --- | --- |
| [collect.py](collect.py) | Headless parallel play. It saves the last `--buffer_s` seconds before every event. |
| [analyze.py](analyze.py) | Offline, numpy only. It flags which fault rules (A–E) each saved window hits, and when. |
| [render.py](render.py) | Replays saved events on the regenerated terrain and writes one robot-tracking mp4 per event. |
| [campaign.py](campaign.py) | Standard library only. Queues one collect + render job per (policy, terrain map) on Euler, and tabulates their `summary.json` files. |
| [taxonomy.py](taxonomy.py) | Offline (numpy, scikit-learn, matplotlib). Per-frame signal labels (rules A–E + privileged state), failure mode per fall, window features, Ward clustering, separability of the known causes. |
| [classify.py](classify.py) | Offline (torch, scikit-learn). Which event acts on the robot from proprioception only, labels from the triggers (class probability vs time since the trigger, onset AUROC); `--falls_csv`: failure mode of the falls from proprioception. |
| [sequences.py](sequences.py) | Offline (torch, scikit-learn, tslearn). Taxonomy methods compared on the same falls: rules, first violated constraint, label shares + Ward (with / without the cause flags), DTW cost metrics (state, event labels, student latent), autoencoder latent (plain, and policy-invariant: decoder conditioned on the policy + adversary), privileged-state student latent; scored by stability, NMI with terrain / policy, physical coherence, detectability beyond the context. Writes `falls.csv` with every method's label. |
| [detect.py](detect.py) | Offline (torch, scikit-learn). Early-detection strategies (supervised hazard, with IMU, with harmless events as negatives, per-mode hazard, forward-model anomaly, tilt, critic value) at a fixed false-alarm rate on normal walking: lead time before the balance loss per failure mode, alarms on harmless events, per-mode signatures. |
| [reflex.py](reflex.py) | Isaac Lab, GPU. Closed-loop test of a precursor reflex for the Robust policy: on each terrain-event onset (privileged flags = a perfect onset detector), after a latency, the command is replaced (back / stop / slow) for a while; falls and forward progress per robot-minute against no reflex. Optional learned gate (`--gate_model`), random firing (`--fire_prob`), and onset logging for training a gate (`--log_onsets`). |
| [counterfactual.py](counterfactual.py) | Isaac Lab, GPU. Restores the logged state before each recorded fall (root, joints, last actions; Robust also the CPG phase), then runs one reaction: command (continue / slow / stop / back, Robust only) or posture override of the joint targets (stand / crouch / wide / pause, both policies). Records base contact, max and final tilt, leg contact and distance travelled; control restores from windows without a fall. |
| [test_analyze.py](test_analyze.py), [test_campaign.py](test_campaign.py) | Checks of the geometry helpers of `analyze.py` and of the summary of `campaign.py`. |

## Usage

### Campaign on the terrain maps

```bash
python scripts/forge/surveillance/campaign.py enqueue --name v1    # 2 policies x 10 terrains = 20 jobs
euler status                                                                 # jobs run one after another
euler fetch surveillance/v1
python scripts/forge/surveillance/campaign.py report results/fetch/isaaclab3/surveillance/v1
```

Each job plays `IsaacContrib-Surveillance-<Type>-{Robust,Energy}` (`forge/surveillance`; robust teacher
`model_2999`, energy teacher rough `model_2999`): the terrain map's difficulty rows 1–3 (d = 0.4, 0.55, 0.7;
inclines 0, 11.25, 22.5°), 20 robots that start every episode on a random tile, forward-only commands with smooth
turns of at most ±45° (robust policy; the energy policy has no command input). On the inclines (`rough_slope`,
`rough_stairs`, `slip_slope`) the robots face straight up or down the slope (±15°) and get no turn command. The
terrain types are `forge/terrain`'s 11 plus `mixed_field`. On slip terrain (`slip_slope`, `slip_field`) the foot
friction is drawn at every touchdown (`forge/terrain` default): low with 25 %, back to 0.8 at the next touchdown
unless drawn low again; `ice_field` has a checkerboard of icy cells instead (see `forge/surveillance/REPORT.md` §3.4).
It stops at
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
| `slip_slope`, `slip_field`, `ice_field` | `foot_slip` | a foot that touched down with a low friction (25 % of the touchdowns, or an icy cell) is loaded |
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
python scripts/forge/surveillance/campaign.py report results/fetch/isaaclab3/surveillance/smoke_trig
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
python scripts/forge/surveillance/campaign.py report results/fetch/isaaclab3/surveillance/v2
# optional: rule flags on the falls of one collection
uv run --no-project --with numpy python scripts/forge/surveillance/analyze.py \
    results/fetch/isaaclab3/surveillance/v2/Robust/slip_slope
```

Record the results in `forge/surveillance/REPORT.md` §5 (a new §5.2, set up as §5.1) and in **Collections** below.

### Campaign `v3` (Hard tasks, 2026-10-31; smoke and `v3_hard` done, `forge/surveillance/REPORT.md` §3.5, §5.4)

The `-Hard` tasks (`forge/surveillance/REPORT.md` §3.5: hardest rows, ice patches, soft deep sand, crusts that drop
to the hardpan, taller stubs) and the new `mixed_field` (§3.6). Collections now also log the terrain event's
per-foot flags and the stone poses (fields below), and `render.py` replays the stones. Checked only on the Mac CPU
container (all Hard tasks build and step, the fields have the right shapes, a stone replay renders); the physics is
unverified until the smoke test.

```bash
# 1. smoke: Robust, the event terrains, 1 min sim each
python scripts/forge/surveillance/campaign.py enqueue --name smoke_hard --hard --policies Robust \
    --terrains slip_field slip_slope sand_field crust_collapse mixed_field gravel_field --minutes 1 --triggers 10 --videos 2
euler fetch surveillance/smoke_hard
```

Check: `summary.json` falls per minute (v2: ~0.01 / min per terrain); in the trigger windows the new
`terrain_on_ice`, `terrain_in_sand`, `terrain_collapse` flags fire; feet sink 12–25 cm in sand (v2: 2.6 cm); the
gravel videos show the stones. If a terrain fails every robot within seconds, lower that knob in `apply_hard`.

```bash
# 2. full: 2 policies x 11 terrains, 10 min or 100 falls each
python scripts/forge/surveillance/campaign.py enqueue --name v3_hard --hard
euler fetch surveillance/v3_hard
# 3. taxonomy and classifiers on it (event labels now come from the logged flags)
uv run --no-project --with numpy --with scikit-learn --with matplotlib python scripts/forge/surveillance/taxonomy.py \
    --out results/reports/surveillance/20261031_hard_terrain/data/taxonomy_v3 --falls results/fetch/isaaclab3/surveillance/v3_hard/*/* \
    --baseline results/fetch/isaaclab3/surveillance/v2/*/{rough_field,pit_field,stub_field,rough_slope,rough_stairs}
uv run --no-project --with torch --with numpy --with scikit-learn --with matplotlib python \
    scripts/forge/surveillance/classify.py results/fetch/isaaclab3/surveillance/v3_hard/Robust/mixed_field \
    --out results/reports/surveillance/20261031_hard_terrain/data/classify_v3/Robust_mixed
```

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

euler fetch fault_analysis   # -> results/fetch/isaaclab3/fault_analysis/... (events, terrain, videos)
```

Replay on the Linux machine, in the local Isaac container. `docker-compose.codes.yaml` mounts `~/hyeslee/results` at `/workspace/results`:

```bash
python scripts/forge/surveillance/render.py \
    /workspace/results/fetch/isaaclab3/fault_analysis/a2_study_robust_teacher/<collection> \
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
| `--trigger_quiet_s` | 1.0 | A rising edge of the trigger counts only after it was off this long (an onset, not a robot re-entering the patch it is crossing). |
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
R=results/fetch/isaaclab3/fault_analysis
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
| `terrain_foot_mu`, `terrain_low_mu`, `terrain_on_ice`, `terrain_in_sand`, `terrain_collapse` | F each | terrain maps, since 2026-10-31: the terrain event's per-foot state (applied friction, slip or ice sample active, on an ice cell, in soft sand, its crust collapsed this step); privileged |
| `rock_pose_w` | K×7 | `gravel_field`, since 2026-10-31: pose of the robot's stones, replayed by `render.py` |

## Rule flags

Only the trigger (base contact; with `--contact_onsets` also non-foot contact onsets) runs online. The rules A–E are labels: `analyze.py` gives each a per-frame mask over the saved window, and they never decide what is recorded. A rule is either *fixed* (a physical threshold, `FIXED` in `analyze.py`) or *spike* (a signal past the `--percentile` of normal walking, `SPIKES`). Foot contact is the contact sensor's own state (`contact_time > 0`); a touchdown is the first contact frame after air time.

### Rule types

The letter of a rule names what it watches. The code does not define the letters in words; this reading follows from
the rules in each group.

| Type | Watches | Question it answers | Rules |
| --- | --- | --- | --- |
| A | body contact and posture | Is the body touching something, or in a failed posture? | `A_base_contact`, `A_leg_contact`, `A_tilt`, `A_low_base` |
| B | whole-body dynamics and the controller's reaction | Is the motion, the critic or the action unusually abrupt, or the support about to be lost? | `B_ang_rate`, `B_gravity_rate`, `B_impact`, `B_support`, `B_value_drop`, `B_action_jump` |
| C | feet and step timing | Is a single step going wrong? | `C_trip`, `C_slip`, `C_late_touchdown`, `C_early_touchdown`, `C_flight`, `C_sinking`, `C_wedge` |
| D | task performance | Is the robot failing to do what it is commanded? | `D_tracking`, `D_heading`, `D_no_progress` |
| E | joints and actuators | Is a joint or motor at its limit or working unusually hard? | `E_torque_saturation`, `E_joint_limit`, `E_pd_error`, `E_power` |

A rule is a label on a frame, not a failure: a spike rule fires on about 0.1 % of normal frames by construction.

### Rules

The exact formula of every rule is under [Implementation](#implementation).

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

### Implementation

The formulas `analyze.py` and `taxonomy.py` compute, per frame $t$.

**Spike threshold** of a signal $x_s$, per policy $\pi$, over the baseline frames $\mathcal D^\pi$ of normal walking,
with $p$ = `--percentile` (99.9 in `taxonomy.py`): $\theta_s = P_p(x_s \mid \mathcal D^\pi)$ for the high tail and
$\theta_s = P_{100-p}(x_s \mid \mathcal D^\pi)$ for the low tail. The values are written to `thresholds.json`.

**Notation.** $s_i(t)$: foot $i$ in stance. $\tau^{air}_i$: air time. $\mathrm{TD}_i(t) = s_i(t) \land \tau^{air}_i(t-1) > 0$:
touchdown. $z^{TD}_i$: foot height at its last touchdown. $v_i$: foot velocity. $F^n_b$: contact normal force on body
$b$. $F_i$: net contact force on foot $i$. $f_c$: `contact_threshold` (1 N). $L$: the hip, thigh and calf bodies. $\bar g$:
gravity direction in the base frame ($\bar g_z = -1$ upright). $h$: base height above the terrain under it. $V$: critic
value. $u$: policy action. $\varphi_i$: CPG phase divided by $\pi$ (swing is $[0, 1]$). $\mathrm{held}_T[x](t)$: $x$ has
been true for the last $T$ seconds, including $t$. $N(t)$: a touchdown within ±1 frame of $t$. $\Delta t$: the policy
step.

#### Rules A–E

| Type | Rule | Kind | True at $t$ when | Threshold |
| --- | --- | --- | --- | --- |
| A | `A_base_contact` | fixed | $\lVert F^n_{base}(t)\rVert > f_c$ | 1 N |
| A | `A_leg_contact` | fixed | $\exists b \in L:\ \lVert F^n_b(t)\rVert > f_c$ | 1 N |
| A | `A_tilt` | fixed | $\bar g_z(t) > -\cos 50°$ | 50° |
| A | `A_low_base` | fixed | $h(t) < 0.5 \cdot \mathrm{median}_{\mathcal D^\pi}(h)$ | 0.5 × baseline median |
| B | `B_ang_rate` | spike | $\lVert\omega_b(t)\rVert > \theta$ | $\theta$ |
| B | `B_gravity_rate` | spike | $\lVert\bar g(t) - \bar g(t-1)\rVert / \Delta t > \theta$ | $\theta$ |
| B | `B_impact` | spike | $\lVert a(t)\rVert > \theta_{td}$ if $N(t)$, else $\lVert a(t)\rVert > \theta_{free}$ | $\theta_{td}$, $\theta_{free}$ |
| B | `B_support` | spike, low | $\mathrm{held}_{0.1\,s}[m(t) < \theta](t)$, with $m$ the signed distance of the capture point $x + v/\omega_0$ to the stance-feet hull (one or two feet: minus the distance to the point or segment) | $\theta$ |
| B | `B_value_drop` | spike, low | $V(t) - V(t - 0.5\,\mathrm{s}) < \theta$ | $\theta$ |
| B | `B_action_jump` | spike | $\lVert u(t) - u(t-1)\rVert > \theta$ | $\theta$ |
| C | `C_trip` | spike | $\max_{i:\,\mathrm{TD}_i(t)} \lVert F_{xy,i}(t)\rVert > \theta$ | $\theta$ |
| C | `C_slip` | spike | $\max_{i:\,s_i(t)} \lVert v_{xy,i}(t)\rVert > \theta$ | $\theta$ |
| C | `C_late_touchdown` | spike | $\exists i:\ \mathrm{TD}_i(t) \land \tau^{air}_i(t-1) > \theta$ | $\theta$ |
| C | `C_early_touchdown` | spike, low | $\min_{i:\,\mathrm{TD}_i(t)} \varphi_i(t) < \theta$; without a CPG, $\min_{i:\,\mathrm{TD}_i(t)} \tau^{air}_i(t-1) < \theta$ | $\theta$ |
| C | `C_flight` | spike | $\big(\forall i:\ \lnot s_i(t)\big) \land \min_i \tau^{air}_i(t) > \theta$ | $\theta$ |
| C | `C_sinking` | fixed | $\exists i:\ s_i(t) \land z^{TD}_i(t) - z_i(t) > 0.02$ | 2 cm |
| C | `C_wedge` | fixed | $\exists i:\ \mathrm{held}_{0.1\,s}\big[s_i \land 0.25 < \varphi_i < 0.75 \land \lVert v_i\rVert < 0.05\big](t)$; CPG policies only | 0.05 m/s, 0.1 s |
| D | `D_tracking` | fixed | $\mathrm{held}_{1\,s}\big[\lVert v^{cmd}_{xy} - v_{xy}\rVert > 0.5\big](t)$ | 0.5 m/s, 1 s |
| D | `D_heading` | fixed | $\mathrm{held}_{1\,s}\big[\lvert\omega^{cmd} - \omega_z\rvert > 0.5\big](t)$ | 0.5 rad/s, 1 s |
| D | `D_no_progress` | fixed | $C(t) \ge 0.5 \land \lVert p_{xy}(t) - p_{xy}(t - 2\,\mathrm{s})\rVert < 0.25\,C(t)$, with $C(t)$ the commanded distance over the last 2 s | 0.5 m, 25 %, 2 s |
| E | `E_torque_saturation` | fixed | $\exists j:\ \mathrm{held}_{0.1\,s}\big[\lvert\tau^{comp}_j - \tau_j\rvert > 0.01\,\tau^{lim}_j\big](t)$ | 1 %, 0.1 s |
| E | `E_joint_limit` | fixed + spike, low | $\exists j:\ x_j(t) < \theta_j \land x_j(t) < 0.05$, with $x_j = \min(q_j - q^{lo}_j,\ q^{hi}_j - q_j) / (q^{hi}_j - q^{lo}_j)$ | $\theta_j$ per joint, 5 % |
| E | `E_pd_error` | spike | $\exists j:\ \lvert q^{des}_j(t) - q_j(t)\rvert > \theta_j$ | $\theta_j$ per joint |
| E | `E_power` | spike | $\sum_j \lvert\tau_j(t)\,\dot q_j(t)\rvert > \theta$ | $\theta$ |

#### Privileged labels of `taxonomy.py`

Per-frame labels from state a real robot does not measure (`taxonomy.privileged`), with fixed thresholds.
$p^{TD}_{xy,i}$ is the foot's planar position at its last touchdown.

| Label | True at $t$ when | Threshold |
| --- | --- | --- |
| `slip_2feet` | $\sum_i \big[s_i(t) \land \lVert v_{xy,i}(t)\rVert > 0.3\big] \ge 2$ | 0.3 m/s |
| `sink_5cm` | $\exists i:\ s_i(t) \land z^{TD}_i(t) - z_i(t) > 0.05 \land \lVert p_{xy,i}(t) - p^{TD}_{xy,i}(t)\rVert < 0.05$ | 5 cm |
| `front_leg_contact` | $\exists b \in L_{FL,FR}:\ \lVert F^n_b(t)\rVert > f_c$ | 1 N |
| `hind_leg_contact` | $\exists b \in L_{RL,RR}:\ \lVert F^n_b(t)\rVert > f_c$ | 1 N |
| `thigh_contact` | $\exists b \in L_{hip,thigh}:\ \lVert F^n_b(t)\rVert > f_c$ | 1 N |
| `stance_1foot` | $\sum_i s_i(t) \le 1 \land h(t) < 0.3$ | 0.3 m |

#### Terrain-event flags

Simulator state per foot, logged since 2026-10-31. The taxonomy label is true when any foot's flag is true. There is no
numeric threshold on the flag itself. $\sigma_i$: the foot is over the slip terrain or the ice grid.
$\mathrm{enter}_i(t) = \sigma_i(t) \land (\lnot\sigma_i(t-1) \lor \mathrm{TD}_i(t))$;
$\mathrm{leave}_i(t) = \lnot\sigma_i(t) \land \sigma_i(t-1)$; $\xi_i \sim U(0, 1)$.

| Label | Per-foot flag |
| --- | --- |
| `terrain_low_mu` | on $\mathrm{enter}_i$: $(\xi_i < p_{slip}) \lor \mathrm{icy}_i$; on $\mathrm{leave}_i$: false; else held |
| `terrain_on_ice` | on $\mathrm{enter}_i$: $\mathrm{icy}_i$; on $\mathrm{leave}_i$: false; else held |
| `terrain_in_sand` | the foot is over ordinary sand or a collapsed crust, and $z_i(t) < z_{surface}(t) + r_{foot}$ |
| `terrain_collapse` | true only on the step the crust patch under the foot collapses |

`terrain_low_mu` and `terrain_on_ice` are latched at touchdown and held through the swing, so they mean "the last
touchdown was on a low-friction spot", not "this foot is on ice now".

#### Physical state of `sequences.py` (z-state)

The continuous per-frame description used by `first_violation`, `dtw`, `ae` and as the target of `priv_latent`
(`sequences.STATE`, `physical_state`). $\bar g$ is the gravity direction and $\omega$ the angular velocity in the base
frame, $v^b$ the base velocity in the base frame. Left and right are merged: absolute values for the sideways
channels, and legs grouped as front / hind only. All channels are privileged sim state.

| # | Channel | Group (`GROUPS`) | Value at frame $t$ |
| --- | --- | --- | --- |
| 1 | `roll` | attitude | $\lvert\mathrm{atan2}(\bar g_y, -\bar g_z)\rvert$ |
| 2 | `pitch` | attitude | $\mathrm{atan2}(\bar g_x, -\bar g_z)$ |
| 3 | `roll_rate` | attitude | $\lvert\omega_x\rvert$ |
| 4 | `pitch_rate` | attitude | $\omega_y$ |
| 5 | `yaw_rate` | attitude | $\lvert\omega_z\rvert$ |
| 6 | `base_height` | body_drop | $h(t)$ |
| 7 | `vx_error` | tracking | $v^b_x - v^{cmd}_x$ |
| 8 | `vy_error` | tracking | $\lvert v^b_y - v^{cmd}_y\rvert$ |
| 9 | `vz` | body_drop | world vertical velocity of the base |
| 10 | `support_margin` | support | $\mathrm{clip}(m(t), -0.3, 0.3)$ with $m$ of `B_support`; $-0.3$ with no foot down |
| 11 | `feet_down` | support | $\sum_i s_i(t)$ |
| 12 | `slip_front` | slip | $\max_{i \in \text{front}} s_i(t)\,\lVert v_{xy,i}(t)\rVert$ |
| 13 | `slip_hind` | slip | $\max_{i \in \text{hind}} s_i(t)\,\lVert v_{xy,i}(t)\rVert$ |
| 14 | `sink_front` | sink | $\max_{i \in \text{front}}$ of $\max(z^{TD}_i - z_i,\ 0)$, counted while the foot is in stance and within 5 cm of its touchdown point |
| 15 | `sink_hind` | sink | the same over the hind feet |
| 16 | `leg_force_front` | leg_contact | $\log\big(1 + \sum_{b \in L_{FL,FR}} \lVert F^n_b(t)\rVert\big)$ |
| 17 | `leg_force_hind` | leg_contact | $\log\big(1 + \sum_{b \in L_{RL,RR}} \lVert F^n_b(t)\rVert\big)$ |
| 18 | `torque_ratio` | torque | $\max_j \lvert\tau_j(t)\rvert / \tau^{lim}_j$ |

From the raw channels to the sequences the methods use:

1. **Cut** (`cut`): the frames $[t_b - 1.5\,\mathrm{s},\ t_b + 0.5\,\mathrm{s})$, 100 frames, edge-padded past the window.
2. **Standardize** (`standardize`), per policy $\pi$ and channel $c$: $z_c = (x_c - \mathrm{median}_{\mathcal N^\pi}\,x_c) / \max(\mathrm{std}_{\mathcal N^\pi}\,x_c,\ 10^{-3})$.
   $\mathcal N^\pi$ is every frame of the policy's `random_walk` trigger windows given by `--normal`, without the windows in
   which the base touches.
3. **Per method:** `dtw` and `ae` use the compressed $\mathrm{sign}(z)\log(1 + \lvert z\rvert)$, and `ae` then clips to ±10.
   The `priv_latent` targets and the mechanism $R^2$ clip $z$ to ±10. `first_violation` uses $z$ as it is.

Example from the fall below (Robust, normal walking of `v3_hard`, 455 windows or 113,750 frames):

| Frame ($t - t_b$) | Channel | Raw | Median | Std | $z$ | Compressed |
| --- | --- | --- | --- | --- | --- | --- |
| 142 (−0.96 s) | `slip_front` | 1.302 m/s | 0.005 | 0.158 | 8.21 | 2.22 |
| 142 (−0.96 s) | `slip_hind` | 0.021 m/s | 0.031 | 0.138 | −0.08 | −0.08 |
| 197 (+0.14 s) | `roll` | 0.913 rad | 0.081 | 0.114 | 7.27 | 2.11 |
| 197 (+0.14 s) | `base_height` | 0.225 m | 0.360 | 0.031 | −4.32 | −1.67 |

The front stance foot slides 8.21 normal standard deviations above this policy's normal walking. The cut of this
fall is frames 115–214.

### Worked example

One real fall: `v3_hard/Robust/slip_field/events/000003.npz`, 250 frames at 50 Hz. Its balance loss is frame 190, the
reference $t_b$ of `taxonomy.py`. The spike thresholds are the Robust values of the `taxonomy_v3` run (the policy's
values for every rule: `results/reports/surveillance/20261031_hard_terrain/LABELS.md`).

| Type | Rule | Frame ($t - t_b$) | Values | Why it is true there |
| --- | --- | --- | --- | --- |
| A | `A_tilt` | 197 (+0.14 s) | $\bar g_z = -0.588$; the frame before, $-0.664$ | $-0.588 > -\cos 50° = -0.643$ |
| B | `B_ang_rate` | 152 (−0.76 s) | $\lVert\omega_b\rVert = 1.233$ rad/s; the frame before, 0.678 | $1.233 > \theta = 1.157$ |
| C | `C_slip` | 142 (−0.96 s) | stance feet FL 1.302 and RR 0.021 m/s; FR 1.616 and RL 2.088 m/s are in swing, so they do not count | $\max(1.302, 0.021) = 1.302 > \theta = 1.221$ |
| D | `D_tracking` | 240 (+1.00 s) | velocity error at least 0.724 m/s on frames 191–240; 0.428 m/s on frame 190 | frame 240 is the first with 1 s (50 frames) above 0.5 m/s |
| E | `E_pd_error` | 12 (−3.56 s) | FL thigh $\lvert q^{des} - q\rvert = 0.565$ rad | $0.565 > \theta_{FL\,thigh} = 0.551$ |
| privileged | `slip_2feet` | 146 (−0.88 s) | all four feet in stance, speeds 0.910, 0.585, 0.562 and 0.041 m/s | three stance feet above 0.3 m/s, and $3 \ge 2$ |
| terrain | `terrain_on_ice` | 190 (0 s) | all four feet in the air, with an applied friction of 0.055 on each | latched at each foot's last touchdown on ice |

A label becomes a feature as the share of its frames in a 0.5 s bin. `C_slip` is true on 10 of the 25 frames of
$[t_b - 0.5\,\mathrm{s},\ t_b)$, so `C_slip@-0.5` = 0.40. Over the four bins `@-1.5`, `@-1.0`, `@-0.5` and `@+0.0`, `A_tilt` is
0, 0, 0 and 0.72, and `terrain_on_ice` is 0.12, 1, 1 and 1. A firing rule alone is not a failure: in this window
`E_pd_error` first fires 3.56 s before the balance loss.

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

- 2026-11-09: "Rule flags" gives the meaning of the types A–E, an "Implementation" part with the formula of every rule
  and the privileged and terrain labels of `taxonomy.py`, the physical state of `sequences.py` (z-state), and a worked
  example from a real fall. Documentation only.
  Reason: the user's request.
- 2026-11-01 (afternoon): `counterfactual.py` adds the reactions `back`, `stand`, `crouch`, `wide` and `pause`, works
  for the Energy policy (last actions restored), takes `--num_envs`, and records `tilt_max_deg`, `tilt_end_deg`,
  `leg_contact` and `travel_m`. `sequences.py` adds `ae_inv`, an autoencoder made invariant to the policy. New
  `reflex.py` (closed-loop precursor reflex). Reason:
  the reaction axis of the taxonomy (`results/reports/surveillance/20261101_reactions_reflex/REPORT.md`). Base contact alone misses robots that
  end on their side propped on their legs (25 % of the non-contact stair restores of Energy end tilted > 60°).
- 2026-11-01: `sequences.py`, `detect.py`, `counterfactual.py`, `test_sequences.py`; `classify.py --stride`.
  Reason: the user's request (methods to categorize the failure sequences and to detect each category; write-up in
  `results/reports/surveillance/20261101_taxonomy_detection/REPORT.md`).
- 2026-10-31 (evening): `collect.py --trigger_quiet_s` (default 1 s; trigger windows start only at onsets);
  `classify.py` labels frames from the logged flags (`frame_labels`); `taxonomy.py` `sink_5cm` only straight down.
  Reason: in `v3_hard` the frames before a slip / sand trigger had the event in 21–77 % of the frames
  (`forge/surveillance/REPORT.md` §5.4), and sliding feet on slopes counted as sinking.
- 2026-10-31: `collect.py` logs the terrain event's per-foot flags and the stone poses; `render.py` replays the
  stones; `campaign.py --hard`; `taxonomy.py`, `classify.py`; `analyze.load_collection` reads a `triggers/` dir with
  its parent's `meta.json`. Reason: the 2026-10-26 meeting (stones missing in the videos, privileged labels for the
  taxonomy, self-supervised failure classes).
- 2026-10-26: `slip_field` (10 terrain types; `campaign.py` reads `EXTRA_TERRAINS` of `forge/surveillance`). Reason: the user's request.
- 2026-10-26: Event-trigger windows (`--max_triggers`, `triggers/`, `render.py --triggers`); campaign defaults 20 robots, 10 min, 100 falls, 100 trigger windows, 10 + 10 videos. Reason: the user's request (check that the terrain events fire, apart from the falls).
- 2026-10-26: Moved to `scripts/forge/surveillance` with the task (`forge/surveillance`, `IsaacContrib-Surveillance-*`); collections now under `logs/surveillance/`; campaign defaults 100 falls, 10 robots. The campaign `terrain_v1` (20 robots, 200 falls) stays under `logs/fault_analysis/terrain_v1`. Reason: the user's request.
- 2026-10-25: Trigger is base contact only (`--terminations`, default `base_contact`; `--contact_onsets` replaces `--terminal_only`). `episodes.csv`, `summary.json`, `left_tile`, and kept terrain-map curricula in `collect.py` (`--num_envs` now defaults to the task's); `render.py --pick`; `analyze.py --rules`; `campaign.py` on the new deploy tasks (`forge/surveillance`). Scripts moved from `scripts/reinforcement_learning/` to `scripts/forge/`. Reason: the user's request (success rates per difficulty on the terrain maps, rules turned on one at a time during analysis).

- 2026-10-24: `analyze.py` flags the rules A–E on saved windows (`flags.csv`, `thresholds.json`), with baseline percentiles from normal walking; flagged `2026-10-24_rough_terminal` against `smoke/rough2`.
- 2026-10-24: Collected `2026-10-24_rough_terminal`. `render.py` gained `--count`, a closer default camera, and hidden command markers.
- 2026-10-24: Contact onsets no longer reset the robot. The task's terminations do, and are recorded as `terminal`. Added the spawn grace, `--terminal_only`, `--sub_terrains`, the Linux replay mount, and `--real_time` for `render.py`. Reason: in the smoke test, leg contacts were mostly harmless scrapes and spawn artifacts (see Decisions).
- 2026-10-23: First version: `collect.py` (parallel play, non-foot-contact trigger, 5 s ring buffer, per-event npz, terrain mesh) and `render.py` (kinematic replay with a tracking camera).
