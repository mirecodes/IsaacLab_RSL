# Motor-failure detector

Collects motor-fault episodes of the frozen A2 robust teacher (task `IsaacContrib-FaultDetector-MotorFailure-A2Robust`,
`forge/fault_detector/motor_failure_detector`, faults from `forge/fault_injector`), then trains and evaluates MLP and
GRU detectors that output one fault probability per joint from the last $H = 3$ policy steps.

| File | Runs in | What it does |
| --- | --- | --- |
| [collect.py](collect.py) | Isaac Lab | Plays the policy with the injector; writes episode shards, `index.csv`, `meta.json`. |
| [detector.py](detector.py) | torch + numpy | Dataset, history windows, sampler, MLP / GRU model, training, metrics. |
| [experiment.py](experiment.py) | torch + numpy (+ matplotlib) | `sanity` (injector / dataset checks), `matrix` (train + evaluate the 12 runs), `report`. |
| [test_detector.py](test_detector.py) | torch + numpy | Window boundaries and padding, split, overfit, alarm rule, metrics on synthetic data. |
| [tmp/explore.py](tmp/explore.py) | torch + numpy | Scratch study: detection within 60 ms (hypotheses and results in [tmp/REPORT.md](tmp/REPORT.md)). |
| [demo.py](demo.py) | Isaac Lab (Kit window) | Live demo: one robot, a random joint fault per episode, ground truth vs the CJ detector on screen. |
| [tmp/torque_check.py](tmp/torque_check.py) | Isaac Lab | Does a fault reach the physics? Torque at the actuator vs Newton's `joint_f`, gait and falls per (joint, h). |

## Usage

On Euler, profile `isaaclab3fd` (own queue and holder, same image and code as `isaaclab3`; the robust-teacher
checkpoint was copied into its `logs/`):

```bash
# `enqueue` quotes each argument, so a command with && goes through bash -c "..."
CKPT=/workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt
COLLECT="python scripts/forge/fault_detector/collect.py --task IsaacContrib-FaultDetector-MotorFailure-A2Robust --checkpoint $CKPT"
euler -p isaaclab3fd enqueue fd_main -- bash -c "$COLLECT --num_episodes 16384 --seed 0 --output_dir logs/fault_detector/data/main"
euler -p isaaclab3fd enqueue fd_test_id -- bash -c "$COLLECT --num_episodes 2048 --num_envs 2048 --seed 1 --output_dir logs/fault_detector/data/test_id"
euler -p isaaclab3fd enqueue fd_test_ood -- bash -c "$COLLECT --num_episodes 2048 --num_envs 2048 --seed 2 \
    --fault_types power_loss=1 --power_loss_health 0.6 0.8 --output_dir logs/fault_detector/data/test_ood"
euler -p isaaclab3fd enqueue fd_matrix -- bash -c "python scripts/forge/fault_detector/experiment.py sanity logs/fault_detector/data/main \
    && python scripts/forge/fault_detector/experiment.py matrix logs/fault_detector/results/v1 \
    && python scripts/forge/fault_detector/experiment.py report logs/fault_detector/results/v1"
euler -p isaaclab3fd fetch fault_detector/results   # -> results/euler/isaaclab3fd/fault_detector/results
```

- `collect.py`: `--num_episodes`, `--output_dir`, injector overrides `--p_fault`, `--fault_types a=x b=y`,
  `--power_loss_health lo hi`, `--passive_damping`, plus `play.py`'s (`--num_envs`, `--seed`, `--checkpoint`).
  The terrain mesh is always seeded with 0; `--seed` sets the env seed and so the fault schedule.
- `experiment.py matrix <exp_dir> [key=value ...]`: any `DetectorCfg` field (`detector.py`), e.g. `epochs=10`,
  `size=large`, `history_length=5`, `extra_terms='["tracking_error"]'`; `--archs`, `--obs_sets`, `--seeds` select runs.
  Finished runs (with `metrics.json`) are skipped, so a rerun resumes an interrupted matrix.
- Checks: `uv run --no-project --with torch --with numpy python scripts/forge/fault_detector/test_detector.py`.

Data layout: `shards/<n>.npz` holds one array per recorded term, `<group>/<term>` of shape (sum of episode lengths,
dim), plus `ep_len` and `episode_id`; `index.csv` has one row per episode (`has_fault`, `fault_type`, `fault_joint`,
`onset_step`, `health`, `terrain_level`, `terrain_type`, `termination_reason`). Run outputs: `<arch>_<obs_set>_s<seed>/`
with `checkpoint.pt` (weights, cfg, normalizer, dataset index md5), `metrics.json`, `episodes_<set>.csv`, `train.log`;
`report.md`, `results.csv`, `sweep.csv`, `analysis.csv` and `figures/` at the top.

## Live demo

`demo.py` plays the robust teacher with one robot on the training terrain. Faults come and go at random: healthy
periods of 2–5 s alternate with fault periods of 2–5 s, each on one random joint (at most one faulty joint at a time;
default power loss with a random $h \in [0.1, 0.6]$). The schedule replaces the injector's one fault per episode; the
episode ends on a fall or after 20 s (`--record`: 40 s). The detector of a `tmp/explore.py` run runs on every PD
sample (200 Hz) with that run's validation rule, drift and threshold (default: CJ seed 0, per-joint CUSUM, 0.5 false
alarms / min).

On a machine with a display (it needs a Kit window), from the Isaac Lab root:

```bash
# once: the CJ run and the policy (the defaults below look in results/euler/... of the workspace)
euler -p isaaclab3fd fetch fault_detector/results/tmp_explore/phase1/CJ_s0
euler fetch rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0
./isaaclab.sh -p scripts/forge/fault_detector/demo.py
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | fetched robust teacher `model_2999.pt` | policy checkpoint |
| `--detector` | `logs/.../tmp_explore/phase1/CJ_s0`, else the fetched copy | a `tmp/explore.py` run (`checkpoint.pt`, `val.json`: its rule) or an `experiment.py matrix` run (`checkpoint.pt`, `metrics.json`: its 0.5 / min threshold and debounce); PD rate |
| `--fault_types` | `power_loss=1` | e.g. `power_out=0.5 power_loss=0.5`; `kt_loss=1` with `--task IsaacContrib-FaultDetector-MotorDegrade-A2Robust` |
| `--power_loss_health lo hi` | `0.1 0.6` | health $h$ (torque factor) of a power loss; the detector was trained on 0.1–0.6, so e.g. `0.2 0.8` includes weaker faults it never saw |
| `--kt_loss_health lo hi` | the task's (`0 0.8`) | health $h$ of a torque-constant loss |
| `--off_s lo hi`, `--on_s lo hi` | `2 5`, `2 5` | healthy / faulty period lengths [s] |
| `--episode_s` | 20 (`--record`: 40) | episode length [s] |
| `--fault_seed` | random (printed at start) | repeat a fault sequence |
| `--no_noise` | off | feed noiseless measurements (the threshold was set with noise on) |
| `--console` | off | no window, console lines only |
| `--record DIR` | off | headless: one mp4 per episode with the panel drawn in, plus `episodes.log` (the console lines) |
| `--episodes N` | 6 | with `--record`: episodes to record |
| `--eye dx dy dz` | `-1.5 -1.5 0.8` | with `--record`: camera offset from the base [m] |

`--task`, `--viz kit` (not with `--console` / `--record`) and `--num_envs 1` are set by the script; other `play.py`
options pass through.

Recording on Euler (no Kit there: frames come from the headless capture visualizer, as in `fault_analysis/render.py`;
the CJ run must be in the profile's `logs/`, copied there from `isaaclab3fd` for `isaaclab3`):

```bash
euler -p isaaclab3 enqueue fd_demo_render -- python scripts/forge/fault_detector/demo.py \
    --record logs/fault_detector/demo/<name> --episodes 1 --episode_s 40 \
    --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt
euler -p isaaclab3 fetch fault_detector/demo   # -> results/euler/isaaclab3/fault_detector/demo/<name>
```

What is shown:

- **Panel** "Motor faults: GT vs detector": one row per joint. Ground-truth light red with `factor=h` while the joint
  runs with $h < 1$. Detector light red while the alarm holds and that joint's statistic is over the threshold, with
  the detection: its latency from the motor failure (the start of the first faulty PD substep; one PD period, 5 ms,
  more than the study's latency, which counts from the first sample after it) in ms and policy steps (20 ms),
  `wrong joint` (another
  joint is faulty) or `false alarm` (no fault); else green with the statistic (CUSUM value; alarm above the threshold
  in the header). Gray = not armed (first 0.5 s of an episode). The status lines give the time, both sets and the
  last fault's detection.
- **Console**, one line per change, one per fault and one per episode:

  ```text
  [t=  4.36 s] GT {FR_calf_joint: factor=0.32} | fault detector {}
  [t=  4.38 s] GT {FR_calf_joint: factor=0.32} | fault detector {FR_calf_joint: 20 ms (1.00 steps)}
  [fault] FR_calf_joint factor=0.32 from t=4.36 s for 3.12 s: FR_calf_joint after 20 ms (1.00 steps)
  ```

The CJ detector reads a per-joint torque (motor-current) signal: the actuator's applied torque one PD step old, plus
uniform noise of ±1 N·m (`tmp/REPORT.md`). Its CUSUM is capped at twice the threshold: the crossing is unchanged, and
the statistic falls back within a few samples when a fault ends (the detector was trained and calibrated on faults
that never end, so its behavior after a recovery is untested). The live detector matches the offline pipeline sample
by sample, and the fault schedule and the detection scoring have a check (`demo.py --selftest`, torch + numpy only).

## Metrics

- **Offline** (every window past the first 0.5 s, natural distribution): top-1 joint accuracy and leg accuracy on
  faulty windows, 12 × 12 confusion, per-joint AUROC / AP, type accuracy, nominal-window false-positive rate
  ($p_{any} > 0.5$), ECE (10 bins) with reliability diagram, top-1 accuracy by steps since onset.
- **Closed loop** (whole episodes, alarm rule of `REPORT.md` §3.3): median / p90 latency, detection within 10 and 25
  steps, miss rate, correct joint / leg at the alarm, false alarms per minute of nominal walking (rising edges before
  the onset or in nominal episodes) and per episode, falls before detection; swept over $\tau \times k$.
- Sets: `test` (10 % of `main`, by episode), `test_id` (same distribution, other seed), `test_ood` (power loss
  $h \in [0.6, 0.8]$); each with input noise on and off. The table uses `test_id`, noise on, $\tau = 0.5$, $k = 2$.

## Decisions

Choices the spec leaves open or that the setup forces, taken in the spirit of the experiment:

| Decision | Choice | Reason |
| --- | --- | --- |
| Terrain | training grid only, all levels uniform | the user's instruction (one environment); the policy's own distribution, so abnormal motion comes from the fault, not from unseen terrain. Harsh-world OOD dropped |
| `test_ood` | power loss only, $h \in [0.6, 0.8]$, same terrain | follows from the terrain decision |
| Actuator path | Isaac Lab DC motor (`use_newton_actuators = False`) | Newton-native actuators reject custom classes; same DC-motor model |
| Onset granularity | policy steps | labels are per policy step; first faulty label = first observation after faulty physics |
| `last_action` | raw 16-dim CPG action | the robust teacher's action space (spec assumed 12) |
| Disturbances | pushes and external wrench off | false alarms should reflect the detector, not random shoves |
| Policy noise | none (teacher trained noiseless) | same behaviour as in training |
| Detector noise | Isaac Lab velocity-task scales, proprio only | spec 5.3; the teacher has no noise of its own |
| Closed loop | replay of whole held-out episodes | the detector is passive, so this equals in-sim per-step inference; one rollout serves all runs and the sweep |
| Alarm arming | 0.5 s after each reset | the reset transient is excluded from training too (`skip_initial_s`) |
| Labels near the end | episodes ending before the onset count as nominal | their labels are all 0 |
| Split | stratified by (has_fault, type, joint), seed 0, shared by all runs | spec 7.2 |
| Validation | 262k windows drawn once with the training sampler, noise on | stable early-stopping signal |
| Model code | `scripts/forge/fault_detector/detector.py`, not the task's `agents/` | it never runs inside the env; plain torch keeps it testable without Isaac |
| Deferred | health head, `softmax13`, stateful GRU, recovery / multi-joint faults, policy swap, power-out videos | the spec defers them until the 2 × 2 results are in |

## Results (2026-10-25, `results/v1`, `v1_h10`, `v1_h25`)

Full tables and figures: `euler -p isaaclab3fd fetch fault_detector/results` →
`results/euler/isaaclab3fd/fault_detector/results/v1*/report.md` and `figures/`. Mean ± std over 3 seeds;
closed loop on `test_id` (2162 episodes, 4446 faulty), noise on, $\tau = 0.5$, $k = 2$; 1 step = 20 ms.

### Data and injector checks (`experiment.py sanity`, `main`)

16,384 episodes / 13.5M steps (3858 env steps × 4096 robots, 171 s on one RTX 4090). Faulty 69 %, 912–998
episodes per joint, onset in [2.0, 8.0] s, power-loss $h$ in [0.10, 0.60]. Labels match the schedule in every
episode; applied torque of a power-out joint after onset is exactly 0; applied / PD torque of a power-loss joint
correlates with $h$ at 0.998. Power out ends in a fall in 66 % of episodes (power loss 14 %, nominal 16 %).
**Actuator path check**: first episode of each of 512 robots, same seed, no faults: falls 31 (Isaac Lab DC motor,
this task) vs 34 (Newton-native, the training setup), same per-level pattern; nominal falls concentrate on the top
two levels of `grid_steps` (0.35 m steps, above A2's rated 0.30 m).

### 2 × 2 (H = 3)

| Config | Top-1 joint | Type acc | Macro AUROC | Median latency | Detect ≤ 10 / ≤ 25 step | Correct joint at alarm | False alarm / min | ECE | Params | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|
| MLP · proprio | 0.794 ± 0.001 | 0.648 | 0.946 | 6 | 0.680 / 0.890 | 0.943 | 8.06 ± 0.20 | 0.002 | 72.7k | 0.032 |
| GRU · proprio | 0.773 ± 0.001 | 0.641 | 0.938 | 7 | 0.644 / 0.874 | 0.931 | 8.51 ± 0.32 | 0.001 | 70.7k | 0.092 |
| MLP · proprio_priv | 0.807 ± 0.001 | 0.662 | 0.950 | 6 | 0.686 / 0.893 | 0.948 | 6.81 ± 0.34 | 0.002 | 75.0k | 0.032 |
| GRU · proprio_priv | 0.781 ± 0.001 | 0.650 | 0.942 | 7 | 0.642 / 0.869 | 0.926 | 7.65 ± 0.22 | 0.001 | 71.8k | 0.091 |

Miss rate ≤ 1 %, falls before detection ≤ 0.3 % in every config. Seed spread is ~0.001 in accuracy, so every gap
above is far outside seed noise.

- **Q1 (MLP vs GRU)**: the MLP is better on every metric at equal parameters and identical information (window
  GRU): +2 pt top-1, 1 step lower latency, fewer false alarms over the whole sweep
  (`figures/latency_vs_false_alarm.png`), and 3 × faster on CPU (0.03 vs 0.09 ms; both far inside 20 ms).
- **Q2 (privileged base state)**: small gain: +1 pt top-1, −15 % false alarms (MLP 8.1 → 6.8 / min), latency
  unchanged. Ground-truth base height and velocity are an upper bound, so a state estimator is not justified by
  detection alone.
- **Q3**: power out is fast (median 4 steps = 80 ms, 85 % within 10 steps, MLP) and power loss slower (10 steps, 49 %
  within 10), degrading with $h$ (6 steps at $h \in [0.1, 0.2)$ to 18 at $[0.5, 0.6)$). Thigh (HFE) faults of the front
  legs are the hardest (FL / FR thigh median 13 / 11 steps vs 3–7 elsewhere); hind legs are faster than front
  (5 vs 7). A fault during stance is found 2 steps sooner than during swing (5 vs 7). Command speed and terrain
  family matter little, except `grid_steps` lowers the correct-joint rate (0.86).
- **Spec target** (power-out median ≤ 3 steps): reached only at $\tau = 0.3, k = 1$ (3 steps, but 63 false alarms /
  min). At the default 4 steps.
- **Calibration**: ECE ≤ 0.002; the reliability curve sits slightly above the diagonal (mildly under-confident in
  0.3–0.8), so $p_j$ can be passed on as a probability.
- **Robustness**: input noise costs ~1 pt top-1 and no latency. `test_ood` (weak power loss $h \in [0.6, 0.8]$) is
  hard: top-1 0.34, median latency 62 steps, 12 % missed (MLP · proprio).

### False alarms and history length (MLP · proprio)

False alarms are the weak point (8 / min at the default, 37 % of episodes see one before the onset). They grow
monotonically with terrain difficulty (3.7 / min on level 0 to 12.9 / min on levels 8–9), and only 6 % of the false
alarms in nominal episodes occur within 1 s before a nominal fall (seed 0): with a 60 ms window the detector confuses
terrain-induced transients with faults.
History-length ablation (the spec's "add after the 2 × 2"):

| H | Params | Top-1 joint | Type acc | test_ood top-1 / latency | $\tau$ 0.5, $k$ 2: latency / FA per min | at ~1 FA / min: setting → latency |
|---|---|---|---|---|---|---|
| 3 | 72.7k | 0.794 | 0.648 | 0.34 / 62 | 6 / 8.1 | $\tau$ 0.9, $k$ 2 → 8.0 (1.07 / min) |
| 10 | 160.5k | 0.872 | 0.756 | 0.46 / 43 | 7 / 6.5 | $\tau$ 0.9, $k$ 2 → 8.3 (0.89 / min) |
| 25 | 348.7k | 0.916 | 0.835 | 0.58 / 45 | 8 / 4.8 | $\tau$ 0.9, $k$ 1 → 9.0 (1.38 / min); $k$ 2 → 10.0 (0.68 / min) |

A longer history clearly improves identification (which joint, which fault, weak faults), but at a matched false-alarm
rate it does not shorten detection: the latency–false-alarm trade-off is set by the terrain transients, not by the
window. (The MLP's parameters grow with $H$; a GRU with stateful hidden state would test long history at fixed size.)

### Suggested next steps

1. Fix the operating point with the supervisor from `latency_vs_false_alarm.png` (e.g. $\tau = 0.9, k = 2$: 8 steps,
   ~1 false alarm / min, 80 % detected within 25 steps).
2. Two-stage use: a fast alarm from the H = 3 MLP, joint / type identification from a longer window.
3. Reduce terrain-driven false alarms: stateful GRU (`gru_mode`), or the policy's own tracking error / foot-scan
   terms as inputs, then re-check the trade-off.

## v2 (2026-10-25, `results/v2`, ablations `v2_a_h5`, `v2_b_h5_grace_hnm`, `v2_c_pd_h5`, `compare/`)

Feedback on v1: compare at 0.5 false alarms / min, history 5, fewer false alarms right after the onset, inputs at the
PD rate, confusion matrices and one-leg (3-motor) accuracy. Data: `main_pd`, `test_id_pd`, `test_ood_pd`, the v1
recipe re-collected with `collect.py --pd_rate` (16,387 / 2,171 / 2,101 episodes; PD-rate samples match the
policy-rate observation at every $m t$ to float16 precision, `experiment.py sanity`).

### Decisions (v2)

| Decision | Choice | Reason |
| --- | --- | --- |
| Comparison | each run's threshold $\tau$ and debounce $k \in$ {0, 20, 40, 60 ms} set on its validation split to ≤ 0.5 false alarms / min (lowest median latency), then scored on the test sets; also 0.25 / 1 / 2 per min | a fixed $\tau = 0.5$ compared runs at different false-alarm rates (v1: 7–9 / min) |
| Success | alarm ≤ 500 ms after the onset whose argmax joint is the faulty joint (`joint`) or on its leg (`leg`) | one number for detection + identification; `leg` answers "which leg's motors" |
| History | 5 policy steps = 100 ms (20 samples at the PD rate) | feedback 2; the MLP input grows ×4 at the PD rate (286k params) |
| PD rate | samples at every actuator step (200 Hz), tapped inside the actuator (`FaultyDCMotor.probe`); terms without a PD-rate array repeated | feedback 4; the state the PD law acts on |
| Onset false alarms | label grace 80 ms (left out of the loss) + 5 hard-negative fine-tuning epochs (nominal windows weighted by $1 + 4\,p_{any}$, lr × 0.1) | feedback 3: the alarms right after the onset are a moving-boundary artifact (grace); the remaining nominal false alarms are hard negatives |
| Temperature scaling | not used | monotone, so it does not move a matched-false-alarm operating point; ECE is already ≤ 0.002 |
| Ablations | MLP · proprio, 3 seeds each: a = H5 only, b = + grace + HNM, c = + PD rate | attribute the gain to each change |
| `applied_torque` at the PD rate | recorded (`pd/applied_torque`, torque of the previous substep) but not a detector input | a motor-current sensor; only in the 60 ms exploration (`tmp/`) as a separate tier |

### Results (test_id, noise on, 0.5 / min, mean over 3 seeds)

| Config | Success joint ≤ 500 ms | Success leg ≤ 500 ms | Median latency | Detect ≤ 200 ms | Miss | FA / min (test) | test_ood success |
|---|---|---|---|---|---|---|---|
| v1 H3 MLP · proprio | 0.742 | 0.742 | 193 ms | 0.504 | 0.057 | 0.38 | 0.058 |
| v2a H5 | 0.770 | 0.771 | 167 ms | 0.557 | 0.045 | 0.43 | 0.076 |
| v2b H5 + grace + HNM | 0.786 | 0.786 | 160 ms | 0.577 | 0.037 | 0.46 | 0.091 |
| v2c H5 + PD rate | 0.792 | 0.793 | 163 ms | 0.562 | 0.033 | 0.40 | 0.088 |
| **v2 MLP · proprio** (all) | **0.820** | **0.821** | **155 ms** | 0.589 | 0.025 | 0.46 | 0.112 |
| v2 GRU · proprio | 0.775 | 0.776 | 192 ms | 0.505 | 0.034 | 0.35 | 0.118 |
| v2 MLP · proprio_priv | 0.820 | 0.822 | 142 ms | 0.611 | 0.026 | 0.43 | 0.113 |
| v2 GRU · proprio_priv | 0.784 | 0.785 | 190 ms | 0.509 | 0.032 | 0.39 | 0.126 |

- **Each change helps, and they add up**: H5 +2.8 pt, grace + HNM +1.6, PD rate +2.2; together +7.8 pt over v1
  (0.742 → 0.820), 38 ms lower median latency, misses 5.7 → 2.5 %. The validation thresholds hold on the test set
  (0.35–0.46 false alarms / min against the 0.5 target).
- **MLP vs GRU**: the MLP stays ahead at the PD rate (+4.5 pt, 37 ms faster); the GRU is slightly better on weak
  faults (`test_ood` 0.118 vs 0.112) and 7 × slower on CPU (0.34 ms, still far inside 5 ms).
- **Confusion / one-leg faults** (`compare/confusion_compare.png`, `v2/figures/confusion.png`): the 12 × 12
  confusion is diagonal; window-level leg accuracy 0.876 vs joint 0.845, and ~20 % of the joint errors stay on the
  right leg. At the alarm the diagnosis is almost always right (joint 0.992, leg 0.995), so success joint ≈ success
  leg: misses come from late detection, not from naming the wrong motor of the leg. Front legs remain harder (leg
  accuracy FL / FR 0.85 vs RL / RR 0.91 / 0.89; front thigh faults succeed in 0.58 / 0.71).
- **Still weak**: power loss at $h \ge 0.4$ (success 0.54 / 0.37) and `test_ood` (0.11). Power out: 0.96, median
  110 ms.
- **60 ms (3 policy steps)**: at these operating points only 13 % (v2 MLP) to 21 % (v2a) of the faults are found
  with the right joint within 60 ms; seeds whose operating point uses a debounce k > 1 (picked for the lowest median
  latency) drop furthest. Proprioception cannot reach 0.9 within 60 ms (half the faults barely change the motion
  in that time); a motor-current reading with a joint-shared network reaches 0.957 on `test_id`: see
  [`tmp/REPORT.md`](tmp/REPORT.md).

## kt_loss (2026-10-26, `results/kt_A`, `kt_B`, `kt_compare`)

The user's request: a fault that scales both the gain and the torque limit, as demagnetized magnets or a worn gear
(lower torque constant $K_t$), $h$ from 0 (motor off) to 0.8, and the v2 comparison repeated on it, with $h \le 0.6$ and
the harder $h \le 0.8$. Task `IsaacContrib-FaultDetector-MotorDegrade-A2Robust` (`forge/fault_detector/
motor_degrade_detector/REPORT.md`), fault type `kt_loss` of `forge/fault_injector`.

**Why the walk barely changes** (`tmp/torque_check.py`, 3 robots per case, 5 s of fault after 3 s of walking): the
torque reaches Newton exactly as computed (sim drive gains 0, max |`joint_f` − actuator output| = 0), and the existing
power loss already scales gain and limit ($h\,\mathrm{clip}_{DC}(\tau_{PD}) = \mathrm{clip}(h\,\tau_{PD}, \pm h\,
\tau_{max})$). But the PD demand grows 3–5× as the joint lags (tracking error 2–3×), so at $h = 0.1$ the joint still
delivers 22–48 % of its healthy torque; walking torques sit far below the 120 / 180 N·m limits. $h = 0$ on one joint
felled 4 of 12 robots, on all joints 2 of 3 within 0.6 s.

### Decisions (kt_loss)

| Decision | Choice | Reason |
| --- | --- | --- |
| Torque law | unchanged ($\tau = h\,\mathrm{clip}_{DC}(\tau_{PD})$) | identical to the requested $\mathrm{clip}(h\,\tau_{PD}, \pm h\,\tau_{max})$ |
| Motor current | `motor_current` = $K_{t,nom}\,i = \mathrm{clip}_{DC}(\tau_{PD})$ of the previous PD substep, ±1 N·m noise | the user's choice: what the drive reports; its current loop holds $i$ at the command, so $h$ is invisible to it. The CJ tier of `tmp/` read $h\,\tau$ instead (a torque sensor), which is why it found faults within 5–20 ms |
| Data | `data_kt/main_pd` 17,104 episodes, `test_id_pd` 4,230 (seed 1); 68 % faulty, $h \sim U[0, 0.8]$ | the v2 recipe (PD rate) with the new fault |
| A / B | A trains, validates and tests on nominal + $h \le 0.6$ episodes (`max_health=0.6`), B on all | the user's choice: each range trained and scored on its own |
| Ablation | `proprio` vs `proprio_current` (+12 inputs per sample); MLP and GRU; 3 seeds | the user's request |
| Recipe | v2: H = 5 policy steps (20 PD samples), 80 ms grace, 5 HNM epochs, thresholds on val at 0.5 / min | comparable with v2 |
| GRU inference | batches of 8192 windows (32768 for the MLP) | cuDNN's GRU workspace for 32k windows ran out of the 24 GB GPU next to B's data |

Checks: `experiment.py sanity` on both sets: labels match, and at every PD sample the applied torque equals
$h \times$ `motor_current` on the faulty joint and `motor_current` elsewhere (max error 7e-4, float16; 280 M samples).

### Results (test_id, noise on, 0.5 / min, mean ± std over 3 seeds)

| Config | A ($h \le 0.6$): success joint / leg ≤ 500 ms | A median latency | A miss | B ($h \le 0.8$): success joint / leg | B median latency | B miss | FA / min (A / B) |
|---|---|---|---|---|---|---|---|
| MLP · proprio | 0.760 / 0.761 | 213 ms | 0.033 | 0.642 / 0.643 | 237 ms | 0.119 | 0.38 / 0.49 |
| GRU · proprio | 0.685 / 0.686 | 287 ms | 0.045 | 0.585 / 0.586 | 310 ms | 0.128 | 0.59 / 0.63 |
| **MLP · proprio + current** | **0.844 / 0.845** | **153 ms** | 0.016 | **0.735 / 0.735** | **178 ms** | 0.066 | 0.42 / 0.47 |
| GRU · proprio + current | 0.808 / 0.809 | 190 ms | 0.016 | 0.691 / 0.691 | 227 ms | 0.065 | 0.51 / 0.59 |

By $h$ (MLP · proprio + current, seeds pooled; success joint ≤ 500 ms / median latency):

| $h$ | [0, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.4) | [0.4, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.8) |
|---|---|---|---|---|---|---|---|---|
| A | 0.96 / 90 | 0.96 / 110 | 0.94 / 125 | 0.86 / 165 | 0.82 / 220 | 0.55 / 400 | – | – |
| B | 0.97 / 95 | 0.96 / 110 | 0.94 / 120 | 0.88 / 155 | 0.84 / 200 | 0.63 / 290 | 0.43 / 622 | 0.22 / 1330 |
| B, MLP · proprio | 0.93 / 125 | 0.90 / 155 | 0.89 / 170 | 0.75 / 222 | 0.68 / 295 | 0.50 / 395 | 0.31 / 852 | 0.14 / 1548 |

By leg (success joint ≤ 500 ms, MLP · proprio + current): A FL 0.79, FR 0.84, RL 0.88, RR 0.86; B 0.63, 0.69, 0.82,
0.79.

- **The motor current helps although it does not carry $h$**: +8.4 pt in A (+9.3 in B), 60 ms faster, half the
  misses. It is the PD command, i.e. the tracking error $q^\ast - q$, which proprioception lacks (the CPG action is not
  the joint target); a lagging joint shows as a growing command at once. An MLP with the tracking error instead of the
  current (`extra_terms=["tracking_error"]`, `results/kt_A_te`, `kt_B_te`) recovers about half of the gain: A 0.806 /
  180 ms, B 0.697 / 203 ms (proprio 0.760 / 0.642, + current 0.844 / 0.735). The rest is the command itself:
  $K_p e - K_d \dot q$ with per-joint gains and the DC saturation, which the network would have to learn from $e$ and
  $\dot q$ (each with its own noise).
- **MLP beats GRU** in every cell (+3.6 to +7.5 pt, 37–74 ms faster), as in v2; the GRU also runs over its
  false-alarm target on the test set (0.59–0.63 / min).
- **B is harder only through its weak faults**: on $h < 0.6$ the B models match or beat the A models (e.g. [0.5, 0.6):
  0.63 vs 0.55), so training on weaker faults costs nothing; B's overall drop comes from $h \ge 0.6$, where 0.43 / 0.22
  are found within 500 ms (the robot hardly notices a 30 % torque loss, see the torque check).
- **Joint vs leg**: the diagnosis at the alarm is right for 0.97–0.99 (joint) / 0.97–0.99 (leg), so the two success
  rates coincide; misses are late detections. Front legs stay harder than hind legs, hip and thigh harder than knee.
- **Renders** (`euler -p isaaclab3 fetch fault_detector/demo`, same fault seed 1144512330, MLP · proprio + current
  seed 0): `2026-10-26_kt_A_h0-0.6` and `2026-10-26_kt_B_h0-0.8`, 6 faults each, all found on the right joint
  (A 55–205 ms, B 60–295 ms), no false alarm.
