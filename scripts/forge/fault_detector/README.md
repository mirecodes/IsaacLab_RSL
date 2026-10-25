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
