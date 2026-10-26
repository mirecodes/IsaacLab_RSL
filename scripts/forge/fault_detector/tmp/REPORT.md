# Exploration: motor-fault detection within 3 policy steps

Scratch study (`tmp/`) next to the main detector pipeline (`../README.md`, v2). Question: can a single-joint motor
fault be found, and the joint (or at least the leg) be named, within **3 policy steps = 60 ms** of its onset, at
**≤ 0.5 false alarms per minute** of nominal walking? Code: `explore.py`; results on Euler under
`logs/fault_detector/results/tmp_explore/`.

## Protocol (fixed before the first result)

- **Data**: the v2 PD-rate datasets (`main_pd`, `test_id_pd`, `test_ood_pd`; 200 Hz, one terrain = the training
  grid). Same episode split of `main_pd` as the main pipeline (80 / 10 / 10, seed 0).
- **Score**: `S60` = share of faulty episodes with an alarm ≤ 60 ms after the first faulty sample whose diagnosis
  (argmax joint at the alarm) is the faulty joint (`S60 joint`) or on the faulty leg (`S60 leg`). Same latency
  convention as the main study (latency 0 = first sample after the first faulty physics substep).
- **"Sufficient"**: `S60 ≥ 0.9` on `test_id` (noise on) at the validation-set 0.5 / min threshold.
- **Selection only on validation.** Every choice (hypothesis kept or dropped, decision rule, CUSUM drift,
  threshold) is made on the validation split. `run` without `--test` does not load the test sets. A configuration
  is tested once, after it was selected; `test_ood` (power loss $h \in [0.6, 0.8]$, outside training) is never used
  for selection. Final configurations: 3 seeds, mean ± std.
- **Fixed budget**: the same training recipe for all classifiers (v2: MLP sizes, 30 epochs, early stop on val
  macro-F1, 5 HNM epochs; label grace 0, see H5), no per-hypothesis hyperparameter tuning. Decision-rule candidates
  are fixed: debounce $k \in \{1, 2, 4, 8\}$ samples, per-joint CUSUM with drift at the 0.9 / 0.99 / 0.999 nominal
  validation quantile of the score.
- **Noise**: the main study's uniform observation noise (joint velocity ±1.5 rad/s, ...) on all raw measurements,
  derived features computed from the noisy ones; motor-current torque ±1.0 N m. Noise-off numbers are reported,
  not selected on.
- **Observability**: per faulty episode, the torque the fault removed in its first 60 ms,
  $I = \sum (1 - h)\,|\mathrm{clip}_{DC}(\tau_{PD})|\,\Delta t$ over the faulty substeps (clean state). Results are
  broken down by quartile of $I$ (edges from validation) and by stance / swing at onset: a fault that removes almost
  no torque is physically silent within the deadline, whatever the detector.

## Hypotheses (with the expected outcome, written before the runs)

| ID | Hypothesis | Run(s) | Expected |
|---|---|---|---|
| H0 | A large share of the 60 ms misses are physically silent (small $I$, swing phase) | breakdown of every run | `S60` near 0 in the lowest $I$ quartile, high in the top one |
| H1 | Explicit physics features (tracking error, PD torque command, joint acceleration) let a network see the missing torque within a few PD samples | `P1` vs `B0` | `S60` up |
| H2 | A joint-shared network (one per-joint model + leg mates + global context) generalizes better than a 12-output MLP, less overfitting | `J1` vs `P1` | `S60` up or equal at ~⅓ the parameters; smaller val→test gap; better `test_ood` |
| H3 | A nominal-dynamics residual detector (trained on nominal samples only, no fault labels) is competitive and cannot overfit the fault distribution | `R1` | lower `S60` than supervised on `test_id`, smaller drop on `test_ood` |
| H4 | A motor-current reading (A2 reports joint torque estimates) makes a power-out / power-loss visible one PD period after onset | `C1`, `CJ` | `S60` ≥ 0.9 (separate sensor tier) |
| H5 | The v2 label grace (80 ms) teaches the model not to fire early and costs 60 ms detections | `B0` (grace 0) vs `V2` (v2 checkpoint, grace 0.08) | `S60` of `B0` > `V2` |
| H6 | Accumulating evidence (CUSUM) beats k-consecutive debounce at matched false alarms | rule chosen on val, all runs | CUSUM selected for most runs |
| H7 | Leg-level diagnosis is much easier than joint-level within 60 ms (most early confusions are within a leg) | `S60 leg` vs `S60 joint` | gap ≥ 10 pt |

Phase 1: one seed, validation only, all runs above. Phase 2: the best proprioceptive run and the best current-tier
run (by validation `S60 joint`), 3 seeds, test sets once.

## Starting point: the main pipeline at 60 ms (test_id, noise on, 3 seeds)

From the main runs' episode tables (`results/<exp>/<run>/episodes_test_id.csv`), at their own validation 0.5 / min
operating point (chosen for the lowest median latency, not for 60 ms):

| Run | Success ≤ 500 ms | `S60 joint` | `S60 leg` | `S60 joint`, power out only |
|---|---|---|---|---|
| v1 H3 MLP · proprio | 0.742 | 0.144 | 0.144 | 0.218 |
| v2a H5 (policy rate) | 0.770 | **0.211** | 0.211 | 0.303 |
| v2b H5 + grace + HNM | 0.786 | 0.189 | 0.189 | 0.272 |
| v2c PD rate, H5 | 0.792 | 0.142 | 0.142 | 0.210 |
| v2 (PD, H5, grace, HNM) MLP | 0.820 | 0.133 | 0.133 | 0.201 |
| v2 GRU | 0.775 | 0.087 | 0.087 | 0.134 |

Far below the 0.9 bar, so the exploration proceeds. Two observations before any new run: the changes that raise the
500 ms success (label grace, PD rate) seem to lower `S60` (H5 was written from this); and almost every 60 ms alarm
names the right joint (`S60 joint` = `S60 leg` ≈ detect ≤ 60 ms), so the bottleneck is
detecting early at a low false-alarm rate, not identification. (At the PD rate latency starts 15 ms earlier in
physical time than at the policy rate; the PD-rate rows have the stricter deadline.)

## Phase 1 (validation, seed 0)

`S60` at the validation 0.5 / min threshold with the rule chosen on validation; `I` quartile edges 0.35 / 0.85 /
1.96 N m s (a mean removed torque of ~6 / 14 / 33 N m over the 60 ms).

| Run | Model / features | Params | `S60 joint` (rule) | power out / loss | stance / swing | `S60` by `I` quartile (low → high) | `S60` at 0.25 / 1 / 2 per min |
|---|---|---|---|---|---|---|---|
| B0 | MLP / proprio (v2 recipe, grace 0) | 286k | 0.165 (debounce 1) | 0.23 / 0.09 | 0.26 / 0.10 | 0.00 / 0.01 / 0.13 / 0.52 | 0.09 / 0.19 / 0.24 |
| P1 | MLP / phys | 470k | 0.271 (debounce 1) | 0.39 / 0.14 | 0.38 / 0.20 | 0.00 / 0.06 / 0.30 / 0.72 | 0.25 / 0.30 / 0.32 |
| J1 | joint-shared / phys | 129k | **0.275** (debounce 1) | 0.39 / 0.15 | 0.34 / 0.23 | 0.00 / 0.09 / 0.28 / 0.73 | 0.25 / 0.31 / 0.34 |
| R1 | nominal residual / phys | 470k | 0.014 (debounce 8) | – | – | – | – |
| C1 | MLP / current | 593k | 0.825 (CUSUM q0.999) | 0.92 / 0.72 | 0.89 / 0.78 | 0.43 / 0.88 / 0.99 / 1.00 | 0.81 / 0.84 / 0.85 |
| CJ | joint-shared / current | 145k | **0.950** (CUSUM q0.999) | 0.99 / 0.91 | 0.98 / 0.93 | 0.81 / 0.99 / 1.00 / 1.00 | 0.94 / 0.95 / 0.96 |
| V2 | the v2 checkpoint (grace 0.08) | 286k | 0.175 (debounce 1) | | | | |

`S60 leg` equals `S60 joint` within 0.005 in every run. Median latency at the chosen rule: CJ 10 ms, C1 25 ms.
Pipeline check of the current tier: on clean nominal samples the recorded torque matches the previous sample's
DC-clipped PD command to median 0.008 N m (p99 0.064 N m), i.e. float16 precision.

Verdict per hypothesis (validation only; final numbers in phase 2):

- **H0 confirmed.** With proprioception, the lowest quartile of removed torque is never found within 60 ms and the
  second one rarely (≤ 0.09); the top quartile reaches 0.72. About half of the faults barely change the motion within
  the deadline, so the 0.9 bar is out of reach for any proprioceptive detector, whatever the model. Swing-phase
  faults (small torque demand) are the hard half.
- **H1 supported.** Explicit physics features: 0.165 → 0.271, the gain in the middle quartiles and in swing.
- **H2 partly.** Joint-shared ≈ MLP on validation (0.275 vs 0.271, proprio) at 27 % of the parameters; with the
  current signal it is far better (0.950 vs 0.825, lowest quartile 0.81 vs 0.43, 24 % of the parameters): one shared per-joint residual detector
  learns from 12× the examples. Generalization is checked on the test sets in phase 2.
- **H3 rejected.** The residual model predicts the next joint velocity well (normalized val MSE 0.12), but its score
  has heavy nominal tails (touch-down impacts, velocity noise): at 0.5 / min almost nothing passes within 60 ms.
- **H4 confirmed** (sensor tier): a motor-current reading takes `S60` from 0.27 to 0.83–0.95.
- **H5 rejected.** Without the 80 ms label grace (B0) the 60 ms score is not higher than with it (V2: 0.175 vs
  0.165, same validation protocol). The low starting-point means came from the main pipeline's operating point:
  it picks the debounce with the lowest median latency, which for some seeds is k > 1, and that costs the 60 ms
  score (per seed, `S60` / k: v2 0.175 / 1, 0.177 / 1, 0.048 / 8; v2b 0.213 / 1, 0.203 / 1, 0.152 / 2; v2c 0.128 / 4,
  0.178 / 1, 0.120 / 4; v1 0.098 / 3, 0.159 / 2, 0.177 / 2). At k = 1 grace does not hurt (v2b 0.21 ≈ v2a 0.21);
  the PD-rate runs stay ~3 pt lower (v2 0.176, v2c 0.178), consistent with their 15 ms stricter deadline.
- **H6 rejected for proprioception, supported with current.** Debounce 1 wins for B0 / P1 / J1; with the current
  signal CUSUM at the 0.999 drift wins narrowly (CJ 0.950 vs 0.936 debounce 1).
- **H7 rejected.** Leg-level is not easier within 60 ms: an early alarm almost always names the right joint already;
  detection is the bottleneck.

Phase 2 candidates by the pre-set rule (best validation `S60 joint` per tier): **J1** (proprioceptive) and **CJ**
(motor current); plus V2 on the test sets as the baseline under the same rule selection.

## Phase 2 (test, 3 seeds)

Validation-chosen rule, drift and threshold applied unchanged to the test sets; mean ± std over seeds (V2: one
checkpoint). `test_ood` = power loss $h \in [0.6, 0.8]$, never trained on.

| Run | val `S60` | `test_id` `S60 joint` / `leg` | FA / min (test) | `S60`, noise off | Median latency | Success ≤ 500 ms | `test_ood` `S60` / ≤ 500 ms |
|---|---|---|---|---|---|---|---|
| V2 (main pipeline, v2 MLP) | 0.175 | 0.172 / 0.172 | 0.46 | 0.170 | 150 ms | 0.809 | 0.000 / 0.103 |
| J1 (joint-shared, phys) | 0.266 ± 0.006 | 0.283 ± 0.005 / 0.283 ± 0.005 | 0.35 ± 0.03 | 0.310 ± 0.010 | 108 ms | 0.859 | 0.002 / 0.120 |
| **CJ (joint-shared, current)** | 0.947 ± 0.002 | **0.957 ± 0.003** / 0.957 ± 0.003 | 0.49 ± 0.03 | 0.953 ± 0.003 | 10 ms | 1.000 | 0.565 ± 0.007 / 0.993 |

J1 on `test_id` by group: power out 0.404, power loss 0.151; stance 0.335, swing 0.242; by quartile of $I$: 0.001 /
0.106 / 0.314 / 0.729. Trade-off: 0.253 / 0.283 / 0.313 / 0.342.

CJ on `test_id` by group: power out 0.983, power loss 0.929; stance 0.974, swing 0.944; by quartile of removed torque
$I$: 0.835 / 1.000 / 0.998 / 1.000. Trade-off (`S60` at 0.25 / 0.5 / 1 / 2 per min): 0.953 / 0.957 / 0.961 / 0.964.

- **No overfitting visible**: the test scores match or exceed validation (CJ 0.947 → 0.957, J1 0.266 → 0.283, V2
  0.175 → 0.172), the validation thresholds hold their false-alarm rate on the test set (0.35–0.49 / min), and the
  seed spread is ≤ 0.005. `test_ood`, never used for any choice, keeps CJ at 0.99 within 500 ms; within 60 ms it finds 0.57 of
  these weak faults (they remove 20–40 % of the torque).
- **V2 has no validation → test gap** under this protocol (0.175 → 0.172), and it matches the main pipeline's own
  number for the same checkpoint (seed 0: 0.175, debounce 1); the 0.133 mean of the starting-point table is pulled
  down by seed 2's debounce-8 operating point (H5 above).

## Conclusions

1. **With proprioception alone, 60 ms is physically out of reach for half of the faults.** The best detector (J1:
   physics features, joint-shared network) finds 0.283 on `test_id` at 0.35 / min (v2: 0.172), 0.73 of the faults
   that remove much torque early and none of those that remove little. No model, decision rule or
   history change moves the lowest quartile; only a signal that measures torque does. (J1 is still the better
   proprioceptive detector overall: 0.859 within 500 ms at its 60 ms operating point, v2 0.809.)
2. **A motor-current (torque) reading solves it**: joint-shared network + current (CJ) finds **0.957 ± 0.003** of
   the faults with the right joint (and leg) within 60 ms at 0.49 false alarms / min, median 10 ms, every fault
   within 500 ms, robust to input noise (0.953 noise off) and to unseen weak power loss (0.99 within 500 ms). This
   assumes a per-joint torque reading (motor current, as Unitree drivers report a torque estimate; the A2's
   actual noise and delay are unverified here) with ±1 N m noise, one PD period old, and a fault that leaves that
   reading intact (a dead driver may stop reporting, which is itself a detectable event).
3. **Leg-level diagnosis gives nothing extra within 60 ms**: early alarms name the right joint already (`S60 leg` −
   `S60 joint` ≤ 0.001 in phase 2, ≤ 0.005 in phase 1).
4. Rejected: nominal-dynamics residual (H3), CUSUM for proprioception (H6), label grace as the cause of slow alarms
   (H5).

Suggested next steps (not run): check the A2's real torque-estimate noise and latency, then re-train CJ with that
noise model (the tier hinges on it); test a dead-driver case where the torque reading freezes or drops out; the
proprioceptive detector's operating point should be selected for the deployment deadline (k = 1 for 60 ms), not
for the median latency.

## Attempt log

- 2026-10-25: protocol and hypotheses fixed; `explore.py` self-test on synthetic data passes (CUSUM vs plain loop,
  every model kind end to end, `--test` path). `collect.py` now also records `pd/applied_torque` (torque of the
  previous substep) so H4 can run on the same episodes; the queued v2 collection jobs were repointed to that code
  (`euler retarget`), so no extra collection was needed.
- 2026-10-25: `load_episodes` fills one preallocated array (the holder's 32 GB were nearly reached by the
  concatenation copy); validation forward passes chunked (the v2 GRU ran out of GPU memory on 262k windows).
- 2026-10-25: phase 1 (B0, P1, J1, R1, C1, CJ, V2; validation only). Phase 2 candidates chosen by the pre-set rule.
- 2026-10-25: phase 2 (J1, CJ × 3 seeds, V2; test sets opened once).
- 2026-10-25: **bug found and fixed after phase 2**: `choose_rules` stored `closed_loop`'s `k` (always 1) over the
  rule's debounce k, so a run whose validation rule was debounce k > 1 was tested at k = 1 with the k-rule's
  threshold. Affected: J1 seed 1 (validation picked debounce 2; test 0.335 at 1.23 false alarms / min) and the R1
  breakdown. Not affected: every validation `S60`, the selection, and all CUSUM / debounce-1 runs. J1 seed 1 was
  re-evaluated with its unchanged validation rule (test 0.278 at 0.39 / min); the self-test now checks the stored k.
- 2026-10-25: the starting-point `S60` gap (v2 0.133 vs V2 0.172) traced to the main pipeline's per-seed debounce
  choice (H5 text corrected after an earlier wrong explanation).
