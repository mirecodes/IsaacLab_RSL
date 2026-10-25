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

## Phase 1 (validation, seed 0)

Pending.

## Phase 2 (test, 3 seeds)

Pending.

## Attempt log

- 2026-10-25: protocol and hypotheses fixed; `explore.py` self-test on synthetic data passes (CUSUM vs plain loop,
  every model kind end to end, `--test` path). `collect.py` now also records `pd/applied_torque` (torque of the
  previous substep) so H4 can run on the same episodes; the queued v2 collection jobs were repointed to that code
  (`euler retarget`), so no extra collection was needed.
