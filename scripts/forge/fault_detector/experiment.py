# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Detector experiment: dataset sanity checks, the training matrix with evaluation, reports and comparisons.

Torch + numpy (+ matplotlib for figures); no Isaac Sim. Settings are ``key=value`` overrides of ``DetectorCfg``::

    python scripts/forge/fault_detector/experiment.py sanity logs/fault_detector/data/main
    python scripts/forge/fault_detector/experiment.py matrix logs/fault_detector/results/v2 rate=pd history_length=5
    python scripts/forge/fault_detector/experiment.py report logs/fault_detector/results/v2
    python scripts/forge/fault_detector/experiment.py compare out_dir v1=<exp> v2=<exp> ...

Every run is compared at matched false alarms: its alarm threshold is set on the validation split so that it
raises at most ``target`` false alarms per minute of nominal walking (``FA_TARGETS``), then scored on the test sets.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector as fd  # noqa: E402

ARCHS, OBS_SETS, SEEDS = ("mlp", "gru"), ("proprio", "proprio_priv"), (0, 1, 2)
FA_TARGETS = (0.25, 0.5, 1.0, 2.0)
"""False alarms per minute the thresholds are set to on the validation split."""
MAIN_TARGET = 0.5


def parse_overrides(pairs: list[str]) -> dict:
    """``key=value`` -> {key: value}, the value parsed as JSON when possible (``epochs=5``, ``extra_terms=["x"]``)."""
    fields = {f.name for f in dataclasses.fields(fd.DetectorCfg)}
    out = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown setting '{key}'")
        with contextlib.suppress(json.JSONDecodeError):
            value = json.loads(value)
        out[key] = tuple(value) if isinstance(value, list) else value
    return out


def joint_legs(joint_names: list[str]) -> np.ndarray:
    """Joint index -> foot index (FL, FR, RL, RR, the foot-contact order)."""
    return np.array([fd.LEGS.index(name[:2]) for name in joint_names])


def joint_kind(name: str) -> str:
    return {"hip": "HAA", "thigh": "HFE", "calf": "KFE"}[name.split("_")[1]]


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(fd.json_safe(rows))


def set_name(dataset_dir: str) -> str:
    """Test-set name of a dataset directory: ``.../test_id_pd`` -> ``test_id``."""
    return os.path.basename(dataset_dir.rstrip("/")).removesuffix("_pd")


def input_columns(data: fd.Episodes, cfg: fd.DetectorCfg) -> list[int]:
    """Columns of ``data`` that belong to the run's inputs."""
    return [i for i, name in enumerate(data.names) if name.split("[")[0] in cfg.inputs]


def subset(data: fd.Episodes, cols: list[int]) -> fd.Episodes:
    """The given input columns; a view when they are a prefix (PROPRIO of PROPRIO + PRIV)."""
    sel = slice(0, len(cols)) if cols == list(range(len(cols))) else cols
    return dataclasses.replace(data, x=data.x[:, sel], noise=data.noise[sel], names=[data.names[i] for i in cols])


# ---------------------------------------------------------------------------------------------------------------------
# matrix: train (or reload) + evaluate every (arch, obs_set, seed)


def evaluate(model, cfg, mean, std, val: fd.Episodes, sets: dict, legs, device) -> tuple[dict, dict]:
    """Thresholds set on ``val`` per false-alarm target, then offline and closed-loop metrics on every test set with
    input noise on and off; also the episode rows at ``MAIN_TARGET`` (noise on)."""
    skip = round(cfg.skip_initial_s / val.sample_dt)
    src = fd.WindowSource(val, cfg, device, mean, std)
    p_any = fd.predict(model, src, True, seed=1)[0].max(axis=1)
    del src
    calib = {target: fd.calibrate(val, p_any, skip, target) for target in FA_TARGETS}
    metrics, rows = {"calibration": [{"target": t, **c} for t, c in calib.items()]}, {}
    for name, data in sets.items():
        src = fd.WindowSource(data, cfg, device, mean, std)
        metrics[name] = {}
        for noise in (True, False):
            prob, type_prob = fd.predict(model, src, noise, seed=1)
            operating = []
            for target, c in calib.items():
                summary, episode_rows = fd.closed_loop(data, prob, legs, skip, c["tau"], c["k"])
                operating.append({"target": target, **summary})
                if noise and target == MAIN_TARGET:
                    rows[name] = episode_rows
            metrics[name]["noise_on" if noise else "noise_off"] = {
                "offline": fd.offline_metrics(data, prob, type_prob, legs, skip),
                "closed_loop": operating,
            }
        del src
    return metrics, rows


def run_one(cfg, run_dir, train_all, val_all, test_all, legs, device, log_file, reevaluate: bool) -> None:
    """Train (or, with ``reevaluate``, reload) one detector, then evaluate it into ``metrics.json`` and
    ``episodes_<set>.csv``."""

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    cols = input_columns(val_all, cfg)
    path = os.path.join(run_dir, "checkpoint.pt")
    if reevaluate:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    else:
        checkpoint = fd.train(cfg, subset(train_all, cols), subset(val_all, cols), device, log)
        checkpoint["dataset_index_md5"] = fd.index_hash(cfg.dataset_dir)
        checkpoint["input_names"] = [val_all.names[i] for i in cols]
        torch.save(checkpoint, path)
    # the checkpoint's own settings (window, skip, noise) drive the evaluation, also for reloaded runs
    model, cfg = fd.load_detector(checkpoint, device)
    sets = {name: subset(data, cols) for name, data in test_all.items()}
    mean, std = checkpoint["mean"], checkpoint["std"]
    metrics, rows = evaluate(model, cfg, mean, std, subset(val_all, cols), sets, legs, device)
    metrics["compute"] = {
        "num_params": checkpoint["num_params"],
        "window": checkpoint.get("window", cfg.history_length),
        "cpu_batch1_ms": fd.inference_time_ms(model, len(cols), checkpoint.get("window", cfg.history_length)),
    }
    metrics["best_val_macro_f1"] = checkpoint["best_val_macro_f1"]
    metrics["epochs_run"] = len(checkpoint["history"])
    metrics["cfg"] = dataclasses.asdict(cfg)
    for name, episode_rows in rows.items():
        write_csv(os.path.join(run_dir, f"episodes_{name}.csv"), episode_rows)
    fd.save_json(os.path.join(run_dir, "metrics.json"), metrics)
    i = FA_TARGETS.index(MAIN_TARGET)
    log(f"[matrix] {run_dir}: " + ", ".join(
        f"{n} success {m['noise_on']['closed_loop'][i]['success_joint_500ms']:.3f} "
        f"(leg {m['noise_on']['closed_loop'][i]['success_leg_500ms']:.3f}) "
        f"lat {m['noise_on']['closed_loop'][i]['median_latency_ms']} ms "
        f"FA/min {m['noise_on']['closed_loop'][i]['fa_per_min']:.2f}"
        for n, m in metrics.items() if isinstance(m, dict) and "noise_on" in m))  # fmt: skip


def cmd_matrix(args) -> None:
    base = fd.DetectorCfg(**parse_overrides(args.overrides))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = fd.read_meta(base.dataset_dir)
    legs = joint_legs(meta["joint_names"])
    split = fd.split_episodes(fd.within_health(fd.read_index(base.dataset_dir), base.max_health), base.split)
    os.makedirs(args.exp_dir, exist_ok=True)
    fd.save_json(os.path.join(args.exp_dir, "split.json"), {k: v.tolist() for k, v in split.items()})
    obs_sets = args.obs_sets or OBS_SETS
    inputs = fd.PROPRIO + fd.PRIV + list(base.extra_terms)
    inputs += ["motor_current"] if "proprio_current" in obs_sets else []
    load = lambda d, ids=None: fd.load_episodes(d, inputs, ids, base.rate)  # noqa: E731
    print(f"[matrix] loading {base.dataset_dir} at the {base.rate} rate "
          f"({', '.join(f'{k} {len(v)}' for k, v in split.items())} episodes)", flush=True)  # fmt: skip
    train_all = None if args.reevaluate else load(base.dataset_dir, split["train"])
    val_all = load(base.dataset_dir, split["val"])
    test_all = {"test": load(base.dataset_dir, split["test"])}
    for d in base.test_dirs:
        if os.path.exists(os.path.join(d, "index.csv")):
            kept = fd.within_health(fd.read_index(d), base.max_health)
            test_all[set_name(d)] = load(d, [r["episode_id"] for r in kept])
    print(f"[matrix] {len(train_all) if train_all is not None else 0} train / {len(val_all)} val samples; test sets: "
          + ", ".join(f"{k} ({len(v.episodes)} episodes)" for k, v in test_all.items()), flush=True)  # fmt: skip
    for obs_set in obs_sets:
        for arch in args.archs or ARCHS:
            for seed in args.seeds or SEEDS:
                cfg = dataclasses.replace(base, arch=arch, obs_set=obs_set, seed=seed)
                run_dir = os.path.join(args.exp_dir, f"{arch}_{obs_set}_s{seed}")
                done = os.path.exists(os.path.join(run_dir, "metrics.json"))
                if args.reevaluate and not os.path.exists(os.path.join(run_dir, "checkpoint.pt")):
                    continue
                if done and not args.reevaluate:
                    print(f"[matrix] {run_dir} done, skipped")
                    continue
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "eval.log" if args.reevaluate else "train.log"), "a") as log_file:
                    run_one(cfg, run_dir, train_all, val_all, test_all, legs, device, log_file, args.reevaluate)


# ---------------------------------------------------------------------------------------------------------------------
# sanity: injector and dataset checks


def motor_current_check(shard, by_id: dict, J: int, m: int) -> dict:
    """PD-rate check of the injector's current reading: the applied torque is ``h`` x the reading on a ``kt_loss``
    joint and equal to it everywhere else (both of the previous substep, so sample ``n`` uses the health label of
    observation ``(n - 1) // m + 1``, the step whose physics ran that substep)."""
    applied = shard["pd/applied_torque"].astype(np.float32)
    current = shard["pd/motor_current"].astype(np.float32)
    labels = shard["fault/labels"]
    err, faulty_ratio = [], []
    start = 0
    for e, n in enumerate(shard["ep_len"]):
        kt = int(by_id[int(shard["episode_id"][e])]["fault_type"]) == fd.FAULT_TYPES.index("kt_loss")
        health = labels[start : start + n, J + 1 : 2 * J + 1]
        s = np.arange(1, n * m)
        s = s[(s - 1) // m + 1 < n]
        h = health[(s - 1) // m + 1] if kt else np.ones((len(s), J))
        a, c = applied[start * m + s], current[start * m + s]
        err.append(np.abs(a - h * c) / (1.0 + np.abs(c)))
        big = (h < 1) & (np.abs(c) > 5.0)
        faulty_ratio += list(zip(h[big], a[big] / c[big]))
        start += n
    err = np.concatenate(err)
    out = {"samples": int(err.size), "max_rel_err_applied_vs_h_current": float(err.max()),
           "note": "|tau - h c| / (1 + |c|), float16 storage"}  # fmt: skip
    if faulty_ratio:
        r = np.array(faulty_ratio)
        out["kt_loss_tau_over_current_vs_h_max_abs_diff"] = float(np.abs(r[:, 1] - r[:, 0]).max())
    return out


def cmd_sanity(args) -> None:
    """Label, schedule, torque and PD-rate alignment checks of one dataset, plus onset plots (``sanity/``)."""
    meta = fd.read_meta(args.dataset_dir)
    names, J = meta["joint_names"], len(meta["joint_names"])
    cfg = meta["fault_cfg"]
    out = os.path.join(args.dataset_dir, "sanity")
    os.makedirs(out, exist_ok=True)
    index = fd.read_index(args.dataset_dir)
    report = {"episodes": len(index), "steps": int(sum(int(r["length"]) for r in index))}
    faulty = [r for r in index if int(r["has_fault"])]
    report["fault_fraction"] = len(faulty) / len(index)
    types = range(1, len(fd.FAULT_TYPES))
    report["fault_type_counts"] = {fd.FAULT_TYPES[t]: sum(int(r["fault_type"]) == t for r in faulty) for t in types}
    report["fault_joint_counts"] = {names[j]: sum(int(r["fault_joint"]) == j for r in faulty) for j in range(J)}
    onset_s = np.array([int(r["onset_step"]) * meta["step_dt"] for r in faulty])
    report["onset_s"] = {"min": onset_s.min(), "max": onset_s.max(), "mean": onset_s.mean(), "cfg": cfg["onset_time_s"]}
    for t, key in ((2, "power_loss_health"), (3, "kt_loss_health")):
        h = np.array([float(r["health"]) for r in faulty if int(r["fault_type"]) == t])
        if len(h):
            report[f"{fd.FAULT_TYPES[t]}_h"] = {"min": h.min(), "max": h.max(), "mean": h.mean(), "cfg": cfg.get(key)}
    reasons: dict[str, dict[str, int]] = {}
    for r in index:
        kind = fd.FAULT_TYPES[int(r["fault_type"])]
        reasons.setdefault(kind, {}).setdefault(r["termination_reason"], 0)
        reasons[kind][r["termination_reason"]] += 1
    report["termination_by_type"] = reasons

    # per-step checks on the first shard
    shard_dir = os.path.join(args.dataset_dir, "shards")
    shard = np.load(os.path.join(shard_dir, sorted(os.listdir(shard_dir))[0]))
    ep_len, ep_ids = shard["ep_len"], shard["episode_id"]
    starts = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    labels, torque = shard["fault/labels"], shard["analysis/applied_torque"]
    target, pos, vel = shard["analysis/joint_pos_target"], shard["detector/joint_pos_rel"], shard["detector/joint_vel"]
    by_id = {int(r["episode_id"]): r for r in index}
    bad_labels, out_torque, ratios, examples = 0, [], [], {}
    kp = np.array([150.0 if "calf" in n else 100.0 for n in names])
    kd = np.array([6.0 if "calf" in n else 4.0 for n in names])
    limit = np.array([180.0 if "calf" in n else 120.0 for n in names])
    for e, (lo, n) in enumerate(zip(starts, ep_len)):
        row = by_id[int(ep_ids[e])]
        mask = labels[lo : lo + n, :J] > 0.5
        onset, joint = int(row["onset_step"]), int(row["fault_joint"])
        expected = np.zeros_like(mask)
        if int(row["has_fault"]):
            expected[onset:, joint] = True
        bad_labels += int((mask != expected).any())
        if not int(row["has_fault"]) or n - onset < 5:
            continue
        tau = torque[lo + onset : lo + n, joint]
        if int(row["fault_type"]) == 1:
            out_torque.append(np.abs(tau).max())
        else:
            ref = np.clip(kp[joint] * (target[lo + onset : lo + n, joint] - pos[lo + onset : lo + n, joint])
                          - kd[joint] * vel[lo + onset : lo + n, joint], -limit[joint], limit[joint])  # fmt: skip
            big = np.abs(ref) > 5.0
            if big.sum() > 5:
                ratios.append((float(row["health"]), float(np.median(tau[big] / ref[big]))))
        kind = fd.FAULT_TYPES[int(row["fault_type"])]
        if kind not in examples and onset >= 50 and n - onset >= 50:
            examples[kind] = (lo, onset, joint, row)
    report["label_mismatch_episodes"] = bad_labels
    report["power_out_max_abs_torque_after_onset"] = float(np.max(out_torque)) if out_torque else None
    if ratios:
        r = np.array(ratios)
        report["power_loss_torque_ratio"] = {
            "episodes": len(r),
            "corr_h_vs_measured_ratio": float(np.corrcoef(r[:, 0], r[:, 1])[0, 1]),
            "median_abs_error": float(np.median(np.abs(r[:, 1] - r[:, 0]))),
            "note": "applied torque / PD torque estimated from the 50 Hz sample (the actuator acts at 200 Hz)",
        }
    if "pd/joint_pos_rel" in shard.files:
        # sample m t of the PD stream is the state of observation o_t (float16 storage)
        m = int(meta["decimation"])
        pd_pos, pd_vel = shard["pd/joint_pos_rel"].astype(np.float32), shard["pd/joint_vel"].astype(np.float32)
        report["pd_alignment"] = {
            "samples": len(pd_pos),
            "expected_samples": int(ep_len.sum() * m),
            "max_abs_joint_pos_diff_at_m_t": float(np.abs(pd_pos[::m] - pos).max()),
            "max_abs_joint_vel_diff_at_m_t": float(np.abs(pd_vel[::m] - vel).max()),
            "median_abs_joint_pos_step_within_policy_step": float(np.median(np.abs(np.diff(pd_pos, axis=0)))),
        }
    if "pd/motor_current" in shard.files:
        report["motor_current"] = motor_current_check(shard, by_id, J, int(meta["decimation"]))
    fd.save_json(os.path.join(out, "sanity.json"), report)
    print(json.dumps(fd.json_safe(report), indent=2))

    plt = _pyplot()
    if plt is None or not examples:
        return
    fig, axes = plt.subplots(3, len(examples), figsize=(5 * len(examples), 7), squeeze=False)
    for c, (kind, (lo, onset, joint, row)) in enumerate(sorted(examples.items())):
        sl = slice(lo + onset - 50, lo + onset + 50)
        t = (np.arange(-50, 50)) * meta["step_dt"]
        series = [("joint pos rel [rad]", pos[sl, joint]), ("joint vel [rad/s]", vel[sl, joint]),
                  ("tracking error q*-q [rad]", target[sl, joint] - pos[sl, joint])]  # fmt: skip
        for r_, (label, value) in enumerate(series):
            ax = axes[r_, c]
            ax.plot(t, value)
            ax.axvline(0, color="k", ls="--", lw=0.8)
            ax.set_ylabel(label)
        axes[0, c].set_title(f"{kind}: {names[joint]} (h = {float(row['health']):.2f})")
        axes[-1, c].set_xlabel("time since onset [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "onset_examples.png"), dpi=120)


# ---------------------------------------------------------------------------------------------------------------------
# report and compare


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print("[report] matplotlib not installed: figures skipped")
        return None


def num(value) -> float:
    """JSON value -> float (None, written for NaN, back to NaN)."""
    return np.nan if value is None else float(value)


def mean_std(values, digits: int = 3) -> str:
    v = np.array([num(x) for x in values], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "–"
    return f"{v.mean():.{digits}f} ± {v.std():.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def load_runs(exp: str) -> dict[tuple[str, str], dict[int, dict]]:
    """(arch, obs_set) -> {seed: metrics} of an experiment directory."""
    runs: dict[tuple[str, str], dict[int, dict]] = {}
    for name in sorted(os.listdir(exp)):
        path = os.path.join(exp, name, "metrics.json")
        if os.path.exists(path):
            arch, rest = name.split("_", 1)
            obs_set, seed = rest.rsplit("_s", 1)
            with open(path) as f:
                runs.setdefault((arch, obs_set), {})[int(seed)] = json.load(f)
    return runs


def run_configs(runs: dict) -> list[tuple[str, str]]:
    """The (arch, obs_set) pairs of ``runs``: ``OBS_SETS`` first, then any other obs set."""
    obs_sets = list(OBS_SETS) + sorted({o for _, o in runs} - set(OBS_SETS))
    return [(a, o) for o in obs_sets for a in ARCHS if (a, o) in runs]


def op(m: dict, set_: str = "test_id", noise: str = "noise_on", target: float = MAIN_TARGET) -> dict:
    """Closed-loop summary of one run at a false-alarm target."""
    return next(x for x in m[set_][noise]["closed_loop"] if x["target"] == target)


def off(m: dict, set_: str = "test_id", noise: str = "noise_on") -> dict:
    return m[set_][noise]["offline"]


def leg_confusion(m: dict, set_: str = "test_id") -> np.ndarray:
    return np.array(off(m, set_)["leg_confusion"], dtype=float)


MAIN_COLUMNS = [
    ("Success, joint ≤ 500 ms", lambda m: op(m)["success_joint_500ms"], 3),
    ("Success, leg ≤ 500 ms", lambda m: op(m)["success_leg_500ms"], 3),
    ("Median latency [ms]", lambda m: op(m)["median_latency_ms"], 0),
    ("Detect ≤ 200 ms", lambda m: op(m)["detect_200ms"], 3),
    ("Miss", lambda m: op(m)["miss_rate"], 3),
    ("FA / min (test)", lambda m: op(m)["fa_per_min"], 2),
    ("Correct joint / leg at alarm", lambda m: op(m)["correct_joint"], 3),
    ("Top-1 joint (windows)", lambda m: off(m)["top1_joint_acc"], 3),
    ("Leg acc (windows)", lambda m: off(m)["leg_acc"], 3),
    ("Same-leg share of joint errors", lambda m: off(m)["same_leg_error_share"], 3),
    ("Type acc", lambda m: off(m)["type_acc"], 3),
    ("ECE", lambda m: off(m)["ece"], 4),
]


def main_table(rows: list[tuple[str, list[dict]]]) -> list[str]:
    """Markdown table of the main metrics, one row per (label, runs of the seeds)."""
    head = "| Config | " + " | ".join(c for c, _, _ in MAIN_COLUMNS) + " | Params | CPU ms |"
    lines = [head, "|" + "---|" * (len(MAIN_COLUMNS) + 3)]
    for label, ms in rows:
        cells = [mean_std([f(m) for m in ms], d) for _, f, d in MAIN_COLUMNS]
        joint_ok, leg_ok = [op(m)["correct_joint"] for m in ms], [op(m)["correct_leg"] for m in ms]
        cells[6] = f"{mean_std(joint_ok)} / {mean_std(leg_ok)}"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {ms[0]['compute']['num_params']} | "
                     f"{mean_std([m['compute']['cpu_batch1_ms'] for m in ms], 3)} |")  # fmt: skip
    return lines


def axis_value(row: dict, axis: str, names: list[str]) -> str | None:
    """Bin of one faulty-episode row (``episodes_<set>.csv``) along a Q3 analysis axis; None = not applicable."""
    joint = int(row["fault_joint"])
    if axis == "fault_type":
        return row["fault_type"]
    if axis == "joint_kind":
        return joint_kind(names[joint])
    if axis == "front_hind":
        return "front" if names[joint][0] == "F" else "hind"
    if axis == "health_h":
        lo = np.floor(float(row["health"]) * 10) / 10
        return f"h in [{lo:.1f}, {lo + 0.1:.1f})" if row["fault_type"] in ("power_loss", "kt_loss") else None
    if axis == "gait_phase":
        return "stance" if row["stance_at_onset"] == "True" else "swing"
    if axis == "terrain_type":
        return row["terrain_type"]
    if axis == "joint":
        return names[joint]
    if axis == "leg":
        return names[joint][:2]
    return None


def pooled_rows(exp: str, config: tuple[str, str], seeds, set_: str = "test_id") -> list[dict]:
    rows = []
    for seed in seeds:
        with open(os.path.join(exp, f"{config[0]}_{config[1]}_s{seed}", f"episodes_{set_}.csv")) as f:
            rows += list(csv.DictReader(f))
    return rows


def latency_values(rows: list[dict]) -> np.ndarray:
    return np.array([float(r["latency_ms"]) if r["latency_ms"] not in ("", "None") else np.inf for r in rows])


def cmd_report(args) -> None:
    exp = args.exp_dir
    runs = load_runs(exp)
    configs = run_configs(runs)
    names = fd.read_meta(args.dataset_dir)["joint_names"]
    sets = [k for k in next(iter(runs[configs[0]].values())) if k.startswith("test")]
    label = {c: f"{c[0].upper()} · {c[1]}" for c in configs}
    cfg0 = next(iter(runs[configs[0]].values()))["cfg"]
    lines = ["# Motor-failure detector results", "",
             f"Rate {cfg0['rate']}, history {cfg0['history_length']} policy steps, label grace "
             f"{cfg0['ignore_after_onset_s']} s, hard-negative epochs {cfg0['hnm_epochs']}. Mean ± std over seeds; "
             f"test_id with input noise; thresholds set on the validation split to {MAIN_TARGET} false alarms / min.",
             ""]  # fmt: skip
    lines += main_table([(label[c], list(runs[c].values())) for c in configs])

    lines += ["", "## Matched false-alarm trade-off (test_id: success joint ≤ 500 ms / median latency ms / FA per min)",
              "", "| Config | " + " | ".join(f"target {t} / min" for t in FA_TARGETS) + " |",
              "|---|" + "---|" * len(FA_TARGETS)]  # fmt: skip
    for c in configs:
        ms = list(runs[c].values())
        cells = [f"{mean_std([op(m, target=t)['success_joint_500ms'] for m in ms])} / "
                 f"{mean_std([op(m, target=t)['median_latency_ms'] for m in ms], 0)} / "
                 f"{mean_std([op(m, target=t)['fa_per_min'] for m in ms], 2)}" for t in FA_TARGETS]  # fmt: skip
        lines.append(f"| {label[c]} | " + " | ".join(cells) + " |")

    lines += ["", f"## Robustness at {MAIN_TARGET} / min (success joint ≤ 500 ms / median latency ms / FA per min)", "",
              "| Config | " + " | ".join(f"{s} {n}" for s in sets for n in ("noise on", "noise off")) + " |",
              "|---|" + "---|" * (2 * len(sets))]  # fmt: skip
    for c in configs:
        ms = list(runs[c].values())
        cells = [f"{mean_std([op(m, s, n)['success_joint_500ms'] for m in ms])} / "
                 f"{mean_std([op(m, s, n)['median_latency_ms'] for m in ms], 0)} / "
                 f"{mean_std([op(m, s, n)['fa_per_min'] for m in ms], 2)}"
                 for s in sets for n in ("noise_on", "noise_off")]  # fmt: skip
        lines.append(f"| {label[c]} | " + " | ".join(cells) + " |")

    analysis_rows = []
    for axis in ("fault_type", "joint_kind", "front_hind", "health_h", "gait_phase", "terrain_type", "joint", "leg"):
        lines += ["", f"## By {axis} (median latency ms / success joint ≤ 500 ms / success leg ≤ 500 ms; seeds pooled)",
                  ""]  # fmt: skip
        pooled = {c: pooled_rows(exp, c, runs[c]) for c in configs}
        values = sorted({axis_value(r, axis, names) for c in configs for r in pooled[c]} - {None})
        lines += ["| Config | " + " | ".join(values) + " |", "|---|" + "---|" * len(values)]
        for c in configs:
            cells = []
            for v in values:
                sel = [r for r in pooled[c] if axis_value(r, axis, names) == v]
                lat = latency_values(sel)
                ok = lat <= 500
                med = np.median(lat[np.isfinite(lat)]) if np.isfinite(lat).any() else np.nan
                sj = np.mean(ok & np.array([r["correct_joint"] == "True" for r in sel])) if sel else np.nan
                sl = np.mean(ok & np.array([r["correct_leg"] == "True" for r in sel])) if sel else np.nan
                cells.append(f"{med:.0f} / {sj:.2f} / {sl:.2f} (n={len(sel)})")
                analysis_rows.append({"arch": c[0], "obs_set": c[1], "axis": axis, "value": v, "n": len(sel),
                                      "median_latency_ms": med, "success_joint": sj, "success_leg": sl})  # fmt: skip
            lines.append(f"| {label[c]} | " + " | ".join(cells) + " |")
    write_csv(os.path.join(exp, "analysis.csv"), analysis_rows)

    table_rows = []
    for c in configs:
        for seed, m in runs[c].items():
            table_rows.append({"arch": c[0], "obs_set": c[1], "seed": seed,
                               **{f"offline_{k}": v for k, v in off(m).items() if not isinstance(v, list)},
                               **{f"cl_{k}": v for k, v in op(m).items()},
                               **m["compute"], "best_val_macro_f1": m["best_val_macro_f1"]})  # fmt: skip
    write_csv(os.path.join(exp, "results.csv"), table_rows)
    with open(os.path.join(exp, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))
    write_figures(exp, configs, runs, label, names)


def write_figures(exp, configs, runs, label, names) -> None:
    """The report's figures (``figures/*.png``)."""
    plt = _pyplot()
    if plt is None:
        return
    fig_dir = os.path.join(exp, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    # success and latency at matched false alarms
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c in configs:
        ms = list(runs[c].values())
        fa = [np.mean([op(m, target=t)["fa_per_min"] for m in ms]) for t in FA_TARGETS]
        axes[0].plot(fa, [np.nanmean([num(op(m, target=t)["success_joint_500ms"]) for m in ms]) for t in FA_TARGETS],
                     "o-", label=label[c])  # fmt: skip
        axes[1].plot(fa, [np.nanmean([num(op(m, target=t)["median_latency_ms"]) for m in ms]) for t in FA_TARGETS],
                     "o-", label=label[c])  # fmt: skip
    for ax, ylabel in zip(axes, ("success: right joint within 500 ms", "median latency [ms]")):
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per minute on test_id (thresholds set on val)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "matched_false_alarm.png"), dpi=130)
    # confusion matrices, joint and leg (summed over seeds)
    confusion_figure(plt, [(label[c], list(runs[c].values())) for c in configs], names,
                     os.path.join(fig_dir, "confusion.png"))  # fmt: skip
    # accuracy vs time since onset
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in configs:
        ms = list(runs[c].values())
        bins = [b["bin"] for b in off(ms[0])["since_onset_top1"]]
        acc = np.nanmean([[num(b["top1"]) for b in off(m)["since_onset_top1"]] for m in ms], axis=0)
        ax.plot(bins, acc, "o-", label=label[c])
    ax.set_xlabel("time since onset")
    ax.set_ylabel("top-1 joint accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "accuracy_since_onset.png"), dpi=130)
    # reliability diagram
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for c in configs:
        rel = off(next(iter(runs[c].values())))["reliability"]
        pts = [(b["confidence"], b["accuracy"]) for b in rel if b["count"] > 0]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=label[c], ms=3)
    ax.set_xlabel("predicted probability p_j")
    ax.set_ylabel("observed fault frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "reliability.png"), dpi=130)
    print(f"[report] {exp}/report.md, figures in {fig_dir}")


def confusion_figure(plt, entries: list[tuple[str, list[dict]]], names: list[str], path: str) -> None:
    """Row-normalized 12 x 12 joint and 4 x 4 leg confusion (summed over seeds), one column per entry."""
    fig, axes = plt.subplots(2, len(entries), figsize=(4.2 * len(entries), 8), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1.4]})  # fmt: skip
    ticks = [n.replace("_joint", "") for n in names]
    for col, (title, ms) in enumerate(entries):
        for row, (cm, tk) in enumerate(((np.sum([np.array(off(m)["confusion"], float) for m in ms], axis=0), ticks),
                                        (np.sum([leg_confusion(m) for m in ms], axis=0), list(fd.LEGS)))):  # fmt: skip
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax = axes[row, col]
            ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            ax.set_xticks(range(len(tk)), tk, rotation=90, fontsize=6 if row == 0 else 8)
            ax.set_yticks(range(len(tk)), tk, fontsize=6 if row == 0 else 8)
            if row == 1:
                for i in range(len(tk)):
                    ax.text(i, i, f"{cm[i, i]:.2f}", ha="center", va="center", fontsize=8, color="white")
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
        acc = np.mean([off(m)["top1_joint_acc"] for m in ms])
        leg = np.mean([off(m)["leg_acc"] for m in ms])
        axes[0, col].set_title(f"{title}\njoint {acc:.3f}, leg {leg:.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def cmd_compare(args) -> None:
    """One table (and confusion / trade-off figures) across experiments: ``label=exp_dir`` pairs, every config."""
    exps = [pair.split("=", 1) for pair in args.experiments]
    names = fd.read_meta(args.dataset_dir)["joint_names"]
    entries = []
    for label, exp in exps:
        runs = load_runs(exp)
        for c in run_configs(runs):
            entries.append((f"{label}: {c[0].upper()} · {c[1]}", list(runs[c].values()), label, c))
    lines = ["# Comparison at matched false alarms", "",
             f"test_id with input noise; thresholds set on each run's validation split to {MAIN_TARGET} false alarms "
             "/ min. Mean ± std over seeds.", ""]  # fmt: skip
    lines += main_table([(e[0], e[1]) for e in entries])
    lines += ["", "## test_ood (weak power loss) at the same thresholds", "",
              "| Config | Success joint ≤ 500 ms | Success leg ≤ 500 ms | Median latency [ms] | Miss | FA / min |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for title, ms, _, _ in entries:
        if all("test_ood" in m for m in ms):
            o = [op(m, "test_ood") for m in ms]
            lines.append(f"| {title} | {mean_std([x['success_joint_500ms'] for x in o])} | "
                         f"{mean_std([x['success_leg_500ms'] for x in o])} | "
                         f"{mean_std([x['median_latency_ms'] for x in o], 0)} | "
                         f"{mean_std([x['miss_rate'] for x in o])} | "
                         f"{mean_std([x['fa_per_min'] for x in o], 2)} |")  # fmt: skip
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "compare.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plt = _pyplot()
    if plt is None:
        return
    focus = [e for e in entries if e[3] == tuple(args.focus.split(","))]
    confusion_figure(plt, [(e[0], e[1]) for e in focus], names, os.path.join(args.out_dir, "confusion_compare.png"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for title, ms, _, _ in focus:
        fa = [np.mean([op(m, target=t)["fa_per_min"] for m in ms]) for t in FA_TARGETS]
        for ax, key in zip(axes, ("success_joint_500ms", "success_leg_500ms")):
            ax.plot(fa, [np.nanmean([num(op(m, target=t)[key]) for m in ms]) for t in FA_TARGETS], "o-", label=title)
    for ax, ylabel in zip(axes, ("success: right joint within 500 ms", "success: right leg within 500 ms")):
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per minute on test_id (thresholds set on val)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "success_compare.png"), dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("matrix", help="Train (or reevaluate) and evaluate every (arch, obs_set, seed).")
    p.add_argument("exp_dir")
    p.add_argument("overrides", nargs="*", help="key=value settings of DetectorCfg")
    p.add_argument("--archs", nargs="+", default=None)
    p.add_argument("--obs_sets", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--reevaluate", action="store_true", help="Reload each run's checkpoint instead of training.")
    p.set_defaults(func=cmd_matrix)
    p = sub.add_parser("sanity", help="Injector and dataset checks of one dataset.")
    p.add_argument("dataset_dir")
    p.set_defaults(func=cmd_sanity)
    p = sub.add_parser("report", help="Tables and figures of a matrix.")
    p.add_argument("exp_dir")
    p.add_argument("--dataset_dir", default="logs/fault_detector/data/main", help="For joint names.")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("compare", help="Main metrics and confusion of several experiments side by side.")
    p.add_argument("out_dir")
    p.add_argument("experiments", nargs="+", help="label=exp_dir pairs")
    p.add_argument("--focus", default="mlp,proprio", help="arch,obs_set drawn in the comparison figures")
    p.add_argument("--dataset_dir", default="logs/fault_detector/data/main", help="For joint names.")
    p.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
