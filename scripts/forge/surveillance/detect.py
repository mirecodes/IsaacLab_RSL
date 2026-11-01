# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strategies to detect falls and their failure modes early, compared at the same false-alarm budget.

Offline (torch, numpy, scikit-learn). Each strategy turns 0.5 s of signals into one danger score per frame:

* ``hazard``: MLP on proprioception (:data:`classify.PROPRIO`), trained on "the balance loss is less than ``H`` s
  ahead" (fall windows) vs normal walking. Supervised by the simulation, no human label.
* ``hazard_ev``: the same with the event windows without a fall as extra negatives (harmless slips, sinks, collapses,
  stones), to tell a harmless event from one that ends in a fall.
* ``hazard_imu``: the same as ``hazard`` plus the body-frame specific force an IMU measures (``root_acc_w`` and
  gravity, rotated).
* ``mode_hazard``: one MLP with a class per failure mode (``--modes``: a column of ``sequences.py``'s ``falls.csv``)
  plus normal; score = 1 - P(normal), and the arg-max mode names the coming fall (competing risks).
* ``forward_error``: MLP trained on normal walking only, predicting the next frame of joint positions and
  velocities, gravity and base rate from 0.5 s of history (with the PD targets); score = its standardized squared error
  (an anomaly detector: no fall data, no labels).
* ``tilt``: the base tilt angle (a rule on the IMU attitude).
* ``value``: minus the critic's value (privileged: the critic sees the full state; an upper reference).

Thresholds: per strategy, the score that keeps false alarms on held-out normal walking (``random_walk`` windows) at
``--fa_per_min`` (an alarm: the score above it for 3 frames; at most one per 1 s). On held-out falls: the first alarm
in ``[-3 s, +0.25 s]`` of the balance loss; lead time = balance loss - alarm. Also the alarm rate on the event
windows without a fall (harmless slips, sand, collapses: a detector that fires on every event is noisy). Per mode
(``--modes``): recall with a lead of at least 0.25 s. ``signatures``: per mode, which proprioceptive channel group
first leaves its normal range (mean |z| > 2 over the mode's falls) and when::

    uv run --no-project --with torch --with numpy --with scikit-learn python scripts/forge/surveillance/detect.py \\
        --falls results/sequences_v3/falls.csv --normal $R/*/{rough_field,...} --events $R/*/{slip_field,...} \\
        --out results/detect_v3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze  # noqa: E402
from classify import PROPRIO  # noqa: E402

DT, HIST, HORIZON_S = 0.02, 25, 1.0
GROUPS = {
    "joint_pos": slice(0, 12), "joint_vel": slice(12, 24), "joint_target": slice(24, 36), "gravity": slice(36, 39),
    "ang_rate": slice(39, 42), "command": slice(42, 45), "imu_acc": slice(45, 48),
}  # fmt: skip
"""Channel groups of :func:`signals`' ``x``."""


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vectors ``v`` [T, 3] into the body frame of ``q`` [T, 4] (x, y, z, w)."""
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def signals(e: dict) -> dict[str, np.ndarray]:
    """Per-frame inputs of one window: proprioception + IMU specific force [T, 48], tilt [T], value [T]."""
    prop = np.concatenate([e[f].reshape(len(e[f]), -1) for f in PROPRIO], 1)
    acc = quat_rotate_inverse(e["root_quat_w"], e["root_acc_w"][:, :3] - np.array([0.0, 0.0, -9.81]))
    tilt = np.degrees(np.arccos(np.clip(-e["projected_gravity_b"][:, 2], -1, 1)))
    return {"x": np.concatenate([prop, acc], 1).astype(np.float32), "tilt": tilt.astype(np.float32),
            "value": e["value"].astype(np.float32)}  # fmt: skip


def load(falls_csv: str, normal: list[str], events: list[str], modes: str):
    """Fall windows (with their balance loss and mode), normal walking windows, and event windows without a
    fall."""
    out = {"falls": [], "normal": [], "events": []}
    with open(falls_csv) as f:
        for r in csv.DictReader(f):
            e = analyze.load_event(r["collection"], {"event": r["event"], "num_frames": 0, "episode_step": 0})
            s = signals(e)
            s.update(ref=int(r["onset"]), mode=r[modes], policy=r["policy"], terrain=r["terrain"])
            out["falls"].append(s)
    for kind, paths in (("normal", normal), ("events", events)):
        for path in paths:
            root = os.path.join(path, "triggers")
            if not os.path.isfile(os.path.join(root, "index.csv")):
                continue
            _, index = analyze.load_collection(root)
            for r in index:
                e = analyze.load_event(root, r)
                if np.linalg.norm(e["contact_normal_force_w"][:, 0], axis=-1).max() > 1.0:
                    continue  # the robot falls inside it
                s = signals(e)
                s.update(policy=path.rstrip("/").split("/")[-2], terrain=path.rstrip("/").split("/")[-1])
                out[kind].append(s)
    return out


def split(n: int, seed: int, frac: float = 0.3) -> np.ndarray:
    """Boolean test mask over ``n`` windows."""
    rng = np.random.default_rng(seed)
    test = np.zeros(n, bool)
    test[rng.permutation(n)[: round(frac * n)]] = True
    return test


def windows(x: np.ndarray, frames) -> np.ndarray:
    return np.stack([x[t - HIST + 1 : t + 1].reshape(-1) for t in frames])


def mlp(din: int, dout: int):
    from torch import nn

    return nn.Sequential(nn.Linear(din, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, dout))


def fit(x, y, dout: int, loss: str, epochs: int, device: str, seed: int):
    """Train an MLP; returns ``predict(x) -> outputs`` (standardized inputs, input noise as in classify.py)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = torch.tensor(x)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-3)
    model = mlp(x.shape[1], dout).to(device)
    y = torch.tensor(y).to(device)
    if loss == "ce":
        weight = 1.0 / torch.bincount(y, minlength=dout).clamp(min=1).float()
        loss_fn = nn.CrossEntropyLoss(weight=(weight / weight.sum() * dout))
    else:
        loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xn = ((x - mean) / std).to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(xn), device=device).split(2048):
            loss = loss_fn(model(xn[idx] + 0.05 * torch.randn_like(xn[idx])), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    def predict(xq: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = model(((torch.tensor(xq) - mean) / std).to(device))
            return (torch.softmax(out, 1) if loss == "ce" else out).cpu().numpy()

    return predict


def alarms(score: np.ndarray, thr: float, sustain: int = 3, refractory: int = 50) -> list[int]:
    """Frames where ``score`` has been above ``thr`` for ``sustain`` frames, at most one per ``refractory``."""
    out, run, last = [], 0, -refractory
    for t, s in enumerate(score):
        run = run + 1 if s > thr else 0
        if run >= sustain and t - last >= refractory:
            out.append(t)
            last = t
    return out


def threshold(normal_scores: list[np.ndarray], fa_per_min: float) -> float:
    """Lowest threshold (on a quantile grid) whose alarm rate over the normal windows is at most ``fa_per_min``."""
    minutes = sum(len(s) for s in normal_scores) * DT / 60
    flat = np.concatenate(normal_scores)
    for q in np.linspace(0.9, 1.0, 201):
        thr = float(np.quantile(flat, q)) if q < 1 else float(flat.max()) + 1e-6
        if sum(len(alarms(s, thr)) for s in normal_scores) / minutes <= fa_per_min:
            return thr
    return float(flat.max()) + 1e-6


def evaluate(score_fn, data, test: dict, fa_per_min: float) -> dict:
    """Threshold on the test normal windows, then lead times on the test falls, alarm rate on event windows."""
    normal = [score_fn(w) for w, t in zip(data["normal"], test["normal"]) if t]
    thr = threshold(normal, fa_per_min)
    leads, per_mode = [], {}
    for w, t in zip(data["falls"], test["falls"]):
        if not t:
            continue
        s = score_fn(w)
        lo, hi = max(w["ref"] - round(3 / DT), HIST - 1), min(w["ref"] + round(0.25 / DT), len(s) - 1)
        hits = [a for a in alarms(s[lo : hi + 1], thr)]
        lead = (w["ref"] - (lo + hits[0])) * DT if hits else np.nan
        leads.append(lead)
        per_mode.setdefault(w["mode"], []).append(lead)
    leads = np.array(leads)
    ev = [score_fn(w) for w, t in zip(data["events"], test["events"]) if t]
    ev_min = sum(len(s) for s in ev) * DT / 60
    return {
        "threshold": round(thr, 4),
        "recall_any": round(float(np.isfinite(leads).mean()), 3),
        "recall_lead_0.25s": round(float((leads >= 0.25).mean()), 3),
        "recall_lead_0.5s": round(float((leads >= 0.5).mean()), 3),
        "median_lead_s": round(float(np.nanmedian(leads)), 3) if np.isfinite(leads).any() else None,
        "event_alarms_per_min": round(sum(len(alarms(s, thr)) for s in ev) / max(ev_min, 1e-9), 2),
        "per_mode_recall_lead_0.25s": {
            m: [round(float((np.array(v) >= 0.25).mean()), 2), len(v)] for m, v in sorted(per_mode.items())
        },  # fmt: skip
    }


def frames_of(w: dict, kind: str):
    """Training frames of a window: (frame, label) with label 1 = balance loss within HORIZON_S, 0 = normal; fall
    windows contribute only their last HORIZON_S before the balance loss (+0.25 s) as positives."""
    if kind == "fall":
        lo = max(w["ref"] - round(HORIZON_S / DT), HIST - 1)
        return range(lo, min(w["ref"] + round(0.25 / DT), len(w["x"]) - 1) + 1)
    return range(HIST - 1, len(w["x"]), 3)


def signatures(data: dict, stats: dict) -> dict:
    """Per mode: |z| (vs the policy's normal walking, ``stats[policy] = (mean, std)``) averaged over the mode's falls,
    from -2 s to +0.25 s of the balance loss; per channel group of :data:`GROUPS` its most deviating channel, the
    first time that exceeds 2 and its value at -0.5 s and 0 s."""
    offsets = np.arange(-round(2 / DT), round(0.25 / DT) + 1)
    out = {}
    for mode in sorted({w["mode"] for w in data["falls"]}):
        z = []
        for w in data["falls"]:
            if w["mode"] == mode:
                mu, sd = stats[w["policy"]]
                z.append(np.abs((w["x"][np.clip(w["ref"] + offsets, 0, len(w["x"]) - 1)] - mu) / sd))
        z = np.mean(z, 0)  # T, C
        entry = {"n": int(sum(w["mode"] == mode for w in data["falls"]))}
        for g, sl in GROUPS.items():
            curve = z[:, sl].max(1)
            above = np.flatnonzero(curve > 2.0)
            entry[g] = {"first_above_2_s": round(float(offsets[above[0]] * DT), 2) if len(above) else None,
                        "at_-0.5s": round(float(curve[offsets == -25][0]), 2),
                        "at_0s": round(float(curve[offsets == 0][0]), 2)}  # fmt: skip
        out[mode] = entry
    return out


def main() -> None:
    global HIST
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--falls", required=True, help="falls.csv of sequences.py (balance loss, modes).")
    parser.add_argument("--modes", default="rules", help="Column of falls.csv with the failure mode.")
    parser.add_argument("--normal", nargs="+", required=True)
    parser.add_argument("--events", nargs="+", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--fa_per_min", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=3000, help="Normal / event windows kept (random subset).")
    parser.add_argument("--history_s", type=float, default=HIST * DT, help="Input history of the learned detectors.")
    args = parser.parse_args()
    HIST = round(args.history_s / DT)
    import pickle

    import torch

    os.makedirs(args.out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = os.path.join(args.out, f"loaded_{args.modes}.pkl")
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            data = pickle.load(f)
    else:
        data = load(args.falls, args.normal, args.events, args.modes)
        with open(cache, "wb") as f:
            pickle.dump(data, f)
    rng = np.random.default_rng(args.seed)
    for kind in ("normal", "events"):  # memory: the training frames of every window are held at once
        if len(data[kind]) > args.max_windows:
            data[kind] = [data[kind][i] for i in sorted(rng.choice(len(data[kind]), args.max_windows, replace=False))]
    test = {k: split(len(v), args.seed + i) for i, (k, v) in enumerate(data.items())}
    print({k: (len(v), int(test[k].sum())) for k, v in data.items()}, flush=True)
    modes = sorted({w["mode"] for w in data["falls"]})
    results = {"counts": {k: len(v) for k, v in data.items()}, "modes": modes}

    def training_set(cols: slice, multiclass: bool, events: bool = False):
        xs, ys = [], []
        kinds = (("normal", 0), ("falls", 1)) + ((("events", 0),) if events else ())
        for kind, label in kinds:
            for w, t in zip(data[kind], test[kind]):
                if t:
                    continue
                fr = list(frames_of(w, "fall" if label else "normal"))
                xs.append(windows(w["x"][:, cols], fr))
                y = (1 + modes.index(w["mode"])) if (label and multiclass) else label
                ys.append(np.full(len(fr), y))
        return np.concatenate(xs).astype(np.float32), np.concatenate(ys)

    def scorer(predict, cols: slice, column=None):
        def score(w):
            s = np.zeros(len(w["x"]), np.float32)
            p = predict(windows(w["x"][:, cols], range(HIST - 1, len(w["x"]))))
            s[HIST - 1 :] = 1 - p[:, 0]
            return s

        return score

    strategies = {}
    proprio, with_imu = slice(0, 45), slice(0, 48)
    for name, cols, events in (("hazard", proprio, False), ("hazard_imu", with_imu, False),
                               ("hazard_ev", proprio, True)):  # fmt: skip
        x, y = training_set(cols, False, events)
        strategies[name] = scorer(fit(x, y, 2, "ce", args.epochs, device, args.seed), cols)
    x, y = training_set(proprio, True)
    mode_predict = fit(x, y, 1 + len(modes), "ce", args.epochs, device, args.seed)
    strategies["mode_hazard"] = scorer(mode_predict, proprio)

    # forward model on normal walking: next joint pos / vel, gravity, base rate (the PD targets carry the action)
    xs, ys = [], []
    for w, t in zip(data["normal"], test["normal"]):
        if t:
            continue
        fr = list(range(HIST - 1, len(w["x"]) - 1, 2))
        xs.append(windows(w["x"][:, proprio], fr))
        ys.append(w["x"][[f + 1 for f in fr], :42] - w["x"][fr, :42])
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)
    y_sd = y.std(0) + 1e-6
    fwd = fit(x, y / y_sd, 42, "mse", args.epochs, device, args.seed)

    def forward_error(w):
        s = np.zeros(len(w["x"]), np.float32)
        fr = np.arange(HIST - 1, len(w["x"]) - 1)
        pred = fwd(windows(w["x"][:, proprio], fr).astype(np.float32))
        err = ((pred - (w["x"][fr + 1, :42] - w["x"][fr, :42]) / y_sd) ** 2).mean(1)
        s[fr + 1] = np.convolve(err, np.ones(5) / 5, "same")  # 0.1 s smoothing
        return s

    strategies["forward_error"] = forward_error
    strategies["tilt"] = lambda w: w["tilt"]
    strategies["value"] = lambda w: -w["value"]

    for name, fn in strategies.items():
        results[name] = evaluate(fn, data, test, args.fa_per_min)
        print(f"== {name}: " + json.dumps(results[name]), flush=True)

    # which mode does mode_hazard name, at the alarm (test falls with an alarm)
    thr = results["mode_hazard"]["threshold"]
    confusion = {m: {} for m in modes}
    for w, t in zip(data["falls"], test["falls"]):
        if not t:
            continue
        s = strategies["mode_hazard"](w)
        lo = max(w["ref"] - round(3 / DT), HIST - 1)
        hits = alarms(s[lo : w["ref"] + round(0.25 / DT) + 1], thr)
        if not hits:
            continue
        f = lo + hits[0]
        p = mode_predict(windows(w["x"][:, proprio], [f]))[0, 1:]
        guess = modes[int(p.argmax())]
        confusion[w["mode"]][guess] = confusion[w["mode"]].get(guess, 0) + 1
    results["mode_at_alarm"] = confusion
    print("mode named at the alarm:", confusion, flush=True)

    stats = {}
    for policy in {w["policy"] for w in data["normal"]}:
        train = [w["x"] for w, t in zip(data["normal"], test["normal"]) if not t and w["policy"] == policy]
        ref = np.concatenate(train)
        stats[policy] = (ref.mean(0), ref.std(0) + 1e-6)
    results["signatures"] = signatures(data, stats)
    for mode, entry in results["signatures"].items():
        first = {g: v["first_above_2_s"] for g, v in entry.items() if g != "n"}
        print(f"   signature {mode} (n {entry['n']}): first |z|>2 at {first}", flush=True)
    with open(os.path.join(args.out, f"detect_{args.modes}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
