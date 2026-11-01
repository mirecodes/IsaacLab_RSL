# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Which terrain event is acting on the robot, from proprioception only, with labels from the simulation.

Offline (torch, numpy). The event-trigger windows of a campaign (``triggers/``, 5 s centered on the trigger) carry
their event for free: the frames from the trigger to ``--label_s`` after it are that event, all frames before it are
``normal`` (the random-walk windows are normal throughout). Windows that log the terrain flags (v3 on) are labeled
per frame from those instead (:func:`frame_labels`). A small MLP sees the last ``--history_s`` of
proprioception that a real A2 measures (joint positions, velocities and PD targets, gravity in the base frame, base
angular rate, the velocity command) and outputs one probability per class. No human label is involved; this is the
self-supervised multi-class detector of the 2026-10-26 meeting, on the v2 events.

Windows are split per window (train / validation / test 70 / 15 / 15, stratified by event). The test result is the
class probability over the time since the trigger (how fast the right class rises and the others fall) and the
accuracy at each delay::

    uv run --no-project --with torch --with numpy --with matplotlib python scripts/forge/surveillance/classify.py \\
        results/euler/isaaclab3/surveillance/v2/Robust/* --out results/classify_v1/Robust
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch import nn

PROPRIO = [
    "joint_pos",
    "joint_vel",
    "joint_pos_target",
    "projected_gravity_b",
    "root_ang_vel_b",
    "command_base_velocity",
]
"""Fields a real robot measures or knows (no base linear velocity, no contact, no terrain)."""


def load(paths: list[str], label: int) -> tuple[list[np.ndarray], list[str], list[int], list[np.ndarray | None]]:
    """Proprioception [T, D] of every full trigger window, its event, its trigger frame and its per-frame labels
    from the logged terrain flags (:func:`frame_labels`; None without flags)."""
    windows, events, triggers, labels = [], [], [], []
    for path in paths:
        root = os.path.join(path, "triggers")
        if not os.path.isfile(os.path.join(root, "index.csv")):
            continue
        with open(os.path.join(path, "summary.json")) as f:
            event = json.load(f)["event_trigger"]
        with open(os.path.join(root, "index.csv")) as f:
            rows = [r for r in csv.DictReader(f) if int(r["num_frames"]) == 250]
        for r in rows:
            e = np.load(os.path.join(root, "events", f"{int(r['event']):06d}.npz"))
            k = int(r["num_frames"]) // 2 - 1
            windows.append(np.concatenate([e[f].reshape(len(e[f]), -1) for f in PROPRIO], 1).astype(np.float32))
            events.append(event if event != "any_event" else mixed_event(e, k))
            triggers.append(k)
            labels.append(frame_labels(e, label) if event in FLAGGED and "terrain_in_sand" in e else None)
    return windows, events, triggers, labels


FLAGGED = ("crust_collapse", "sand_sink", "foot_slip", "any_event")
"""Triggers whose event is in the logged terrain flags (not ``stone_step`` and ``random_walk``)."""


def frame_labels(e, label: int) -> np.ndarray:
    """The event acting at each frame, from the privileged terrain flags of any foot: ``crust_collapse`` for
    ``label`` frames from a collapse, else ``sand_sink`` while a foot is in sand, else ``foot_slip`` while a foot
    has a low-friction sample, else ``normal``. In v3 the frames before a trigger are often not normal: a foot was
    already on ice or in sand (the trigger re-fired while the robot crossed the patch)."""
    out = np.where(e["terrain_low_mu"].any(1), "foot_slip", "normal").astype(object)
    out[e["terrain_in_sand"].any(1)] = "sand_sink"
    for c in np.flatnonzero(e["terrain_collapse"].any(1)):
        out[c : c + label] = "crust_collapse"
    return out


def mixed_event(e, k: int) -> str:
    """Which event fired an ``any_event`` window (``mixed_field``): the logged terrain flags at the trigger frame."""
    for flag, name in (("terrain_collapse", "crust_collapse"), ("terrain_in_sand", "sand_sink"),
                       ("terrain_low_mu", "foot_slip")):  # fmt: skip
        if e[flag][k].any():
            return name
    return "unknown"


def samples(windows, events, triggers, ids, classes, history: int, label: int, before: int, labels=None,
            stride: int = 1):  # fmt: skip
    """(x, y, delay) for every ``stride``-th frame from ``before`` frames before the trigger to ``label`` after it: x
    the last ``history`` frames flattened, y the window's event from the trigger on (``normal`` before it and for
    random walks), or with ``labels`` the flagged event of the frame (a class outside ``classes`` counts as
    ``normal``), delay the frame minus the trigger frame."""
    xs, ys, ds = [], [], []
    for i in ids:
        w, k = windows[i], triggers[i]
        for t in range(max(k - before, history - 1), min(k + label, len(w)), stride):
            xs.append(w[t - history + 1 : t + 1].reshape(-1))
            if labels is not None and labels[i] is not None:
                ys.append(classes.index(labels[i][t]) if labels[i][t] in classes else 0)
            else:
                ys.append(classes.index(events[i]) if t >= k and events[i] in classes else 0)
            ds.append(t - k)
    return torch.tensor(np.array(xs)), torch.tensor(ys), np.array(ds)


def split(events: list[str], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for event in sorted(set(events)):
        ids = rng.permutation([i for i, e in enumerate(events) if e == event])
        a, b = int(0.7 * len(ids)), int(0.85 * len(ids))
        out["train"] += ids[:a].tolist()
        out["val"] += ids[a:b].tolist()
        out["test"] += ids[b:].tolist()
    return out


def train(x, y, xv, yv, classes, epochs: int, device: str, seed: int):
    """MLP 2 x 256 (ELU), class-balanced cross entropy, Adam; the epoch with the best validation balanced accuracy."""
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-3)
    model = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                          nn.Linear(256, len(classes))).to(device)  # fmt: skip
    weight = 1.0 / torch.bincount(y, minlength=len(classes)).clamp(min=1).float()
    loss_fn = nn.CrossEntropyLoss(weight=(weight / weight.sum() * len(classes)).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xn, xvn = ((x - mean) / std).to(device), ((xv - mean) / std).to(device)
    y, best, state = y.to(device), -1.0, None
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(xn), device=device).split(1024):
            # input noise as a stand-in for sensor noise and against overfitting on 700 windows
            loss = loss_fn(model(xn[idx] + 0.1 * torch.randn_like(xn[idx])), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xvn).argmax(1).cpu()
        score = np.mean([(pred[yv == c] == c).float().mean().item() for c in range(len(classes)) if (yv == c).any()])
        if score > best:
            best, state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, mean, std, best


def onset_auroc(windows, events, triggers, labels, ids, h, lab, bef, args, device) -> dict:
    """Per event, within its own terrain: a binary MLP, "the event started in the last ``label_s``" vs the frames
    before the trigger of the same windows, so the terrain cannot give the answer away (in v2 each event has its own
    terrain). Pre-trigger frames with a flagged event are left out (they are neither onset nor normal). Test AUROC of
    the frames at each delay against all pre-trigger test frames."""
    from sklearn.metrics import roc_auc_score

    flagged = ["normal", "crust_collapse", "sand_sink", "foot_slip"]

    def subset(sub, classes, stride=1):
        x, y, d = samples(windows, events, triggers, sub, classes, h, lab, bef, stride=stride)
        _, f, _ = samples(windows, events, triggers, sub, flagged, h, lab, bef, labels, stride=stride)
        keep = (torch.from_numpy(d) >= 0) | (f == 0)
        return x[keep], y[keep], d[keep.numpy()]

    out = {}
    for event in sorted(set(events) - {"random_walk"}):
        classes = ["normal", event]
        sub = {k: [i for i in v if events[i] == event] for k, v in ids.items()}
        if not all(sub.values()):
            print(f"[onset] {event}: too few windows for a train / val / test split", flush=True)
            continue
        x, y, _ = subset(sub["train"], classes, args.stride)
        xv, yv, _ = subset(sub["val"], classes, args.stride)
        xt, yt, d = subset(sub["test"], classes)
        model, mean, std, _ = train(x, y, xv, yv, classes, args.epochs, device, args.seed)
        with torch.no_grad():
            p = torch.softmax(model(((xt - mean) / std).to(device)), 1)[:, 1].cpu().numpy()
        neg, out[event] = p[d < 0], {}
        for s in (0.0, 0.1, 0.2, 0.5, 1.0):
            pos = p[d == round(s / 0.02)]
            auroc = roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos])
            out[event][f"{s:.2f}"] = round(float(auroc), 3)
        print(f"[onset] {event}: AUROC at delay {out[event]}", flush=True)
    return out


def fall_modes(falls_csv: str, out: str, history: int = 25) -> dict:
    """Failure mode of each fall (``taxonomy.py``'s ``falls.csv``) from the proprioception of the ``history`` frames
    ending ``delta`` s from its balance loss; 5-fold balanced accuracy of a random forest on per-channel mean, std,
    change and last value. The reference "context" sees only the policy and the terrain type."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    with open(falls_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] != "other"]
    y = np.array([r["mode"] for r in rows])
    data = [np.load(os.path.join(r["collection"], "events", f"{int(r['event']):06d}.npz")) for r in rows]
    terrains = sorted({r["terrain"] for r in rows})
    context = np.array([[r["policy"] == "Robust"] + [r["terrain"] == t for t in terrains] for r in rows], float)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def score(X):
        model = RandomForestClassifier(400, class_weight="balanced", random_state=0)
        pred = cross_val_predict(model, X, y, cv=cv)
        return round(float(balanced_accuracy_score(y, pred)), 3), pred

    result = {"counts": {m: int((y == m).sum()) for m in sorted(set(y))}, "context_only": score(context)[0]}
    for delta in (-1.0, -0.5, -0.25, 0.0, 0.25):
        X = []
        for r, e in zip(rows, data):
            t = min(max(int(r["onset"]) + round(delta / 0.02), history - 1), len(e["joint_pos"]) - 1)
            w = np.concatenate([e[k][t - history + 1 : t + 1].reshape(history, -1) for k in PROPRIO], 1)
            X.append(np.concatenate([w.mean(0), w.std(0), w[-1] - w[0], w[-1]]))
        acc, pred = score(np.array(X))
        result[f"proprio_{delta:+.2f}s"] = acc
    labels = sorted(set(y))
    result["confusion_+0.25s"] = {"labels": labels, "matrix": confusion_matrix(y, pred, labels=labels).tolist()}
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "fall_modes.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collections", nargs="*", help="Collections with trigger windows (one policy).")
    parser.add_argument("--falls_csv", default=None, help="Instead: failure modes of taxonomy.py's falls.csv.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--history_s", type=float, default=0.5)
    parser.add_argument("--label_s", type=float, default=1.5, help="Frames after the trigger labeled as the event.")
    parser.add_argument("--before_s", type=float, default=1.0, help="Normal frames used before each trigger.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1, help="Training frames: every stride-th (memory).")
    args = parser.parse_args()
    if args.falls_csv:
        fall_modes(args.falls_csv, args.out)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dt = 0.02
    h, lab, bef = round(args.history_s / dt), round(args.label_s / dt), round(args.before_s / dt)
    windows, events, triggers, labels = load(args.collections, lab)
    classes = ["normal"] + sorted(set(events) - {"random_walk"})
    ids = split(events, args.seed)
    x, y, _ = samples(windows, events, triggers, ids["train"], classes, h, lab, bef, labels, args.stride)
    xv, yv, _ = samples(windows, events, triggers, ids["val"], classes, h, lab, bef, labels, args.stride)
    xt, yt, dtest = samples(windows, events, triggers, ids["test"], classes, h, lab, bef, labels)
    model, mean, std, val = train(x, y, xv, yv, classes, args.epochs, device, args.seed)
    with torch.no_grad():
        prob = torch.softmax(model(((xt - mean) / std).to(device)), 1).cpu().numpy()
    # per test window's true event: mean class probabilities and accuracy at each delay
    owner = np.concatenate([[events[i]] * len(range(max(triggers[i] - bef, h - 1), min(triggers[i] + lab, 250)))
                            for i in ids["test"]])  # fmt: skip
    delays = np.arange(-bef, lab)
    curves, accuracy = {}, {}
    for event in sorted(set(events)):
        sel = owner == event
        curves[event] = np.array([prob[sel & (dtest == d)].mean(0) for d in delays])
        accuracy[event] = np.array([(prob[m].argmax(1) == yt.numpy()[m]).mean() for m in
                                    (sel & (dtest == d) for d in delays)])  # fmt: skip
    result = {
        "classes": classes, "history_s": args.history_s, "label_s": args.label_s, "val_balanced_accuracy": val,
        "windows": {k: len(v) for k, v in ids.items()},
        "accuracy_at_delay_s": {event: {f"{d * dt:.2f}": round(float(accuracy[event][list(delays).index(d)]), 3)
                                        for d in (0, 5, 10, 25, 50, 74)} for event in accuracy},
        "normal_false_positive": round(float(1 - np.mean(prob[yt.numpy() == 0].argmax(1) == 0)), 3),
        "test_balanced_accuracy": round(float(np.mean([np.mean(prob[yt.numpy() == c].argmax(1) == c)
                                                       for c in range(len(classes)) if (yt == c).any()])), 3),
        "test_recall": {name: round(float(np.mean(prob[yt.numpy() == c].argmax(1) == c)), 3)
                        for c, name in enumerate(classes) if (yt == c).any()},
    }  # fmt: skip
    result["onset_auroc"] = onset_auroc(windows, events, triggers, labels, ids, h, lab, bef, args, device)
    with open(os.path.join(args.out, "classify.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))

    fig, axes = plt.subplots(1, len(curves), figsize=(3.2 * len(curves), 3), sharey=True)
    for ax, (event, curve) in zip(axes, curves.items()):
        for c, name in enumerate(classes):
            ax.plot(delays * dt, curve[:, c], label=name, lw=2 if name == event or (event == "random_walk" and c == 0)
                    else 1)  # fmt: skip
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"true: {event}", fontsize=9)
        ax.set_xlabel("time since trigger [s]")
    axes[0].set_ylabel("mean class probability (test)")
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_probability.png"), dpi=130)


if __name__ == "__main__":
    main()
