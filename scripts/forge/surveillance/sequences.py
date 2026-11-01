# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ways to turn failure sequences into a taxonomy, compared on the same falls with the same scores.

Offline (numpy, scikit-learn, torch, tslearn). Every fall is cut to ``[-1.5 s, +0.5 s]`` around its balance loss
(``taxonomy.balance_loss``) and described per frame by a privileged **physical state** (:data:`STATE`: attitude and
rates, base height, velocity tracking error, capture-point support margin, feet down, front / hind foot slip speed and
sinkage, front / hind leg contact force, torque saturation), standardized by the same policy's normal walking (the
``random_walk`` trigger windows), so 1 means "one normal standard deviation off". Left and right are merged (absolute
roll, front / hind pairs): a mirrored fall is the same mechanism.

Methods (each gives one label per fall):

* ``rules``: ``taxonomy.failure_mode``, the priority list over privileged labels (the 2026-10-31 taxonomy).
* ``label_ward``: ``taxonomy.features`` (label shares in 0.5 s bins) + Ward, k by silhouette (the 2026-10-31 clusters).
* ``first_violation``: the mechanism group (:data:`GROUPS`) whose state first leaves its normal range (|z| > 4 for
  3 frames) in ``[-1.5 s, +0.3 s]``: an ordering, not a distance, so no k to choose.
* ``dtw``: a cost between two sequences, dynamic time warping over the standardized state (squared Euclidean frame
  cost, Sakoe-Chiba band 0.2 s), then average-linkage clustering of the distance matrix, k by silhouette.
* ``ae``: latent of a 1-D convolutional autoencoder of the standardized state sequence (unsupervised), k-means.
* ``priv_latent``: latent of an encoder from 0.5 s of **proprioception only** trained to output the current
  physical state (a student that estimates the privileged state, as in teacher-student locomotion); the falls'
  latents at the balance loss, k-means. The only representation that runs on the real robot.

Scores per method: k; stability (mean ARI of the method refit on 80 % subsamples, on the shared falls); NMI with the
terrain (cause), the policy, and the ``rules`` modes; ``mechanism_r2`` (share of the variance of the per-fall mean
state explained by the labels: physically coherent groups score high); ``outcome_r2`` (same for how the robot ends:
tip direction and time to the ground); detectability (5-fold balanced accuracy of a random forest that names the label
from proprioception 0.5 s before and at the balance loss)::

    R=results/euler/isaaclab3/surveillance/v3_hard
    uv run --no-project --with torch --with numpy --with scikit-learn --with matplotlib --with tslearn python \\
        scripts/forge/surveillance/sequences.py --out results/sequences_v3 --falls $R/*/* \\
        --normal $R/*/{rough_field,pit_field,stub_field,rough_slope,rough_stairs}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze  # noqa: E402
import taxonomy  # noqa: E402
from classify import PROPRIO  # noqa: E402

STATE = [
    "roll", "pitch", "roll_rate", "pitch_rate", "yaw_rate", "base_height", "vx_error", "vy_error", "vz",
    "support_margin", "feet_down", "slip_front", "slip_hind", "sink_front", "sink_hind", "leg_force_front",
    "leg_force_hind", "torque_ratio",
]  # fmt: skip
"""Privileged physical state per frame (see :func:`physical_state`)."""
GROUPS = {
    "attitude": ["roll", "pitch", "roll_rate", "pitch_rate", "yaw_rate"],
    "body_drop": ["base_height", "vz"],
    "tracking": ["vx_error", "vy_error"],
    "support": ["support_margin", "feet_down"],
    "slip": ["slip_front", "slip_hind"],
    "sink": ["sink_front", "sink_hind"],
    "leg_contact": ["leg_force_front", "leg_force_hind"],
    "torque": ["torque_ratio"],
}
"""Mechanism groups of :data:`STATE` for ``first_violation``."""
BEFORE_S, AFTER_S = 1.5, 0.5


def physical_state(e: dict, meta: dict) -> np.ndarray:
    """[T, len(STATE)] privileged physical state of one window (raw units)."""
    ev = analyze.Event(e, meta)
    g = e["projected_gravity_b"]
    names = meta["contact_body_names"]
    front = np.array([names[i].startswith(("FL", "FR")) for i in ev.feet])
    feet = e["body_pos_w"][:, ev.foot_bodies]
    speed = np.where(ev.stance, np.linalg.norm(e["body_lin_vel_w"][:, ev.foot_bodies, :2], axis=-1), 0.0)
    # sinkage below the touchdown point, straight down (as taxonomy.privileged)
    ref, sink = feet[0].copy(), np.zeros(ev.stance.shape)
    for t in range(ev.n):
        ref = np.where(ev.touchdown[t][:, None], feet[t], ref)
        moved = np.linalg.norm(feet[t, :, :2] - ref[:, :2], axis=-1)
        sink[t] = np.where(ev.stance[t] & (moved < 0.05), np.maximum(ref[:, 2] - feet[t, :, 2], 0.0), 0.0)
    force = np.linalg.norm(e["contact_normal_force_w"], axis=-1)
    leg_front = [i for i in ev.legs if names[i].startswith(("FL", "FR"))]
    leg_hind = [i for i in ev.legs if i not in leg_front]
    cmd = e["command_base_velocity"]
    margin = np.nan_to_num(ev.support_margin(), nan=-0.3)
    effort = np.asarray(meta["joint_effort_limits"])
    cols = [
        np.abs(np.arctan2(g[:, 1], -g[:, 2])), np.arctan2(g[:, 0], -g[:, 2]),
        np.abs(e["root_ang_vel_b"][:, 0]), e["root_ang_vel_b"][:, 1], np.abs(e["root_ang_vel_b"][:, 2]),
        ev.base_height(), e["root_lin_vel_b"][:, 0] - cmd[:, 0], np.abs(e["root_lin_vel_b"][:, 1] - cmd[:, 1]),
        e["root_lin_vel_w"][:, 2], np.clip(margin, -0.3, 0.3), ev.stance.sum(1).astype(float),
        speed[:, front].max(1), speed[:, ~front].max(1), sink[:, front].max(1), sink[:, ~front].max(1),
        np.log1p(force[:, leg_front].sum(1)), np.log1p(force[:, leg_hind].sum(1)),
        (np.abs(e["applied_torque"]) / effort).max(1),
    ]  # fmt: skip
    return np.stack(cols, 1).astype(np.float32)


def proprio(e: dict) -> np.ndarray:
    return np.concatenate([e[f].reshape(len(e[f]), -1) for f in PROPRIO], 1).astype(np.float32)


def cut(x: np.ndarray, ref: int, dt: float) -> np.ndarray:
    """Frames ``[ref - BEFORE_S, ref + AFTER_S)``, edge-padded past the window."""
    idx = np.clip(np.arange(ref - round(BEFORE_S / dt), ref + round(AFTER_S / dt)), 0, len(x) - 1)
    return x[idx]


def load_falls(paths: list[str], thresholds: dict) -> tuple[list[dict], dict]:
    """Every fall with ``BEFORE_S`` of history before its balance loss: rows, and arrays ``state`` [N, T, S],
    ``proprio`` [N, W, P] (the whole window), ``label_x`` (``taxonomy.features``), ``label_x_mech`` (the same without
    the terrain-event flags, i.e. without the cause), ``label_seq`` [N, T, L] (those mechanism labels per frame),
    ``outcome`` [N, 3]."""
    bins = [(-1.5 + i * taxonomy.BIN_S, -1.5 + (i + 1) * taxonomy.BIN_S) for i in range(4)]
    rows, state, prop, label_x, outcome, refs = [], [], [], [], [], []
    label_x_mech, label_seq = [], []
    for path in paths:
        if not os.path.isfile(os.path.join(path, "index.csv")):
            continue
        meta, index = analyze.load_collection(path)
        policy, terrain = path.rstrip("/").split("/")[-2:]
        dt = meta["step_dt"]
        for r in index:
            if r["terminal"] != "True":
                continue
            e = analyze.load_event(path, r)
            n = int(r["num_frames"])
            ref = taxonomy.balance_loss(e, n - 1, dt)
            if ref * dt < BEFORE_S:
                continue
            masks = analyze.rule_masks(e, meta, thresholds[policy], grace_s=0.0)
            labels = {name: masks[name] for name in taxonomy.LABELS} | taxonomy.privileged(e, meta)
            x = taxonomy.features(labels, e, ref, bins, dt)
            label_x.append(np.append(x, (n - 1 - ref) * dt / 3.0))
            mech = {k: v for k, v in labels.items() if k not in taxonomy.EVENT_FLAGS}
            label_x_mech.append(np.append(taxonomy.features(mech, e, ref, bins, dt), (n - 1 - ref) * dt / 3.0))
            label_seq.append(cut(np.stack(list(mech.values()), 1).astype(np.float32), ref, dt))
            outcome.append(np.append(taxonomy.tip_direction(e, n - 1, round(3 / dt)), (n - 1 - ref) * dt))
            state.append(cut(physical_state(e, meta), ref, dt))
            p = proprio(e)
            prop.append(np.pad(p, ((250 - len(p), 0), (0, 0)), mode="edge") if len(p) < 250 else p)
            refs.append(ref + 250 - len(p))
            rows.append({"policy": policy, "terrain": terrain, "event": int(r["event"]), "collection": path,
                         "mode": taxonomy.failure_mode(labels, ref, n - 1, dt), "onset": ref})  # fmt: skip
    arrays = {"state": np.array(state), "proprio": np.array(prop), "label_x": np.array(label_x),
              "outcome": np.array(outcome), "ref": np.array(refs), "label_x_mech": np.array(label_x_mech),
              "label_seq": np.array(label_seq)}  # fmt: skip
    return rows, arrays


def load_normal(paths: list[str]) -> dict[str, dict]:
    """Per policy: the physical state and proprioception of every frame of its normal-walking trigger windows."""
    out: dict = {}
    for path in paths:
        root = os.path.join(path, "triggers")
        if not os.path.isfile(os.path.join(root, "index.csv")):
            continue
        meta, index = analyze.load_collection(root)
        policy = path.rstrip("/").split("/")[-2]
        for r in index:
            e = analyze.load_event(root, r)
            if np.linalg.norm(e["contact_normal_force_w"][:, 0], axis=-1).max() > 1.0:
                continue  # a base contact inside the window: not normal
            d = out.setdefault(policy, {"state": [], "proprio": []})
            d["state"].append(physical_state(e, meta))
            d["proprio"].append(proprio(e))
    return out


def standardize(rows, state, normal) -> np.ndarray:
    """State in units of the policy's normal-walking standard deviation, centered on its normal median."""
    out = np.empty_like(state)
    for policy in {r["policy"] for r in rows}:
        ref = np.concatenate(normal[policy]["state"])
        mu, sd = np.median(ref, 0), np.maximum(ref.std(0), 1e-3)
        sel = np.array([r["policy"] == policy for r in rows])
        out[sel] = (state[sel] - mu) / sd
    return out


# -- methods


def first_violation(z: np.ndarray, dt: float, thr: float = 4.0, sustain: int = 3) -> np.ndarray:
    """Per fall, the group whose channel first stays beyond ``thr`` normal deviations for ``sustain`` frames, in
    ``[-BEFORE_S, +0.3 s]`` of the balance loss; ``none`` if none does."""
    names = list(GROUPS)
    group_of = np.array([next(i for i, g in enumerate(names) if s in GROUPS[g]) for s in STATE])
    end = round((BEFORE_S + 0.3) / dt)
    out = []
    for seq in z:
        over = np.abs(seq[:end]) > thr
        run = np.ones_like(over)
        for k in range(sustain):
            run[: len(over) - k] &= over[k:]
            run[len(over) - k :] = False
        first = np.where(run.any(0), run.argmax(0), np.inf)
        out.append("none" if np.isinf(first).all() else names[group_of[np.argmin(first)]])
    return np.array(out)


def dtw_matrix(z: np.ndarray, step: int, band_s: float, dt: float) -> np.ndarray:
    from tslearn.metrics import cdist_dtw

    x = z[:, ::step].astype(np.float64)
    return cdist_dtw(x, sakoe_chiba_radius=max(round(band_s / (dt * step)), 1), n_jobs=-1)


def kmedoids(d: np.ndarray, k: int, seed: int = 0, iters: int = 50, restarts: int = 5) -> np.ndarray:
    """k-medoids on a distance matrix (k-means++ seeding, alternate assignment / medoid update); the restart with
    the lowest total distance. Unlike average linkage it does not spend clusters on single outliers."""
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    for _ in range(restarts):
        med = [rng.integers(len(d))]
        for _ in range(k - 1):
            p = d[:, med].min(1) ** 2
            med.append(rng.choice(len(d), p=p / p.sum()))
        med = np.array(med)
        for _ in range(iters):
            lab = d[:, med].argmin(1)
            new = np.array([np.flatnonzero(lab == c)[d[np.ix_(lab == c, lab == c)].sum(1).argmin()]
                            if (lab == c).any() else med[c] for c in range(k)])  # fmt: skip
            if (new == med).all():
                break
            med = new
        cost = d[:, med].min(1).sum()
        if cost < best_cost:
            best, best_cost = d[:, med].argmin(1), cost
    return best


def compress(z: np.ndarray) -> np.ndarray:
    """sign(z) log(1 + |z|): a 10-sigma frame no longer outweighs ten 1-sigma frames in a distance."""
    return np.sign(z) * np.log1p(np.abs(z))


def best_k(fit, score, k_range=range(3, 11)) -> tuple[int, dict]:
    """k with the best silhouette, ignoring partitions where one cluster holds more than 90 % of the falls."""
    scores = {}
    for k in k_range:
        labels = fit(k)
        if np.bincount(labels).max() > 0.9 * len(labels):
            continue
        scores[k] = float(score(labels))
    return (max(scores, key=scores.get) if scores else k_range[0]), scores


def kmeans(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(k, n_init=10, random_state=seed).fit_predict(x)


def train_ae(z: np.ndarray, latent: int, epochs: int, seed: int, device: str) -> np.ndarray:
    """1-D convolutional autoencoder of [N, T, S] sequences; returns the [N, latent] codes."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x = torch.tensor(np.clip(z, -10, 10), dtype=torch.float32).transpose(1, 2).to(device)  # N, S, T
    s, t = x.shape[1], x.shape[2]
    enc = nn.Sequential(nn.Conv1d(s, 32, 5, 2, 2), nn.ELU(), nn.Conv1d(32, 32, 5, 2, 2), nn.ELU(), nn.Flatten())
    flat = enc(x[:1].cpu()).shape[1]
    enc = nn.Sequential(enc, nn.Linear(flat, latent)).to(device)
    dec = nn.Sequential(nn.Linear(latent, 128), nn.ELU(), nn.Linear(128, s * t)).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3, weight_decay=1e-5)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(64):
            loss = ((dec(enc(x[idx])).view(-1, s, t) - x[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return enc(x).cpu().numpy()


def train_priv_latent(xs: list[np.ndarray], ys: list[np.ndarray], history: int, latent: int, epochs: int,
                      seed: int, device: str):  # fmt: skip
    """MLP from ``history`` frames of proprioception to a ``latent`` bottleneck to the current standardized physical
    state. ``xs`` / ``ys``: per window [T, P] / [T, S]. Returns ``encode(windows [N, T, P], frames [N]) -> [N,
    latent]`` and the validation R² per state channel."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    val = set(order[: len(xs) // 10].tolist())
    data = {"train": ([], []), "val": ([], [])}
    for i, (x, y) in enumerate(zip(xs, ys)):
        part = data["val" if i in val else "train"]
        for t in range(history - 1, len(x), 2):
            part[0].append(x[t - history + 1 : t + 1].reshape(-1))
            part[1].append(y[t])
    x, y = (torch.tensor(np.array(a)) for a in data["train"])
    xv, yv = (torch.tensor(np.array(a)) for a in data["val"])
    mean, std = x.mean(0), x.std(0).clamp(min=1e-3)
    y, yv = y.clamp(-10, 10), yv.clamp(-10, 10)
    enc = nn.Sequential(nn.Linear(x.shape[1], 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, latent))
    head = nn.Sequential(nn.ELU(), nn.Linear(latent, y.shape[1]))
    enc, head = enc.to(device), head.to(device)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=1e-3, weight_decay=1e-5)
    xn, yd = ((x - mean) / std).to(device), y.to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(xn), device=device).split(1024):
            loss = ((head(enc(xn[idx] + 0.05 * torch.randn_like(xn[idx]))) - yd[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = head(enc(((xv - mean) / std).to(device))).cpu()
    r2 = 1 - ((pred - yv) ** 2).mean(0) / yv.var(0).clamp(min=1e-6)

    def encode(windows: np.ndarray, frames: np.ndarray) -> np.ndarray:
        w = np.stack([win[t - history + 1 : t + 1].reshape(-1) for win, t in zip(windows, frames)])
        with torch.no_grad():
            return enc(((torch.tensor(w) - mean) / std).to(device)).cpu().numpy()

    return encode, {s: round(float(v), 3) for s, v in zip(STATE, r2)}


# -- scores


def variance_explained(x: np.ndarray, labels: np.ndarray) -> float:
    """Share of the total variance of ``x`` [N, D] (columns standardized) explained by the group means."""
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    within = sum(((x[labels == c] - x[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    return float(1 - within / (x**2).sum())


def rf_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    """5-fold balanced accuracy of a random forest naming ``labels`` from ``x``; classes with < 5 falls dropped."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    labels = np.unique(labels, return_inverse=True)[1]
    keep = np.isin(labels, [c for c in np.unique(labels) if (labels == c).sum() >= 5])
    if len(np.unique(labels[keep])) < 2:
        return float("nan")
    model = RandomForestClassifier(300, class_weight="balanced", random_state=0, n_jobs=-1)
    pred = cross_val_predict(model, x[keep], labels[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return round(float(balanced_accuracy_score(labels[keep], pred)), 3)


def detectability(prop: np.ndarray, refs: np.ndarray, labels: np.ndarray, delta_s: float, dt: float) -> float:
    """:func:`rf_accuracy` from 0.5 s of proprioception ending ``delta_s`` from the balance loss (per-channel mean,
    std, change, last value)."""
    h = round(0.5 / dt)
    t = np.clip(refs + round(delta_s / dt), h - 1, prop.shape[1] - 1)
    w = np.stack([p[i - h + 1 : i + 1] for p, i in zip(prop, t)])
    return rf_accuracy(np.concatenate([w.mean(1), w.std(1), w[:, -1] - w[:, 0], w[:, -1]], 1), labels)


def stability(refit, n: int, k: int, repeats: int = 8, frac: float = 0.8, seed: int = 0) -> float:
    """Mean ARI between the full-data labels and the labels refit on random ``frac`` subsamples (on the shared
    falls). ``refit(idx, k) -> labels of idx``."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    full = refit(np.arange(n), k)
    scores = []
    for _ in range(repeats):
        idx = np.sort(rng.choice(n, round(frac * n), replace=False))
        scores.append(adjusted_rand_score(full[idx], refit(idx, k)))
    return round(float(np.mean(scores)), 3)


def profile(z: np.ndarray, labels: np.ndarray, top: int = 4) -> dict:
    """Per label: count and the state channels with the largest mean |z| over ``[-1 s, +0.3 s]``."""
    out = {}
    for c in np.unique(labels):
        m = np.abs(z[labels == c][:, 25:90]).mean((0, 1))
        order = np.argsort(-m)[:top]
        out[str(c)] = {"n": int((labels == c).sum()), "top": [f"{STATE[i]} {m[i]:.1f}" for i in order]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--falls", nargs="+", required=True)
    parser.add_argument("--normal", nargs="+", required=True, help="Collections with random_walk trigger windows.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtw_max", type=int, default=1500, help="Falls in the DTW matrix (random subset beyond).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self_baseline", action="store_true", help="Subtract each fall's first 0.5 s.")
    parser.add_argument("--max_start_tilt_deg", type=float, default=None, help="Drop falls already tilted this much "
                        "1.5 s before the balance loss (already down).")  # fmt: skip
    parser.add_argument("--cache", default=None, help="Loaded-data pickle (default: <out>/loaded.pkl).")
    args = parser.parse_args()
    import torch
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    os.makedirs(args.out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dt = 0.02
    policies = sorted({p.rstrip("/").split("/")[-2] for p in args.falls})
    thresholds = {
        policy: analyze.baseline_thresholds(
            [os.path.join(p, "triggers") for p in args.normal if p.rstrip("/").split("/")[-2] == policy],
            99.9, 0.0, 1.0)
        for policy in policies
    }  # fmt: skip
    cache = args.cache or os.path.join(args.out, "loaded.pkl")
    if os.path.isfile(cache):
        with open(cache, "rb") as f:
            rows, a, normal = pickle.load(f)
    else:
        rows, a = load_falls(args.falls, thresholds)
        normal = load_normal(args.normal)
        with open(cache, "wb") as f:
            pickle.dump((rows, a, normal), f)
    if args.max_start_tilt_deg is not None:
        # robots already down at the window start (propped on their legs, the base not yet touching) are a late
        # base-contact termination, not a mechanism
        roll, pitch = a["state"][:, 0, STATE.index("roll")], a["state"][:, 0, STATE.index("pitch")]
        keep = np.degrees(np.arccos(np.cos(roll) * np.cos(pitch))) <= args.max_start_tilt_deg
        rows, a = [r for r, k in zip(rows, keep) if k], {k: v[keep] for k, v in a.items()}
        print(f"kept {keep.sum()} of {len(keep)} falls (tilt <= {args.max_start_tilt_deg} deg at the start)")
    z = standardize(rows, a["state"], normal)
    if args.self_baseline:
        # change from the robot's own first 0.5 s of the window (its gait on this terrain), not from flat normal
        z = z - np.median(z[:, : round(0.5 / dt)], 1, keepdims=True)
    zc = np.clip(z, -10, 10)
    n = len(rows)
    print(f"falls {n}, normal windows " + ", ".join(f"{p} {len(v['state'])}" for p, v in normal.items()), flush=True)
    mech = zc[:, 25:90].mean(1)  # per-fall mean state over [-1 s, +0.3 s]
    rng = np.random.default_rng(args.seed)
    results, labels = {}, {}

    labels["rules"] = np.array([r["mode"] for r in rows])
    labels["first_violation"] = first_violation(z, dt)
    # clustering methods: name -> (fit(idx, k) -> labels, silhouette(labels) on all falls of the method)
    clusterers = {}
    zl = compress(z)
    sub = np.sort(rng.choice(n, min(n, args.dtw_max), replace=False))  # DTW is quadratic: a subset

    from sklearn.decomposition import PCA

    lx = PCA(n_components=0.95, random_state=0).fit_transform(a["label_x"])

    def ward(idx, k):
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(lx[idx])

    clusterers["label_ward"] = (ward, lambda lab: silhouette_score(lx, lab), np.arange(n))
    lxm = PCA(n_components=0.95, random_state=0).fit_transform(a["label_x_mech"])

    def ward_mech(idx, k):
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(lxm[idx])

    clusterers["label_ward_mech"] = (ward_mech, lambda lab: silhouette_score(lxm, lab), np.arange(n))
    # cost metric on the binary event sequence: DTW with the number of differing labels as the frame cost
    dlab = dtw_matrix(a["label_seq"][sub], step=2, band_s=0.2, dt=dt)
    clusterers["label_dtw"] = (lambda idx, k: kmedoids(dlab[np.ix_(idx, idx)], k),
                               lambda lab: silhouette_score(dlab, lab, metric="precomputed"), sub)  # fmt: skip

    d = dtw_matrix(zl[sub], step=2, band_s=0.2, dt=dt)
    np.save(os.path.join(args.out, "dtw.npy"), d)
    clusterers["dtw"] = (lambda idx, k: kmedoids(d[np.ix_(idx, idx)], k),
                         lambda lab: silhouette_score(d, lab, metric="precomputed"), sub)  # fmt: skip

    code = train_ae(zl, latent=8, epochs=150, seed=args.seed, device=device)
    clusterers["ae"] = (lambda idx, k: kmeans(code[idx], k), lambda lab: silhouette_score(code, lab), np.arange(n))

    # privileged-state student on normal windows + falls (whole windows, standardized per policy)
    xs, ys = [], []
    for policy, v in normal.items():
        ref = np.concatenate(v["state"])
        mu, sd = np.median(ref, 0), np.maximum(ref.std(0), 1e-3)
        keep = rng.permutation(len(v["state"]))[:800]  # memory: every second frame of each window is a sample
        xs += [v["proprio"][i] for i in keep]
        ys += [(v["state"][i] - mu) / sd for i in keep]
    # fall windows: the cut state lines up with the proprio frames [ref - 75, ref + 25)
    for i in range(n):
        lo = a["ref"][i] - round(BEFORE_S / dt)
        hi = min(a["ref"][i] + round(AFTER_S / dt), 250)
        xs.append(a["proprio"][i][lo:hi])
        ys.append(z[i][: hi - lo])
    encode, r2 = train_priv_latent(xs, ys, history=25, latent=16, epochs=30, seed=args.seed, device=device)
    pl = encode(a["proprio"], a["ref"])
    clusterers["priv_latent"] = (lambda idx, k: kmeans(pl[idx], k), lambda lab: silhouette_score(pl, lab),
                                 np.arange(n))  # fmt: skip
    results["priv_latent_r2"] = r2
    print("student R2 per state:", r2, flush=True)
    # cost metric in the student's latent: DTW over its latent trajectory through [-1.5 s, +0.5 s]
    offsets = np.arange(-round(BEFORE_S / dt), round(AFTER_S / dt), 2)
    traj = np.stack([encode(a["proprio"], np.clip(a["ref"] + o, 24, 249)) for o in offsets], 1)
    traj = (traj - traj.mean((0, 1))) / traj.std((0, 1))
    dl = dtw_matrix(traj[sub], step=1, band_s=0.2, dt=2 * dt)
    clusterers["dtw_latent"] = (lambda idx, k: kmedoids(dl[np.ix_(idx, idx)], k),
                                lambda lab: silhouette_score(dl, lab, metric="precomputed"), sub)  # fmt: skip

    refits = {}
    for name, (fit, score, idx) in clusterers.items():
        k, scores = best_k(lambda k: fit(np.arange(len(idx)), k), score)
        for tag, kk in (("", k), ("@6", 6)):
            lab = np.full(n, -1)
            lab[idx] = fit(np.arange(len(idx)), kk)
            labels[name + tag], refits[name + tag] = lab, (fit, kk)
        results.setdefault("silhouette", {})[name] = scores

    terrain = np.array([f"{r['policy']}/{r['terrain']}" for r in rows])
    policy = np.array([r["policy"] for r in rows])
    context = (terrain[:, None] == np.unique(terrain)[None]).astype(float)
    for name, lab in labels.items():
        sel = lab.astype(str) != "-1"  # DTW methods: only their subset
        lab_s = lab[sel].astype(str)
        k = len(np.unique(lab_s))
        res = {
            "n": int(sel.sum()), "k": k,
            "nmi_terrain": round(float(normalized_mutual_info_score(terrain[sel], lab_s)), 3),
            "nmi_policy": round(float(normalized_mutual_info_score(policy[sel], lab_s)), 3),
            "nmi_rules": round(float(normalized_mutual_info_score(labels["rules"][sel], lab_s)), 3),
            "mechanism_r2": round(variance_explained(mech[sel], lab_s), 3),
            "outcome_r2": round(variance_explained(a["outcome"][sel], lab_s), 3),
            "detect_-0.5s": detectability(a["proprio"][sel], a["ref"][sel], lab_s, -0.5, dt),
            "detect_+0.0s": detectability(a["proprio"][sel], a["ref"][sel], lab_s, 0.0, dt),
            # the same forest from the policy and terrain alone: how much of the label is just the context
            "detect_context": rf_accuracy(context[sel], lab_s),
            "profile": profile(zc[sel], lab_s),
        }  # fmt: skip
        # detectability above chance: (balanced accuracy - 1/k) / (1 - 1/k), comparable across k
        for key in ("detect_-0.5s", "detect_+0.0s", "detect_context"):
            res[key + "_kappa"] = round((res[key] - 1 / k) / (1 - 1 / k), 3)
        if name in refits:
            fit, kk = refits[name]
            res["stability_ari"] = stability(fit, int(sel.sum()), kk)
        results[name] = res
        print(f"== {name}: " + json.dumps({k: v for k, v in res.items() if k != "profile"}), flush=True)
        for c, p in res["profile"].items():
            print(f"     {c}: {p}", flush=True)
    # agreement between methods (on the DTW subset, where all have labels)
    names = [name for name in labels if not name.endswith("@6")]
    results["ari_between"] = {
        f"{p}~{q}": round(float(adjusted_rand_score(labels[p][sub].astype(str), labels[q][sub].astype(str))), 3)
        for i, p in enumerate(names) for q in names[i + 1 :]
    }  # fmt: skip
    print("ARI between methods:", results["ari_between"], flush=True)
    with open(os.path.join(args.out, "sequences.json"), "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(os.path.join(args.out, "sequences.npz"), z=z, ae=code, priv_latent=pl, sub=sub,
                        **{f"labels_{k}": v.astype(str) for k, v in labels.items()})  # fmt: skip
    with open(os.path.join(args.out, "falls.csv"), "w") as f:
        f.write(",".join([*rows[0], *labels]) + "\n")
        for i, r in enumerate(rows):
            f.write(",".join([str(v) for v in r.values()] + [str(labels[k][i]) for k in labels]) + "\n")


if __name__ == "__main__":
    main()
