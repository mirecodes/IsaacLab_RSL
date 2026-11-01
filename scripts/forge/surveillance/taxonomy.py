# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Failure taxonomy from privileged simulation state: per-frame signal labels, window features, clustering.

Offline (numpy, scikit-learn, matplotlib). Every frame of a saved window gets **signal labels**: the rule masks of
``analyze.py`` (A-E, thresholds from normal walking) plus privileged per-foot signals that a real robot does not
measure (stance-foot slip speed, sinkage below the touchdown height, which leg segment touches, the terrain event's
flags when the collection logged them). A window becomes one feature vector: the share of frames with each label in
0.5 s bins before the reference frame (the fall, or the trigger), and the direction the base tips over. Falls are
then clustered (Ward, k by silhouette) and each cluster is named by the labels that set it apart. The same features
on the event-trigger windows, whose event is known, check that the feature space separates the events at all::

    uv run --no-project --with numpy --with scikit-learn --with matplotlib python \\
        scripts/forge/surveillance/taxonomy.py --out results/taxonomy_v1 \\
        --falls results/euler/isaaclab3/fault_analysis/terrain_v1/*/* results/euler/isaaclab3/surveillance/v2/*/* \\
        --baseline results/euler/isaaclab3/surveillance/v2/*/{rough_field,pit_field,stub_field}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import analyze  # noqa: E402

LABELS = [
    "A_leg_contact", "A_tilt", "A_low_base", "B_ang_rate", "B_impact", "B_support", "B_value_drop", "B_action_jump",
    "C_trip", "C_slip", "C_late_touchdown", "C_flight", "C_sinking", "C_wedge", "D_tracking", "D_no_progress",
    "E_torque_saturation", "E_pd_error", "E_power",
]  # fmt: skip
"""Rules of ``analyze.py`` used as signal labels (A_base_contact is the fall itself; D_heading never hits)."""
PRIVILEGED = ["slip_2feet", "sink_5cm", "front_leg_contact", "hind_leg_contact", "thigh_contact", "stance_1foot"]
"""Privileged per-frame labels from the full sim state (see :func:`privileged`)."""
EVENT_FLAGS = ["terrain_low_mu", "terrain_on_ice", "terrain_in_sand", "terrain_collapse"]
"""Per-foot terrain-event flags of collections since 2026-10-31 (``collect.py``); any foot -> one label."""
BIN_S = 0.5


def privileged(e: dict, meta: dict) -> dict[str, np.ndarray]:
    """Per-frame boolean labels from state a real robot does not measure.

    ``slip_2feet``: at least two stance feet slide faster than 0.3 m/s at once (one sliding foot is common).
    ``sink_5cm``: a stance foot 5 cm below its touchdown point, less than 5 cm away from it horizontally.
    ``front_/hind_leg_contact``: a hip, thigh or calf of a front / hind leg touches something. ``thigh_contact``: a
    hip or thigh does (higher than a shin scrape).
    ``stance_1foot``: at most one foot on the ground while the base is below 0.3 m above the terrain under it.
    Terrain-event flags (any foot) when the collection has them.
    """
    ev = analyze.Event(e, meta)
    names = meta["contact_body_names"]
    speed = np.linalg.norm(e["body_lin_vel_w"][:, ev.foot_bodies, :2], axis=-1)
    pos = e["body_pos_w"][:, ev.foot_bodies]
    ref, sink = pos[0].copy(), np.zeros_like(ev.stance)
    for t in range(ev.n):
        ref = np.where(ev.touchdown[t][:, None], pos[t], ref)
        # straight down only: a foot sliding down a slope or off a bump also ends up lower than its touchdown
        moved = np.linalg.norm(pos[t, :, :2] - ref[:, :2], axis=-1)
        sink[t] = ev.stance[t] & (ref[:, 2] - pos[t, :, 2] > 0.05) & (moved < 0.05)
    touch = np.linalg.norm(e["contact_normal_force_w"], axis=-1) > meta["args"]["contact_threshold"]
    legs = [i for i in ev.legs]
    front = [i for i in legs if names[i].startswith(("FL", "FR"))]
    thigh = [i for i in legs if names[i].endswith(("_hip", "_thigh"))]
    out = {
        "slip_2feet": ((speed > 0.3) & ev.stance).sum(1) >= 2,
        "sink_5cm": sink.any(1),
        "front_leg_contact": touch[:, front].any(1),
        "hind_leg_contact": touch[:, [i for i in legs if i not in front]].any(1),
        "thigh_contact": touch[:, thigh].any(1),
        "stance_1foot": (ev.stance.sum(1) <= 1) & (ev.base_height() < 0.3),
    }
    for flag in EVENT_FLAGS:
        if flag in e:
            out[flag] = np.asarray(e[flag]).reshape(ev.n, -1).any(1)
    return out


def tip_direction(e: dict, ref: int, span: int) -> np.ndarray:
    """Where the base tips, in the body frame: the horizontal gravity direction (x: nose down, |y|: sideways) at the
    first frame of ``[ref - span, ref]`` past 30 deg of tilt, scaled by that frame's tilt; zeros if it stays
    upright (a base hit)."""
    g = e["projected_gravity_b"][max(ref - span, 0) : ref + 1]
    tilted = np.nonzero(g[:, 2] > -math.cos(math.radians(30.0)))[0]
    if len(tilted) == 0:
        return np.zeros(2)
    gx, gy = g[tilted[0], :2] / max(np.linalg.norm(g[tilted[0], :2]), 1e-6)
    return np.array([gx, abs(gy)])


def balance_loss(e: dict, ref: int, dt: float, tilt_deg: float = 20.0, span_s: float = 3.0) -> int:
    """Frame where the fall at ``ref`` starts: the first frame of the last unbroken stretch up to ``ref`` with the
    base tilted more than ``tilt_deg``, within ``span_s``; ``ref`` itself if the base stays below that (an upright
    base hit). Aligning windows here, not at the base contact, keeps how long a robot takes to go down (0.2 s for a
    trip, 2 s for a slide) out of the mechanism."""
    tilted = e["projected_gravity_b"][: ref + 1, 2] > -math.cos(math.radians(tilt_deg))
    if not tilted[ref]:
        return ref
    t, first = ref, max(ref - round(span_s / dt), 0)
    while t > first and tilted[t - 1]:
        t -= 1
    return t


MODES = ["upright_hit", "slide", "foot_trap", "leg_collision", "support_loss", "other"]
"""Failure modes of :func:`failure_mode`, in its priority order."""


def failure_mode(labels: dict[str, np.ndarray], ref: int, end: int, dt: float) -> str:
    """Physical failure mode of a fall from its privileged labels around the balance loss ``ref`` (fall at ``end``).

    First match, over ``[ref - 1 s, ref + 0.3 s]`` (``[ref - 0.5 s, ...]`` for the contact-type modes):

    * ``upright_hit``: the base touched while tilted less than 20 deg (no balance loss: an obstacle under the body);
    * ``slide``: two or more stance feet slid faster than 0.3 m/s at once in 20 % of the frames;
    * ``foot_trap``: a stance foot sank 5 cm below its touchdown, or stayed down while commanded into the swing;
    * ``leg_collision``: a hip, thigh or calf touched something in 20 % of the 0.5 s before the balance loss, or a
      touchdown with a trip-size horizontal force;
    * ``support_loss``: at most one foot down with the base low, or all feet in the air;
    * ``other``.

    Base rates on normal walking (v2 ``random_walk`` windows, a balance loss pretended at the trigger frame), Robust /
    Energy: slide 0 / 0 %, sink or wedge 0.8 / 0 %, sustained leg contact 3.4 / 0.2 % (any leg contact: 15 / 6 %,
    which is why a single touch does not count), trip 1.2 / 0.4 %, one foot or flight 1.2 / 1.2 %.
    """
    if ref == end:
        return "upright_hit"

    def span(name, before_s, after_s=0.3):
        x = labels[name]
        return x[max(ref - round(before_s / dt), 0) : min(ref + round(after_s / dt), end) + 1]

    if span("slip_2feet", 1.0).mean() >= 0.2:
        return "slide"
    if span("sink_5cm", 1.0).any() or span("C_wedge", 1.0).any():
        return "foot_trap"
    if span("A_leg_contact", 0.5, 0.0).mean() >= 0.2 or span("C_trip", 0.5).any():
        return "leg_collision"
    if span("stance_1foot", 0.5).any() or span("C_flight", 0.5).any():
        return "support_loss"
    return "other"


def features(labels: dict[str, np.ndarray], e: dict, ref: int, bins: list[tuple[float, float]], dt: float):
    """Share of frames with each label in each bin (seconds relative to ``ref``), then the tip direction."""
    out, n = [], len(e["root_pos_w"])
    for name in labels:
        x = labels[name].astype(float)
        for a, b in bins:
            lo, hi = max(ref + round(a / dt), 0), min(ref + round(b / dt), n)
            out.append(x[lo:hi].mean() if hi > lo else 0.0)
    return np.concatenate([out, tip_direction(e, n - 1, round(3 / dt))])


def load_windows(paths: list[str], thresholds: dict, triggers: bool, bins, min_s: float):
    """Feature rows of every window of ``paths`` with at least ``min_s`` of history before its reference frame."""
    rows, X, frames = [], [], []
    for path in paths:
        root = os.path.join(path, "triggers") if triggers else path
        if not os.path.isfile(os.path.join(root, "index.csv")):
            continue
        meta, index = analyze.load_collection(root)
        policy, terrain = path.rstrip("/").split("/")[-2:]
        with open(os.path.join(path, "summary.json")) as f:
            trigger = json.load(f).get("event_trigger")
        for r in index:
            n, dt = int(r["num_frames"]), meta["step_dt"]
            if not triggers and r["terminal"] != "True":
                continue
            e = analyze.load_event(root, r)
            # trigger windows: the trigger frame; falls: the balance loss before the base contact
            ref = n // 2 - 1 if triggers else balance_loss(e, n - 1, dt)
            if ref * dt < min_s:
                continue
            masks = analyze.rule_masks(e, meta, thresholds[policy], grace_s=0.0)
            labels = {name: masks[name] for name in LABELS} | privileged(e, meta)
            x = features(labels, e, ref, bins, dt)
            if not triggers:
                # how long the fall takes from the balance loss to the base contact, scaled to [0, 1] over 3 s
                x = np.append(x, (n - 1 - ref) * dt / 3.0)
            X.append(x)
            frames.append(labels)
            tilt = float(np.degrees(np.arccos(-np.clip(e["projected_gravity_b"][ref, 2], -1, 1))))
            rows.append({"policy": policy, "terrain": terrain, "trigger": trigger, "event": int(r["event"]),
                         "mode": "" if triggers else failure_mode(labels, ref, n - 1, dt), "onset": ref,
                         "level": int(r["terrain_level"]), "collection": path, "tilt_ref": tilt})  # fmt: skip
    names = [f"{name}@{a:+.1f}" for name in frames[0] for a, _ in bins] + ["tip_forward", "tip_side"]
    names += [] if triggers else ["fall_duration"]
    return rows, np.array(X), names, frames


def ward_clusters(X: np.ndarray, k_range=range(3, 11), seed: int = 0):
    """PCA to 95 % variance, Ward clustering; k with the best silhouette. Returns labels, k, scores, Z.

    Not standardized: the features are shares of frames (and the tip direction), all on [0, 1]; standardizing
    blows up rare labels, and single outlier windows then take a cluster of their own."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    Z = PCA(n_components=0.95, random_state=seed).fit_transform(X)
    scores = {}
    for k in k_range:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        scores[k] = float(silhouette_score(Z, labels))
    k = max(scores, key=scores.get)
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z), k, scores, Z


def describe(X: np.ndarray, names: list[str], labels: np.ndarray, top: int = 5) -> dict[int, list[str]]:
    """Per cluster, the features most above the overall mean (in overall standard deviations)."""
    mu, sd = X.mean(0), np.maximum(X.std(0), 1e-6)
    out = {}
    for c in np.unique(labels):
        z = (X[labels == c].mean(0) - mu) / sd
        order = np.argsort(-z)[:top]
        out[int(c)] = [f"{names[i]} {X[labels == c, i].mean():.2f} (z {z[i]:+.1f})" for i in order]
    return out


def separability(X: np.ndarray, truth: list[str]) -> dict:
    """How well the features tell the known causes apart: 5-fold cross-validated balanced accuracy and per-cause
    recall of a random forest (the supervised upper reference for the unsupervised clusters)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    y = np.array(truth)
    keep = np.isin(y, [c for c in set(truth) if truth.count(c) >= 5])
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    pred = cross_val_predict(model, X[keep], y[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    classes = sorted(set(y[keep]))
    recall = recall_score(y[keep], pred, labels=classes, average=None)
    return {"balanced_accuracy": float(balanced_accuracy_score(y[keep], pred)), "chance": 1 / len(classes),
            "recall": {c: round(float(r), 2) for c, r in zip(classes, recall)}}  # fmt: skip


def crosstab(rows: list[dict], labels: np.ndarray, key) -> dict:
    out: dict = {}
    for r, c in zip(rows, labels):
        out.setdefault(int(c), {}).setdefault(key(r), 0)
        out[int(c)][key(r)] += 1
    return out


def scatter(Z: np.ndarray, rows: list[dict], clusters: np.ndarray, triggers: bool, path: str) -> None:
    """First two principal components, colored by the known cause, the failure mode (falls) and the cluster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colorings = [("known cause", [r["trigger"] if triggers else f"{r['policy']}/{r['terrain']}" for r in rows])]
    if not triggers:
        colorings.append(("failure mode", [r["mode"] for r in rows]))
    colorings.append(("Ward cluster", [f"cluster {c}" for c in clusters]))
    fig, axes = plt.subplots(1, len(colorings), figsize=(5.5 * len(colorings), 4.5))
    for ax, (title, values) in zip(axes, colorings):
        for i, value in enumerate(sorted(set(values))):
            sel = np.array(values) == value
            ax.scatter(Z[sel, 0], Z[sel, 1], s=8, alpha=0.7, label=f"{value} ({sel.sum()})", color=plt.cm.tab20(i % 20))
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.legend(fontsize=6, markerscale=2)
    axes[0].set_ylabel("PC 2")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--falls", nargs="+", required=True, help="Collections whose falls are clustered.")
    parser.add_argument("--baseline", nargs="+", required=True, help="Collections with normal-walking trigger windows.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from sklearn.metrics import adjusted_rand_score

    os.makedirs(args.out, exist_ok=True)
    # thresholds of the spike rules per policy, from its normal walking: the random_walk trigger windows (no event)
    policies = sorted({p.rstrip("/").split("/")[-2] for p in args.falls})
    thresholds = {
        policy: analyze.baseline_thresholds(
            [os.path.join(p, "triggers") for p in args.baseline if p.rstrip("/").split("/")[-2] == policy],
            99.9, 0.0, 1.0)
        for policy in policies
    }  # fmt: skip
    with open(os.path.join(args.out, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=1)

    # falls: 1.5 s before the balance loss and 0.5 s into it
    fall_bins = [(-1.5 + i * BIN_S, -1.5 + (i + 1) * BIN_S) for i in range(4)]
    trig_bins = [(-0.5 + i * BIN_S, -0.5 + (i + 1) * BIN_S) for i in range(4)]
    result = {}
    # trigger windows per policy (does the feature space separate the known events?), falls of all policies
    runs = [(f"triggers_{p}", trig_bins, True, [c for c in args.falls if f"/{p}/" in c]) for p in policies]
    for kind, bins, triggers, paths in [*runs, ("falls", fall_bins, False, args.falls)]:
        rows, X, names, _ = load_windows(paths, thresholds, triggers, bins, min_s=1.5 if not triggers else 0.5)
        labels, k, scores, Z = ward_clusters(X)
        # the known cause: the trigger (event) of a trigger window, the terrain of a fall
        truth = [r["trigger"] if triggers else r["terrain"] for r in rows]
        result[kind] = {
            "n": len(rows), "k": k, "silhouette": scores,
            "ari_known_cause": adjusted_rand_score(truth, labels),
            "cv_known_cause": separability(X, truth),
            "profiles": describe(X, names, labels),
            "by_cause": crosstab(rows, labels, lambda r: f"{r['policy']}/{r['trigger'] if triggers else r['terrain']}"),
        }  # fmt: skip
        if not triggers:
            modes = np.array([r["mode"] for r in rows])
            result[kind]["ari_mode"] = adjusted_rand_score(modes, labels)
            result[kind]["by_mode"] = crosstab(rows, labels, lambda r: r["mode"])
            result[kind]["mode_by_terrain"] = {
                f"{p}/{t}": dict(Counter(r["mode"] for r in rows if (r["policy"], r["terrain"]) == (p, t)))
                for p, t in sorted({(r["policy"], r["terrain"]) for r in rows})
            }
            print("   failure modes per terrain:")
            for key, counts in result[kind]["mode_by_terrain"].items():
                print(f"     {key}: {counts}")
            print(
                f"   ARI clusters vs modes {result[kind]['ari_mode']:.2f}; clusters by mode: {result[kind]['by_mode']}"
            )
        scatter(Z, rows, labels, triggers, os.path.join(args.out, f"{kind}.png"))
        np.savez_compressed(os.path.join(args.out, f"{kind}.npz"), X=X, Z=Z, labels=labels, names=np.array(names))
        with open(os.path.join(args.out, f"{kind}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[*rows[0], "cluster"])
            w.writeheader()
            w.writerows([r | {"cluster": int(c)} for r, c in zip(rows, labels)])
        cv = result[kind]["cv_known_cause"]
        print(
            f"== {kind}: n {len(rows)}, k {k}, ARI vs known cause {result[kind]['ari_known_cause']:.2f}, "
            f"5-fold balanced accuracy {cv['balanced_accuracy']:.2f} (chance {cv['chance']:.2f})"
        )
        print("   per cause recall:", cv["recall"])
        for c, desc in result[kind]["profiles"].items():
            print(f"  cluster {c} ({(labels == c).sum()}): {result[kind]['by_cause'][c]}")
            print("     " + "; ".join(desc))
    with open(os.path.join(args.out, "taxonomy.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
