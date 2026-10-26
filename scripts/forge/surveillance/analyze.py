# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flag which fault rules (A-E) each saved event window of a collection hits.

The rules are labels for analysis, not recording triggers. Each rule gives a per-frame boolean mask over the
event window (:func:`rule_masks`); ``flags.csv`` stores, per event and rule, whether it hit and the first and
last hit time relative to the event step (seconds, <= 0). Rules are of two kinds:

* fixed: a physical threshold (tilt angle, joint-limit margin, torque clipping, ...), see :data:`FIXED`;
* spike: a signal above the ``--percentile`` (or below ``100 - --percentile``) of its baseline distribution, i.e.
  of normal walking. Baseline frames are those at least ``--baseline_s`` before the event step of the
  ``--baseline`` collections (default: the analyzed one). The thresholds are written to ``thresholds.json``.

Frames within ``grace_s`` of a spawn are never flagged or used as baseline. Needs only numpy::

    python scripts/forge/surveillance/analyze.py <collection> [--baseline <collection> ...]
"""

import argparse
import csv
import json
import math
import os
import warnings

import numpy as np

FIXED = {
    "tilt_deg": 50.0,  # A: the task terminates at 1 rad (57 deg), so a 60 deg limit would never be seen
    "base_height_ratio": 0.5,  # A: base height above terrain below this fraction of the baseline median
    "sustain_s": 0.1,  # B support margin, C wedge, E torque saturation: how long a condition must hold
    "wedge_phase": (0.25, 0.75),  # C wedge: CPG swing part in which the foot is commanded well above the ground
    "wedge_speed": 0.05,  # C wedge: foot speed [m/s] counted as stuck
    "sink_m": 0.02,  # C sinking: foot drop [m] below its touchdown height while in contact
    "track_err": 0.5,  # D tracking: |v_cmd_xy - v_xy| [m/s]
    "yaw_err": 0.5,  # D heading: |w_cmd - w_z| [rad/s]
    "task_sustain_s": 1.0,  # D tracking / heading: how long the error must hold
    "progress_s": 2.0,  # D no progress: span over which displacement is compared with the command
    "progress_min_cmd": 0.5,  # D no progress: commanded distance [m] in the span for the rule to apply
    "progress_ratio": 0.25,  # D no progress: displacement below this fraction of the commanded distance
    "clip_ratio": 0.01,  # E saturation: |computed - applied| torque above this fraction of the effort limit
    "limit_margin": 0.05,  # E joint limit: within this fraction of the joint range from a limit (and past baseline)
    "value_horizon_s": 0.5,  # B value drop: V(t) - V(t - horizon)
}
"""Fixed rule parameters."""

SPIKES = {
    # signal: (rule, tail); "high" flags values above the percentile, "low" below its mirror
    "ang_rate": ("B_ang_rate", "high"),
    "gravity_rate": ("B_gravity_rate", "high"),
    "impact_touchdown": ("B_impact", "high"),
    "impact_free": ("B_impact", "high"),
    "support_margin": ("B_support", "low"),
    "value_change": ("B_value_drop", "low"),
    "action_jump": ("B_action_jump", "high"),
    "touchdown_force_xy": ("C_trip", "high"),
    "foot_slip": ("C_slip", "high"),
    "swing_time_late": ("C_late_touchdown", "high"),
    "touchdown_phase": ("C_early_touchdown", "low"),
    "swing_time_early": ("C_early_touchdown", "low"),
    "flight_time": ("C_flight", "high"),
    "limit_distance": ("E_joint_limit", "low"),
    "pd_error": ("E_pd_error", "high"),
    "power": ("E_power", "high"),
}
"""Percentile-thresholded signals and the rule each one feeds. A signal that is all NaN (e.g. ``touchdown_phase``
without a CPG policy) never hits."""

RULES = [
    "A_base_contact", "A_leg_contact", "A_tilt", "A_low_base",
    "B_ang_rate", "B_gravity_rate", "B_impact", "B_support", "B_value_drop", "B_action_jump",
    "C_trip", "C_slip", "C_late_touchdown", "C_early_touchdown", "C_flight", "C_sinking", "C_wedge",
    "D_tracking", "D_no_progress", "D_heading",
    "E_torque_saturation", "E_joint_limit", "E_pd_error", "E_power",
]  # fmt: skip
"""Rule names, in ``flags.csv`` column order."""


def load_collection(path: str) -> tuple[dict, list[dict]]:
    """Return ``meta.json`` and the ``index.csv`` rows of a collection."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(path, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    return meta, rows


def load_event(path: str, row: dict) -> dict:
    """Return the arrays of one event, plus ``spawn_age`` (frames since the robot's last spawn)."""
    e = dict(np.load(os.path.join(path, "events", f"{int(row['event']):06d}.npz")))
    n = int(row["num_frames"])
    e["spawn_age"] = int(row["episode_step"]) - (n - 1) + np.arange(n)
    return e


def sustained(mask: np.ndarray, k: int) -> np.ndarray:
    """True at frames where ``mask`` has held for the last ``k`` frames (axis 0)."""
    run = np.zeros(mask.shape[1:], dtype=int)
    out = np.zeros_like(mask)
    for t in range(len(mask)):
        run = np.where(mask[t], run + 1, 0)
        out[t] = run >= k
    return out


def hull_margin(p: np.ndarray, pts: np.ndarray) -> float:
    """Signed distance of 2D point ``p`` to the convex hull of ``pts`` (positive inside, NaN without points)."""
    if len(pts) == 0:
        return math.nan
    if len(pts) == 1:
        return -float(np.linalg.norm(p - pts[0]))
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # monotone chain, counter-clockwise
    lower, upper = [], []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    a, b = hull, np.roll(hull, -1, axis=0)
    ab = b - a
    s = np.clip(((p - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-12), 0.0, 1.0)
    dist = np.linalg.norm(a + s[:, None] * ab - p, axis=1).min()
    inside = len(hull) >= 3 and all(cross(a[i], b[i], p) > 0 for i in range(len(hull)))
    return dist if inside else -dist


class Event:
    """Signals of one event window derived from the logged fields."""

    def __init__(self, e: dict, meta: dict):
        self.e, self.meta = e, meta
        self.dt = meta["step_dt"]
        self.n = len(e["root_pos_w"])
        names = meta["contact_body_names"]
        self.feet = [names.index(b) for b in names if b not in meta["trigger_body_names"]]
        self.foot_bodies = [meta["body_names"].index(names[i]) for i in self.feet]
        self.base = names.index(meta["body_names"][0])
        self.legs = [names.index(b) for b in meta["trigger_body_names"] if names.index(b) != self.base]
        # foot contact state as the sensor sees it; touchdown = first contact frame after air
        self.stance = e["contact_time"][:, self.feet] > 0.0
        air = e["contact_air_time"][:, self.feet]
        self.touchdown = np.zeros_like(self.stance)
        self.touchdown[1:] = self.stance[1:] & (air[:-1] > 0.0)
        self.swing_time = np.full(self.stance.shape, np.nan)
        self.swing_time[1:][self.touchdown[1:]] = air[:-1][self.touchdown[1:]]
        phase = next((v for k, v in e.items() if k.startswith("action_") and k.endswith("_phase")), None)
        self.phase = None if phase is None else phase / math.pi  # swing is [0, 1], stance (1, 2)

    def diff(self, x: np.ndarray) -> np.ndarray:
        """Backward difference along time; the first frame is 0."""
        out = np.zeros_like(x)
        out[1:] = x[1:] - x[:-1]
        return out

    def signals(self) -> dict[str, np.ndarray]:
        """Per-frame spike signals (NaN where a signal does not apply)."""
        e, dt = self.e, self.dt
        foot_vel = e["body_lin_vel_w"][:, self.foot_bodies]
        near_td = self.touchdown.any(1)
        near_td = near_td | np.roll(near_td, 1) | np.roll(near_td, -1)
        acc = np.linalg.norm(e["root_acc_w"][:, :3], axis=1)
        k = round(FIXED["value_horizon_s"] / dt)
        value = e.get("value")
        value_change = np.full(self.n, np.nan)
        if value is not None and self.n > k:
            value_change[k:] = value[k:] - value[:-k]
        slip = np.where(self.stance, np.linalg.norm(foot_vel[..., :2], axis=-1), np.nan)
        swing = self.swing_time
        # touchdown: horizontal net foot force (a ratio to the vertical one blows up on first-frame grazes);
        # the CPG phase if the policy has one, else the swing time
        f_xy = np.linalg.norm(e["contact_force_w"][:, self.feet, :2], axis=-1)
        f_xy = np.where(self.touchdown, f_xy, -np.inf).max(1)
        phase = np.full(self.n, np.inf)
        if self.phase is not None:
            phase = np.where(self.touchdown, self.phase, np.inf).min(1)
            swing = np.full_like(swing, np.nan)
        lim = np.asarray(self.meta["joint_pos_limits"])
        q = e["joint_pos"]
        limit_distance = np.minimum(q - lim[:, 0], lim[:, 1] - q) / (lim[:, 1] - lim[:, 0])
        return {
            "ang_rate": np.linalg.norm(e["root_ang_vel_b"], axis=1),
            "gravity_rate": np.linalg.norm(self.diff(e["projected_gravity_b"]), axis=1) / dt,
            "impact_touchdown": np.where(near_td, acc, np.nan),
            "impact_free": np.where(near_td, np.nan, acc),
            "support_margin": self.support_margin(),
            "value_change": value_change,
            "action_jump": np.linalg.norm(self.diff(e["action"]), axis=1),
            # -inf / inf mark frames without a stance foot or touchdown; they never flag nor enter the baseline
            "touchdown_force_xy": f_xy,
            "foot_slip": np.nanmax(np.where(np.isnan(slip), -np.inf, slip), axis=1),
            "swing_time_late": np.nanmax(np.where(np.isnan(self.swing_time), -np.inf, self.swing_time), axis=1),
            "touchdown_phase": phase,
            "swing_time_early": np.nanmin(np.where(np.isnan(swing), np.inf, swing), axis=1),
            "flight_time": np.where(self.stance.any(1), 0.0, e["contact_air_time"][:, self.feet].min(1)),
            "limit_distance": limit_distance,
            "pd_error": np.abs(e["joint_pos_target"] - e["joint_pos"]),
            "power": np.abs(e["applied_torque"] * e["joint_vel"]).sum(1),
        }

    def base_height(self) -> np.ndarray:
        return self.e["root_pos_w"][:, 2] - self.e["base_terrain_height"]

    def support_margin(self) -> np.ndarray:
        """Signed distance [m] of the capture point to the stance feet hull; NaN with no foot down."""
        e = self.e
        mass = np.asarray(self.meta["body_mass"])
        com = (e["body_com_pos_w"] * mass[:, None]).sum(1) / mass.sum()
        com_vel = np.gradient(com, self.dt, axis=0) if self.n > 1 else np.zeros_like(com)
        feet = e["body_pos_w"][:, self.foot_bodies]
        out = np.full(self.n, np.nan)
        for t in range(self.n):
            down = feet[t, self.stance[t]]
            if len(down) == 0:
                continue
            height = max(com[t, 2] - down[:, 2].mean(), 0.05)
            capture = com[t, :2] + com_vel[t, :2] / math.sqrt(9.81 / height)
            out[t] = hull_margin(capture, down[:, :2])
        return out

    def fixed_masks(self) -> dict[str, np.ndarray]:
        """Masks of the fixed-threshold rules (``A_low_base`` needs a threshold and is set in :func:`rule_masks`)."""
        e, meta, dt = self.e, self.meta, self.dt
        sustain = max(round(FIXED["sustain_s"] / dt), 1)
        force = np.linalg.norm(e["contact_normal_force_w"], axis=-1) > meta["args"]["contact_threshold"]
        masks = {
            "A_base_contact": force[:, self.base],
            "A_leg_contact": force[:, self.legs].any(1),
            "A_tilt": e["projected_gravity_b"][:, 2] > -math.cos(math.radians(FIXED["tilt_deg"])),
        }
        # C sinking: foot drops below its touchdown height during one stance
        z = e["body_pos_w"][:, self.foot_bodies, 2]
        ref = z[0].copy()
        sink = np.zeros_like(self.stance)
        for t in range(self.n):
            ref = np.where(self.touchdown[t], z[t], ref)
            sink[t] = self.stance[t] & (ref - z[t] > FIXED["sink_m"])
        masks["C_sinking"] = sink.any(1)
        # C wedge: foot stays down and still while the CPG commands it well into the swing
        if self.phase is not None:
            lo, hi = FIXED["wedge_phase"]
            speed = np.linalg.norm(e["body_lin_vel_w"][:, self.foot_bodies], axis=-1)
            stuck = self.stance & (self.phase > lo) & (self.phase < hi) & (speed < FIXED["wedge_speed"])
            masks["C_wedge"] = sustained(stuck, sustain).any(1)
        else:
            # ponytail: wedge needs a commanded swing; for non-CPG policies derive one from joint_pos_target
            masks["C_wedge"] = np.zeros(self.n, dtype=bool)
        # D: velocity and yaw-rate tracking, progress over a span
        cmd = e.get("command_base_velocity")
        task_sustain = round(FIXED["task_sustain_s"] / dt)
        if cmd is not None:
            v_err = np.linalg.norm(cmd[:, :2] - e["root_lin_vel_b"][:, :2], axis=1)
            w_err = np.abs(cmd[:, 2] - e["root_ang_vel_b"][:, 2])
            masks["D_tracking"] = sustained(v_err > FIXED["track_err"], task_sustain)
            masks["D_heading"] = sustained(w_err > FIXED["yaw_err"], task_sustain)
            k = round(FIXED["progress_s"] / dt)
            stall = np.zeros(self.n, dtype=bool)
            if self.n > k:
                commanded = np.convolve(np.linalg.norm(cmd[:, :2], axis=1) * dt, np.ones(k), "valid")[1:]
                moved = np.linalg.norm(e["root_pos_w"][k:, :2] - e["root_pos_w"][:-k, :2], axis=1)
                stall[k:] = (commanded >= FIXED["progress_min_cmd"]) & (moved < FIXED["progress_ratio"] * commanded)
            masks["D_no_progress"] = stall
        else:
            for name in ("D_tracking", "D_heading", "D_no_progress"):
                masks[name] = np.zeros(self.n, dtype=bool)
        # E: torque clipped by the actuator model (the A2 motor curve clips below the effort limit)
        effort = np.asarray(meta["joint_effort_limits"])
        clipped = np.abs(e["computed_torque"] - e["applied_torque"]) > FIXED["clip_ratio"] * effort
        masks["E_torque_saturation"] = sustained(clipped, sustain).any(1)
        return masks


def baseline_thresholds(paths: list[str], percentile: float, baseline_s: float, grace_s: float) -> dict:
    """Percentile thresholds of every spike signal over the baseline frames of ``paths``."""
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in SPIKES}
    heights = []
    for path in paths:
        meta, rows = load_collection(path)
        old = round(baseline_s / meta["step_dt"])
        grace = round(grace_s / meta["step_dt"])
        for row in rows:
            e = load_event(path, row)
            keep = (np.arange(len(e["spawn_age"])) < len(e["spawn_age"]) - old) & (e["spawn_age"] > grace)
            if not keep.any():
                continue
            ev = Event(e, meta)
            for name, x in ev.signals().items():
                pooled[name].append(x[keep])
            if "base_terrain_height" in e:
                heights.append(ev.base_height()[keep])
    out = {"percentile": percentile, "baseline": paths, "baseline_s": baseline_s, "fixed": FIXED, "spikes": {}}
    for name, chunks in pooled.items():
        x = np.concatenate(chunks)
        x = x.reshape(len(x), -1)
        q = percentile if SPIKES[name][1] == "high" else 100.0 - percentile
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN signals give NaN thresholds, which never hit
            values = np.nanpercentile(np.where(np.isfinite(x), x, np.nan), q, axis=0)
        out["spikes"][name] = values.tolist() if values.size > 1 else float(values[0])
        out.setdefault("baseline_frames", int(len(x)))
    if heights:
        out["base_height_min"] = FIXED["base_height_ratio"] * float(np.median(np.concatenate(heights)))
    return out


def rule_masks(e: dict, meta: dict, thresholds: dict, grace_s: float) -> dict[str, np.ndarray]:
    """Per-frame boolean mask of every rule in :data:`RULES` for one event (frames inside the spawn grace are False)."""
    ev = Event(e, meta)
    masks = ev.fixed_masks()
    masks["A_low_base"] = (
        ev.base_height() < thresholds["base_height_min"]
        if "base_height_min" in thresholds
        else np.zeros(ev.n, dtype=bool)
    )
    for name, x in ev.signals().items():
        rule, tail = SPIKES[name]
        thr = np.asarray(thresholds["spikes"][name])
        with np.errstate(invalid="ignore"):
            hit = x > thr if tail == "high" else x < thr
        if rule == "E_joint_limit":
            # near a limit and nearer than normal walking gets: A2's calves reach their fold limit in every swing
            hit &= x < FIXED["limit_margin"]
        hit = hit.reshape(ev.n, -1).any(1)
        if rule == "B_support":
            hit = sustained(hit, max(round(FIXED["sustain_s"] / ev.dt), 1))
        masks[rule] = masks.get(rule, np.zeros(ev.n, dtype=bool)) | hit
    valid = e["spawn_age"] > round(grace_s / ev.dt)
    return {rule: masks[rule] & valid for rule in RULES}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("collection", help="Collection directory written by collect.py.")
    parser.add_argument("--baseline", nargs="+", default=None, help="Collections for the normal-walking baseline.")
    parser.add_argument("--baseline_s", type=float, default=2.0, help="Baseline frames: this long before the event.")
    parser.add_argument("--percentile", type=float, default=99.9, help="Spike threshold percentile.")
    parser.add_argument("--grace_s", type=float, default=None, help="Spawn grace (default: the collection's).")
    parser.add_argument("--thresholds", default=None, help="Reuse a thresholds.json instead of a baseline.")
    parser.add_argument(
        "--rules", nargs="+", default=[""], help="Flag only rules starting with these, e.g. A_tilt C (default: all)."
    )
    args = parser.parse_args()
    rules = [rule for rule in RULES if rule.startswith(tuple(args.rules))]

    meta, rows = load_collection(args.collection)
    grace_s = meta["args"]["grace_s"] if args.grace_s is None else args.grace_s
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)
    else:
        baseline = args.baseline or [args.collection]
        thresholds = baseline_thresholds(baseline, args.percentile, args.baseline_s, grace_s)
    with open(os.path.join(args.collection, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    dt = meta["step_dt"]
    header = ["event", "terminal", "terminations", "num_frames"]
    header += [f"{r}{s}" for r in rules for s in ("", "_first_s", "_last_s")]
    second = round(1.0 / dt)
    table = []
    # summary counts: rule hit in the window, in its last second, and in the first second of full windows
    # (normal walking 4-5 s before the event, the reference rate of a rule)
    count = {rule: np.zeros(3, dtype=int) for rule in rules}
    num_full = 0
    for row in rows:
        e = load_event(args.collection, row)
        masks = rule_masks(e, meta, thresholds, grace_s)
        n = len(e["spawn_age"])
        full = n == meta["num_frames"]
        num_full += full
        out = [row["event"], row["terminal"], row["terminations"], n]
        for rule in rules:
            hits = np.flatnonzero(masks[rule])
            out += [int(len(hits) > 0)]
            out += [f"{(hits[0] - n + 1) * dt:.2f}", f"{(hits[-1] - n + 1) * dt:.2f}"] if len(hits) else ["", ""]
            count[rule] += [len(hits) > 0, masks[rule][-second:].any(), full and masks[rule][:second].any()]
        table.append(out)
    with open(os.path.join(args.collection, "flags.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)

    print(f"[analyze] {len(table)} events ({num_full} full windows), baseline {thresholds['baseline_frames']} frames")
    print(f"{'rule':22s} {'window':>7s} {'last 1 s':>9s} {'first 1 s':>10s}")
    for rule in rules:
        c = count[rule]
        print(f"{rule:22s} {c[0] / len(table):7.1%} {c[1] / len(table):9.1%} {c[2] / max(num_full, 1):10.1%}")
    print(f"[analyze] wrote flags.csv and thresholds.json to {args.collection}")


if __name__ == "__main__":
    main()
