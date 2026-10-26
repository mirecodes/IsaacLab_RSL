# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exploration: find a motor fault (joint, or leg) within 3 policy steps (60 ms) of its onset.

Scratch study next to the main pipeline (see ``REPORT.md`` here for the hypotheses and the results). Reuses
``detector.py`` (data, training, alarm statistics) on the PD-rate datasets; adds physics features, a joint-shared
model, a nominal-dynamics residual detector, a motor-current tier and a CUSUM decision rule.

Selection discipline: ``run`` without ``--test`` evaluates on the validation split only (the test sets are not even
loaded); a configuration goes to ``--test`` once, after it was chosen on validation. ::

    python scripts/forge/fault_detector/tmp/explore.py run OUT B0=mlp/proprio P1=mlp/phys J1=joint/phys
    python scripts/forge/fault_detector/tmp/explore.py run OUT J1=joint/phys --test --seeds 0 1 2
    python scripts/forge/fault_detector/tmp/explore.py summarize OUT
    python scripts/forge/fault_detector/tmp/explore.py selftest
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import detector as fd  # noqa: E402
import experiment as ex  # noqa: E402

DEADLINE_MS = 60.0
"""3 policy steps; latency counts from the first faulty sample, as everywhere in the study."""
TARGET = 0.5
"""False alarms per minute, set on the validation split."""
FA_TARGETS = (0.25, 0.5, 1.0, 2.0)
TORQUE_NOISE = 1.0
"""Motor-current torque reading noise, uniform half-width [N m] (A2 joints: 120 / 180 N m peak)."""
# PD gains and DC-motor limits of the A2 robust teacher (dataset env.yaml, actuator ``base_legs``):
# kind -> (stiffness, damping, saturation = effort limit, velocity limit)
GAINS = {"hip": (100.0, 4.0, 120.0, 22.0), "thigh": (100.0, 4.0, 120.0, 22.0), "calf": (150.0, 6.0, 180.0, 14.67)}

fd.INPUTS["applied_torque"] = ("analysis/applied_torque", "pd/applied_torque", None, TORQUE_NOISE)
RAW = fd.PROPRIO + ["tracking_error", "applied_torque"]
GLOBAL = ("base_ang_vel", "projected_gravity", "last_action", "velocity_command")
FEATURES = {
    "proprio": ("q", "qd"),
    "phys": ("q", "qd", "e", "tau", "qdd"),
    "current": ("q", "qd", "e", "tau", "qdd", "tau_meas", "tau_res"),
}
"""Per-joint features: position and velocity (rel. default), tracking error q* - q, PD torque command
clip_DC(Kp e - Kd qd), finite-difference acceleration; with a motor-current sensor the measured torque (previous
substep) and its residual against the previous command."""
BASE = {"dataset_dir": "logs/fault_detector/data/main_pd", "rate": "pd", "history_length": 5,
        "ignore_after_onset_s": 0.0, "hnm_epochs": 5,
        "test_dirs": ("logs/fault_detector/data/test_id_pd", "logs/fault_detector/data/test_ood_pd")}  # fmt: skip


def columns(names: list[str], term: str) -> list[int]:
    return [i for i, n in enumerate(names) if n.split("[")[0] == term]


def joint_gains(joint_names: list[str], device) -> torch.Tensor:
    """[4, J]: stiffness, damping, saturation, velocity limit."""
    return torch.tensor([GAINS[n.split("_")[1]] for n in joint_names], device=device).T.contiguous()


def dc_clip(tau: torch.Tensor, qd: torch.Tensor, sat: torch.Tensor, vlim: torch.Tensor) -> torch.Tensor:
    """Isaac Lab's DC-motor torque-speed clipping (effort limit = saturation here)."""
    hi = (sat * (1.0 - qd / vlim)).clamp(min=0.0).minimum(sat)
    lo = (sat * (-1.0 - qd / vlim)).clamp(max=0.0).maximum(-sat)
    return torch.maximum(torch.minimum(tau, hi), lo)


class FeatureSource(fd.WindowSource):
    """Windows of [global terms, per-joint features (feature-major: f blocks of J)]; noise goes on the raw
    measurements first, so derived features carry it as a real estimator would."""

    def __init__(self, data, cfg, device, mean=None, std=None, features="phys", joint_names=()):
        super().__init__(data, cfg, device, 0.0, 1.0)
        self.g_cols = torch.tensor(sum((columns(data.names, n) for n in GLOBAL), []), device=device)
        self.cols = {n: torch.tensor(columns(data.names, n), device=device) for n in RAW}
        self.per_joint = FEATURES[features]
        self.num_joints = self.mask.shape[1]
        self.g, self.f = len(self.g_cols), len(self.per_joint)
        self.dim = self.g + self.f * self.num_joints
        self.kp, self.kd, self.sat, self.vlim = joint_gains(list(joint_names), device)
        if mean is None:
            gen = torch.Generator(device=device).manual_seed(0)
            rows = torch.randint(len(self.t), (1 << 17,), device=device, generator=gen)
            z = torch.cat([self.features(self.raw_windows(r, cfg.train_noise, gen)).flatten(0, 1)
                           for r in rows.split(8192)])  # fmt: skip
            mean, std = z.mean(0), z.std(0).clamp(min=1e-3)
        self.mean, self.std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)

    def raw_windows(self, rows, noise: bool, generator=None) -> torch.Tensor:
        """[B, L + 1, D_raw]: one extra sample for the differences."""
        x = self.x[fd.window_rows(rows, self.start[rows], self.length + 1)].float()
        if noise:
            x = x + (torch.rand(x.shape, device=self.device, generator=generator) * 2 - 1) * self.noise
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L + 1, D_raw] -> [B, L, dim]."""
        c = lambda n: x[..., self.cols[n]]  # noqa: E731
        q, qd, e = c("joint_pos_rel"), c("joint_vel"), c("tracking_error")
        tau = dc_clip(self.kp * e - self.kd * qd, qd, self.sat, self.vlim)
        per = {"q": q, "qd": qd, "e": e, "tau": tau,
               "qdd": torch.diff(qd, dim=1, prepend=qd[:, :1]) / self.data.sample_dt}  # fmt: skip
        if "tau_meas" in self.per_joint:
            # sample n holds the torque of substep n - 1, computed from the state of sample n - 1
            per["tau_meas"] = c("applied_torque")
            per["tau_res"] = per["tau_meas"] - torch.cat([tau[:, :1], tau[:, :-1]], dim=1)
        joint = torch.stack([per[k] for k in self.per_joint], dim=2).flatten(2)
        return torch.cat([x[..., self.g_cols], joint], dim=2)[:, 1:]

    def windows(self, rows, noise: bool, generator=None) -> torch.Tensor:
        z = self.features(self.raw_windows(rows, noise, generator))
        return ((z - self.mean) / self.std).clamp(-self.cfg.clip, self.cfg.clip)


def mlp(dims: list[int]) -> nn.Sequential:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    return nn.Sequential(*layers)


class JointNet(nn.Module):
    """One network shared by the 12 joints: a joint's logit from its own feature window, its two leg mates'
    windows, a global-context embedding and a joint one-hot. Type head on the joint-softmax-pooled features."""

    def __init__(self, cfg, input_dim, num_joints, window, g: int, f: int, legs: np.ndarray):
        super().__init__()
        self.g, self.f, self.num_joints, self.window = g, f, num_joints, window
        mates = [[k for k in range(num_joints) if legs[k] == legs[j] and k != j] for j in range(num_joints)]
        self.register_buffer("mates", torch.tensor(mates))
        self.register_buffer("onehot", torch.eye(num_joints))
        self.context = mlp([window * g, 128, 64])
        self.joint_net = mlp([3 * window * f + 64 + num_joints, 128, 64])
        self.joint_head = nn.Linear(64, 1)
        self.type_head = nn.Linear(64, len(fd.FAULT_TYPES))

    def forward(self, x):
        b, n = x.shape[0], self.num_joints
        ctx = self.context(x[..., : self.g].flatten(1))
        per = x[..., self.g :].reshape(b, self.window, self.f, n).permute(0, 3, 1, 2).flatten(2)  # [B, J, L f]
        z = torch.cat([per, per[:, self.mates[:, 0]], per[:, self.mates[:, 1]], ctx.unsqueeze(1).expand(b, n, -1),
                       self.onehot.expand(b, n, n)], dim=2)  # fmt: skip
        h = self.joint_net(z)
        logits = self.joint_head(h).squeeze(-1)
        pooled = (torch.softmax(logits, dim=1).unsqueeze(-1) * h).sum(1)
        return logits, self.type_head(pooled)


# ---------------------------------------------------------------------------------------------------------------------
# Nominal-dynamics residual (trained on nominal samples only: no fault labels)


class ResidualNet(nn.Module):
    """Predicts every joint's next velocity sample from the window ending one sample earlier."""

    def __init__(self, input_dim, num_joints, window):
        super().__init__()
        self.net = nn.Sequential(mlp([input_dim * window, 256, 128]), nn.Linear(128, num_joints))

    def forward(self, x):
        return self.net(x.flatten(1))


def residual_target(src: FeatureSource, rows, noise: bool, gen):
    """Measured (noisy) joint velocity at ``rows`` in the source's normalized units, and the window input."""
    prev = torch.maximum(rows - 1, src.start[rows])
    qd = src.x[rows][:, src.cols["joint_vel"]].float()
    if noise:
        qd = qd + (torch.rand(qd.shape, device=src.device, generator=gen) * 2 - 1) * src.noise[src.cols["joint_vel"]]
    k = src.g + src.per_joint.index("qd") * src.num_joints
    sl = slice(k, k + src.num_joints)
    return src.windows(prev, noise, gen), (qd - src.mean[sl]) / src.std[sl]


def train_residual(cfg, src: FeatureSource, val: FeatureSource, device, log, epochs: int = 15) -> dict:
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    nominal = lambda s: ((s.mask.sum(1) == 0) & (s.t >= s.seconds(cfg.skip_initial_s))).nonzero().squeeze(-1)  # noqa
    rows, val_rows = nominal(src), nominal(val)
    val_rows = val_rows[torch.randperm(len(val_rows), device=device, generator=gen)[: cfg.val_windows // 4]]
    net = ResidualNet(src.dim, src.num_joints, src.length).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    iters = cfg.samples_per_epoch // cfg.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iters)
    gv = torch.Generator(device=device).manual_seed(12345)
    vx, vy = residual_target(val, val_rows, cfg.train_noise, gv)
    best, best_loss, history = None, float("inf"), []
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(iters):
            x, y = residual_target(src, rows[torch.randint(len(rows), (cfg.batch_size,), device=device, generator=gen)],
                                   cfg.train_noise, gen)  # fmt: skip
            loss = nn.functional.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            total += loss.item()
        net.eval()
        with torch.no_grad():
            val_loss = sum(nn.functional.mse_loss(net(a), b, reduction="sum").item()
                           for a, b in zip(vx.split(65536), vy.split(65536))) / vy.numel()  # fmt: skip
        history.append({"epoch": epoch, "train_mse": total / iters, "val_mse": val_loss})
        log(f"[residual] epoch {epoch}: mse {total / iters:.4f}, val mse {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best = val_loss, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best)
    with torch.no_grad():
        res = torch.cat([net(a) - b for a, b in zip(vx.split(65536), vy.split(65536))])
    return {"model": best, "sigma": res.std(0).cpu(), "history": history, "best_val_mse": best_loss,
            "num_params": fd.num_params(net)}  # fmt: skip


@torch.no_grad()
def residual_scores(net, sigma, src: FeatureSource, noise: bool, seed: int = 1, batch: int = 65536) -> np.ndarray:
    """Squared normalized velocity residual per sample and joint (nominal ~ chi-square with 1 dof)."""
    net.eval()
    gen = torch.Generator(device=src.device).manual_seed(seed)
    sigma = sigma.to(src.device)
    out = []
    for rows in torch.arange(len(src.t), device=src.device).split(batch):
        x, y = residual_target(src, rows, noise, gen)
        out.append((((y - net(x)) / sigma) ** 2).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Decision rules and the 60 ms score


def cusum(data: fd.Episodes, z: np.ndarray, drift: float, skip: int, device) -> np.ndarray:
    """Per-joint one-sided CUSUM ``C = max(0, C + z - drift)``, from 0 at sample ``skip`` of every episode."""
    num_t = int(data.ep_len.max())
    offs = np.arange(num_t)
    valid = offs[None] < data.ep_len[:, None]
    rows = torch.from_numpy(np.where(valid, data.ep_start[:, None] + offs[None], 0)).to(device)
    zt = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(device)
    out = torch.zeros(*rows.shape, z.shape[1], device=device)
    c = torch.zeros(rows.shape[0], z.shape[1], device=device)
    for t in range(skip, num_t):
        c = (c + zt[rows[:, t]] - drift).clamp(min=0.0)
        out[:, t] = c
    return out[torch.from_numpy(valid).to(device)].cpu().numpy()


def alarms(stat: np.ndarray, thr: float, k: int) -> np.ndarray:
    return fd.held(stat.max(axis=1) > thr, k)


def fa_rate(data, stat, thr, k, skip) -> float:
    return fd.alarm_stats(data, alarms(stat, thr, k).astype(np.float32), 0.5, 1, skip)["fa_per_min"]


def threshold(data, stat, k, skip, target) -> float:
    """Lowest threshold whose false alarms stay at or under ``target`` per minute (bisection)."""
    armed = stat.max(axis=1)[data.t >= skip]
    lo, hi = float(armed.min()), float(armed.max()) + 1e-3
    for _ in range(40):
        mid = (lo + hi) / 2
        if fa_rate(data, stat, mid, k, skip) > target:
            lo = mid
        else:
            hi = mid
    return hi


def score_rows(data, stat, thr, k, legs, skip) -> tuple[dict, list[dict]]:
    """Closed-loop summary + 60 ms successes of the alarm ``max_j stat > thr`` held ``k`` samples; the diagnosis is
    the argmax joint of ``stat`` at the alarm sample."""
    on = alarms(stat, thr, k)
    prob = np.zeros(stat.shape, np.float32)
    idx = np.flatnonzero(on)
    prob[idx, stat[idx].argmax(axis=1)] = 1.0
    summary, rows = fd.closed_loop(data, prob, legs, skip, 0.5, 1)
    del prob
    lat = np.array([r["latency_ms"] if r["latency_ms"] is not None else np.inf for r in rows])
    ok_j = np.array([r["correct_joint"] for r in rows], bool)
    ok_l = np.array([r["correct_leg"] for r in rows], bool)
    within = lat <= DEADLINE_MS + 1e-6
    summary.update({"detect_60ms": float(within.mean()), "success_joint_60ms": float((within & ok_j).mean()),
                    "success_leg_60ms": float((within & ok_l).mean())})  # fmt: skip
    for r, w in zip(rows, within):
        r["within_60ms"] = bool(w)
    return summary, rows


def candidate_rules(val, z, skip) -> list[dict]:
    """Debounce k on the raw score, and per-joint CUSUM at three drifts (nominal quantiles of the validation
    score; fixed from then on)."""
    nominal = ~val.mask.any(axis=1) & (val.t >= skip)
    rules = [{"rule": f"debounce_{k}", "k": k, "drift": None} for k in (1, 2, 4, 8)]
    for q in (0.9, 0.99, 0.999):
        rules.append({"rule": f"cusum_q{q}", "k": 1, "drift": float(np.quantile(z[nominal][::7], q))})
    return rules


def statistic(data, z, rule: dict, skip, device) -> np.ndarray:
    return z if rule["drift"] is None else cusum(data, z, rule["drift"], skip, device)


def choose_rules(val, z, legs, skip, device) -> tuple[dict, dict]:
    """On validation: per objective (joint, leg), the rule with the most 60 ms successes at ``TARGET``, and the
    trade-off of the joint rule over ``FA_TARGETS``. Also returns the rules' statistics on ``val``."""
    results, stats = [], {}
    for rule in candidate_rules(val, z, skip):
        stat = stats[rule["rule"]] = statistic(val, z, rule, skip, device)
        thr = threshold(val, stat, rule["k"], skip, TARGET)
        # the rule last: closed_loop's summary has its own k (1, the alarm is already held)
        results.append({**score_rows(val, stat, thr, rule["k"], legs, skip)[0], **rule, "thr": thr})
    best = {obj: max(results, key=lambda r: r[f"success_{obj}_60ms"]) for obj in ("joint", "leg")}
    trade = []
    stat, k = stats[best["joint"]["rule"]], best["joint"]["k"]
    for target in FA_TARGETS:
        thr = threshold(val, stat, k, skip, target)
        trade.append({"target": target, "thr": thr, **score_rows(val, stat, thr, k, legs, skip)[0]})
    return {"rules": results, "best": best, "trade_off": trade}, stats


def missing_impulse(data: fd.Episodes, joint_names) -> np.ndarray:
    """Per faulty episode: the torque the fault removed over its first 60 ms [N m s], from the clean state:
    sum over the faulty substeps of (1 - h) |clip_DC(tau_PD)| dt. Near 0: physically silent within the deadline."""
    gains = joint_gains(joint_names, "cpu").numpy()
    kp, kd, sat, vlim = gains
    q_cols, e_cols = columns(data.names, "joint_vel"), columns(data.names, "tracking_error")
    n = int(round(DEADLINE_MS / 1000 / data.sample_dt))
    out = []
    for e in np.flatnonzero(data.onset < data.ep_len):
        ep = data.episodes[e]
        j, h = int(ep["fault_joint"]), float(ep["health"])
        lo = data.ep_start[e] + max(data.onset[e] - 1, 0)
        hi = min(lo + n, data.ep_start[e] + data.ep_len[e])
        qd = data.x[lo:hi, q_cols[j]].astype(np.float32)
        tau = kp[j] * data.x[lo:hi, e_cols[j]].astype(np.float32) - kd[j] * qd
        tau = dc_clip(torch.from_numpy(tau), torch.from_numpy(qd), torch.tensor(sat[j]), torch.tensor(vlim[j])).numpy()
        out.append((1 - h) * np.abs(tau).sum() * data.sample_dt)
    return np.array(out)


def breakdown(rows: list[dict], impulse: np.ndarray, edges) -> dict:
    """60 ms success by fault type, stance / swing and missing-impulse bin (edges from the validation split)."""
    ok = lambda sel, key: float(np.mean([r[key] and r["within_60ms"] for r, s in zip(rows, sel) if s])) \
        if any(sel) else None  # noqa: E731
    out = {}
    for name, sel in {
        "power_out": [r["fault_type"] == "power_out" for r in rows],
        "power_loss": [r["fault_type"] == "power_loss" for r in rows],
        "stance": [r["stance_at_onset"] for r in rows],
        "swing": [not r["stance_at_onset"] for r in rows],
        **{f"impulse[{lo:.3g},{hi:.3g})": list((impulse >= lo) & (impulse < hi)) for lo, hi in zip(edges, edges[1:])},
    }.items():
        out[name] = {"count": int(sum(sel)), "joint": ok(sel, "correct_joint"), "leg": ok(sel, "correct_leg")}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Runs


@dataclasses.dataclass
class Spec:
    name: str
    model: str  # mlp | joint | residual | ckpt
    features: str  # a FEATURES key, or a checkpoint path for ckpt


def parse_spec(text: str) -> Spec:
    name, rest = text.split("=", 1)
    model, features = rest.split("/", 1)
    return Spec(name, model, features)


def scores(spec, state, cfg, data, device, joint_names, legs, noise: bool) -> np.ndarray:
    """Per-sample, per-joint score of one trained run on ``data`` (logits, or squared residuals)."""
    if spec.model == "ckpt":
        model, ccfg = fd.load_detector(state, device)
        cols = [data.names.index(n) for n in state["input_names"]]
        src = fd.WindowSource(ex.subset(data, cols), ccfg, device, state["mean"], state["std"])
        prob = fd.predict(model, src, noise, seed=1)[0]
        return np.log(np.clip(prob, 1e-7, 1)) - np.log(np.clip(1 - prob, 1e-7, 1))
    src = FeatureSource(data, cfg, device, state["mean"], state["std"], spec.features, joint_names)
    if spec.model == "residual":
        net = ResidualNet(src.dim, src.num_joints, src.length).to(device)
        net.load_state_dict(state["model"])
        return residual_scores(net, state["sigma"], src, noise)
    model = build(spec, cfg, src, legs).to(device)
    fd.load_state(model, state["model"])
    prob = fd.predict(model, src, noise, seed=1)[0]
    return np.log(np.clip(prob, 1e-7, 1)) - np.log(np.clip(1 - prob, 1e-7, 1))


def build(spec, cfg, src, legs):
    if spec.model == "joint":
        return JointNet(cfg, src.dim, src.num_joints, src.length, src.g, src.f, legs)
    return fd.Detector(cfg, src.dim, src.num_joints, src.length)


def train_run(spec, cfg, train_data, val_data, device, joint_names, legs, log) -> dict:
    if spec.model == "ckpt":
        return torch.load(spec.features, map_location="cpu", weights_only=False)
    source = functools.partial(FeatureSource, features=spec.features, joint_names=joint_names)
    if spec.model == "residual":
        src = source(train_data, cfg, device)
        val = source(val_data, cfg, device, src.mean, src.std)
        state = train_residual(cfg, src, val, device, log)
        state.update(mean=src.mean.cpu(), std=src.std.cpu())
        return state
    g, f = len(sum((columns(val_data.names, n) for n in GLOBAL), [])), len(FEATURES[spec.features])
    if spec.model == "joint":
        make = lambda c, d, n, w: JointNet(c, d, n, w, g, f, legs)  # noqa: E731
    else:
        make = fd.Detector
    return fd.train(cfg, train_data, val_data, device, log, source=source, build=make)


def evaluate(spec, state, cfg, val, tests, device, joint_names, legs, log) -> tuple[dict, dict]:
    skip = round(cfg.skip_initial_s / val.sample_dt)
    t0 = time.time()
    z = scores(spec, state, cfg, val, device, joint_names, legs, True)
    chosen, stats = choose_rules(val, z, legs, skip, device)
    impulse = missing_impulse(val, joint_names)
    edges = [0.0, *np.quantile(impulse, [0.25, 0.5, 0.75]).tolist(), float("inf")]
    val_out = {**chosen, "impulse_edges": edges}
    for obj in ("joint", "leg"):
        b = chosen["best"][obj]
        _, rows = score_rows(val, stats[b["rule"]], b["thr"], b["k"], legs, skip)
        val_out[f"breakdown_{obj}"] = breakdown(rows, impulse, edges)
    del stats
    if "tau_res" in FEATURES.get(spec.features, ()):
        val_out["torque_consistency"] = torque_consistency(val, cfg, device, joint_names)
    best = chosen["best"]
    parts = [f"{o} {best[o]['rule']} S60 {best[o][f'success_{o}_60ms']:.3f}" for o in ("joint", "leg")]
    log(f"[eval] {spec.name} val: {', '.join(parts)} ({time.time() - t0:.0f} s)")
    test_out = {}
    for name, data in tests.items():
        imp = missing_impulse(data, joint_names)
        test_out[name] = {}
        for noise in (True, False):
            z = scores(spec, state, cfg, data, device, joint_names, legs, noise)
            # the validation rule, drift and threshold, unchanged
            stats = {b["rule"]: statistic(data, z, b, skip, device) for b in chosen["best"].values()}
            entry = {}
            for obj in ("joint", "leg"):
                b = chosen["best"][obj]
                summary, rows = score_rows(data, stats[b["rule"]], b["thr"], b["k"], legs, skip)
                entry[obj] = {"rule": b["rule"], **summary, "breakdown": breakdown(rows, imp, edges)}
                if noise and obj == "joint":
                    ex.write_csv(os.path.join(state["_dir"], f"episodes_{name}.csv"), rows)
            entry["trade_off"] = []
            stat, k = stats[chosen["best"]["joint"]["rule"]], chosen["best"]["joint"]["k"]
            for t in chosen["trade_off"]:
                entry["trade_off"].append({"target": t["target"], **score_rows(data, stat, t["thr"], k, legs, skip)[0]})
            test_out[name]["noise_on" if noise else "noise_off"] = entry
            log(f"[eval] {spec.name} {name} noise {noise}: S60 joint {entry['joint']['success_joint_60ms']:.3f}, "
                f"leg {entry['leg']['success_leg_60ms']:.3f}, FA/min {entry['joint']['fa_per_min']:.2f}")
    return val_out, test_out


def torque_consistency(data, cfg, device, joint_names) -> dict:
    """Pipeline check of the current tier on clean nominal samples: |tau_meas - clip_DC(tau_PD)(n - 1)| [N m]."""
    src = FeatureSource(data, cfg, device, 0.0, 1.0, "current", joint_names)
    rows = ((src.mask.sum(1) == 0) & (src.t >= 4)).nonzero().squeeze(-1)[::97]
    k = src.g + src.per_joint.index("tau_res") * src.num_joints
    res = src.features(src.raw_windows(rows, False))[:, -1, k : k + src.num_joints].abs()
    return {"median": res.median().item(), "p99": res.flatten().quantile(0.99).item(), "samples": len(rows)}


def cmd_run(args) -> None:
    cfg0 = fd.DetectorCfg(**{**BASE, **ex.parse_overrides(args.overrides)})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = fd.read_meta(cfg0.dataset_dir)
    joint_names = meta["joint_names"]
    legs = ex.joint_legs(joint_names)
    split = fd.split_episodes(fd.read_index(cfg0.dataset_dir), cfg0.split)
    load = lambda d, ids=None: fd.load_episodes(d, RAW, ids, cfg0.rate)  # noqa: E731
    specs = [parse_spec(s) for s in args.specs]
    trains = any(s.model != "ckpt" for s in specs)
    train_data = load(cfg0.dataset_dir, split["train"]) if trains else None
    val_data = load(cfg0.dataset_dir, split["val"])
    tests = {ex.set_name(d): load(d) for d in cfg0.test_dirs} if args.test else {}
    os.makedirs(args.out, exist_ok=True)
    for spec in specs:
        for seed in args.seeds:
            cfg = dataclasses.replace(cfg0, seed=seed)
            run_dir = os.path.join(args.out, f"{spec.name}_s{seed}")
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.log"), "a") as log_file:

                def log(msg):
                    print(msg, flush=True)
                    log_file.write(msg + "\n")
                    log_file.flush()

                path = os.path.join(run_dir, "checkpoint.pt")
                if os.path.exists(path):
                    state = torch.load(path, map_location="cpu", weights_only=False)
                else:
                    state = train_run(spec, cfg, train_data, val_data, device, joint_names, legs, log)
                    if spec.model != "ckpt":
                        torch.save(state, path)
                state["_dir"] = run_dir
                val_out, test_out = evaluate(spec, state, cfg, val_data, tests, device, joint_names, legs, log)
                info = {"spec": dataclasses.asdict(spec), "seed": seed, "num_params": state.get("num_params"),
                        "cfg": dataclasses.asdict(cfg)}  # fmt: skip
                fd.save_json(os.path.join(run_dir, "val.json"), {**info, **val_out})
                if test_out:
                    fd.save_json(os.path.join(run_dir, "test.json"), {**info, **test_out})


def fmt(values, digits=3) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "–"
    return f"{np.mean(values):.{digits}f}" + (f" ± {np.std(values):.{digits}f}" if len(values) > 1 else "")


def cmd_summarize(args) -> None:
    runs: dict[str, list] = {}
    for path in sorted(glob.glob(os.path.join(args.out, "*", "val.json"))):
        with open(path) as f:
            v = json.load(f)
        t = None
        t_path = path.replace("val.json", "test.json")
        if os.path.exists(t_path):
            with open(t_path) as f:
                t = json.load(f)
        runs.setdefault(v["spec"]["name"], []).append((v, t))
    lines = ["| Run | Model / features | Params | Seeds | val S60 joint (rule) | val S60 leg (rule) | "
             "test_id S60 joint | test_id S60 leg | test_id FA/min | test_id S60 joint, noise off | "
             "test_ood S60 joint / leg |", "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, entries in runs.items():
        v0 = entries[0][0]
        val_j = [v["best"]["joint"]["success_joint_60ms"] for v, _ in entries]
        val_l = [v["best"]["leg"]["success_leg_60ms"] for v, _ in entries]
        tests = [t for _, t in entries if t]
        tid = lambda key, obj, noise="noise_on": [t["test_id"][noise][obj][key] for t in tests]  # noqa: E731
        ood = lambda obj: [t["test_ood"]["noise_on"][obj][f"success_{obj}_60ms"] for t in tests if "test_ood" in t]  # noqa
        spec, best = v0["spec"], v0["best"]
        lines.append(
            f"| {name} | {spec['model']} / {os.path.basename(spec['features'])} | {v0.get('num_params') or '–'} | "
            f"{len(entries)} | {fmt(val_j)} ({best['joint']['rule']}) | {fmt(val_l)} ({best['leg']['rule']})"
            f" | {fmt(tid('success_joint_60ms', 'joint'))} | {fmt(tid('success_leg_60ms', 'leg'))} | "
            f"{fmt(tid('fa_per_min', 'joint'), 2)} | {fmt(tid('success_joint_60ms', 'joint', 'noise_off'))} | "
            f"{fmt(ood('joint'))} / {fmt(ood('leg'))} |"
        )
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    print(text)


def cmd_selftest(_args) -> None:
    """CUSUM against a plain loop, and every model kind end to end on the synthetic dataset of test_detector."""
    import test_detector as td

    rng = np.random.default_rng(0)
    data = fd.Episodes(x=None, mask=None, ftype=None, since=None, t=np.concatenate([np.arange(5), np.arange(7)]),
                       start=None, contact=None, command=None, ep_start=np.array([0, 5]), ep_len=np.array([5, 7]),
                       onset=None, episodes=[], noise=None, names=[], sample_dt=0.005, per_step=4)  # fmt: skip
    z = rng.normal(size=(12, 2)).astype(np.float32)
    got = cusum(data, z, 0.3, 1, "cpu")
    for s, n in ((0, 5), (5, 7)):
        c = np.zeros(2)
        for t in range(n):
            c = np.maximum(0, c + z[s + t] - 0.3) if t >= 1 else np.zeros(2)
            assert np.allclose(got[s + t], c, atol=1e-6), (s, t)
    with tempfile.TemporaryDirectory() as root:
        td.make_dataset(root, num_episodes=400)  # >= 1 validation episode per (type, joint) stratum
        out = os.path.join(root, "out")
        overrides = [f"dataset_dir={root}", "test_dirs=[]", "epochs=1", "samples_per_epoch=8192", "batch_size=512",
                     "val_windows=2048", "hnm_epochs=1", "skip_initial_s=0.05", "onset_window_s=0.1"]  # fmt: skip
        # the synthetic joint names are the real ones (meta.json), so the gains resolve
        args = argparse.Namespace(out=out, specs=["M=mlp/phys", "J=joint/current", "R=residual/phys"], seeds=[0],
                                  test=False, overrides=overrides)  # fmt: skip
        cmd_run(args)
        for name in ("M", "J", "R"):
            with open(os.path.join(out, f"{name}_s0", "val.json")) as f:
                v = json.load(f)
            assert v["best"]["joint"]["fa_per_min"] <= TARGET + 1e-9, (name, v["best"]["joint"])
            assert all(r["k"] == int(r["rule"].split("_")[1]) for r in v["rules"] if r["drift"] is None), name
        test_root = os.path.join(root, "test_id")
        td.make_dataset(test_root, num_episodes=60, seed=1)
        args.specs, args.test = ["J=joint/current"], True
        args.overrides = [o for o in overrides if not o.startswith("test_dirs")] + [f'test_dirs=["{test_root}"]']
        cmd_run(args)
        with open(os.path.join(out, "J_s0", "test.json")) as f:
            t = json.load(f)["test_id"]
        assert set(t) == {"noise_on", "noise_off"} and len(t["noise_on"]["trade_off"]) == len(FA_TARGETS)
        cmd_summarize(argparse.Namespace(out=out))
    print("selftest passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("out")
    p.add_argument("specs", nargs="+", help="name=model/features (model: mlp | joint | residual | ckpt)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--test", action="store_true", help="also evaluate on the test sets (final runs only)")
    p.add_argument("--set", dest="overrides", nargs="*", default=[], help="DetectorCfg key=value overrides")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("summarize")
    p.add_argument("out")
    p.set_defaults(func=cmd_summarize)
    p = sub.add_parser("selftest")
    p.set_defaults(func=cmd_selftest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
