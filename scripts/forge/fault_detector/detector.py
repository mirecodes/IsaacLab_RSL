# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor-failure detector: dataset, history windows, MLP / GRU model, training and metrics (torch + numpy only).

Datasets are written by ``collect.py``. A sample is (episode, sample i) at the policy rate (50 Hz) or the PD rate
(200 Hz, ``rate="pd"``): the input is the episode's last ``history_length`` policy steps of samples ending at i
(oldest first, padded with the first sample), the labels are sample i's. See ``README.md``.
"""

from __future__ import annotations

import csv
import dataclasses
import glob
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

# input name -> (policy-rate array, PD-rate array or None to repeat the policy-rate one, columns, noise half-width):
# the policy-observation noise of Isaac Lab's velocity tasks (uniform), on proprioception only; privileged terms
# are ground truth
INPUTS = {
    "base_ang_vel": ("detector/base_ang_vel", "pd/base_ang_vel", None, 0.2),
    "projected_gravity": ("detector/projected_gravity", "pd/projected_gravity", None, 0.05),
    "joint_pos_rel": ("detector/joint_pos_rel", "pd/joint_pos_rel", None, 0.01),
    "joint_vel": ("detector/joint_vel", "pd/joint_vel", None, 1.5),
    "last_action": ("detector/last_action", "pd/last_action", None, 0.0),
    "velocity_command": ("detector/velocity_command", "pd/velocity_command", None, 0.0),
    "base_height": ("privileged/base_height", None, None, 0.0),
    "base_lin_vel_xy": ("privileged/base_lin_vel", "pd/base_lin_vel", slice(0, 2), 0.0),
    "imu_lin_acc": ("analysis/imu_lin_acc", None, None, 0.0),
    # q* - q, from the recorded target and position (both relative to the default pose)
    "tracking_error": ("analysis/joint_pos_target", "pd/joint_pos_target", None, 0.01),
}
PROPRIO = ["base_ang_vel", "projected_gravity", "joint_pos_rel", "joint_vel", "last_action", "velocity_command"]
PRIV = ["base_height", "base_lin_vel_xy"]
SIZES = {"small": ([128, 64], 64), "medium": ([256, 128], 128), "large": ([512, 256, 128], 256)}
FAULT_TYPES = ("none", "power_out", "power_loss")
LEGS = ("FL", "FR", "RL", "RR")


@dataclasses.dataclass
class DetectorCfg:
    """Every setting of one run; ``key=value`` overrides on the command line (see ``experiment.py``)."""

    # -- data
    dataset_dir: str = "logs/fault_detector/data/main"
    test_dirs: tuple = ("logs/fault_detector/data/test_id", "logs/fault_detector/data/test_ood")
    split: tuple = (0.8, 0.1, 0.1)
    rate: str = "policy"  # "policy" (50 Hz samples) | "pd" (200 Hz, the actuator rate; needs collect.py --pd_rate)
    # -- input
    obs_set: str = "proprio"  # "proprio" | "proprio_priv"
    extra_terms: tuple = ()  # e.g. ("tracking_error",), ("imu_lin_acc",)
    history_length: int = 3  # in policy steps (20 ms each); x 4 samples at the PD rate
    clip: float = 10.0
    train_noise: bool = True
    noise_scale_mult: float = 1.0
    # -- model
    arch: str = "mlp"  # "mlp" | "gru"
    size: str = "medium"
    mlp_hidden_dims: tuple | None = None
    gru_hidden_dim: int | None = None
    gru_num_layers: int = 1
    layer_norm: bool = False
    dropout: float = 0.0
    use_type_head: bool = True
    lambda_type: float = 0.5
    pos_weight: float = 1.0
    ignore_after_onset_s: float = 0.0  # label grace: samples this soon after the onset are left out of the loss
    # -- sampling and optimization
    skip_initial_s: float = 0.5
    fault_ratio: float = 0.5
    onset_window_s: float = 0.5
    onset_oversample: float = 4.0
    batch_size: int = 4096
    samples_per_epoch: int = 2_000_000
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 5
    val_windows: int = 262_144
    hnm_epochs: int = 0  # hard-negative fine-tuning epochs after the main phase (0: off)
    hnm_weight: float = 4.0  # nominal window weight 1 + hnm_weight * p_any of the main-phase model
    seed: int = 0

    @property
    def inputs(self) -> list[str]:
        return PROPRIO + (PRIV if self.obs_set == "proprio_priv" else []) + list(self.extra_terms)


# ---------------------------------------------------------------------------------------------------------------------
# Data


def read_index(dataset_dir: str) -> list[dict]:
    with open(os.path.join(dataset_dir, "index.csv")) as f:
        return list(csv.DictReader(f))


def read_meta(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "meta.json")) as f:
        return json.load(f)


def index_hash(dataset_dir: str) -> str:
    with open(os.path.join(dataset_dir, "index.csv"), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def split_episodes(index: list[dict], fractions=(0.8, 0.1, 0.1), seed: int = 0) -> dict[str, np.ndarray]:
    """Episode-level train / val / test split, stratified by (has_fault, fault_type, fault_joint)."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple, list[int]] = {}
    for row in index:
        key = (row["has_fault"], row["fault_type"], row["fault_joint"])
        strata.setdefault(key, []).append(int(row["episode_id"]))
    out = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        ids = rng.permutation(strata[key])
        n_train = round(fractions[0] * len(ids))
        n_val = round(fractions[1] * len(ids))
        out["train"] += ids[:n_train].tolist()
        out["val"] += ids[n_train : n_train + n_val].tolist()
        out["test"] += ids[n_train + n_val :].tolist()
    return {k: np.sort(np.array(v, dtype=np.int64)) for k, v in out.items()}


@dataclasses.dataclass
class Episodes:
    """Samples of a set of episodes, concatenated episode by episode (S samples, E episodes, J joints)."""

    x: np.ndarray  # [S, D] raw detector inputs (float16 at the PD rate)
    mask: np.ndarray  # [S, J] fault mask
    ftype: np.ndarray  # [S] fault type (0 before onset)
    since: np.ndarray  # [S] samples since onset (-1 before)
    t: np.ndarray  # [S] sample within the episode
    start: np.ndarray  # [S] row of the episode's first sample
    contact: np.ndarray  # [S, 4] foot contact (analysis)
    command: np.ndarray  # [S, 3] velocity command (analysis)
    ep_start: np.ndarray  # [E]
    ep_len: np.ndarray  # [E] in samples
    onset: np.ndarray  # [E] first faulty sample (= ep_len for an episode without an observed fault)
    episodes: list[dict]  # [E] index rows
    noise: np.ndarray  # [D] noise half-width per input column
    names: list[str]  # [D] column names
    sample_dt: float  # [s] per sample
    per_step: int  # samples per policy step (1 or the decimation)

    def __len__(self) -> int:
        return len(self.t)


def load_episodes(dataset_dir: str, inputs: list[str], episode_ids=None, rate: str = "policy") -> Episodes:
    """Load the given episodes (default: all) of a dataset written by ``collect.py`` at the policy or PD rate.

    At the PD rate (decimation m), sample ``m t + k`` is the state before physics substep ``k`` of policy step ``t``;
    policy-rate terms without a PD-rate array are repeated over the step. The first faulty sample of an episode
    with fault onset step ``s`` (first faulty observation) is ``m s - (m - 1)``: right after the first faulty
    substep.
    """
    meta = read_meta(dataset_dir)
    m = int(meta.get("decimation", 4)) if rate == "pd" else 1
    index = {int(r["episode_id"]): r for r in read_index(dataset_dir)}
    keep = None if episode_ids is None else list(set(int(i) for i in episode_ids))
    parts: dict[str, list] = {k: [] for k in ("contact", "command", "ep_len", "ids")}
    noise, names, x, filled = None, None, None, 0
    paths = sorted(glob.glob(os.path.join(dataset_dir, "shards", "*.npz")))
    # the inputs are filled into one preallocated array: no second copy of the largest array while loading
    selected = {}
    for path in paths:
        with np.load(path) as shard:
            ep_len, ep_ids = shard["ep_len"], shard["episode_id"]
        sel = np.ones(len(ep_ids), bool) if keep is None else np.isin(ep_ids, keep)
        if sel.any():
            selected[path] = (ep_len, ep_ids, sel)
    total = sum(int(ep_len[sel].sum()) * m for ep_len, _, sel in selected.values())
    for path, (ep_len, ep_ids, sel) in selected.items():
        shard = np.load(path)
        rows = np.repeat(sel, ep_len)
        columns, noise, names = [], [], []
        for name in inputs:
            step_key, pd_key, cols, scale = INPUTS[name]
            if m > 1 and pd_key is not None:
                value = shard[pd_key][np.repeat(rows, m)]
                if name == "tracking_error":
                    value = value - shard["pd/joint_pos_rel"][np.repeat(rows, m)]
            else:
                value = shard[step_key][rows]
                if name == "tracking_error":
                    value = value - shard["detector/joint_pos_rel"][rows]
                value = np.repeat(value, m, axis=0)
            if cols is not None:
                value = value[:, cols]
            columns.append(value.astype(np.float16 if m > 1 else np.float32))
            noise += [scale] * value.shape[1]
            names += [f"{name}[{i}]" for i in range(value.shape[1])]
        if x is None:
            x = np.empty((total, sum(c.shape[1] for c in columns)), dtype=columns[0].dtype)
        np.concatenate(columns, axis=1, out=x[filled : filled + len(columns[0])])
        filled += len(columns[0])
        del columns
        parts["contact"].append(np.repeat(shard["analysis/foot_contact"][rows] > 0.5, m, axis=0))
        parts["command"].append(np.repeat(shard["detector/velocity_command"][rows], m, axis=0))
        parts["ep_len"].append(ep_len[sel] * m)
        parts["ids"].append(ep_ids[sel])
    ep_len = np.concatenate(parts["ep_len"]).astype(np.int64)
    ep_start = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
    episodes = [index[int(i)] for i in np.concatenate(parts["ids"])]
    num_joints = len(meta["joint_names"])
    has_fault = np.array([int(e["has_fault"]) for e in episodes], bool)
    onset_step = np.array([int(e["onset_step"]) for e in episodes])
    onset = np.where(has_fault, np.maximum(m * onset_step - (m - 1), 0), ep_len)
    t = np.concatenate([np.arange(n, dtype=np.int32) for n in ep_len])
    since = (t - np.repeat(onset, ep_len)).astype(np.int32)
    faulty = since >= 0
    since[~faulty] = -1
    mask = np.zeros((len(t), num_joints), bool)
    joint = np.repeat(np.array([max(int(e["fault_joint"]), 0) for e in episodes]), ep_len)
    mask[np.flatnonzero(faulty), joint[faulty]] = True
    ftype = np.where(faulty, np.repeat(np.array([int(e["fault_type"]) for e in episodes]), ep_len), 0).astype(np.int8)
    return Episodes(
        x=x,
        mask=mask,
        ftype=ftype,
        since=since,
        t=t,
        start=np.repeat(ep_start, ep_len),
        contact=np.concatenate(parts["contact"]),
        command=np.concatenate(parts["command"]),
        ep_start=ep_start,
        ep_len=ep_len,
        onset=onset,
        episodes=episodes,
        noise=np.array(noise, dtype=np.float32),
        names=names,
        sample_dt=meta["step_dt"] / m,
        per_step=m,
    )


def window_rows(rows: torch.Tensor, start: torch.Tensor, length: int) -> torch.Tensor:
    """Rows of the windows ending at ``rows``: [B, L], oldest first, clamped to the episode start (repeat first)."""
    offsets = torch.arange(length - 1, -1, -1, device=rows.device)
    return torch.maximum(rows.unsqueeze(1) - offsets, start.unsqueeze(1))


def chunked_mean_std(x: torch.Tensor, chunk: int = 1 << 22) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column mean and std of a large (possibly float16) tensor without a full float32 copy."""
    total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    square = torch.zeros_like(total)
    for lo in range(0, len(x), chunk):
        part = x[lo : lo + chunk].double()
        total += part.sum(0)
        square += (part * part).sum(0)
    mean = total / len(x)
    std = (square / len(x) - mean * mean).clamp(min=0).sqrt()
    return mean.float(), std.float().clamp(min=1e-3)


class WindowSource:
    """Episodes on the device, with normalization and noise; ``windows(rows)`` -> model input [B, L, D]."""

    def __init__(self, data: Episodes, cfg: DetectorCfg, device: str, mean=None, std=None):
        self.data, self.cfg, self.device = data, cfg, device
        self.x = torch.from_numpy(data.x).to(device)
        self.start = torch.from_numpy(data.start).to(device)
        self.mask = torch.from_numpy(data.mask).to(device).float()
        self.ftype = torch.from_numpy(data.ftype.astype(np.int64)).to(device)
        self.since = torch.from_numpy(data.since).to(device)
        self.t = torch.from_numpy(data.t).to(device)
        self.length = cfg.history_length * data.per_step
        self.dim = self.x.shape[1]
        """Model inputs per sample."""
        if mean is None:
            mean, std = chunked_mean_std(self.x)
        self.mean, self.std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
        self.noise = torch.from_numpy(data.noise).to(device) * cfg.noise_scale_mult

    def seconds(self, s: float) -> int:
        """A duration in samples."""
        return round(s / self.data.sample_dt)

    def windows(self, rows: torch.Tensor, noise: bool, generator: torch.Generator | None = None) -> torch.Tensor:
        x = self.x[window_rows(rows, self.start[rows], self.length)].float()
        if noise:
            u = torch.rand(x.shape, device=self.device, generator=generator) * 2 - 1
            x = x + u * self.noise
        return ((x - self.mean) / self.std).clamp(-self.cfg.clip, self.cfg.clip)


def sampling_cdf(src: WindowSource, cfg: DetectorCfg, nominal_weight: torch.Tensor | None = None) -> torch.Tensor:
    """CDF over all rows: 50 % nominal / 50 % faulty windows, the faulty half equal per (joint, type), onset windows
    oversampled, the first ``skip_initial_s`` of each episode excluded (see ``README.md``). ``nominal_weight``
    reweights nominal rows within their half (hard-negative mining)."""
    num_joints = src.mask.shape[1]
    faulty = src.mask.any(dim=1)
    joint = src.mask.argmax(dim=1)
    group = torch.where(faulty, joint * 2 + (src.ftype - 1).clamp(min=0) + 1, 0)  # 0 nominal, 1.. combos
    w = torch.ones(len(src.t), device=src.device, dtype=torch.float64)
    w[faulty & (src.since < src.seconds(cfg.onset_window_s))] = cfg.onset_oversample
    if nominal_weight is not None:
        w = torch.where(faulty, w, w * nominal_weight.double())
    w[src.t < src.seconds(cfg.skip_initial_s)] = 0.0
    group_sum = torch.zeros(2 * num_joints + 1, device=src.device, dtype=torch.float64).index_add_(0, group, w)
    present = (group_sum[1:] > 0).sum().clamp(min=1)
    target = torch.full_like(group_sum, cfg.fault_ratio / present.item())
    target[0] = 1.0 - cfg.fault_ratio
    w = w * target[group] / group_sum[group].clamp(min=1e-12)
    cdf = torch.cumsum(w, dim=0)
    return cdf / cdf[-1]


def sample_rows(cdf: torch.Tensor, n: int, generator: torch.Generator) -> torch.Tensor:
    u = torch.rand(n, device=cdf.device, generator=generator, dtype=torch.float64)
    return torch.searchsorted(cdf, u).clamp(max=len(cdf) - 1)


# ---------------------------------------------------------------------------------------------------------------------
# Model


class Detector(nn.Module):
    """MLP or GRU trunk -> per-joint fault logits (12, sigmoid each) + fault-type logits (3, softmax)."""

    def __init__(self, cfg: DetectorCfg, input_dim: int, num_joints: int = 12, window: int | None = None):
        super().__init__()
        self.arch = cfg.arch
        mlp_dims, gru_dim = SIZES[cfg.size]
        mlp_dims = list(cfg.mlp_hidden_dims or mlp_dims)
        gru_dim = cfg.gru_hidden_dim or gru_dim
        if cfg.arch == "mlp":
            layers, dim = [], input_dim * (window or cfg.history_length)
            for hidden in mlp_dims:
                layers += [nn.Linear(dim, hidden)]
                layers += [nn.LayerNorm(hidden)] if cfg.layer_norm else []
                layers += [nn.ELU()]
                layers += [nn.Dropout(cfg.dropout)] if cfg.dropout > 0 else []
                dim = hidden
            self.trunk = nn.Sequential(*layers)
        elif cfg.arch == "gru":
            # window mode: the hidden state starts at 0 for every window, so it sees the MLP's window only
            self.trunk = nn.GRU(input_dim, gru_dim, num_layers=cfg.gru_num_layers, batch_first=True)
            dim = gru_dim
        else:
            raise ValueError(f"unknown arch '{cfg.arch}'")
        self.joint_head = nn.Linear(dim, num_joints)
        self.type_head = nn.Linear(dim, len(FAULT_TYPES)) if cfg.use_type_head else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: [B, L, D] -> joint logits [B, J], type logits [B, 3] (or None)."""
        if self.arch == "mlp":
            feature = self.trunk(x.flatten(1))
        else:
            feature = self.trunk(x)[0][:, -1]
        return self.joint_head(feature), (self.type_head(feature) if self.type_head is not None else None)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def chunked(model: Detector, x: torch.Tensor, batch: int = 32768):
    """``model(x)`` in batches (a GRU's workspace for a whole validation set exceeds the GPU)."""
    outs = [model(part) for part in x.split(batch)]
    types = [t for _, t in outs]
    return torch.cat([j for j, _ in outs]), (torch.cat(types) if types[0] is not None else None)


def loss_fn(model: Detector, cfg: DetectorCfg, x, mask, ftype, since, grace: int = 0) -> torch.Tensor:
    """BCE over joints + lambda_type CE over the type; samples ``0 <= since < grace`` are left out (label grace)."""
    joint_logits, type_logits = model(x) if model.training else chunked(model, x)
    pos_weight = torch.full((mask.shape[1],), cfg.pos_weight, device=x.device)
    bce = nn.functional.binary_cross_entropy_with_logits(joint_logits, mask, pos_weight=pos_weight, reduction="none")
    keep = ((since < 0) | (since >= grace)).float()
    loss = (bce.mean(dim=1) * keep).sum() / keep.sum().clamp(min=1)
    if type_logits is not None:
        ce = nn.functional.cross_entropy(type_logits, ftype, reduction="none")
        loss = loss + cfg.lambda_type * (ce * keep).sum() / keep.sum().clamp(min=1)
    return loss


@torch.no_grad()
def predict(model: Detector, src: WindowSource, noise: bool, seed: int = 0, batch: int = 32768, rows=None):
    """Joint probabilities [S, J] and type probabilities [S, 3] at every row of ``src`` (or the given rows)."""
    model.eval()
    g = torch.Generator(device=src.device).manual_seed(seed)
    rows = torch.arange(len(src.t), device=src.device) if rows is None else rows
    probs, types = [], []
    for lo in range(0, len(rows), batch):
        part = rows[lo : lo + batch]
        joint_logits, type_logits = model(src.windows(part, noise, g))
        probs.append(torch.sigmoid(joint_logits))
        types.append(
            torch.softmax(type_logits, dim=1)
            if type_logits is not None
            else torch.zeros(len(part), 3, device=src.device)
        )
    return torch.cat(probs).cpu().numpy(), torch.cat(types).cpu().numpy()


def macro_f1(prob: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred = prob > threshold
    tp = (pred & (mask > 0.5)).sum(0).float()
    fp = (pred & (mask < 0.5)).sum(0).float()
    fn = (~pred & (mask > 0.5)).sum(0).float()
    return (2 * tp / (2 * tp + fp + fn).clamp(min=1)).mean().item()


def train(cfg: DetectorCfg, train_data: Episodes, val_data: Episodes, device: str, log=print,
          source=WindowSource, build=Detector):  # fmt: skip
    """Train one detector; returns the checkpoint dict (best val macro-F1 of the main phase, then the hard-negative
    fine-tuning if ``hnm_epochs`` > 0) and the per-epoch history. ``source`` / ``build`` replace the window source
    and the model (same interfaces; the model gets ``source.dim`` inputs per sample)."""
    torch.manual_seed(cfg.seed)
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    src = source(train_data, cfg, device)
    val = source(val_data, cfg, device, src.mean, src.std)
    grace = src.seconds(cfg.ignore_after_onset_s)
    cdf = sampling_cdf(src, cfg)
    gv = torch.Generator(device=device).manual_seed(12345)
    val_rows = sample_rows(sampling_cdf(val, cfg), cfg.val_windows, gv)
    val_x = val.windows(val_rows, cfg.train_noise, gv)
    model = build(cfg, src.dim, src.mask.shape[1], src.length).to(device)
    log(f"[train] {cfg.arch} {cfg.size} {cfg.obs_set} {cfg.rate} H={cfg.history_length} seed {cfg.seed}: "
        f"{num_params(model)} params, input {src.dim} x {src.length}, {len(src.t)} train rows")  # fmt: skip
    iters = cfg.samples_per_epoch // cfg.batch_size
    history = []

    def run_phase(name: str, epochs: int, lr: float, cdf: torch.Tensor, select: bool) -> None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * iters)
        best, best_f1, bad = None, -1.0, 0
        for epoch in range(epochs):
            model.train()
            start, total = time.time(), 0.0
            for _ in range(iters):
                rows = sample_rows(cdf, cfg.batch_size, g)
                x = src.windows(rows, cfg.train_noise, g)
                loss = loss_fn(model, cfg, x, src.mask[rows], src.ftype[rows], src.since[rows], grace)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                total += loss.item()
            model.eval()
            with torch.no_grad():
                y = (val.mask[val_rows], val.ftype[val_rows], val.since[val_rows])
                val_loss = loss_fn(model, cfg, val_x, *y, grace).item()
                f1 = macro_f1(torch.sigmoid(chunked(model, val_x)[0]), val.mask[val_rows])
            history.append({"phase": name, "epoch": epoch, "train_loss": total / iters, "val_loss": val_loss,
                            "val_macro_f1": f1, "wall_s": round(time.time() - start, 1)})  # fmt: skip
            log(f"[train] {name} epoch {epoch}: loss {total / iters:.4f}, val loss {val_loss:.4f}, val F1 {f1:.4f}")
            if not select:
                continue
            if f1 > best_f1:
                best_f1, bad = f1, 0
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.early_stop_patience:
                    log(f"[train] early stop at epoch {epoch}")
                    break
        if select:
            model.load_state_dict(best)

    run_phase("main", cfg.epochs, cfg.lr, cdf, select=True)
    if cfg.hnm_epochs > 0:
        # nominal windows the main-phase model scores high get more weight (the same 50 % nominal share in total)
        nominal = (src.mask.sum(dim=1) == 0) & (src.t >= src.seconds(cfg.skip_initial_s))
        rows = nominal.nonzero().squeeze(-1)
        p_any = torch.zeros(len(src.t), device=device)
        prob, _ = predict(model, src, cfg.train_noise, seed=cfg.seed + 7, rows=rows)
        p_any[rows] = torch.from_numpy(prob.max(axis=1)).to(device)
        log(f"[train] hard negatives: {(p_any > 0.5).sum().item()} of {len(rows)} nominal windows above 0.5")
        run_phase("hnm", cfg.hnm_epochs, cfg.lr * 0.1, sampling_cdf(src, cfg, 1.0 + cfg.hnm_weight * p_any), False)
    model.eval()
    with torch.no_grad():
        final_f1 = macro_f1(torch.sigmoid(chunked(model, val_x)[0]), val.mask[val_rows])
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "cfg": dataclasses.asdict(cfg),
        "input_dim": src.dim,
        "num_joints": src.mask.shape[1],
        "window": src.length,
        "mean": src.mean.cpu(),
        "std": src.std.cpu(),
        "num_params": num_params(model),
        "best_val_macro_f1": final_f1,
        "history": history,
    }


def load_detector(checkpoint: dict, device: str) -> tuple[Detector, DetectorCfg]:
    """Model and cfg of a checkpoint; settings this version no longer has are dropped (v1: label grace 0 and a
    25-step onset window, the current defaults)."""
    fields = {f.name for f in dataclasses.fields(DetectorCfg)}
    saved = {k: tuple(v) if isinstance(v, list) else v for k, v in checkpoint["cfg"].items() if k in fields}
    cfg = DetectorCfg(**saved)
    window = checkpoint.get("window", cfg.history_length)
    model = Detector(cfg, checkpoint["input_dim"], checkpoint["num_joints"], window).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), cfg


# ---------------------------------------------------------------------------------------------------------------------
# Metrics


def auroc_ap(score: np.ndarray, label: np.ndarray) -> tuple[float, float]:
    """Area under the ROC curve (rank statistic) and average precision of one binary task."""
    pos = label.sum()
    neg = len(label) - pos
    if pos == 0 or neg == 0:
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="stable")
    hits = label[order].astype(np.float64)
    tp = np.cumsum(hits)
    ap = float((tp / np.arange(1, len(hits) + 1))[hits > 0].sum() / pos)
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    auroc = float((ranks[label > 0].sum() - pos * (pos + 1) / 2) / (pos * neg))
    return auroc, ap


def calibration(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error over all (sample, joint) probabilities, and the reliability-diagram bins."""
    p, y = prob.ravel(), label.ravel().astype(np.float64)
    which = np.minimum((p * bins).astype(int), bins - 1)
    count = np.bincount(which, minlength=bins)
    conf = np.bincount(which, weights=p, minlength=bins) / np.maximum(count, 1)
    acc = np.bincount(which, weights=y, minlength=bins) / np.maximum(count, 1)
    ece = float((count / len(p) * np.abs(acc - conf)).sum())
    return ece, [{"bin": b, "count": int(count[b]), "confidence": float(conf[b]), "accuracy": float(acc[b])}
                 for b in range(bins)]  # fmt: skip


SINCE_BINS_MS = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 10**9)]


def offline_metrics(data: Episodes, prob: np.ndarray, type_prob: np.ndarray, legs: np.ndarray, skip: int) -> dict:
    """Window-level metrics over the samples past ``skip`` of each episode (natural distribution, no resampling)."""
    valid = data.t >= skip
    faulty = data.mask.any(axis=1) & valid
    nominal = ~data.mask.any(axis=1) & valid
    true_joint = data.mask.argmax(axis=1)
    pred_joint = prob.argmax(axis=1)
    num_joints = prob.shape[1]
    confusion = np.zeros((num_joints, num_joints), dtype=np.int64)
    np.add.at(confusion, (true_joint[faulty], pred_joint[faulty]), 1)
    leg_confusion = np.zeros((len(LEGS), len(LEGS)), dtype=np.int64)
    np.add.at(leg_confusion, (legs[true_joint[faulty]], legs[pred_joint[faulty]]), 1)
    wrong = faulty & (pred_joint != true_joint)
    same_leg = float((legs[pred_joint[wrong]] == legs[true_joint[wrong]]).mean()) if wrong.any() else 0.0
    per_joint = [auroc_ap(prob[valid, j], data.mask[valid, j]) for j in range(num_joints)]
    ece, reliability = calibration(prob[valid], data.mask[valid])
    p_any = prob.max(axis=1)
    since_ms = data.since * data.sample_dt * 1000
    since_acc = []
    for lo, hi in SINCE_BINS_MS:
        sel = faulty & (since_ms >= lo) & (since_ms < hi)
        top1 = float((pred_joint[sel] == true_joint[sel]).mean()) if sel.any() else float("nan")
        leg = float((legs[pred_joint[sel]] == legs[true_joint[sel]]).mean()) if sel.any() else float("nan")
        since_acc.append({"bin": f"[{lo},{hi if hi < 10**9 else 'inf'}) ms", "count": int(sel.sum()), "top1": top1,
                          "leg": leg})  # fmt: skip
    return {
        "num_windows": int(valid.sum()),
        "num_fault_windows": int(faulty.sum()),
        "top1_joint_acc": float((pred_joint[faulty] == true_joint[faulty]).mean()),
        "leg_acc": float((legs[pred_joint[faulty]] == legs[true_joint[faulty]]).mean()),
        "same_leg_error_share": same_leg,
        "type_acc": float((type_prob[faulty].argmax(axis=1) == data.ftype[faulty]).mean()),
        "nominal_fp_rate": float((p_any[nominal] > 0.5).mean()),
        "macro_auroc": float(np.nanmean([a for a, _ in per_joint])),
        "macro_ap": float(np.nanmean([b for _, b in per_joint])),
        "per_joint_auroc": [a for a, _ in per_joint],
        "per_joint_ap": [b for _, b in per_joint],
        "ece": ece,
        "reliability": reliability,
        "since_onset_top1": since_acc,
        "confusion": confusion.tolist(),
        "leg_confusion": leg_confusion.tolist(),
    }


def held(above: np.ndarray, k: int) -> np.ndarray:
    """True where ``above`` held for the last ``k`` samples (the alarm condition)."""
    if k <= 1:
        return above
    run = np.lib.stride_tricks.sliding_window_view(above, k).all(axis=1)
    return np.concatenate([np.zeros(min(k - 1, len(above)), bool), run])


def alarm_stats(data: Episodes, p_any: np.ndarray, tau: float, k: int, skip: int) -> dict:
    """Alarms of ``p_any > tau`` held for ``k`` samples, armed ``skip`` samples after each reset (vectorized).

    A false alarm is a rising edge before the episode's fault onset (or anywhere in a nominal episode); latency is
    from the onset to the first alarm at or after it (inf = missed). ``skip >= k`` keeps windows inside episodes.
    """
    armed = data.t >= skip
    on = held(p_any > tau, k) & armed
    rising = on & ~np.concatenate([[False], on[:-1]])
    onset = np.repeat(data.onset, data.ep_len)
    pre = armed & (data.t < onset)
    false_alarms = rising & pre
    minutes = pre.sum() * data.sample_dt / 60
    fa_per_episode = np.add.reduceat(false_alarms.astype(np.int64), data.ep_start) if len(data.ep_start) else []
    faulty = data.onset < data.ep_len
    onset_g = (data.ep_start + data.onset)[faulty]
    end_g = (data.ep_start + data.ep_len)[faulty]
    post = np.flatnonzero(on & (data.t >= onset))
    pos = np.searchsorted(post, onset_g)
    cand = post[np.minimum(pos, max(len(post) - 1, 0))] if len(post) else np.full(len(onset_g), -1)
    hit = (pos < len(post)) & (cand < end_g) & (cand >= onset_g)
    return {
        "fa_per_min": float(false_alarms.sum() / max(minutes, 1e-9)),
        "fa_episode_rate": float(np.mean(np.asarray(fa_per_episode) > 0)) if len(data.ep_start) else float("nan"),
        "nominal_minutes": float(minutes),
        "faulty": np.flatnonzero(faulty),
        "latency": np.where(hit, cand - onset_g, np.inf),
        "alarm_row": np.where(hit, cand, -1),
    }


def calibrate(data: Episodes, p_any: np.ndarray, skip: int, target: float, debounce_ms=(0, 20, 40, 60)) -> dict:
    """The (tau, k) whose false alarms on ``data`` stay at or under ``target`` per minute with the lowest median
    latency: for each debounce k, the smallest tau that meets the target (bisection in logit space)."""
    best = None
    for k in sorted({max(1, round(ms / 1000 / data.sample_dt)) for ms in debounce_ms}):
        lo, hi = 0.0, 18.0  # logit range of tau; sigmoid(18) ~ 1 - 1.5e-8
        if alarm_stats(data, p_any, 1 / (1 + math.exp(-hi)), k, skip)["fa_per_min"] > target:
            continue
        for _ in range(30):
            mid = (lo + hi) / 2
            if alarm_stats(data, p_any, 1 / (1 + math.exp(-mid)), k, skip)["fa_per_min"] > target:
                lo = mid
            else:
                hi = mid
        tau = 1 / (1 + math.exp(-hi))
        stats = alarm_stats(data, p_any, tau, k, skip)
        lat = stats["latency"]
        median = float(np.median(lat)) if len(lat) else float("inf")
        key = (median, -float(np.isfinite(lat).mean()) if len(lat) else 0.0)
        if best is None or key < best["key"]:
            best = {"key": key, "tau": tau, "k": k, "val_fa_per_min": stats["fa_per_min"], "val_median_latency_ms":
                    median * data.sample_dt * 1000}  # fmt: skip
    if best is None:
        return {"tau": 1.0, "k": 1, "val_fa_per_min": float("nan"), "val_median_latency_ms": float("nan")}
    best.pop("key")
    return best


def closed_loop(data: Episodes, prob: np.ndarray, legs: np.ndarray, skip: int, tau: float, k: int,
                fall_terms=("base_contact", "bad_orientation")) -> tuple[dict, list[dict]]:  # fmt: skip
    """Episode-level alarm metrics at (tau, k) (see ``README.md``). Returns the summary and one row per faulty
    episode. Times in ms; success = an alarm within 500 ms of the onset that names the right joint (or leg)."""
    s = alarm_stats(data, prob.max(axis=1), tau, k, skip)
    ms = data.sample_dt * 1000
    rows = []
    for e, lat, alarm in zip(s["faulty"], s["latency"], s["alarm_row"]):
        ep = data.episodes[e]
        joint = int(ep["fault_joint"])
        diag = int(prob[alarm].argmax()) if alarm >= 0 else -1
        onset_row = int(data.ep_start[e] + data.onset[e])
        fell = ep["termination_reason"] in fall_terms
        rows.append({
            "episode_id": int(ep["episode_id"]),
            "fault_type": FAULT_TYPES[int(ep["fault_type"])],
            "fault_joint": joint,
            "health": float(ep["health"]),
            "terrain_type": ep["terrain_type"],
            "terrain_level": int(ep["terrain_level"]),
            "latency_ms": float(lat * ms) if np.isfinite(lat) else None,
            "correct_joint": diag == joint,
            "correct_leg": diag >= 0 and bool(legs[diag] == legs[joint]),
            "fell": fell,
            "fall_before_detection": fell and not np.isfinite(lat),
            "stance_at_onset": bool(data.contact[onset_row, legs[joint]]),
            "command_speed": float(np.linalg.norm(data.command[onset_row, :2])),
        })  # fmt: skip
    lat = np.array([r["latency_ms"] if r["latency_ms"] is not None else np.inf for r in rows])
    detected = np.isfinite(lat)
    joint_ok = np.array([r["correct_joint"] for r in rows], bool)
    leg_ok = np.array([r["correct_leg"] for r in rows], bool)
    n = max(len(rows), 1)
    summary = {
        "tau": tau,
        "k": k,
        "k_ms": k * ms,
        "num_fault_episodes": len(rows),
        "fa_per_min": s["fa_per_min"],
        "fa_episode_rate": s["fa_episode_rate"],
        "median_latency_ms": float(np.median(lat[detected])) if detected.any() else float("nan"),
        "p90_latency_ms": float(np.percentile(lat[detected], 90)) if detected.any() else float("nan"),
        "detect_200ms": float((lat <= 200).sum() / n),
        "detect_500ms": float((lat <= 500).sum() / n),
        "success_joint_500ms": float(((lat <= 500) & joint_ok).sum() / n),
        "success_leg_500ms": float(((lat <= 500) & leg_ok).sum() / n),
        "miss_rate": float((~detected).sum() / n),
        "correct_joint": float(joint_ok[detected].mean()) if detected.any() else float("nan"),
        "correct_leg": float(leg_ok[detected].mean()) if detected.any() else float("nan"),
        "fall_before_detection": float(np.mean([r["fall_before_detection"] for r in rows])) if rows else float("nan"),
        "nominal_minutes": s["nominal_minutes"],
    }
    return summary, rows


def inference_time_ms(model: Detector, input_dim: int, window: int, repeats: int = 2000) -> float:
    """Median CPU latency of one batch-1 forward pass, single thread."""
    model = model.cpu().eval()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(1, window, input_dim)
    times = []
    with torch.no_grad():
        for i in range(repeats + 100):
            t0 = time.perf_counter()
            model(x)
            if i >= 100:
                times.append(time.perf_counter() - t0)
    torch.set_num_threads(threads)
    return float(np.median(times) * 1000)


def json_safe(obj):
    """NaN / inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not math.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2)
