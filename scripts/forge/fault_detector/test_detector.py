# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks of ``detector.py`` on a synthetic dataset in ``collect.py``'s format (torch + numpy only).

Run: ``uv run --no-project --with torch --with numpy python scripts/forge/fault_detector/test_detector.py``
"""

import csv
import dataclasses
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector as fd  # noqa: E402

J = 12
DIMS = {"detector/base_ang_vel": 3, "detector/projected_gravity": 3, "detector/joint_pos_rel": J,
        "detector/joint_vel": J, "detector/last_action": 16, "detector/velocity_command": 3,
        "privileged/base_height": 1, "privileged/base_lin_vel": 3, "analysis/imu_lin_acc": 3,
        "analysis/joint_pos_target": J, "analysis/foot_contact": 4, "analysis/applied_torque": J}  # fmt: skip


PD_DIMS = {"pd/base_ang_vel": 3, "pd/projected_gravity": 3, "pd/joint_pos_rel": J, "pd/joint_vel": J,
           "pd/base_lin_vel": 3, "pd/joint_pos_target": J, "pd/last_action": 16, "pd/velocity_command": 3,
           "pd/applied_torque": J}  # fmt: skip
M = 4  # decimation


def make_dataset(root: str, num_episodes: int = 60, seed: int = 0) -> list[dict]:
    """Episodes whose joint velocity of the faulty joint jumps to 5 at onset; step counter in base_ang_vel[0].

    PD-rate arrays (decimation 4): sample counter in ``pd/base_ang_vel[0]``, the faulty joint's velocity jumps at
    sample ``4 onset - 3``, as after the first faulty physics substep.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "shards"))
    arrays = {k: [] for k in (*DIMS, *PD_DIMS)}
    arrays["fault/labels"] = []
    index, lengths = [], []
    for e in range(num_episodes):
        n = int(rng.integers(40, 80))
        has_fault = e % 3 != 0
        kind, joint, onset = int(rng.integers(1, 3)), int(rng.integers(J)), int(rng.integers(30, 35))
        for key, dim in DIMS.items():
            arrays[key].append(rng.normal(size=(n, dim)).astype(np.float32) * 0.1)
        for key, dim in PD_DIMS.items():
            arrays[key].append(rng.normal(size=(M * n, dim)).astype(np.float32) * 0.1)
        arrays["pd/base_ang_vel"][-1][:, 0] = np.arange(M * n)  # sample marker
        arrays["pd/base_ang_vel"][-1][:, 1] = e
        if has_fault:
            arrays["pd/joint_vel"][-1][M * onset - (M - 1) :, joint] += 5.0
        arrays["detector/base_ang_vel"][-1][:, 0] = np.arange(n)  # step marker
        arrays["detector/base_ang_vel"][-1][:, 1] = e  # episode marker
        labels = np.zeros((n, 2 * J + 4), np.float32)
        labels[:, J + 1 : 2 * J + 1] = 1.0
        labels[:, 2 * J + 2] = -1
        if has_fault:
            labels[onset:, joint] = 1
            labels[onset:, J] = kind
            labels[onset:, J + 1 + joint] = 0.0
            labels[onset:, 2 * J + 2] = np.arange(n - onset)
            arrays["detector/joint_vel"][-1][onset:, joint] += 5.0
            arrays["detector/last_action"][-1][onset:, 0] += 3.0 * kind
        labels[:, 2 * J + 1] = onset if has_fault else -1
        arrays["fault/labels"].append(labels)
        lengths.append(n)
        index.append({"episode_id": e, "env_id": e, "length": n, "has_fault": int(has_fault),
                      "fault_type": kind if has_fault else 0, "fault_joint": joint if has_fault else -1,
                      "onset_step": onset if has_fault else -1, "health": 0.0, "terrain_level": 0,
                      "terrain_type": "flat", "termination_reason": "time_out"})  # fmt: skip
    out = {k: np.concatenate(v).astype(np.float16 if k.startswith("pd/") else np.float32) for k, v in arrays.items()}
    out["ep_len"], out["episode_id"] = np.array(lengths), np.arange(num_episodes)
    np.savez(os.path.join(root, "shards", "0000.npz"), **out)
    with open(os.path.join(root, "index.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    with open(os.path.join(root, "meta.json"), "w") as f:
        names = [f"{leg}_{kind}_joint" for leg in ("FL", "FR", "RL", "RR") for kind in ("hip", "thigh", "calf")]
        json.dump({"step_dt": 0.02, "decimation": M, "joint_names": names}, f)
    return index


def test_windows(root):
    data = fd.load_episodes(root, fd.PROPRIO + fd.PRIV)
    assert data.x.shape[1] == 3 + 3 + J + J + 16 + 3 + 1 + 2
    rows = torch.arange(len(data))
    w = fd.window_rows(rows, torch.from_numpy(data.start), 3).numpy()
    step, episode = data.x[:, 0], data.x[:, 1]
    # every window stays in its episode, oldest first, padded with the first step
    assert (episode[w] == episode[:, None]).all()
    expected = np.maximum(step[:, None] - np.array([2, 1, 0]), 0)
    assert (step[w] == expected).all()
    # labels of a window are those of its last step: the mask turns on exactly at the onset row
    for e, ep in enumerate(data.episodes):
        if int(ep["has_fault"]):
            lo = data.ep_start[e]
            assert not data.mask[lo + int(ep["onset_step"]) - 1].any()
            assert data.mask[lo + int(ep["onset_step"]), int(ep["fault_joint"])]
    print("windows: ok")


def test_pd_rate(root):
    """PD-rate samples: 4 per step in order, labels on from sample 4 onset - 3, policy-rate terms repeated."""
    data = fd.load_episodes(root, fd.PROPRIO + fd.PRIV, rate="pd")
    policy = fd.load_episodes(root, fd.PROPRIO + fd.PRIV)
    assert len(data) == M * len(policy) and data.per_step == M and abs(data.sample_dt - 0.005) < 1e-12
    sample = data.x[:, 0].astype(np.float32)
    assert (sample == data.t).all()
    height = data.names.index("base_height[0]")
    assert (data.x[:, height] == np.repeat(policy.x[:, height], M).astype(np.float16)).all()
    for e, ep in enumerate(data.episodes):
        lo = data.ep_start[e]
        if int(ep["has_fault"]):
            first = M * int(ep["onset_step"]) - (M - 1)
            assert data.onset[e] == first and data.since[lo + first] == 0
            assert not data.mask[lo + first - 1].any() and data.mask[lo + first, int(ep["fault_joint"])]
        else:
            assert data.onset[e] == data.ep_len[e] and not data.mask[lo : lo + data.ep_len[e]].any()
    cfg = fd.DetectorCfg(rate="pd", history_length=5)
    src = fd.WindowSource(data, cfg, "cpu")
    assert src.length == 20 and src.windows(torch.arange(8), False).shape == (8, 20, data.x.shape[1])
    print("pd rate: ok")


def test_split(root):
    index = fd.read_index(root)
    split = fd.split_episodes(index, (0.8, 0.1, 0.1), seed=0)
    sets = [set(v.tolist()) for v in split.values()]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(len(s) for s in sets) == len(index)
    again = fd.split_episodes(index, (0.8, 0.1, 0.1), seed=0)
    assert all((split[k] == again[k]).all() for k in split)
    print("split: ok")


def test_overfit(root):
    """Both trunks drive the loss of 1000 fixed windows close to 0."""
    for arch in ("mlp", "gru"):
        cfg = fd.DetectorCfg(arch=arch, obs_set="proprio_priv", train_noise=False)
        data = fd.load_episodes(root, cfg.inputs)
        src = fd.WindowSource(data, cfg, "cpu")
        torch.manual_seed(0)
        rows = torch.randint(len(data), (1000,))
        x, mask, ftype, since = src.windows(rows, False), src.mask[rows], src.ftype[rows], src.since[rows]
        model = fd.Detector(cfg, data.x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(600):
            loss = fd.loss_fn(model, cfg, x, mask, ftype, since)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert loss.item() < 0.02, (arch, loss.item())
        print(f"overfit {arch}: loss {loss.item():.4f}, {fd.num_params(model)} params")


def test_param_counts():
    """The medium presets match the spec's ~70k parameters (D = 48 there)."""
    for arch, expected in (("mlp", 72_000), ("gru", 70_000)):
        n = fd.num_params(fd.Detector(fd.DetectorCfg(arch=arch), 48))
        assert abs(n - expected) / expected < 0.05, (arch, n)
    print("param counts: ok")


def test_closed_loop(root):
    data = fd.load_episodes(root, fd.PROPRIO)
    prob = np.zeros((len(data), J), np.float32)
    # perfect detector that fires 2 steps after onset, plus one false alarm in episode 0 (nominal)
    for e, ep in enumerate(data.episodes):
        lo = data.ep_start[e]
        if int(ep["has_fault"]):
            prob[lo + int(ep["onset_step"]) + 2 : lo + data.ep_len[e], int(ep["fault_joint"])] = 0.9
    prob[data.ep_start[0] + 30 : data.ep_start[0] + 33, 0] = 0.9
    legs = np.repeat(np.arange(4), 3)
    summary, rows = fd.closed_loop(data, prob, legs, skip=25, tau=0.5, k=2)
    assert summary["median_latency_ms"] == 60 and summary["miss_rate"] == 0 and summary["correct_joint"] == 1
    assert summary["success_joint_500ms"] == 1 and all(r["latency_ms"] == 60 for r in rows)
    nominal_steps = sum(
        (int(ep["onset_step"]) if int(ep["has_fault"]) else int(ep["length"])) - 25 for ep in data.episodes
    )
    assert abs(summary["fa_per_min"] - 1 / (nominal_steps * 0.02 / 60)) < 1e-9
    assert fd.held(np.array([0.9, 0.1, 0.9, 0.9, 0.9]) > 0.5, 2).tolist() == [False, False, False, True, True]
    # the false alarm (3 steps at 0.9) is as strong as the true alarms, so meeting 0.5 / min needs tau above 0.9
    calib = fd.calibrate(data, prob.max(axis=1), 25, target=0.5)
    assert calib["val_fa_per_min"] <= 0.5 and calib["tau"] > 0.9 - 1e-6
    prob[data.ep_start[0] + 30 : data.ep_start[0] + 33, 0] = 0.6
    calib = fd.calibrate(data, prob.max(axis=1), 25, target=0.5)
    assert calib["val_fa_per_min"] == 0 and 0.6 - 1e-6 <= calib["tau"] < 0.9 and calib["val_median_latency_ms"] == 40
    print("closed loop: ok")


def test_auroc():
    score = np.array([0.1, 0.4, 0.35, 0.8])
    label = np.array([0, 0, 1, 1])
    auroc, ap = fd.auroc_ap(score, label)
    assert abs(auroc - 0.75) < 1e-9 and abs(ap - (1 + 2 / 3) / 2) < 1e-9
    print("auroc/ap: ok")


def test_train_smoke(root):
    """Both rates, label grace and hard-negative fine-tuning run end to end and reload."""
    for rate, arch in (("policy", "gru"), ("pd", "mlp")):
        cfg = fd.DetectorCfg(arch=arch, rate=rate, history_length=5, epochs=2, samples_per_epoch=8192,
                             batch_size=1024, val_windows=2048, ignore_after_onset_s=0.08, hnm_epochs=1)  # fmt: skip
        split = fd.split_episodes(fd.read_index(root))
        tr = fd.load_episodes(root, cfg.inputs, split["train"], rate)
        va = fd.load_episodes(root, cfg.inputs, split["val"], rate)
        ckpt = fd.train(cfg, tr, va, "cpu", log=lambda m: None)
        model, cfg2 = fd.load_detector(ckpt, "cpu")
        assert dataclasses.asdict(cfg2) == dataclasses.asdict(cfg) and ckpt["window"] == 5 * va.per_step
        assert [h["phase"] for h in ckpt["history"]] == ["main", "main", "hnm"]
        print(f"train smoke {rate}: val macro-F1 {ckpt['best_val_macro_f1']:.3f}, {ckpt['num_params']} params")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "data")
        make_dataset(root)
        test_windows(root)
        test_pd_rate(root)
        test_split(root)
        test_param_counts()
        test_auroc()
        test_closed_loop(root)
        test_overfit(root)
        test_train_smoke(root)
    print("all checks passed")
