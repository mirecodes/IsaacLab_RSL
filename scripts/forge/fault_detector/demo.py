# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Live demo: one robot walks, a random joint loses power, and the CJ detector runs next to the ground truth.

The frozen policy plays ``IsaacContrib-FaultDetector-MotorFailure-A2Robust`` with one robot; faults come and go at
random (healthy ``--off_s``, faulty ``--on_s``), one random joint at a time (power loss by default, random health).
The faulty actuator's probe feeds every PD-rate
sample (200 Hz) to the detector of a ``tmp/explore.py`` run (default: CJ, motor current + joint-shared network), with
that run's validation rule, drift and threshold. A Kit panel shows every joint with a ground-truth and a detector
light (green: healthy, red: fault, gray: detector not armed yet); the console prints a line on every change::

    [t=  4.38 s] GT {FR_calf_joint: factor=0.32} | fault detector {FR_calf_joint: 20 ms (1.00 steps)}

Run on a machine with a display (see ``README.md``, Live demo)::

    ./isaaclab.sh -p scripts/forge/fault_detector/demo.py

Check of the live detector against the offline pipeline (torch + numpy only)::

    uv run --no-project --with torch --with numpy python scripts/forge/fault_detector/demo.py --selftest
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tmp"))
import detector as fd  # noqa: E402
import experiment as ex  # noqa: E402
import explore as xp  # noqa: E402

TASK = "IsaacContrib-FaultDetector-MotorFailure-A2Robust"
DETECTORS = (
    "logs/fault_detector/results/tmp_explore/phase1/CJ_s0",  # on Euler, or copied there
    "../../results/euler/isaaclab3fd/fault_detector/results/tmp_explore/phase1/CJ_s0",  # `euler fetch` from the repo
)
POLICY = ("../../results/euler/isaaclab3/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/"
          "model_2999.pt")  # fmt: skip
"""The robust teacher, fetched; used when ``--checkpoint`` is not given and it exists."""
# omni.ui colors are 0xAABBGGRR
GREEN, RED, GRAY = 0xFF40C040, 0xFF3030FF, 0xFF808080


class LiveDetector:
    """A ``tmp/explore.py`` run (``mlp`` / ``joint`` model, PD rate) fed one raw sample at a time.

    Reproduces the offline pipeline for one episode: the window of the last ``history_length`` policy steps of
    samples (padded with the first), noise drawn per window as in the evaluation, the logit score and the run's
    validation rule (debounce or per-joint CUSUM), armed after ``skip_initial_s``.
    """

    def __init__(self, run_dir: str, names: list[str], joint_names: list[str], sample_dt: float, device, noise=True):
        state = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
        with open(os.path.join(run_dir, "val.json")) as f:
            val = json.load(f)
        fields = {f.name for f in dataclasses.fields(fd.DetectorCfg)}
        saved = {k: tuple(v) if isinstance(v, list) else v for k, v in state["cfg"].items() if k in fields}
        self.cfg, self.spec = fd.DetectorCfg(**saved), xp.Spec(**val["spec"])
        if self.cfg.rate != "pd" or self.spec.model not in ("mlp", "joint"):
            raise ValueError(f"{run_dir}: needs a PD-rate mlp / joint run, got {self.cfg.rate} {self.spec.model}")
        self.rule = val["best"]["joint"]
        self.length = state["window"]
        n, J = self.length + 1, len(joint_names)
        noise_cols = np.array([fd.INPUTS[c.split("[")[0]][3] for c in names], np.float32) * self.cfg.noise_scale_mult
        # a one-window dataset: ``src.x`` is the live ring buffer, so the offline window / feature code runs as is
        window = fd.Episodes(x=np.zeros((n, len(names)), np.float32), mask=np.zeros((n, J), bool),
                             ftype=np.zeros(n, np.int8), since=np.full(n, -1, np.int32), t=np.zeros(n, np.int32),
                             start=np.zeros(n, np.int64), contact=None, command=None, ep_start=np.zeros(1, np.int64),
                             ep_len=np.array([n]), onset=np.array([n]), episodes=[], noise=noise_cols, names=names,
                             sample_dt=sample_dt, per_step=self.length // self.cfg.history_length)  # fmt: skip
        self.src = xp.FeatureSource(window, self.cfg, device, state["mean"], state["std"], self.spec.features,
                                    joint_names)  # fmt: skip
        if self.src.dim != state["input_dim"]:
            raise ValueError(f"{run_dir}: {self.src.dim} live inputs, the checkpoint has {state['input_dim']}")
        self.model = xp.build(self.spec, self.cfg, self.src, ex.joint_legs(joint_names)).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.noise, self.skip = noise, round(self.cfg.skip_initial_s / sample_dt)
        self.cap = 2 * self.rule["thr"]
        """CUSUM ceiling: the threshold crossing is unchanged, and the statistic falls back within a few samples once
        a fault ends (offline, a fault never ends; uncapped, the sum would stay over the threshold for seconds)."""
        self.last = torch.tensor([self.length], device=device)
        self.reset()

    def reset(self) -> None:
        self.n = self.run = 0
        self.stat = torch.zeros(self.src.num_joints, device=self.src.device)
        self.flags = torch.zeros(self.src.num_joints, dtype=torch.bool, device=self.src.device)

    @property
    def armed(self) -> bool:
        return self.n > self.skip

    @torch.no_grad()
    def update(self, sample: torch.Tensor) -> torch.Tensor:
        """One raw sample (``names`` order) -> per-joint alarm flags (the alarm holds and the joint's statistic is
        over the threshold; the diagnosis of the first alarm is the argmax joint)."""
        x = self.src.x
        if self.n == 0:
            x[:] = sample
        else:
            x[:-1] = x[1:].clone()
            x[-1] = sample
        p = torch.sigmoid(self.model(self.src.windows(self.last, self.noise))[0][0])
        z = p.clamp(1e-7, 1).log() - (1 - p).clamp(1e-7, 1).log()
        drift, thr = self.rule["drift"], self.rule["thr"]
        if drift is None:
            self.stat = z
        elif self.n >= self.skip:
            self.stat = (self.stat + z - drift).clamp(min=0.0, max=self.cap)
        self.run = self.run + 1 if self.stat.max() > thr else 0
        alarm = self.n >= self.skip and self.run >= self.rule["k"]
        self.flags = (self.stat > thr) & alarm
        self.n += 1
        return self.flags


class Panel:
    """Kit window: per joint a ground-truth light (+ health) and a detector light (+ statistic)."""

    def __init__(self, joint_names: list[str], rule: dict):
        import omni.ui as ui

        self.window = ui.Window("Motor faults: GT vs detector", width=420, height=560)
        light = lambda: ui.Circle(radius=7, width=18, height=18, style={"background_color": GRAY})  # noqa: E731
        with self.window.frame:
            with ui.VStack(spacing=4, height=0):
                self.status = ui.Label("", word_wrap=True, height=80)
                with ui.HStack(height=20):
                    ui.Label("joint", width=150)
                    ui.Label("ground truth", width=120)
                    ui.Label(f"detector ({rule['rule']} > {rule['thr']:.1f})")
                self.rows = []
                for name in joint_names:
                    with ui.HStack(height=22, spacing=4):
                        ui.Label(name, width=150)
                        gt, gt_text = light(), ui.Label("", width=98)
                        det, det_text = light(), ui.Label("")
                    self.rows.append((gt, gt_text, det, det_text))

    def update(self, health, marks, stat, armed: bool, status: str) -> None:
        """``marks``: per joint the detection text (latency) while it is flagged, else None."""
        self.status.text = status
        for (gt, gt_text, det, det_text), h, m, s in zip(self.rows, health, marks, stat):
            gt.set_style({"background_color": RED if h < 1 else GREEN})
            gt_text.text = f"factor={h:.2f}" if h < 1 else ""
            det.set_style({"background_color": (RED if m else GREEN) if armed else GRAY})
            det_text.text = (m or f"{s:.1f}") if armed else "arming"


class Overlay:
    """The panel drawn into recorded frames (same rows and status as :class:`Panel`)."""

    def __init__(self, joint_names: list[str], rule: dict):
        from PIL import ImageFont

        self.names, self.state = joint_names, None
        self.header = f"detector ({rule['rule']} > {rule['thr']:.1f})"
        try:
            self.font = ImageFont.load_default(size=15)
        except TypeError:  # Pillow < 10.1: fixed-size bitmap font
            self.font = ImageFont.load_default()

    def update(self, health, marks, stat, armed: bool, status: str) -> None:
        self.state = (health, marks, stat, armed, status)

    def clear(self) -> None:
        self.state = None

    def draw(self, frame) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.ascontiguousarray(np.asarray(frame)[..., :3]))
        if self.state is None:
            return np.asarray(img)
        health, marks, stat, armed, status = self.state
        d = ImageDraw.Draw(img, "RGBA")
        rgb = lambda c: (c & 0xFF, (c >> 8) & 0xFF, (c >> 16) & 0xFF)  # noqa: E731
        lines = status.split("\n")
        x0, y0, row = 10, 10, 22
        top = y0 + 8 + row * len(lines)
        d.rectangle([x0, y0, x0 + 530, top + row * (len(self.names) + 1) + 8], fill=(0, 0, 0, 160))
        for i, text in enumerate(lines):
            d.text((x0 + 10, y0 + 8 + row * i), text, fill="white", font=self.font)
        for x, text in ((x0 + 10, "joint"), (x0 + 160, "ground truth"), (x0 + 290, self.header)):
            d.text((x, top), text, fill=(200, 200, 200), font=self.font)
        for i, (name, h, m, s) in enumerate(zip(self.names, health, marks, stat)):
            y = top + row * (i + 1)
            d.text((x0 + 10, y), name, fill="white", font=self.font)
            det = ((RED, m) if m else (GREEN, f"{s:.1f}")) if armed else (GRAY, "arming")
            for x, color, text in ((x0 + 160, *((RED, f"factor={h:.2f}") if h < 1 else (GREEN, ""))), (x0 + 290, *det)):
                d.ellipse([x, y + 2, x + 14, y + 16], fill=rgb(color))
                d.text((x + 22, y), text, fill="white", font=self.font)
        return np.asarray(img)


class FaultSchedule:
    """Healthy and faulty periods of random length, one faulty joint at most: replaces the injector's one fault per
    episode. Each fault picks a joint, a type (``fault_types``) and a health (``power_loss_health``) at random."""

    def __init__(self, fault_cfg, num_joints: int, off_s, on_s, step_dt: float, seed: int):
        self.cfg, self.num_joints, self.off_s, self.on_s, self.dt = fault_cfg, num_joints, off_s, on_s, step_dt
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> None:
        self.t, self.fault = 0.0, None
        self.until = self.rng.uniform(*self.off_s)

    def step(self) -> list[float]:
        """Advance one policy step; the health of every actuator column for the next step."""
        self.t += self.dt
        if self.t >= self.until:
            if self.fault is None:
                kind = self.rng.choices(list(self.cfg.fault_types), weights=list(self.cfg.fault_types.values()))[0]
                h = 0.0 if kind == "power_out" else self.rng.uniform(*self.cfg.power_loss_health)
                self.fault, self.until = (self.rng.randrange(self.num_joints), h), self.t + self.rng.uniform(*self.on_s)
            else:
                self.fault, self.until = None, self.t + self.rng.uniform(*self.off_s)
        health = [1.0] * self.num_joints
        if self.fault is not None:
            health[self.fault[0]] = self.fault[1]
        return health


class FaultView:
    """Ground truth and detector of env 0, per PD sample; the panel and the console line refresh per policy step.

    Faults may come and go within an episode. A detection (a joint's flag turning on) is scored against the fault
    active at that sample: its latency from that fault's onset, a wrong joint, or a false alarm (no fault)."""

    def __init__(self, detector: LiveDetector, joint_names: list[str], sample_dt: float, step_dt: float, panel=None,
                 log=print):  # fmt: skip
        self.det, self.names, self.dt, self.step_dt, self.panel, self.log = (
            detector, joint_names, sample_dt, step_dt, panel, log
        )  # fmt: skip
        self.episode = 0
        self.reset()

    def reset(self) -> None:
        self.det.reset()
        self.n, self.health, self.line = 0, None, None
        self.fault: tuple[int, float, int] | None = None
        """Active fault: joint, health, onset sample."""
        self.detected: str | None = None
        """The active (or last) fault's detection."""
        self.marks: dict[int, str] = {}
        """Flagged joint -> its detection text, kept while the flag holds."""

    def ms(self, samples: int) -> str:
        ms = samples * self.dt * 1000
        return f"{ms:.0f} ms ({ms / (self.step_dt * 1000):.2f} steps)"

    def update(self, sample: torch.Tensor, health: torch.Tensor) -> None:
        # ``health`` is that of the substep starting at this sample: a fault starts (the motor fails) at the sample
        # before its first faulty substep, so latencies count from the failure itself, one PD period (5 ms) more than
        # the study's metric (from the first sample after the faulty substep, the first one that can show it)
        faulty = (health < 1).nonzero().flatten().tolist()
        if faulty and (self.fault is None or self.fault[0] != faulty[0]):
            self.end_fault()
            self.fault, self.detected = (faulty[0], float(health[faulty[0]]), self.n), None
        elif not faulty and self.fault is not None:
            self.end_fault()
        self.health = health
        flags = self.det.update(sample).tolist()
        for j, f in enumerate(flags):
            if not f:
                self.marks.pop(j, None)
            elif j not in self.marks:
                if self.fault is None:
                    self.marks[j] = "false alarm"
                elif self.fault[0] != j:
                    self.marks[j] = f"wrong joint ({self.names[self.fault[0]]} is faulty)"
                else:
                    self.marks[j] = self.ms(self.n - self.fault[2])
                    self.detected = self.detected or f"{self.names[j]} after {self.marks[j]}"
        self.n += 1

    def end_fault(self) -> None:
        if self.fault is None:
            return
        j, h, onset = self.fault
        self.log(f"[fault] {self.names[j]} factor={h:.2f} from t={onset * self.dt:.2f} s for "
                 f"{(self.n - onset) * self.dt:.2f} s: {self.detected or 'MISSED'}")  # fmt: skip
        self.fault = None

    def refresh(self) -> None:
        if self.health is None:
            return
        health, stat = self.health.tolist(), self.det.stat.tolist()
        marks = [self.marks.get(j) for j in range(len(self.names))]
        gt = ", ".join(f"{n}: factor={h:.2f}" for n, h in zip(self.names, health) if h < 1)
        det = ", ".join(f"{n}: {m}" for n, m in zip(self.names, marks) if m)
        line = f"GT {{{gt}}} | fault detector {{{det}}}"
        t = self.n * self.dt
        if line != self.line:
            self.log(f"[t={t:6.2f} s] {line}")
            self.line = line
        if self.panel is not None:
            status = (f"episode {self.episode}, t = {t:.2f} s\nGT: {{{gt}}}\ndetector: {{{det}}}\n"
                      f"last fault detected: {self.detected or '-'}")  # fmt: skip
            self.panel.update(health, marks, stat, self.det.armed, status)

    def end_episode(self) -> None:
        self.end_fault()
        self.log(f"[episode {self.episode}] ended at t={self.n * self.dt:.2f} s")
        self.episode += 1
        if isinstance(self.panel, Overlay):
            self.panel.clear()
        self.reset()


def selftest() -> None:
    """The live detector on validation episodes of a synthetic run equals the offline statistic and first alarm."""
    import tempfile

    import test_detector as td

    with tempfile.TemporaryDirectory() as root:
        td.make_dataset(root, num_episodes=400)
        out = os.path.join(root, "out")
        overrides = [f"dataset_dir={root}", "test_dirs=[]", "epochs=1", "samples_per_epoch=8192", "batch_size=512",
                     "val_windows=2048", "hnm_epochs=1", "skip_initial_s=0.05", "onset_window_s=0.1"]  # fmt: skip
        xp.cmd_run(argparse.Namespace(out=out, specs=["J=joint/current"], seeds=[0], test=False, overrides=overrides))
        run_dir = os.path.join(out, "J_s0")
        joint_names = fd.read_meta(root)["joint_names"]
        cfg0 = fd.DetectorCfg(**{**xp.BASE, **ex.parse_overrides(overrides)})
        val = fd.load_episodes(root, xp.RAW, fd.split_episodes(fd.read_index(root), cfg0.split)["val"], "pd")
        live = LiveDetector(run_dir, val.names, joint_names, val.sample_dt, "cpu", noise=False)
        state = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu", weights_only=False)
        z = xp.scores(live.spec, state, live.cfg, val, "cpu", joint_names, ex.joint_legs(joint_names), False)
        with open(os.path.join(run_dir, "val.json")) as f:
            rules = {r["rule"]: r for r in json.load(f)["rules"]}
        for rule in (rules["debounce_4"], rules["cusum_q0.99"]):
            # uncapped: the offline CUSUM has no ceiling
            live.rule, live.cap = rule, float("inf")
            stat = xp.statistic(val, z, rule, live.skip, "cpu")
            on = xp.alarms(stat, rule["thr"], rule["k"]) & (val.t >= live.skip)
            alarmed = 0
            for e in range(min(len(val.ep_len), 20)):
                s, n = int(val.ep_start[e]), int(val.ep_len[e])
                live.reset()
                got, first = [], None
                for i in range(n):
                    if live.update(torch.from_numpy(val.x[s + i].astype(np.float32))).any() and first is None:
                        first = i
                    got.append(live.stat.numpy().copy())
                assert np.allclose(np.array(got), stat[s : s + n], rtol=1e-4, atol=1e-3), (rule["rule"], e)
                hits = np.flatnonzero(on[s : s + n])
                assert first == (int(hits[0]) if len(hits) else None), (rule["rule"], e, first, hits[:3])
                alarmed += first is not None
            assert alarmed > 0, rule["rule"]
    check_faults()
    print("demo selftest passed")


def check_faults() -> None:
    """FaultSchedule alternates healthy / one-joint periods; FaultView scores detections against the active fault."""
    cfg = argparse.Namespace(fault_types={"power_out": 0.5, "power_loss": 0.5}, power_loss_health=(0.1, 0.6))
    schedule = FaultSchedule(cfg, 12, (0.1, 0.3), (0.1, 0.3), 0.02, seed=0)
    runs, last = [], None
    for _ in range(500):
        health = schedule.step()
        assert sum(h < 1 for h in health) <= 1, health
        fault = next((j for j, h in enumerate(health) if h < 1), None)
        if fault != last:
            runs.append(fault)
            last = fault
    assert len(runs) > 10 and all((a is None) != (b is None) for a, b in zip(runs, runs[1:])), runs[:6]

    class Stub:  # flags[n]: joints flagged at sample n
        flags_at: dict[int, list[int]] = {}

        def reset(self):
            self.n = 0

        def update(self, _sample):
            flags = torch.zeros(12, dtype=torch.bool)
            flags[self.flags_at.get(self.n, [])] = True
            self.n += 1
            return flags

    lines: list[str] = []
    view = FaultView(Stub(), [f"j{i}" for i in range(12)], 0.005, 0.02, log=lines.append)
    # joint 2 fails at sample 10 (its substep is faulty, seen from sample 11 on) and recovers at 30; joint 7 fails at
    # 40 for 10 samples, never detected
    Stub.flags_at = {14: [2], 15: [2, 5], 33: [9]}
    for n in range(60):
        health = torch.ones(12)
        if 10 <= n < 30:
            health[2] = 0.4
        if 40 <= n < 50:
            health[7] = 0.0
        view.update(None, health)
        if n == 15:
            assert view.marks == {2: "20 ms (1.00 steps)", 5: "wrong joint (j2 is faulty)"}, view.marks
        if n == 33:
            assert view.marks == {9: "false alarm"}, view.marks
    view.end_episode()
    faults = [line for line in lines if line.startswith("[fault]")]
    assert faults == ["[fault] j2 factor=0.40 from t=0.05 s for 0.10 s: j2 after 20 ms (1.00 steps)",
                      "[fault] j7 factor=0.00 from t=0.20 s for 0.05 s: MISSED"], faults  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Live motor-fault demo: ground truth vs detector, one robot.")
    parser.add_argument("--detector", default=None, help=f"Run directory of tmp/explore.py (default: {DETECTORS}).")
    parser.add_argument("--fault_types", nargs="+", default=["power_loss=1"], help="e.g. power_out=0.5 power_loss=0.5")
    parser.add_argument("--power_loss_health", type=float, nargs=2, default=None, help="h range (task: 0.1 0.6).")
    parser.add_argument("--off_s", type=float, nargs=2, default=(2.0, 5.0), help="Healthy period range [s].")
    parser.add_argument("--on_s", type=float, nargs=2, default=(2.0, 5.0), help="Fault period range [s].")
    parser.add_argument("--fault_seed", type=int, default=None, help="Seed of the fault schedule (default: random).")
    parser.add_argument("--no_noise", action="store_true", help="Feed the detector noiseless measurements.")
    parser.add_argument("--console", action="store_true", help="No window (e.g. on Euler): console lines only.")
    parser.add_argument("--record", default=None, help="Headless: write one mp4 per episode (+ episodes.log) here.")
    parser.add_argument("--episodes", type=int, default=6, help="With --record: episodes to record.")
    parser.add_argument("--episode_s", type=float, default=None, help="Episode length [s] (default: task / --record).")
    parser.add_argument("--eye", type=float, nargs=3, default=(-1.5, -1.5, 0.8), help="With --record: camera offset.")
    parser.add_argument("--selftest", action="store_true", help="Check the live detector offline and exit.")
    demo_args, argv = parser.parse_known_args(sys.argv[1:])
    if demo_args.selftest:
        selftest()
        return
    run_dir = demo_args.detector or next((d for d in DETECTORS if os.path.isdir(d)), None)
    if run_dir is None or not os.path.isfile(os.path.join(run_dir, "checkpoint.pt")):
        sys.exit(f"[demo] no detector run at {run_dir or DETECTORS}; pass --detector <dir with checkpoint.pt>")
    if demo_args.record:
        demo_args.console = True
    viz = [] if demo_args.console or "--viz" in argv else ["--viz", "kit"]
    argv = (["--task", TASK] if "--task" not in argv else []) + viz + argv
    if "--checkpoint" not in argv and os.path.isfile(POLICY):
        argv += ["--checkpoint", POLICY]

    # set before importing anything that defines Warp kernels (see play.py)
    import warp as wp

    wp.config.enable_backward = False

    from collect import pd_sample
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint
    from isaaclab_rl.entrypoints.common import (
        apply_env_overrides,
        close_env,
        create_isaaclab_env,
        pre_launch_video_config,
        run_playback,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    args_cli = _parse_args(argv)
    # the demo only makes sense at wall-clock speed
    args_cli.real_time = not demo_args.console
    args_cli.num_envs = 1

    # the collection's env (play_mode=False), so the detector sees its training distribution
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    if demo_args.record:
        # a capture-capable headless visualizer (Newton GL unless --viz picks one), as in fault_analysis/render.py
        args_cli.video = "viz"
        pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        # frames are captured here, not by the env's recorder
        env_cfg.video_recorders = []
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        env_cfg.scene.terrain.terrain_generator.seed = 0
        fault_cfg = env_cfg.events.motor_fault.params["cfg"]
        # the injector keeps the faulty actuator but schedules nothing: FaultSchedule sets the health each step
        fault_cfg.p_fault = 0.0
        fault_cfg.fault_types = {k: float(v) for k, v in (s.split("=") for s in demo_args.fault_types)}
        if demo_args.power_loss_health is not None:
            fault_cfg.power_loss_health = tuple(demo_args.power_loss_health)
        if demo_args.episode_s is not None:
            env_cfg.episode_length_s = demo_args.episode_s
        elif demo_args.record:
            env_cfg.episode_length_s = 40.0
        seed = demo_args.fault_seed if demo_args.fault_seed is not None else random.randrange(1 << 31)
        # camera follows the robot
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (-2.5, -2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            base = env.unwrapped
            robot = base.scene["robot"]
            injector = base.event_manager.get_term_cfg("motor_fault").func
            sample_dt = base.step_dt / base.cfg.decimation

            def raw_sample() -> tuple[torch.Tensor, torch.Tensor]:
                """Env 0's raw detector inputs (``xp.RAW`` order) and the health of the substep about to run."""
                s = pd_sample(base)
                s["tracking_error"] = s["joint_pos_target"] - s["joint_pos_rel"]
                health = torch.ones(robot.num_joints, device=base.device)
                health[injector.columns] = injector.actuator.health[0]
                return torch.cat([s[name][0] for name in xp.RAW]).float(), health

            shapes = {k: v.shape[1] for k, v in pd_sample(base).items()}
            shapes["tracking_error"] = robot.num_joints
            names = [f"{name}[{i}]" for name in xp.RAW for i in range(shapes[name])]
            detector = LiveDetector(run_dir, names, robot.joint_names, sample_dt, base.device, not demo_args.no_noise)
            panel = Overlay(robot.joint_names, detector.rule) if demo_args.record else None
            if not demo_args.console:
                try:
                    panel = Panel(robot.joint_names, detector.rule)
                except Exception as err:  # no Kit window: console only
                    print(f"[demo] no panel ({err}); console output only", flush=True)
            log_path = None
            if demo_args.record:
                os.makedirs(demo_args.record, exist_ok=True)
                log_path = os.path.join(demo_args.record, "episodes.log")
                open(log_path, "w").close()

            def log(msg: str) -> None:
                print(msg, flush=True)
                if log_path is not None:
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")

            view = FaultView(detector, robot.joint_names, sample_dt, base.step_dt, panel, log)
            actuator = injector.actuator
            schedule = FaultSchedule(
                fault_cfg, actuator.num_joints, demo_args.off_s, demo_args.on_s, base.step_dt, seed
            )
            pending = []
            actuator.probe = lambda: pending.append(raw_sample())
            log(f"[demo] policy {resume_path}\n[demo] detector {os.path.abspath(run_dir)} ({detector.rule['rule']}, "
                f"thr {detector.rule['thr']:.2f}), noise {'off' if demo_args.no_noise else 'on'}, fault seed {seed}, "
                f"faults {fault_cfg.fault_types}, healthy {demo_args.off_s} s / faulty {demo_args.on_s} s")  # fmt: skip

            obs = env.get_observations()
            pending.clear()

            def step() -> None:
                nonlocal obs
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                # the samples of this step belong to the episode that ran it, even if it just reset
                for sample, health in pending:
                    view.update(sample, health)
                pending.clear()
                view.refresh()
                if dones[0]:
                    view.end_episode()
                    schedule.reset()
                # after the injector's event (which wrote h = 1): the health of the next step's substeps
                actuator.health[0] = torch.tensor(schedule.step(), device=base.device)

            if demo_args.record:
                record(base, step, view, panel, demo_args, log)
            else:
                run_playback(step, dt=base.step_dt, args_cli=args_cli, env_cfg=env_cfg)
        finally:
            close_env(env)


def record(base, step, view: FaultView, overlay: Overlay, demo_args, log) -> None:
    """Step ``--episodes`` episodes; every policy step renders the robot from ``--eye`` with the overlay into
    ``<record>/episode_<k>.mp4`` (frames streamed to ffmpeg, so long clips need no memory)."""
    import imageio_ffmpeg

    from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES

    viz = next(v for v in base.sim.visualizers if v.cfg.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES)
    robot = base.scene["robot"]
    eye = torch.tensor(demo_args.eye, device=base.device)
    fps = round(1.0 / base.step_dt)
    writer, path = None, None
    with torch.inference_mode():
        while view.episode < demo_args.episodes:
            episode = view.episode
            step()
            if view.episode != episode:  # it just ended; the scene already shows the next episode's reset
                writer.close()
                log(f"[demo] wrote {path}")
                writer = None
                continue
            target = robot.data.root_pos_w.torch[0]
            base.sim.set_camera_view(tuple((target + eye).tolist()), tuple(target.tolist()))
            base.sim.render()
            frame = overlay.draw(viz.render_rgb_array())
            if writer is None:
                path = os.path.abspath(os.path.join(demo_args.record, f"episode_{episode:02d}.mp4"))
                writer = imageio_ffmpeg.write_frames(path, (frame.shape[1], frame.shape[0]), fps=fps)
                writer.send(None)
            writer.send(np.ascontiguousarray(frame))
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
