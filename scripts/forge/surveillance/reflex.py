# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Closed-loop test of a precursor reflex: the Robust policy walks a terrain map, and when a foot enters a terrain
event (low friction, sand, or a collapse; the privileged flags, i.e. a perfect onset detector) the velocity command is
replaced, ``--latency_s`` later, by a reflex command for ``--hold_s``; then the policy's command again, with
``--refractory_s`` before the next reflex. Reports falls (base contact) and forward progress per robot-minute, against
the same run with ``--reflex none``. Unlike the counterfactual restores this also pays for the reflexes on harmless
events::

    L=/workspace/isaaclab/logs/surveillance
    python scripts/forge/surveillance/reflex.py $L/v3_hard/Robust/slip_field --reflex back --out $L/reflex/slip_back
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from collect import RUNNERS, configure_terrain, left_tile  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

REFLEXES = {"none": None, "stop": (0.0, 0.0, 0.0), "back": (-0.3, 0.0, 0.0), "slow": "half"}


def proprio(robot, command) -> torch.Tensor:
    """One frame of ``detect.signals``' input [N, 48]: the logged proprioception fields (collect.py) + the body-frame
    specific force of the base."""
    d = robot.data
    q = d.root_link_quat_w.torch  # x, y, z, w
    acc = d.body_com_acc_w.torch[:, 0, :3] - torch.tensor([0.0, 0.0, -9.81], device=q.device)
    xyz, w = q[:, :3], q[:, 3:4]
    t = 2 * torch.cross(xyz, acc, dim=-1)
    imu = acc - w * t + torch.cross(xyz, t, dim=-1)
    return torch.cat([d.joint_pos.torch, d.joint_vel.torch, d.joint_pos_target.torch, d.projected_gravity_b.torch,
                      d.root_link_ang_vel_b.torch, command.vel_command_b, imu], 1)  # fmt: skip


def summarize(h: torch.Tensor) -> torch.Tensor:
    """Gate features of a [N, T, 48] history: mean, std, last, last - first per channel."""
    return torch.cat([h.mean(1), h.std(1, unbiased=False), h[:, -1], h[:, -1] - h[:, 0]], 1)


def load_gate(path: str, threshold: str, device) -> dict:
    """The gate of gate_train.py: ``predict(history [N, window, 48]) -> P(harmful) [N]`` on the summary features
    (mean, std, last, last - first per channel)."""
    from torch import nn

    ckpt = torch.load(path, map_location=device)
    din = ckpt["din"]
    net = nn.Sequential(nn.Linear(din, 128), nn.ELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.ELU(),
                        nn.Linear(64, 1)).to(device)  # fmt: skip
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    mean, std = ckpt["mean"].to(device), ckpt["std"].to(device)

    def predict(h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(net((summarize(h) - mean) / std)).squeeze(1)

    value = ckpt["thresholds"][threshold] if threshold in ckpt["thresholds"] else float(threshold)
    return {"predict": predict, "threshold": value, "window": ckpt["window"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collection", help="A collect.py collection: its task, checkpoint and seed are used.")
    parser.add_argument("--reflex", choices=list(REFLEXES), default="back")
    parser.add_argument("--latency_s", type=float, default=0.3)
    parser.add_argument("--hold_s", type=float, default=1.0)
    parser.add_argument("--refractory_s", type=float, default=1.0)
    parser.add_argument("--gate_model", default=None, help="Learned gate (gate_train.py of the reactions report): "
                        "fire only if it calls the event harmful at the end of the latency.")  # fmt: skip
    parser.add_argument("--gate_threshold", default="recall_0.75", help="A probability, or a key of the model's "
                        "thresholds.")  # fmt: skip
    parser.add_argument("--fire_prob", type=float, default=1.0, help="Fire on a random fraction of the onsets (the "
                        "control for a gate that fires on the same fraction).")  # fmt: skip
    parser.add_argument("--log_onsets", action="store_true", help="Save the gate features of every onset (at the "
                        "end of the latency) and whether the robot falls within 3 s, to <out>/onsets.npz.")  # fmt: skip
    parser.add_argument("--minutes", type=float, default=2.0, help="Simulated minutes per robot.")
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--out", required=True)
    rf_args, argv = parser.parse_known_args(sys.argv[1:])
    with open(os.path.join(rf_args.collection, "meta.json")) as f:
        meta = json.load(f)
    args_cli = _parse_args(["--task", meta["task"], "--checkpoint", meta["checkpoint"], "--num_envs",
                            str(rf_args.num_envs), "--seed", str(meta["seed"]), *argv])  # fmt: skip
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = meta["seed"]
        # as collect.configure_env: base contact is the only failure, leaving the tile is a success, no pushes
        configure_terrain(env_cfg, meta["seed"], meta["args"].get("sub_terrains"))
        for name, term in vars(env_cfg.terminations).items():
            if term is not None and not term.time_out and name != "base_contact":
                setattr(env_cfg.terminations, name, None)
        world = getattr(env_cfg, "terrain_world", None)
        os.makedirs(rf_args.out, exist_ok=True)
        if world is not None:
            env_cfg.scene.terrain.meta_path = os.path.join(rf_args.out, "terrain_map.yaml")
            env_cfg.terminations.left_tile = DoneTerm(func=left_tile, params={"half_size": world.tile_size / 2},
                                                      time_out=True)  # fmt: skip
        for name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            dev = env.unwrapped.device
            runner = RUNNERS[agent_cfg.class_name](env, to_dict(agent_cfg), log_dir=None, device=dev)
            runner.load(resume_path, map_location=dev)
            policy = runner.get_inference_policy(device=dev)
            base_env = env.unwrapped
            n, dt = base_env.num_envs, base_env.step_dt
            robot = base_env.scene["robot"]
            terrain = base_env.event_manager.get_term_cfg("terrain").func
            command = base_env.command_manager.get_term("base_velocity")
            # reflex state per robot: frames since the event onset (-1: none), and the policy's own command
            since = torch.full((n,), -1, dtype=torch.long, device=dev)
            ok = torch.ones(n, dtype=torch.bool, device=dev)  # the gate's decision for the current onset
            own = torch.zeros(n, 3, device=dev)
            gate = load_gate(rf_args.gate_model, rf_args.gate_threshold, dev) if rf_args.gate_model else None
            window = gate["window"] if gate else 25 if rf_args.log_onsets else 1
            history = torch.zeros(n, window, 48, device=dev)
            logged = {"x": [], "env": [], "step": []}  # onsets for --log_onsets
            fall_steps = [[] for _ in range(n)]
            latency, hold = round(rf_args.latency_s / dt), round(rf_args.hold_s / dt)
            end = latency + hold + round(rf_args.refractory_s / dt)
            reflex = REFLEXES[rf_args.reflex]
            compute = command.compute

            def compute_with_reflex(step_dt):
                compute(step_dt)
                own[:] = command.vel_command_b
                if reflex == "half":
                    command.vel_command_b[firing()] *= 0.5
                elif reflex is not None:
                    command.vel_command_b[firing()] = torch.tensor(reflex, device=dev)

            def firing():
                return (since >= latency) & (since < latency + hold) & ok

            command.compute = compute_with_reflex
            falls = progress = triggers = reflexes = reflex_frames = 0.0
            prev = torch.zeros(n, dtype=torch.bool, device=dev)
            obs = env.get_observations()
            steps = round(rf_args.minutes * 60 / dt)
            with torch.inference_mode():
                for step in range(steps):
                    heading = own[:, :2] / own[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-3)
                    obs, _, dones, _ = env.step(policy(obs))
                    # forward progress along the policy's own command (a reflex backwards counts negative)
                    v = robot.data.root_link_lin_vel_b.torch[:, :2]
                    progress += ((v * heading).sum(-1) * dt).sum().item()
                    falls += base_env.reset_terminated.sum().item()
                    event = terrain.low_mu.any(1) | terrain.in_sand.any(1) | terrain.collapse_event.any(1)
                    start = event & ~prev & (since < 0)
                    triggers += start.sum().item()
                    since = torch.where(start, 0, torch.where(since >= 0, since + 1, since))
                    since = torch.where((since >= end) | dones.bool(), -1, since)
                    if gate is not None or rf_args.log_onsets:
                        history = torch.cat([history[:, 1:], proprio(robot, command)[:, None]], 1)
                    decide = since == latency
                    if rf_args.log_onsets:
                        for i in base_env.reset_terminated.nonzero().flatten().tolist():
                            fall_steps[i].append(step)
                        if decide.any():
                            logged["x"].append(summarize(history[decide]).cpu())
                            logged["env"] += decide.nonzero().flatten().tolist()
                            logged["step"] += [step] * int(decide.sum())
                    if decide.any():
                        ok[decide] = torch.rand(int(decide.sum()), device=dev) < rf_args.fire_prob
                        if gate is not None:
                            ok[decide] &= gate["predict"](history[decide]) > gate["threshold"]
                    reflex_frames += firing().sum().item()
                    reflexes += ((since == latency) & ok).sum().item()
                    prev = event & ~dones.bool()
                    if (step + 1) % 1500 == 0:
                        print(f"[reflex] {rf_args.reflex} step {step + 1}/{steps}: falls {falls:.0f}, "
                              f"progress {progress:.0f} m, triggers {triggers:.0f}", flush=True)  # fmt: skip
            minutes = n * steps * dt / 60
            result = {"collection": rf_args.collection, "reflex": rf_args.reflex, "latency_s": rf_args.latency_s,
                      "hold_s": rf_args.hold_s, "gate_model": rf_args.gate_model,
                      "gate_threshold": gate["threshold"] if gate else None, "fire_prob": rf_args.fire_prob,
                      "robot_minutes": minutes,
                      "falls": falls, "falls_per_min": falls / minutes, "progress_m_per_min": progress / minutes,
                      "triggers_per_min": triggers / minutes, "reflexes_per_min": reflexes / minutes,
                      "reflex_time_fraction": reflex_frames / (n * steps)}  # fmt: skip
            print("[reflex]", json.dumps(result), flush=True)
            if rf_args.log_onsets and logged["x"]:
                horizon = round(3.0 / dt)
                onsets = zip(logged["env"], logged["step"])
                label = [any(s < f <= s + horizon for f in fall_steps[e]) for e, s in onsets]
                np.savez_compressed(os.path.join(rf_args.out, "onsets.npz"), x=torch.cat(logged["x"]).numpy(),
                                    env=np.array(logged["env"]), step=np.array(logged["step"]),
                                    fall_3s=np.array(label))  # fmt: skip
                print(f"[reflex] logged {len(label)} onsets, {sum(label)} followed by a fall", flush=True)
            with open(os.path.join(rf_args.out, "result.json"), "w") as f:
                json.dump(result, f, indent=1)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
