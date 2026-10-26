# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Does a motor fault reach the physics? One scenario per env, the frozen policy walking.

Every env walks healthy for ``--healthy_s``, then holds one fault (a joint set and a health ``h``) for ``--fault_s``.
Per scenario it reports, faulty healthy window vs fault window:

* the torque the actuator outputs (``applied_effort``) and the torque handed to Newton (``control.joint_f``),
  their ratio to the healthy window, and the largest |difference| between the two (should be 0);
* the PD demand (``computed_effort``), the tracking error |q* - q| of the faulty joints;
* base-height drop, xy velocity-tracking error, and whether the robot fell (terminated).

Run (Euler, profile isaaclab3)::

    python scripts/forge/fault_detector/tmp/torque_check.py --checkpoint <model.pt> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from demo import POLICY, TASK  # noqa: E402

JOINTS = ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RL_calf_joint")
HEALTHS = (0.6, 0.3, 0.1, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--healthy_s", type=float, default=3.0)
    parser.add_argument("--fault_s", type=float, default=5.0)
    parser.add_argument("--reps", type=int, default=3, help="Envs per scenario.")
    parser.add_argument("--out", default=None, help="Directory for results.json.")
    args, argv = parser.parse_known_args(sys.argv[1:])
    argv = (["--task", TASK] if "--task" not in argv else []) + argv
    if "--checkpoint" not in argv and os.path.isfile(POLICY):
        argv += ["--checkpoint", POLICY]

    import warp as wp

    wp.config.enable_backward = False

    from collect import pd_sample
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint
    from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.utils import resolve_task_config

    scenarios = [("none", (), 1.0)] + [(j, (j,), h) for j in JOINTS for h in HEALTHS]
    scenarios += [("all", "all", h) for h in (0.1, 0.0)]
    args_cli = _parse_args(argv)
    args_cli.num_envs = len(scenarios) * args.reps
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        env_cfg.episode_length_s = args.healthy_s + args.fault_s + 5.0
        fault_cfg = env_cfg.events.motor_fault.params["cfg"]
        fault_cfg.p_fault = 0.0
        resume_path = _resolve_checkpoint(
            args_cli, agent_cfg, env_cfg, os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        )
        env = RslRlVecEnvWrapper(
            create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False),
            clip_actions=agent_cfg.clip_actions,
        )
        try:
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            base = env.unwrapped
            robot = base.scene["robot"]
            act = base.event_manager.get_term_cfg("motor_fault").func.actuator
            dev, n = base.device, base.num_envs
            arti_cols = act.joint_indices if isinstance(act.joint_indices, torch.Tensor) else slice(None)

            def t(x):
                return x.torch if hasattr(x, "torch") else x

            print(f"[check] policy {resume_path}\n[check] actuator {type(act).__name__} joints {act.joint_names}")
            print(f"[check] passive_damping {act.passive_damping}, joint_ordering {robot.data.has_joint_ordering}")
            print(f"[check] sim drive stiffness {t(robot.data.joint_stiffness)[0].tolist()}")
            print(f"[check] sim drive damping {t(robot.data.joint_damping)[0].tolist()}")
            print(f"[check] actuator kp {t(act.stiffness)[0].tolist()} kd {t(act.damping)[0].tolist()}")
            print(f"[check] effort limit {t(act.effort_limit)[0].tolist()}")

            # per-env target health (actuator columns) and faulty-joint mask (actuator columns)
            health = torch.ones(n, act.num_joints, device=dev)
            mask = torch.zeros(n, act.num_joints, dtype=torch.bool, device=dev)
            for k, (_, joints, h) in enumerate(scenarios):
                cols = range(act.num_joints) if joints == "all" else [act.joint_names.index(j) for j in joints]
                for c in cols:
                    health[k * args.reps : (k + 1) * args.reps, c] = h
                    mask[k * args.reps : (k + 1) * args.reps, c] = True

            phase = {"v": 0}  # 0 healthy, 1 fault, -1 warm-up
            keys = ("applied", "sim", "demand", "track", "count")
            acc = {p: {k: torch.zeros(n, act.num_joints, device=dev) for k in keys} for p in (0, 1)}
            max_diff = torch.zeros(n, device=dev)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)

            def probe() -> None:
                """At each PD substep: the previous substep's actuator output and Newton input (same substep)."""
                p = phase["v"]
                if p < 0:
                    return
                try:
                    sim = wp.to_torch(robot.data._sim_bind_joint_effort)
                except Exception:  # an indexed view
                    sim = wp.to_torch(wp.clone(robot.data._sim_bind_joint_effort))
                if robot.data.has_joint_ordering:
                    sim = sim[:, robot.data.joint_ordering.user_to_backend_indices]
                sim = sim[:, arti_cols]
                applied = act.applied_effort
                # envs that fell are reset and walk again: only the steps before the fall count
                w = (~fell).float()[:, None]
                max_diff.copy_(torch.maximum(max_diff, ((sim - applied).abs() * w).max(dim=1).values))
                s = pd_sample(base)
                track = (s["joint_pos_target"] - s["joint_pos_rel"])[:, arti_cols]
                a = acc[p]
                a["applied"] += w * applied**2
                a["sim"] += w * sim**2
                a["demand"] += w * act.computed_effort**2
                a["track"] += w * track**2
                a["count"] += w

            act.probe = probe
            steps = {0: round(args.healthy_s / base.step_dt), 1: round(args.fault_s / base.step_dt)}
            warm = round(1.0 / base.step_dt)
            height = {p: torch.zeros(n, device=dev) for p in (0, 1)}
            vel_err = {p: torch.zeros(n, device=dev) for p in (0, 1)}
            fell_t = torch.full((n,), float("nan"), device=dev)
            obs = env.get_observations()
            for p, count in ((-1, warm), (0, steps[0] - warm), (1, steps[1])):
                phase["v"] = p
                for i in range(count):
                    with torch.inference_mode():
                        obs, _, dones, extras = env.step(policy(obs))
                    if p == 1:
                        new = dones.bool() & ~fell
                        fell_t[new] = (i + 1) * base.step_dt
                        fell |= dones.bool()
                    # set after the step: the injector event (p_fault = 0) has just reset h to 1
                    act.health[:] = health if p >= 0 and (p == 1 or i == count - 1) else 1.0
                    if p >= 0:
                        alive = ~fell
                        height[p] += t(robot.data.root_pos_w)[:, 2] * alive
                        cmd = base.command_manager.get_command("base_velocity")[:, :2]
                        vel_err[p] += (cmd - t(robot.data.root_lin_vel_b)[:, :2]).norm(dim=1) * alive
            # the fault-phase means count only the steps before a fall
            alive_steps = torch.where(fell, fell_t / base.step_dt, torch.full_like(fell_t, steps[1]))
            alive_steps = alive_steps.clamp(min=1)

            results = []
            cols = "scenario h | applied sim demand (x healthy) | track track0 [rad] | dz [m] v_err v_err0 | fell"
            print("\n" + cols + " | max|sim-applied| [N*m]")
            for k, (name, joints, h) in enumerate(scenarios):
                rows = slice(k * args.reps, (k + 1) * args.reps)
                m = mask[rows] if joints else torch.ones_like(mask[rows])

                def rms(p, key, rows=rows, m=m):
                    a = acc[p]
                    return float(((a[key][rows] / a["count"][rows].clamp(min=1)) * m).sum() / m.sum()) ** 0.5

                ratio = {key: rms(1, key) / max(rms(0, key), 1e-9) for key in ("applied", "sim", "demand")}
                n0 = steps[0] - warm
                dz = float((height[1][rows] / alive_steps[rows] - height[0][rows] / n0).mean())
                v1 = float((vel_err[1][rows] / alive_steps[rows]).mean())
                v0 = float((vel_err[0][rows] / n0).mean())
                fall_times = [round(float(x), 2) for x in fell_t[rows] if x == x]
                row = dict(scenario=name, h=h, ratio=ratio, track=rms(1, "track"), track_healthy=rms(0, "track"))
                row |= dict(dz=dz, vel_err=v1, vel_err_healthy=v0, fell=int(fell[rows].sum()), fell_t=fall_times)
                row |= dict(reps=args.reps, max_sim_minus_applied=float(max_diff[rows].max()))
                results.append(row)
                r = ratio
                print(
                    f"{name:<15}{h:>4.1f} | {r['applied']:5.2f} {r['sim']:5.2f} {r['demand']:5.2f} | "
                    f"{row['track']:.3f} {row['track_healthy']:.3f} | {dz:+.3f} {v1:.2f} {v0:.2f} | "
                    f"{row['fell']}/{args.reps} {fall_times} | {row['max_sim_minus_applied']:.1e}"
                )
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, "results.json"), "w") as f:
                    json.dump(results, f, indent=1)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
