# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a policy of ``IsaacContrib-Velocity-A2FaultAware-*`` (H6): falls and velocity tracking per injected
condition, and the mode observation's alarms in the closed loop.

Fixed evaluation setting: robots on all terrain levels (no terrain curriculum), no pushes or external wrench, the
task's fault probabilities. Per finished episode it records the condition (none / motor power out / motor power loss /
stuck foot / motor + stuck, from the injectors' labels of the episode's last step), whether it fell (any termination
but the time-out), and its velocity tracking error before and after the fault onset. ``--mode_source`` overrides the
mode input (e.g. a detector-trained policy fed the oracle).

    python scripts/forge/fault_aware/evaluate.py --task IsaacContrib-Velocity-A2FaultAware-Oracle \
        --checkpoint <model.pt> --num_envs 1024 --headless --steps 3000 --output <dir>
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.forge.velocity.a2_fault_aware.mdp.fault_mode import DETECTOR_PATH  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

CONDITIONS = ("none", "power_out", "power_loss", "stuck", "motor+stuck")
MODE_DIM = 12


def condition(motor_type: torch.Tensor, stuck_seen: torch.Tensor) -> list[str]:
    """Condition name per env from the motor fault type (0 none, 1 out, 2 loss) and whether a foot was stuck."""
    names = []
    for m, s in zip(motor_type.tolist(), stuck_seen.tolist()):
        if m and s:
            names.append("motor+stuck")
        elif m:
            names.append("power_out" if m == 1 else "power_loss")
        else:
            names.append("stuck" if s else "none")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a fault-aware policy.")
    parser.add_argument("--steps", type=int, default=3000, help="Policy steps (50 Hz).")
    parser.add_argument("--mode_source", default=None, choices=["blind", "oracle", "detector"])
    parser.add_argument("--output", required=True, help="Directory for summary.json.")
    eval_args, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        if eval_args.mode_source:
            env_cfg.observations.policy.fault_mode.params["source"] = eval_args.mode_source
        # fixed setting: every terrain level, no curriculum, no disturbances but the faults
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.curriculum_factor = None
        generator = env_cfg.scene.terrain.terrain_generator
        env_cfg.scene.terrain.max_init_terrain_level = generator.num_rows - 1
        for name in ("push_robot", "base_external_force_torque"):
            if getattr(env_cfg.events, name, None) is not None:
                setattr(env_cfg.events, name, None)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path, load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False})
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            base = env.unwrapped
            motor_inj = base.event_manager.get_term_cfg("motor_fault").func
            stuck_inj = base.event_manager.get_term_cfg("foot_stuck").func
            robot = base.scene["robot"]
            n, dev = env.num_envs, base.device
            j = motor_inj.num_joints

            # per-env episode accumulators
            motor_type = torch.zeros(n, dtype=torch.long, device=dev)
            stuck_seen = torch.zeros(n, dtype=torch.bool, device=dev)
            ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
            err = {k: torch.zeros(n, device=dev) for k in ("pre", "post")}
            cnt = {k: torch.zeros(n, device=dev) for k in ("pre", "post")}
            speed = {k: torch.zeros(n, device=dev) for k in ("pre", "post")}
            stats = defaultdict(lambda: defaultdict(float))
            alarm = defaultdict(float)  # closed-loop alarms of the mode input
            # the detector term (when the mode input comes from it), for its alarm threshold and input statistics
            term = base.observation_manager._group_obs_term_cfgs["policy"][
                base.observation_manager._group_obs_term_names["policy"].index("fault_mode")
            ].func
            term = term if getattr(term, "source", "") == "detector" else None
            any_threshold = 0.5
            if term is not None:
                path = term.cfg.params.get("detector_path") or DETECTOR_PATH
                any_threshold = float(torch.load(path, map_location="cpu", weights_only=True)["any_threshold"])
                z_sum = torch.zeros(term.buffer.shape[-1], device=dev)
                z2_sum, z_n = torch.zeros_like(z_sum), 0

            obs = env.get_observations()
            start = time.time()
            with torch.inference_mode():
                for step in range(eval_args.steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    # labels of the step just taken (before this step's resets they describe the finished episode;
                    # the injectors' label buffers are updated in the interval event after the reset, so read the
                    # cached ones of the previous step for the envs that are done)
                    done = dones.bool()
                    for i in torch.nonzero(done).flatten().tolist():
                        c = condition(motor_type[i : i + 1], stuck_seen[i : i + 1])[0]
                        s = stats[c]
                        s["episodes"] += 1
                        s["falls"] += float(base.reset_terminated[i])
                        s["robot_s"] += float(ep_steps[i]) * base.step_dt
                        for k in ("pre", "post"):
                            if cnt[k][i] > 0:
                                s[f"err_{k}_sum"] += float(err[k][i] / cnt[k][i])
                                s[f"speed_{k}_sum"] += float(speed[k][i] / cnt[k][i])
                                s[f"n_{k}"] += 1
                    ep_steps += 1
                    for d in (motor_type, stuck_seen, ep_steps):
                        d[done] = 0
                    for k in ("pre", "post"):
                        err[k][done], cnt[k][done], speed[k][done] = 0.0, 0.0, 0.0
                    # this step's labels and tracking
                    m_lab = motor_inj.labels()
                    s_lab = stuck_inj.labels()
                    active_m = m_lab[:, :j].any(1)
                    active_s = s_lab[:, :4].any(1)
                    motor_type = torch.where(active_m, m_lab[:, j].long(), motor_type)
                    stuck_seen |= active_s
                    post = motor_type.bool() | stuck_seen
                    cmd = base.command_manager.get_command("base_velocity")[:, :2]
                    v = robot.data.root_link_lin_vel_b.torch[:, :2]
                    e = torch.linalg.norm(v - cmd, dim=1)
                    for k, mask in (("pre", ~post), ("post", post)):
                        err[k] += torch.where(mask, e, 0.0)
                        speed[k] += torch.where(mask, torch.linalg.norm(v, dim=1), 0.0)
                        cnt[k] += mask.float()
                    p_any = obs["policy"][:, -MODE_DIM]
                    fault_now = active_m | active_s
                    alarm["fault_steps"] += float(fault_now.sum())
                    alarm["nominal_steps"] += float((~post).sum())
                    alarm["p_any_fault_sum"] += float(p_any[fault_now].sum())
                    alarm["p_any_nominal_sum"] += float(p_any[~post].sum())
                    for name, th in (("0.5", 0.5), ("th", any_threshold)):
                        alarm[f"fault_alarm_{name}"] += float((fault_now & (p_any > th)).sum())
                        alarm[f"nominal_alarm_{name}"] += float((~post & (p_any > th)).sum())
                    # mode head on the alarmed steps of a single injected fault: actuator for a motor fault,
                    # obstruction for a stuck foot (indices in the mode head, see fault_mode.py)
                    mode = obs["policy"][:, -MODE_DIM + 2 : -4].argmax(1)
                    for name, mask, want in (("motor", active_m & ~active_s, 5), ("stuck", active_s & ~active_m, 3)):
                        hit = mask & (p_any > any_threshold)
                        alarm[f"{name}_alarmed"] += float(hit.sum())
                        alarm[f"{name}_mode_right"] += float((hit & (mode == want)).sum())
                    if term is not None and step > 50:
                        # normalized input of nominal steps: ~N(0, 1) per channel if it matches the training data
                        z = term.buffer[~post, -1]
                        z_sum += z.sum(0)
                        z2_sum += (z * z).sum(0)
                        z_n += len(z)
                    if step % 500 == 0:
                        eps = sum(s["episodes"] for s in stats.values())
                        print(f"[eval] step {step}, {eps:.0f} episodes, {time.time() - start:.0f} s", flush=True)

            summary = {"task": args_cli.task, "checkpoint": resume_path, "mode_source": eval_args.mode_source
                       or env_cfg.observations.policy.fault_mode.params["source"], "steps": eval_args.steps,
                       "num_envs": n, "conditions": {}}  # fmt: skip
            for c in CONDITIONS:
                s = stats.get(c)
                if not s:
                    continue
                summary["conditions"][c] = {
                    "episodes": int(s["episodes"]),
                    "fall_rate": round(s["falls"] / s["episodes"], 4),
                    "falls_per_robot_h": round(s["falls"] / max(s["robot_s"] / 3600, 1e-9), 2),
                    "track_err_pre": round(s["err_pre_sum"] / s["n_pre"], 4) if s["n_pre"] else None,
                    "track_err_post": round(s["err_post_sum"] / s["n_post"], 4) if s["n_post"] else None,
                    "speed_pre": round(s["speed_pre_sum"] / s["n_pre"], 4) if s["n_pre"] else None,
                    "speed_post": round(s["speed_post_sum"] / s["n_post"], 4) if s["n_post"] else None,
                }
            fs, ns = max(alarm["fault_steps"], 1), max(alarm["nominal_steps"], 1)
            summary["alarm"] = {
                "any_threshold": any_threshold,
                "p_any_mean_fault": round(alarm["p_any_fault_sum"] / fs, 4),
                "p_any_mean_nominal": round(alarm["p_any_nominal_sum"] / ns, 4),
                "fault_steps_alarmed_0.5": round(alarm["fault_alarm_0.5"] / fs, 4),
                "nominal_steps_alarmed_0.5": round(alarm["nominal_alarm_0.5"] / ns, 4),
                "fault_steps_alarmed_th": round(alarm["fault_alarm_th"] / fs, 4),
                "nominal_steps_alarmed_th": round(alarm["nominal_alarm_th"] / ns, 4),
                "motor_alarmed_mode_right": round(alarm["motor_mode_right"] / max(alarm["motor_alarmed"], 1), 4),
                "stuck_alarmed_mode_right": round(alarm["stuck_mode_right"] / max(alarm["stuck_alarmed"], 1), 4),
            }
            if term is not None and z_n:
                mean = z_sum / z_n
                rms = (z2_sum / z_n).sqrt()
                groups = {"q": 12, "dq": 12, "tau": 12, "taucmd": 12, "act": 12, "grav": 3, "gyro": 3, "acc": 3,
                          "cmd": 3, "fk": 12, "fkv": 12}  # fmt: skip
                i, zs = 0, {}
                for g, k in groups.items():
                    zs[g] = {
                        "mean": round(float(mean[i : i + k].abs().mean()), 3),
                        "rms": round(float(rms[i : i + k].mean()), 3),
                    }
                    i += k
                summary["detector_input_z_nominal"] = zs
            os.makedirs(eval_args.output, exist_ok=True)
            with open(os.path.join(eval_args.output, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print("[eval] " + json.dumps(summary), flush=True)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
