# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an ``IsaacContrib-Velocity-A2LegGuard-Stand`` policy on its own task (training flag schedule, pushes kept).

Per flagged leg and time since the flag came on: the flagged foot's contact fraction (``F_z > 5 N``), its mean
``F_z`` and its height above the lowest other foot; falls per robot-minute with and without a flag on. Writes
``result.json``::

    python scripts/forge/leg_guard/stand_eval.py --task IsaacContrib-Velocity-A2LegGuard-Stand \
        --checkpoint <model.pt> --num_envs 1024 --out <dir>
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "surveillance"))
from collect import RUNNERS  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
FEET = [f"{leg}_foot" for leg in LEGS]
THRESHOLD = 5.0
"""Contact threshold [N], as the task's penalty."""
BINS_S = (0.0, 0.25, 0.5, 1.0, 2.0, 1e9)
"""Time since the flag came on [s]."""


def _t(x):
    return getattr(x, "torch", x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=2.0, help="Simulated minutes per robot.")
    parser.add_argument("--out", required=True)
    ga, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        os.makedirs(ga.out, exist_ok=True)
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            dev = env.unwrapped.device
            runner = RUNNERS[agent_cfg.class_name](env, to_dict(agent_cfg), log_dir=None, device=dev)
            load_cfg = {"actor": True, "critic": False, "optimizer": False, "iteration": False}
            runner.load(resume_path, load_cfg=load_cfg, map_location=dev)
            policy = runner.get_inference_policy(device=dev)
            base = env.unwrapped
            n, dt = base.num_envs, base.step_dt
            robot = base.scene["robot"]
            sensor = base.scene.sensors["contact_forces"]
            foot_sensor = sensor.find_bodies(FEET, preserve_order=True)[0]
            foot_body = robot.find_bodies(FEET, preserve_order=True)[0]
            nb, nl = len(BINS_S) - 1, len(LEGS)
            # per (leg, bin): steps, contact steps, sum F_z, sum height
            acc = torch.zeros(nl, nb, 4, device=dev)
            onset = torch.full((n,), -1, dtype=torch.long, device=dev)
            prev = torch.zeros(n, nl, device=dev)
            falls = torch.zeros(2, device=dev)
            robot_steps = torch.zeros(2, device=dev)
            bins = torch.tensor(BINS_S, device=dev)
            obs = env.get_observations()
            steps = int(ga.minutes * 60 / dt)
            with torch.inference_mode():
                for step in range(steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    # falls: dones that are not time-outs, split by the flag before the step (a reset redraws it)
                    was_on = prev.any(1)
                    fell = dones.bool() & ~base.termination_manager.time_outs
                    for k, mask in enumerate((~was_on, was_on)):
                        falls[k] += (fell & mask).sum()
                        robot_steps[k] += mask.sum()
                    flag = base.leg_flag.clone()
                    on = flag.any(1)
                    onset[on & ~(was_on & ~dones.bool())] = step
                    onset[~on] = -1
                    prev = flag
                    if not on.any():
                        continue
                    leg = flag.argmax(1)
                    fz = _t(sensor.data.net_forces_w)[:, foot_sensor, 2]
                    pos = _t(robot.data.body_pos_w)[:, foot_body, 2]
                    rows = torch.arange(n, device=dev)
                    f_leg = fz[rows, leg].clamp(min=0.0)
                    others = pos.clone()
                    others[rows, leg] = float("inf")
                    height = pos[rows, leg] - others.min(1).values
                    age = (step - onset).float() * dt
                    b = (torch.bucketize(age, bins, right=True) - 1).clamp(0, nb - 1)
                    idx = (leg * nb + b)[on]
                    vals = torch.stack((torch.ones_like(f_leg), (f_leg > THRESHOLD).float(), f_leg, height), 1)[on]
                    acc.view(nl * nb, 4).index_add_(0, idx, vals)
            acc, falls, robot_steps = acc.cpu(), falls.cpu(), robot_steps.cpu()
            minutes = robot_steps * dt / 60
            table = {}
            for li, leg in enumerate(LEGS):
                table[leg] = {}
                for bi in range(nb):
                    c = acc[li, bi]
                    if c[0] == 0:
                        continue
                    key = f"{BINS_S[bi]}-{BINS_S[bi + 1] if BINS_S[bi + 1] < 1e9 else 'inf'}s"
                    table[leg][key] = {
                        "steps": int(c[0]),
                        "contact_frac": float(c[1] / c[0]),
                        "mean_fz": float(c[2] / c[0]),
                        "mean_height": float(c[3] / c[0]),
                    }
            result = {
                "checkpoint": resume_path,
                "robots": n,
                "minutes_per_robot": ga.minutes,
                "flagged": table,
                "falls_per_robot_min": {
                    "no_flag": float(falls[0] / minutes[0].clamp(min=1e-9)),
                    "flag": float(falls[1] / minutes[1].clamp(min=1e-9)),
                },
                "falls": {"no_flag": int(falls[0]), "flag": int(falls[1])},
                "robot_minutes": {"no_flag": float(minutes[0]), "flag": float(minutes[1])},
            }
            with open(os.path.join(ga.out, "result.json"), "w") as f:
                json.dump(result, f, indent=2)
            print(json.dumps(result, indent=2))
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
