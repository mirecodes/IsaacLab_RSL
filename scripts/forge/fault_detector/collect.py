# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play the frozen policy with the motor-fault injector and save every finished episode for the detector.

Every policy step stores the noiseless ``detector``, ``privileged``, ``analysis`` and ``fault`` observation groups of
``IsaacContrib-FaultDetector-MotorFailure-A2Robust`` (see its ``REPORT.md``). Episodes are written as shards
``shards/<n>.npz`` (one ``<group>/<term>`` array of shape (sum of episode lengths, dim) per term, plus ``ep_len``
and ``episode_id``) and listed in ``index.csv``; ``meta.json`` holds term dims, joint names and the run's arguments.
With ``--pd_rate``, ``pd/<name>`` arrays (float16, decimation x the steps) also hold the state at every actuator (PD)
step, sampled by the faulty actuator's probe (see :class:`EpisodeWriter`).
Runs headless until ``--num_episodes`` episodes have finished, e.g. on Euler::

    euler enqueue fd_collect_main -- python scripts/forge/fault_detector/collect.py \\
        --task IsaacContrib-FaultDetector-MotorFailure-A2Robust --checkpoint <model.pt> \\
        --num_episodes 16384 --seed 0 --output_dir logs/fault_detector/data/main
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

GROUPS = ("detector", "privileged", "analysis", "fault")
INDEX_FIELDS = ["episode_id", "env_id", "length", "has_fault", "fault_type", "fault_joint", "onset_step", "health",
                "terrain_level", "terrain_type", "termination_reason"]  # fmt: skip


def terrain_columns(generator_cfg) -> list[str]:
    """Sub-terrain name of each column of a curriculum terrain (same rule as TerrainGenerator).

    Copied from ``scripts/forge/fault_analysis/collect.py``.
    """
    names = list(generator_cfg.sub_terrains)
    proportions = np.array([cfg.proportion for cfg in generator_cfg.sub_terrains.values()])
    cumulative = np.cumsum(proportions / proportions.sum())
    return [
        names[np.min(np.where(col / generator_cfg.num_cols + 0.001 < cumulative)[0])]
        for col in range(generator_cfg.num_cols)
    ]


def pd_sample(env) -> dict[str, torch.Tensor]:
    """State the PD law acts on at one actuator step (``pd/<name>`` arrays; see the module doc)."""
    d = env.scene["robot"].data
    return {
        "base_ang_vel": d.root_ang_vel_b.torch,
        "projected_gravity": d.projected_gravity_b.torch,
        "joint_pos_rel": d.joint_pos.torch - d.default_joint_pos.torch,
        "joint_vel": d.joint_vel.torch,
        "base_lin_vel": d.root_lin_vel_b.torch,
        "joint_pos_target": d.joint_pos_target.torch - d.default_joint_pos.torch,
        # torque of the previous substep (incl. the fault's health): a motor-current reading one PD period old
        "applied_torque": d.applied_torque.torch,
        # the action being executed in this policy step
        "last_action": env.action_manager.action,
        "velocity_command": env.command_manager.get_command("base_velocity"),
    }


class EpisodeWriter:
    """GPU buffers of the running episodes; finished episodes go to CPU and are written in shards.

    With ``pd_rate``, the faulty actuator's probe also stores :func:`pd_sample` at every actuator step: sample
    ``4 t + k`` (decimation 4) is the state before physics substep ``k`` of policy step ``t``, so sample ``4 t`` is
    the state of observation ``o_t``.
    """

    def __init__(self, env, output_dir: str, shard_episodes: int, pd_rate: bool = False):
        self.env, self.output_dir, self.shard_episodes = env, output_dir, shard_episodes
        om = env.observation_manager
        self.terms = {g: list(zip(om.active_terms[g], [d[0] for d in om.group_obs_term_dim[g]])) for g in GROUPS}
        self.num_joints = env.scene["robot"].num_joints
        n, t_max = env.num_envs, env.max_episode_length + 1
        dims = {g: sum(d for _, d in self.terms[g]) for g in GROUPS}
        self.buffers = {g: torch.zeros(n, t_max, dims[g], device=env.device) for g in GROUPS}
        self.decimation = env.cfg.decimation
        self.pd_buffers: dict[str, torch.Tensor] | None = None
        self._pd_call = 0
        if pd_rate:
            shapes = {k: v.shape[1] for k, v in pd_sample(env).items()}
            self.pd_buffers = {
                k: torch.zeros(n, t_max * self.decimation, dim, device=env.device, dtype=torch.float16)
                for k, dim in shapes.items()
            }
            self.terms["pd"] = list(shapes.items())
            env.scene["robot"].actuators[env.cfg.events.motor_fault.params["actuator"]].probe = self.record_pd
        self.length = torch.zeros(n, dtype=torch.long, device=env.device)
        self.rows = torch.arange(n, device=env.device)
        self.pending: list[dict[str, np.ndarray]] = []
        self.index: list[dict] = []
        self.num_episodes = self.num_shards = 0
        generator = env.cfg.scene.terrain.terrain_generator
        self.columns = terrain_columns(generator) if generator is not None and generator.curriculum else None
        os.makedirs(os.path.join(output_dir, "shards"), exist_ok=True)

    def record(self, obs) -> None:
        """Append the observation of every env to its running episode."""
        t = self.length.clamp(max=self.buffers["fault"].shape[1] - 1)
        for g in GROUPS:
            self.buffers[g][self.rows, t] = obs[g]
        self.length += 1
        self._pd_call = 0

    def record_pd(self) -> None:
        """Actuator probe: store the PD-step sample ``4 t + k`` of the policy step ``t`` being executed."""
        k = self._pd_call
        self._pd_call += 1
        if k >= self.decimation:  # a write outside env.step (none expected)
            return
        t = (self.length - 1).clamp(min=0, max=self.buffers["fault"].shape[1] - 1)
        for name, value in pd_sample(self.env).items():
            self.pd_buffers[name][self.rows, t * self.decimation + k] = value.half()

    def finish(self, ids: torch.Tensor, reasons: list[str]) -> None:
        """Move the episodes of ``ids`` (finished in the last step) to the pending shard."""
        if len(ids) == 0:
            return
        lengths = self.length[ids].cpu().numpy()
        t = int(lengths.max())
        data = {g: self.buffers[g][ids, :t].cpu().numpy() for g in GROUPS}
        if self.pd_buffers is not None:
            data.update({f"pd/{k}": v[ids, : t * self.decimation].cpu().numpy() for k, v in self.pd_buffers.items()})
        terrain = self.env.scene.terrain
        levels = terrain.terrain_levels[ids].cpu().numpy()
        types = terrain.terrain_types[ids].cpu().numpy()
        J = self.num_joints
        for k, env_id in enumerate(ids.cpu().numpy()):
            length = int(lengths[k])
            episode = {g: data[g][k, :length] for g in GROUPS}
            episode.update({g: v[k, : length * self.decimation] for g, v in data.items() if g.startswith("pd/")})
            labels = episode["fault"]
            mask = labels[:, :J]
            faulty = mask.any(axis=1)
            has_fault = bool(faulty.any())
            last = labels[-1]
            self.index.append({
                "episode_id": self.num_episodes,
                "env_id": int(env_id),
                "length": length,
                "has_fault": int(has_fault),
                "fault_type": int(last[J]) if has_fault else 0,
                "fault_joint": int(mask[faulty][0].argmax()) if has_fault else -1,
                "onset_step": int(np.argmax(faulty)) if has_fault else -1,
                "health": float(last[-1]) if has_fault else 1.0,
                "terrain_level": int(levels[k]),
                "terrain_type": self.columns[types[k]] if self.columns else int(types[k]),
                "termination_reason": reasons[k],
            })  # fmt: skip
            episode["episode_id"] = self.num_episodes
            self.pending.append(episode)
            self.num_episodes += 1
        self.length[ids] = 0
        if len(self.pending) >= self.shard_episodes:
            self.write_shard()

    def write_shard(self) -> None:
        if not self.pending:
            return
        arrays = {
            "ep_len": np.array([len(e["fault"]) for e in self.pending], dtype=np.int64),
            "episode_id": np.array([e["episode_id"] for e in self.pending], dtype=np.int64),
        }
        for g in GROUPS:
            group = np.concatenate([e[g] for e in self.pending]).astype(np.float32)
            start = 0
            for name, dim in self.terms[g]:
                arrays[f"{g}/{name}"] = group[:, start : start + dim]
                start += dim
        if self.pd_buffers is not None:
            # float16 halves the 4x larger PD-rate arrays; its resolution is far below the detector's input noise
            for name in self.pd_buffers:
                arrays[f"pd/{name}"] = np.concatenate([e[f"pd/{name}"] for e in self.pending])
        np.savez(os.path.join(self.output_dir, "shards", f"{self.num_shards:04d}.npz"), **arrays)
        self.num_shards += 1
        self.pending = []
        with open(os.path.join(self.output_dir, "index.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(self.index)


def termination_reasons(env, ids: torch.Tensor) -> list[str]:
    """First termination term hit by each env of ``ids`` in the last step (failure terms before time-outs)."""
    manager = env.termination_manager
    names = sorted(manager.active_terms, key=lambda name: manager.get_term_cfg(name).time_out)
    hits = torch.stack([manager.get_term(name)[ids] for name in names], dim=1).cpu().numpy()
    return [names[int(np.argmax(row))] if row.any() else "unknown" for row in hits]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect motor-fault episodes of a frozen policy.")
    parser.add_argument("--num_episodes", type=int, default=16384, help="Stop after this many finished episodes.")
    parser.add_argument("--output_dir", required=True, help="Dataset directory.")
    parser.add_argument("--shard_episodes", type=int, default=1024, help="Episodes per shard file.")
    parser.add_argument("--p_fault", type=float, default=None, help="Override the fault probability.")
    parser.add_argument("--fault_types", nargs="+", default=None, help="Override, e.g. power_out=0.5 power_loss=0.5.")
    parser.add_argument("--power_loss_health", type=float, nargs=2, default=None, help="Override the h range.")
    parser.add_argument("--passive_damping", type=float, default=None, help="Override d [N*m*s/rad].")
    parser.add_argument("--pd_rate", action="store_true", help="Also store the state at every actuator (PD) step.")
    fd_args, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        # the terrain mesh is seeded too, so every collection walks the same grid
        env_cfg.scene.terrain.terrain_generator.seed = 0
        fault_cfg = env_cfg.events.motor_fault.params["cfg"]
        if fd_args.p_fault is not None:
            fault_cfg.p_fault = fd_args.p_fault
        if fd_args.fault_types is not None:
            fault_cfg.fault_types = {k: float(v) for k, v in (s.split("=") for s in fd_args.fault_types)}
        if fd_args.power_loss_health is not None:
            fault_cfg.power_loss_health = tuple(fd_args.power_loss_health)
        if fd_args.passive_damping is not None:
            fault_cfg.passive_damping = fd_args.passive_damping

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        output_dir = os.path.abspath(fd_args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            base = env.unwrapped
            writer = EpisodeWriter(base, output_dir, fd_args.shard_episodes, fd_args.pd_rate)
            meta = {
                "task": args_cli.task,
                "checkpoint": resume_path,
                "seed": env_cfg.seed,
                "num_envs": base.num_envs,
                "step_dt": base.step_dt,
                "decimation": base.cfg.decimation,
                "joint_names": base.scene["robot"].joint_names,
                "terms": writer.terms,
                "fault_cfg": to_dict(fault_cfg),
                "args": vars(fd_args),
                "argv": sys.argv[1:],
            }
            with open(os.path.join(output_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            dump_yaml(os.path.join(output_dir, "env.yaml"), base.cfg)

            print(f"[fd] collecting {fd_args.num_episodes} episodes of {resume_path} into {output_dir}", flush=True)
            obs = env.get_observations()
            step, start = 0, time.time()
            with torch.inference_mode():
                while writer.num_episodes < fd_args.num_episodes:
                    writer.record(obs)
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    ids = dones.nonzero().squeeze(-1)
                    writer.finish(ids, termination_reasons(base, ids))
                    step += 1
                    if step % 250 == 0:
                        print(
                            f"[fd] step {step} ({time.time() - start:.0f} s wall): {writer.num_episodes} episodes",
                            flush=True,
                        )
            writer.write_shard()
            print(f"[fd] done: {writer.num_episodes} episodes in {step} steps, {time.time() - start:.0f} s", flush=True)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
