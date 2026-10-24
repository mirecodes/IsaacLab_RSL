# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL locomotion policy on many robots and save the seconds before every failure.

Every robot keeps a ring buffer of the last ``--buffer_s`` seconds of its state. An event saves that buffer to
``events/<id>.npz`` and indexes it in ``index.csv``. Events are:

* the onset of contact on a non-foot body (base, hip, thigh, calf), after a ``--grace_s`` spawn grace and at most
  once per robot per buffer length; the robot keeps walking;
* a task termination other than a time-out (e.g. base contact, large tilt), always; the robot is reset.

See ``README.md`` next to this file for the data layout. Runs headless until ``--max_events`` events or
``--max_steps`` env steps, e.g. on Euler::

    euler enqueue fault_a2_robust -- python scripts/reinforcement_learning/fault_analysis/collect.py \\
        --task IsaacContrib-Velocity-A2Study-RobustTeacher \\
        --checkpoint /workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/<run>/model_2999.pt
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import RecorderTerm, RecorderTermCfg  # noqa: E402
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg  # noqa: E402
from isaaclab.utils import configclass, to_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

logger = logging.getLogger(__name__)

RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}


class FaultRecorder(RecorderTerm):
    """Keeps the last ``buffer_s`` seconds of every robot on the GPU and saves them on events (see module doc).

    Frame ``t`` is sampled at the end of env step ``t``, before resets: the state ``s_t``, the observation
    ``o_t`` computed from it, the critic value ``V(o_t)``, and the action ``a_{t-1}`` that led to ``s_t``.
    """

    def __init__(self, cfg: "FaultRecorderCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.contact = env.scene["contact_forces"]
        self.scanner = env.scene.sensors.get("height_scanner")
        self.trigger_ids, self.trigger_names = self.contact.find_bodies(cfg.trigger_bodies, preserve_order=True)
        self.num_frames = round(cfg.buffer_s / env.step_dt)
        self.value_fn = None
        """Critic ``obs_buf -> V`` [num_envs]; set by the script once the policy is loaded."""
        self.num_events = 0
        self.num_terminal = 0
        self._buffers: dict[str, torch.Tensor] | None = None
        self._ptr = 0
        self.grace_frames = round(cfg.grace_s / env.step_dt)
        self._length = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._since_event = torch.full_like(self._length, self.num_frames)
        self._in_contact = torch.zeros(env.num_envs, len(self.trigger_ids), dtype=torch.bool, device=env.device)
        os.makedirs(os.path.join(cfg.output_dir, "events"), exist_ok=True)
        self._index_path = os.path.join(cfg.output_dir, "index.csv")
        with open(self._index_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["event", "env_id", "step", "sim_time_s", "episode_step", "num_frames", "terrain_level",
                 "terrain_type", "terminal", "terminations", "base_contact", "contact_bodies"]
            )  # fmt: skip

    def sample(self) -> dict[str, torch.Tensor]:
        """Everything logged per step, each of shape (num_envs, ...)."""
        env, d, c = self._env, self.robot.data, self.contact.data
        frame = {
            # base
            "root_pos_w": d.root_link_pos_w,
            "root_quat_w": d.root_link_quat_w,
            "root_lin_vel_w": d.root_link_lin_vel_w,
            "root_ang_vel_w": d.root_link_ang_vel_w,
            "root_lin_vel_b": d.root_link_lin_vel_b,
            "root_ang_vel_b": d.root_link_ang_vel_b,
            "projected_gravity_b": d.projected_gravity_b,
            # IMU-like base acceleration (linear, angular) at the base CoM
            "root_acc_w": d.body_com_acc_w.torch[:, 0],
            # joints
            "joint_pos": d.joint_pos,
            "joint_vel": d.joint_vel,
            "joint_acc": d.joint_acc,
            "joint_pos_target": d.joint_pos_target,
            "computed_torque": d.computed_torque,
            "applied_torque": d.applied_torque,
            # bodies (articulation body order)
            "body_pos_w": d.body_link_pos_w,
            "body_quat_w": d.body_link_quat_w,
            "body_lin_vel_w": d.body_link_lin_vel_w,
            "body_com_pos_w": d.body_com_pos_w,
            # contacts (contact-sensor body order)
            "contact_force_w": c.net_forces_w,
            "contact_normal_force_w": c.net_normal_forces_w,
            "contact_friction_force_w": c.net_friction_forces_w,
            "contact_air_time": c.current_air_time,
            "contact_time": c.current_contact_time,
            # policy
            "action": env.action_manager.action,
        }
        frame = {k: v.torch if hasattr(v, "torch") else v for k, v in frame.items() if v is not None}
        if self.scanner is not None:
            # terrain height below the base: mean of the height-scanner hits, as in mdp.base_height_l2
            frame["base_terrain_height"] = self.scanner.data.ray_hits_w.torch[..., 2].mean(dim=1)
        for name in env.command_manager.active_terms:
            frame[f"command_{name}"] = env.command_manager.get_command(name)
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            if hasattr(term, "phase"):
                frame[f"action_{name}_phase"] = term.phase
        for group, obs in env.obs_buf.items():
            if isinstance(obs, dict):
                frame.update({f"obs_{group}_{term}": value for term, value in obs.items()})
            else:
                frame[f"obs_{group}"] = obs
        if self.value_fn is not None:
            frame["value"] = self.value_fn(env.obs_buf)
        return frame

    def record_post_step(self):
        env = self._env
        frame = self.sample()
        if self._buffers is None:
            n = (env.num_envs, self.num_frames)
            self._buffers = {k: torch.zeros(n + v.shape[1:], dtype=v.dtype, device=v.device) for k, v in frame.items()}
        for key, value in frame.items():
            self._buffers[key][:, self._ptr] = value
        self._ptr = (self._ptr + 1) % self.num_frames
        self._length += 1
        self._since_event += 1
        # non-foot contact over the sensor history, as in mdp.illegal_contact
        history = self.contact.data.net_normal_forces_w_history.torch[:, :, self.trigger_ids]
        in_contact = history.norm(dim=-1).amax(dim=1) > self.cfg.threshold
        onset = (in_contact & ~self._in_contact).any(dim=1)
        self._in_contact = in_contact
        # reset_terminated excludes time-outs; it is computed before this hook within env.step
        save = env.reset_terminated.clone()
        if not self.cfg.terminal_only:
            save |= onset & (self._length > self.grace_frames) & (self._since_event >= self.num_frames)
        ids = save.nonzero().squeeze(-1)
        if len(ids) > 0:
            self._save(ids, in_contact[ids])
            self._since_event[ids] = 0
        return None, None

    def record_pre_reset(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device)
        self._length[ids] = 0
        self._since_event[ids] = self.num_frames
        return None, None

    def _save(self, ids: torch.Tensor, in_contact: torch.Tensor) -> None:
        env = self._env
        manager = env.termination_manager
        # oldest -> newest; _ptr is the next slot to write, i.e. the oldest frame
        order = (torch.arange(self.num_frames, device=env.device) + self._ptr) % self.num_frames
        data = {k: v[ids][:, order].cpu().numpy() for k, v in self._buffers.items()}
        lengths = self._length[ids].clamp(max=self.num_frames).tolist()
        in_contact = in_contact.cpu().numpy()
        failures = {
            name: manager.get_term(name)[ids].cpu().numpy()
            for name in manager.active_terms
            if not manager.get_term_cfg(name).time_out
        }
        terrain = env.scene.terrain
        levels = getattr(terrain, "terrain_levels", None)
        types = getattr(terrain, "terrain_types", None)
        rows = []
        for i, env_id in enumerate(ids.tolist()):
            n = lengths[i]
            bodies = [name for name, hit in zip(self.trigger_names, in_contact[i]) if hit]
            terminations = [name for name, hit in failures.items() if hit[i]]
            level = int(levels[env_id]) if levels is not None else -1
            col = int(types[env_id]) if types is not None else -1
            np.savez_compressed(
                os.path.join(self.cfg.output_dir, "events", f"{self.num_events:06d}.npz"),
                **{k: v[i, -n:] for k, v in data.items()},
                env_id=env_id,
                step=env.common_step_counter,
                env_origin=terrain.env_origins[env_id].cpu().numpy(),
                terrain_level=level,
                terrain_type=col,
                terminations=np.array(terminations),
                contact_bodies=np.array(bodies),
            )
            rows.append(
                [self.num_events, env_id, env.common_step_counter, f"{env.common_step_counter * env.step_dt:.2f}",
                 int(env.episode_length_buf[env_id]), n, level, col, bool(terminations), "|".join(terminations),
                 self.robot.body_names[0] in bodies, "|".join(bodies)]
            )  # fmt: skip
            self.num_events += 1
            self.num_terminal += bool(terminations)
        with open(self._index_path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


@configclass
class FaultRecorderCfg(RecorderTermCfg):
    class_type: type = FaultRecorder
    output_dir: str = ""
    buffer_s: float = 5.0
    trigger_bodies: str = ""
    threshold: float = 1.0
    grace_s: float = 1.0
    terminal_only: bool = False


@configclass
class FaultRecorderManagerCfg(RecorderManagerBaseCfg):
    # the term keeps its own files; the manager's episode export stays off
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    fault: FaultRecorderCfg = FaultRecorderCfg()


def configure_env(env_cfg, fault_args: argparse.Namespace, output_dir: str) -> None:
    """Training env (own terminations) on a fixed terrain, without disturbances, with the fault recorder."""
    configure_terrain(env_cfg, env_cfg.seed, fault_args.sub_terrains)
    # disturbances: off unless asked for, so failures come from policy and terrain
    if not fault_args.disturbances:
        for name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
    env_cfg.recorders = FaultRecorderManagerCfg(
        fault=FaultRecorderCfg(
            output_dir=output_dir,
            buffer_s=fault_args.buffer_s,
            trigger_bodies=f"(?!{fault_args.foot_bodies}$).*",
            threshold=fault_args.contact_threshold,
            grace_s=fault_args.grace_s,
            terminal_only=fault_args.terminal_only,
        )
    )


def configure_terrain(env_cfg, seed: int, sub_terrains: list[str] | None = None) -> None:
    """Training terrain grid, optionally only the given sub-terrains, with robots spread over all levels and
    kept on their tile. The generator is seeded, so the same arguments give the same mesh."""
    env_cfg.curriculum = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.seed = seed
        if sub_terrains:
            generator.sub_terrains = {name: generator.sub_terrains[name] for name in sub_terrains}


def terrain_mesh(prim_path: str = "/World/ground") -> tuple[np.ndarray, np.ndarray] | None:
    """World-frame vertices and triangles of the terrain mesh on the stage, or None for a ground plane."""
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    prim = sim_utils.get_first_matching_child_prim(prim_path, lambda p: p.GetTypeName() == "Mesh")
    if prim is None:
        return None
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    transform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = points @ transform[:3, :3].T + transform[:3, 3]
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    return points.astype(np.float32), faces


def terrain_columns(generator_cfg) -> list[str]:
    """Sub-terrain name of each column of a curriculum terrain (same rule as TerrainGenerator)."""
    names = list(generator_cfg.sub_terrains)
    proportions = np.array([cfg.proportion for cfg in generator_cfg.sub_terrains.values()])
    cumulative = np.cumsum(proportions / proportions.sum())
    return [
        names[np.min(np.where(col / generator_cfg.num_cols + 0.001 < cumulative)[0])]
        for col in range(generator_cfg.num_cols)
    ]


def write_metadata(env, recorder: FaultRecorder, output_dir: str, info: dict) -> None:
    """meta.json (names, limits, run info), env.yaml (full config) and terrain.npz (mesh, origins)."""
    robot = recorder.robot
    terrain = env.scene.terrain
    generator = env.cfg.scene.terrain.terrain_generator
    meta = {
        **info,
        "step_dt": env.step_dt,
        "num_frames": recorder.num_frames,
        "joint_names": robot.joint_names,
        "body_names": robot.body_names,
        "contact_body_names": recorder.contact.body_names,
        "trigger_body_names": recorder.trigger_names,
        "body_mass": robot.data.body_mass.torch[0].cpu().tolist(),
        "joint_pos_limits": robot.data.joint_pos_limits.torch[0].cpu().tolist(),
        "joint_effort_limits": robot.data.joint_effort_limits.torch[0].cpu().tolist(),
        "terrain_columns": terrain_columns(generator) if generator is not None and generator.curriculum else None,
        "quat_convention": "xyzw",
        "fields": {k: list(v.shape[2:]) for k, v in recorder._buffers.items()},
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    dump_yaml(os.path.join(output_dir, "env.yaml"), env.cfg)
    mesh = terrain_mesh()
    arrays = {"env_origins": terrain.env_origins.cpu().numpy()}
    if terrain.terrain_origins is not None:
        arrays["terrain_origins"] = terrain.terrain_origins.cpu().numpy()
    if mesh is not None:
        arrays["vertices"], arrays["faces"] = mesh
    np.savez_compressed(os.path.join(output_dir, "terrain.npz"), **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect the motion before failures of a trained policy.")
    parser.add_argument("--buffer_s", type=float, default=5.0, help="Seconds kept before each failure.")
    parser.add_argument("--max_events", type=int, default=1000, help="Stop after this many events.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many env steps.")
    parser.add_argument("--foot_bodies", default=".*_foot", help="Regex of the foot bodies; all others trigger.")
    parser.add_argument("--contact_threshold", type=float, default=1.0, help="Trigger contact force [N].")
    parser.add_argument("--grace_s", type=float, default=1.0, help="Ignore contact onsets this long after a reset.")
    parser.add_argument("--terminal_only", action="store_true", help="Record only task terminations.")
    parser.add_argument("--disturbances", action="store_true", help="Keep the task's pushes and external wrench.")
    parser.add_argument("--sub_terrains", nargs="+", default=None, help="Keep only these sub-terrains.")
    parser.add_argument("--output_dir", default=None, help="Default: logs/fault_analysis/<experiment>/<time>_<run>.")
    fault_args, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)
    if args_cli.num_envs is None:
        args_cli.num_envs = 4096

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return
        run_name = os.path.basename(os.path.dirname(resume_path))
        checkpoint = os.path.splitext(os.path.basename(resume_path))[0]
        output_dir = os.path.abspath(
            fault_args.output_dir
            or os.path.join(
                "logs", "fault_analysis", agent_cfg.experiment_name,
                f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}_{checkpoint}",
            )
        )  # fmt: skip
        configure_env(env_cfg, fault_args, output_dir)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = RUNNERS[agent_cfg.class_name](env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            # private access: the recorder manager has no public term getter
            recorder: FaultRecorder = env.unwrapped.recorder_manager._terms["fault"]
            critic = getattr(runner.alg, "critic", None)
            if critic is not None:
                recorder.value_fn = lambda obs: critic(TensorDict(obs, batch_size=[env.num_envs])).squeeze(-1)

            logger.info(f"Collecting failures of {resume_path} into {output_dir}")
            obs = env.get_observations()
            step, start = 0, time.time()
            with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
                while recorder.num_events < fault_args.max_events and step != fault_args.max_steps:
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    step += 1
                    if step == 1:
                        info = {
                            "task": args_cli.task,
                            "checkpoint": resume_path,
                            "seed": env_cfg.seed,
                            "num_envs": env.num_envs,
                            "args": vars(fault_args),
                            "argv": sys.argv[1:],
                        }
                        write_metadata(env.unwrapped, recorder, output_dir, info)
                    if step % 500 == 0:
                        sim_s = step * env.unwrapped.step_dt
                        robot_h = sim_s * env.num_envs / 3600
                        print(
                            f"[fault] step {step} ({sim_s:.0f} s sim, {robot_h:.1f} robot-h, "
                            f"{time.time() - start:.0f} s wall): {recorder.num_events} events "
                            f"({recorder.num_terminal} terminal)",
                            flush=True,
                        )
            print(
                f"[fault] done: {recorder.num_events} events ({recorder.num_terminal} terminal) in {step} steps"
                f" -> {output_dir}",
                flush=True,
            )
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
