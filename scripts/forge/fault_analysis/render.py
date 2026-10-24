# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render events saved by ``collect.py`` as videos that follow the robot over the terrain.

The motion is replayed kinematically: every frame writes the logged base pose and joint positions and renders
without stepping physics. The terrain is regenerated from the task and seed in ``meta.json`` and checked
against the saved ``terrain.npz``. Videos land in ``<collection>/videos/<event>.mp4``::

    euler enqueue fault_render -- python scripts/forge/fault_analysis/render.py \\
        /workspace/isaaclab/logs/fault_analysis/<experiment>/<collection> --events 0 3 17
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
from collect import configure_terrain, terrain_mesh  # noqa: E402
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES  # noqa: E402

from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    pre_launch_video_config,
)

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render events saved by collect.py.")
    parser.add_argument("collection", help="Collection directory written by collect.py.")
    parser.add_argument("--events", type=int, nargs="*", default=None, help="Event ids (default: all).")
    parser.add_argument("--count", type=int, default=None, help="Without --events: the first N full-window events.")
    parser.add_argument("--eye", type=float, nargs=3, default=(-1.0, -1.0, 0.6), help="Camera offset [m].")
    render_args, argv = parser.parse_known_args(sys.argv[1:])
    collection = os.path.abspath(render_args.collection)
    with open(os.path.join(collection, "meta.json")) as f:
        meta = json.load(f)
    events = render_args.events
    if events is None:
        with open(os.path.join(collection, "index.csv")) as f:
            rows = list(csv.DictReader(f))
        if render_args.count is not None:
            rows = [r for r in rows if int(r["num_frames"]) == meta["num_frames"]][: render_args.count]
        events = [int(r["event"]) for r in rows]

    args_cli = _parse_args(["--task", meta["task"], "--num_envs", "1", "--seed", str(meta["seed"]), *argv])
    # a capture-capable visualizer (headless Newton GL unless --viz picks one)
    args_cli.video = args_cli.video or "viz"
    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = meta["seed"]
        configure_terrain(env_cfg, meta["seed"], meta["args"].get("sub_terrains"))
        # no command markers over the robot
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "debug_vis"):
                term.debug_vis = False
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False).unwrapped
        try:
            saved = np.load(os.path.join(collection, "terrain.npz"))
            mesh = terrain_mesh()
            same = mesh is not None and mesh[0].shape == saved["vertices"].shape
            if "vertices" in saved and not (same and np.allclose(mesh[0], saved["vertices"], atol=1e-4)):
                raise RuntimeError("Regenerated terrain differs from terrain.npz: was the task code changed?")

            viz = next(v for v in env.sim.visualizers if v.cfg.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES)
            robot = env.scene["robot"]
            env_ids = torch.tensor([0], device=env.device)
            zeros_root = torch.zeros(1, 6, device=env.device)
            zeros_joint = torch.zeros(1, robot.num_joints, device=env.device)
            eye = np.array(render_args.eye)
            os.makedirs(os.path.join(collection, "videos"), exist_ok=True)
            for event in events:
                data = np.load(os.path.join(collection, "events", f"{event:06d}.npz"))
                pose = torch.as_tensor(np.concatenate([data["root_pos_w"], data["root_quat_w"]], -1), device=env.device)
                joint_pos = torch.as_tensor(data["joint_pos"], device=env.device)
                frames = []
                for t in range(len(pose)):
                    robot.write_root_pose_to_sim_index(root_pose=pose[t : t + 1], env_ids=env_ids)
                    robot.write_root_velocity_to_sim_index(root_velocity=zeros_root, env_ids=env_ids)
                    robot.write_joint_position_to_sim_index(position=joint_pos[t : t + 1], env_ids=env_ids)
                    robot.write_joint_velocity_to_sim_index(velocity=zeros_joint, env_ids=env_ids)
                    env.sim.forward()
                    base = data["root_pos_w"][t]
                    env.sim.set_camera_view(tuple(base + eye), tuple(base))
                    env.sim.render()
                    frames.append(viz.render_rgb_array())
                    if args_cli.real_time:
                        time.sleep(meta["step_dt"])
                path = os.path.join(collection, "videos", f"{event:06d}.mp4")
                ImageSequenceClip(frames, fps=round(1.0 / meta["step_dt"])).write_videofile(path, logger=None)
                print(
                    f"[fault] {path}: contact {data['contact_bodies']}, terminations {data['terminations']}", flush=True
                )
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
