# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record a soft-step policy from a camera that follows robot 0, with the leg signal and foot loads drawn in.

Runs the training env (random signals and fault injectors on); ``--p_touchdown`` raises the random signal rate so a
short clip shows several signals. Writes ``<out>/soft_step.mp4`` and ``<out>/soft_step.log`` (signal and reset
times)::

    euler enqueue soft_record -- python scripts/forge/soft_step/record.py --task IsaacContrib-Velocity-A2SoftStep \\
        --checkpoint logs/wandb_checkpoints/a2_study/<run>/model_<N>.pt --out logs/soft_step/videos
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--seconds", type=float, default=30.0, help="Clip length [s].")
    parser.add_argument("--eye", type=float, nargs=3, default=(-1.5, -1.5, 0.8), help="Camera offset [m].")
    parser.add_argument("--p_touchdown", type=float, default=0.1, help="Random signal chance per touchdown.")
    args, argv = parser.parse_known_args()

    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES
    from isaaclab.utils import to_dict

    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint
    from isaaclab_rl.entrypoints.common import (
        apply_env_overrides,
        close_env,
        create_isaaclab_env,
        pre_launch_video_config,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    from isaaclab_tasks.forge.velocity.a2_soft_step.mdp.signal import LEGS, _t, get_flag
    from isaaclab_tasks.utils import resolve_task_config

    args_cli = _parse_args(argv)
    if args_cli.num_envs is None:
        args_cli.num_envs = 1
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    args_cli.video = "viz"
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.video_recorders = []  # frames are captured here
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        env_cfg.events.leg_signal.params["p_touchdown"] = args.p_touchdown
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
            sensor = base.scene["contact_forces"]
            feet = base.reward_manager.get_term_cfg("load_ramp").params["sensor_cfg"].body_ids
            viz = next(v for v in base.sim.visualizers if v.cfg.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES)
            try:
                font = ImageFont.load_default(size=22)
            except TypeError:  # Pillow < 10.1
                font = ImageFont.load_default()

            os.makedirs(args.out, exist_ok=True)
            path = os.path.join(args.out, "soft_step.mp4")
            log = [f"policy {resume_path}", f"p_touchdown {args.p_touchdown}"]
            eye = torch.tensor(args.eye, device=base.device)
            writer, prev_flag = None, -1
            obs = env.get_observations()
            with torch.inference_mode():
                for k in range(round(args.seconds / base.step_dt)):
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    t = (k + 1) * base.step_dt
                    flag = get_flag(base)[0]
                    leg = int(flag.argmax()) if flag.any() else -1
                    if leg != prev_flag:
                        log.append(f"{t:6.2f} s  signal {LEGS[leg] if leg >= 0 else 'off'}")
                        prev_flag = leg
                    if dones[0]:
                        log.append(f"{t:6.2f} s  reset")

                    target = robot.data.root_pos_w.torch[0]
                    base.sim.set_camera_view(tuple((target + eye).tolist()), tuple(target.tolist()))
                    base.sim.render()
                    img = Image.fromarray(viz.render_rgb_array())
                    draw = ImageDraw.Draw(img)
                    fz = _t(sensor.data.net_forces_w)[0, feet, 2].clamp(min=0.0)
                    vx = base.command_manager.get_command("base_velocity")[0, 0]
                    draw.text((12, 10), f"t {t:5.1f} s   cmd vx {vx:+.2f} m/s", fill=(0, 0, 0), font=font)
                    for i, name in enumerate(LEGS):
                        color = (220, 0, 0) if i == leg else (0, 0, 0)
                        mark = " SIGNAL" if i == leg else ""
                        draw.text((12, 40 + 28 * i), f"{name} {fz[i]:5.0f} N{mark}", fill=color, font=font)
                    frame = np.ascontiguousarray(np.asarray(img)[..., :3])
                    if writer is None:
                        fps = round(1.0 / base.step_dt)
                        writer = imageio_ffmpeg.write_frames(path, (frame.shape[1], frame.shape[0]), fps=fps)
                        writer.send(None)
                    writer.send(frame)
            if writer is not None:
                writer.close()
            with open(os.path.join(args.out, "soft_step.log"), "w") as f:
                f.write("\n".join(log) + "\n")
            print(f"[record] wrote {path}", flush=True)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
