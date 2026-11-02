# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the foot-stuck injector (``forge/fault_injector/foot_stuck.py``): one robot, one clip per (mode, F_esc).

The frozen policy walks one robot; its ``--foot`` sticks at ``--onset`` s in each mode, once with every
``--escape_force``. Each clip follows the robot for ``--seconds`` (or until it falls) and overlays the event state,
the anchor force and the robot's force on the anchor against the escape cone. While stuck, the escape cone is drawn
translucent from the anchor (amber; green while the robot's force is inside it above F_esc). Headless, e.g. on Euler::

    CKPT=/workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt
    euler enqueue foot_stuck_render -- python scripts/forge/fault_injector/render_foot_stuck.py \\
        --task IsaacContrib-Velocity-A2Study-RobustTeacher --checkpoint $CKPT --out logs/foot_stuck_test/render

Writes ``<out>/<mode>_F<F_esc>.mp4`` and ``<out>/clips.log``.
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    pre_launch_video_config,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.forge.fault_injector.foot_stuck import (  # noqa: E402
    ESCAPED,
    STUCK,
    STUCK_MODES,
    FootStuckInjectorCfg,
    add_foot_stuck,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

STATE_COLORS = {0: (120, 200, 120), STUCK: (255, 80, 80), ESCAPED: (90, 170, 255)}


def overlay(frame, lines: list[tuple[str, tuple]], font):
    """Text lines (``(text, rgb)``) on a dark box in the top-left corner."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(np.asarray(frame)[..., :3]))
    d = ImageDraw.Draw(img, "RGBA")
    row = 36
    width = max(d.textlength(text, font=font) for text, _ in lines)
    d.rectangle([12, 12, 40 + width, 24 + row * len(lines)], fill=(0, 0, 0, 170))
    for i, (text, color) in enumerate(lines):
        d.text((26, 18 + row * i), text, fill=color, font=font)
    return np.asarray(img)


def foot_marker():
    """Sphere marker at the stuck foot: green before the onset, red while stuck, blue after the escape."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    spheres = {
        name: sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        for name, color in (("waiting", (0.1, 0.8, 0.1)), ("stuck", (1.0, 0.0, 0.0)), ("escaped", (0.1, 0.4, 1.0)))
    }
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/FootStuck", markers=spheres))


def log_cone(viz, device: str, apex=None, axis=None, half_angle_deg: float = 0.0, color=(1.0, 0.75, 0.1)) -> None:
    """Translucent escape cone (apex at the anchor, open along the cone axis) in a Newton viewer; hidden without apex.

    Markers ignore opacity, so the cone is a mesh logged to the viewer. Viewers without ``log_mesh`` (Kit) skip it.
    """
    if not hasattr(viz, "log_mesh"):
        return
    # the GL viewer cannot update a mesh to zero points, so a hidden cone keeps a placeholder shape
    hidden = apex is None
    if hidden:
        apex, axis, half_angle_deg = np.zeros(3), np.array((0.0, 0.0, 1.0)), 5.0
    length, segments = 0.4, 48
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phi = np.linspace(0.0, 2.0 * math.pi, segments + 1)[:, None]
    radius = length * math.tan(math.radians(half_angle_deg))
    ring = apex + length * a + radius * (np.cos(phi) * u + np.sin(phi) * v)
    # one vertex triple per face: flat normals
    faces = np.stack((np.broadcast_to(apex, (segments, 3)), ring[:-1], ring[1:]), axis=1)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals = np.repeat(normals / np.linalg.norm(normals, axis=1, keepdims=True), 3, axis=0)
    viz.log_mesh(
        "/foot_stuck_cone",
        wp.array(faces.reshape(-1, 3).astype(np.float32), dtype=wp.vec3, device=device),
        wp.array(np.arange(3 * segments, dtype=np.int32), dtype=wp.int32, device=device),
        normals=wp.array(normals.astype(np.float32), dtype=wp.vec3, device=device),
        hidden=hidden,
        backface_culling=False,
        color=color,
        dynamic=True,
        opacity=0.35,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the foot-stuck injector with a frozen policy.")
    parser.add_argument("--out", required=True, help="Output directory of the clips.")
    parser.add_argument("--modes", nargs="+", default=list(STUCK_MODES[1:]), help="Stuck modes to render.")
    parser.add_argument("--escape_force", type=float, nargs="+", default=[math.inf, 250.0], help="F_esc [N] per clip.")
    parser.add_argument("--foot", default="FR", help="Stuck leg prefix (FL, FR, RL, RR); FR faces the camera.")
    parser.add_argument("--cone_deg", type=float, default=None, help="Escape cone half-angle [deg] (default: cfg).")
    parser.add_argument("--onset", type=float, default=2.0, help="Onset time [s].")
    parser.add_argument("--seconds", type=float, default=8.0, help="Clip length [s].")
    parser.add_argument(
        "--eye", type=float, nargs=3, default=(0.5, -1.3, 0.45), help="Camera offset from the base-foot midpoint."
    )
    st_args, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)
    args_cli.num_envs = 1

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    # a capture-capable headless visualizer, as in fault_detector/demo.py
    args_cli.video = "viz"
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.video_recorders = []
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        # easiest terrain rows, no pushes: the stuck foot is the only disturbance
        env_cfg.curriculum = None
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None
        env_cfg.episode_length_s = st_args.seconds + 2.0
        cfg = FootStuckInjectorCfg(
            p_stuck=1.0,
            onset_time_s=(st_args.onset, st_args.onset),
            feet=[f"{st_args.foot}_foot"],
            calves=[f"{st_args.foot}_calf"],
        )
        if st_args.cone_deg is not None:
            cfg.escape_cone_angle_deg = st_args.cone_deg
        add_foot_stuck(env_cfg, cfg)

        log_root_path = f"logs/rsl_rl/{agent_cfg.experiment_name}"
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            os.makedirs(st_args.out, exist_ok=True)
            marker = foot_marker()
            # the reset between clips writes tensors created under inference mode
            with torch.inference_mode():
                for mode in st_args.modes:
                    for force in st_args.escape_force:
                        record(env, policy, st_args, mode, force, marker)
        finally:
            close_env(env)


def record(env, policy, st_args, mode: str, escape_force: float, marker) -> None:
    """One clip: reset with the given mode and F_esc, then render every policy step."""
    import imageio_ffmpeg
    from PIL import ImageFont

    from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES

    base = env.unwrapped
    viz = next(v for v in base.sim.visualizers if v.cfg.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES)
    inj = base.event_manager.get_term_cfg("foot_stuck").func
    robot = inj.robot
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    # the next reset samples this mode and threshold
    inj.mode_probs[:] = torch.tensor([float(m == mode) for m in STUCK_MODES[1:]], device=base.device)
    inj.stuck_cfg.escape_force = (escape_force, escape_force)
    obs, _ = env.reset()
    policy.reset(torch.ones(1, dtype=torch.bool, device=base.device))
    eye = torch.tensor(st_args.eye, device=base.device)
    name = f"{mode}_F{'inf' if math.isinf(escape_force) else round(escape_force)}"
    path = os.path.abspath(os.path.join(st_args.out, f"{name}.mp4"))
    writer = None  # opened at the first frame, which sets the size
    t, peak_push, result = 0.0, 0.0, "held to the end"
    cos = inj.cone_cos
    cone_deg, hold_s = inj.stuck_cfg.escape_cone_angle_deg, inj.stuck_cfg.escape_hold_s
    with torch.inference_mode():
        for _ in range(round(st_args.seconds / base.step_dt)):
            obs, _, dones, _ = env.step(policy(obs))
            if dones[0]:
                result = f"episode ended at {t:.2f} s (fall)"
                break
            t = float(base.episode_length_buf[0]) * base.step_dt
            state = int(inj.state[0])
            push = -inj.force[0]
            magnitude = float(push.norm())
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(push @ inj.cone_axis[0]) / max(magnitude, 1e-9)))))
            if state == STUCK:
                peak_push = max(peak_push, magnitude)
            body = inj.foot_ids[inj.foot[0]]
            drift = float((robot.data.body_link_pos_w.torch[0, body] - inj.anchor_pos[0]).norm())
            state_text = ("waiting for onset + foot contact", "STUCK", "ESCAPED")[state]
            if state == STUCK:
                state_text += f" for {(base.episode_length_buf[0] - inj.stuck_step[0]).item() * base.step_dt:.2f} s"
            if state == ESCAPED:
                esc = inj.escape_push[0]
                how = "slide-out" if float(esc.norm()) == 0 else f"|f| = {float(esc.norm()):.0f} N"
                state_text += f" at {float(inj.escape_step[0]) * base.step_dt:.2f} s ({how})"
                result = state_text
            in_cone = magnitude > 0 and float(push @ inj.cone_axis[0]) >= cos * magnitude
            f_esc = "inf" if math.isinf(escape_force) else f"{escape_force:.0f} N"
            lines = [
                (
                    f"foot stuck: {mode}, {st_args.foot} foot, F_esc {f_esc}, cone {cone_deg:g} deg, hold {hold_s:g} s",
                    (255, 255, 255),
                ),
                (f"t = {t:5.2f} s   {state_text}", STATE_COLORS[state]),
            ]
            if state == STUCK:
                lines += [
                    (
                        f"robot force on anchor |f| = {magnitude:6.0f} N, {angle:5.1f} deg from the cone axis",
                        (255, 220, 80) if in_cone and magnitude >= escape_force else (230, 230, 230),
                    ),
                    (
                        f"in cone: {'yes' if in_cone else 'no '}   held {int(inj.hold[0]) * base.physics_dt:.2f}"
                        f"/{hold_s:g} s   drift {100 * drift:4.1f} cm",
                        (230, 230, 230),
                    ),  # fmt: skip
                ]
            foot = robot.data.body_link_pos_w.torch[0, body]
            marked = inj.anchor_pos[0] if state > 0 else foot
            marker.visualize(translations=marked.view(1, 3), marker_indices=[state])
            if state == STUCK:
                # amber; green while the robot's force is in the cone above F_esc
                color = (0.2, 1.0, 0.3) if in_cone and magnitude >= escape_force else (1.0, 0.75, 0.1)
                apex, axis = inj.anchor_pos[0].cpu().numpy(), inj.cone_axis[0].cpu().numpy()
                log_cone(viz, base.device, apex, axis, cone_deg, color)
            else:
                log_cone(viz, base.device)
            target = 0.5 * (robot.data.root_pos_w.torch[0] + foot)
            base.sim.set_camera_view(tuple((target + eye).tolist()), tuple(target.tolist()))
            base.sim.render()
            frame = overlay(viz.render_rgb_array(), lines, font)
            if writer is None:
                writer = imageio_ffmpeg.write_frames(
                    path, (frame.shape[1], frame.shape[0]), fps=round(1.0 / base.step_dt)
                )
                writer.send(None)
            writer.send(np.ascontiguousarray(frame))
    if writer is not None:
        writer.close()
    msg = f"[render] {name}: {result}; peak |f| while stuck {peak_push:.0f} N -> {path}"
    print(msg, flush=True)
    with open(os.path.join(st_args.out, "clips.log"), "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    main()
