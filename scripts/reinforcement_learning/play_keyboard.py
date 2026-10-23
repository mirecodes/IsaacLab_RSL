# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL velocity policy and drive it live from the keyboard.

Spawns one robot (``--num_envs`` overrides), follows it with the viewport camera and replaces the
``base_velocity`` command with the keyboard state every step:

    W / S   forward / backward  (v_x = +/- max of the task's lin_vel_x range)
    A / D   turn left / right   (w_z = +/- max of the task's ang_vel_z range)
    L       zero the command

Keys add while held and subtract on release. Needs a Kit window (the keyboard is read through carb), e.g.::

    ./isaaclab.sh -p scripts/reinforcement_learning/play_keyboard.py \\
        --task IsaacContrib-Velocity-Go2Study-Base --checkpoint logs/rsl_rl/<exp>/<run>/model_1500.pt --viz kit
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    run_playback,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

logger = logging.getLogger(__name__)

COMMAND_NAME = "base_velocity"
RUNNERS = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}


def main() -> None:
    args_cli = _parse_args(sys.argv[1:])
    # the keyboard is read through carb, and playback only makes sense at wall-clock speed
    args_cli.require_kit = True
    args_cli.real_time = True
    if args_cli.num_envs is None:
        args_cli.num_envs = 1

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli):
        # carb / omni exist only once Kit is running
        from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed

        # the keyboard owns the command: zero ranges so a reset resample stays still, no heading or standing envs
        cmd_cfg = getattr(env_cfg.commands, COMMAND_NAME)
        v_x, w_z = cmd_cfg.ranges.lin_vel_x[1], cmd_cfg.ranges.ang_vel_z[1]
        cmd_cfg.ranges.lin_vel_x = cmd_cfg.ranges.lin_vel_y = cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
        cmd_cfg.ranges.heading = None
        cmd_cfg.heading_command = False
        cmd_cfg.rel_standing_envs = 0.0
        cmd_cfg.resampling_time_range = (1e6, 1e6)
        # no time-out resets while driving (1 h)
        env_cfg.episode_length_s = 3600.0
        # camera follows robot 0
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (-2.5, -2.5, 1.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return
        env_cfg.log_dir = os.path.dirname(resume_path)

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = RUNNERS[agent_cfg.class_name](env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            keyboard = Se2Keyboard(Se2KeyboardCfg(sim_device=env.unwrapped.device))
            # replace the stock arrow/numpad bindings; "L" (zero command) is kept by Se2Keyboard
            keyboard._INPUT_KEY_MAPPING = {
                "W": np.array([v_x, 0.0, 0.0]),
                "S": np.array([-v_x, 0.0, 0.0]),
                "A": np.array([0.0, 0.0, w_z]),
                "D": np.array([0.0, 0.0, -w_z]),
            }
            command = env.unwrapped.command_manager.get_term(COMMAND_NAME).vel_command_b
            logger.info(f"Keyboard play: W/S v_x = +/-{v_x} m/s, A/D w_z = +/-{w_z} rad/s, L = stop")

            obs = env.get_observations()

            def step() -> None:
                nonlocal obs
                # (v_x, v_y, w_z) for every env; seen by the policy from the next observation on
                command[:] = keyboard.advance()
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)

            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
