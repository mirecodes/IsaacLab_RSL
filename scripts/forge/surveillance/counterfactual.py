# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Counterfactual reactions: restore the state some time before a recorded fall, then let the policy go on with
different commands, and see which reaction avoids the fall.

For each fall of one collection (rows of ``sequences.py``'s ``falls.csv``) the robot is put back into its logged state
``--lead_s`` before the balance loss: root pose and velocity, joint positions and velocities, the last actions, and
the CPG phase and joint-target history of the Robust policy's action term. Then one reaction runs for
``--horizon_s``, without terminations; the outcome is whether the base touches the ground. Reactions:

- command (Robust only; the Energy policy has no command input): ``continue`` (the logged command), ``slow`` (half),
  ``stop`` (zero), ``back`` (0.3 m/s backwards);
- posture (both policies; the joint targets are overwritten after the action term): ``stand`` (default pose),
  ``crouch`` (lower base), ``wide`` (hips abducted), each held for the whole horizon; ``pause`` (default pose for
  0.5 s), ``crouch_1s`` and ``wide_1s`` (1 s), then the policy again with the logged command.

The same is done from the trigger frame of event windows that did not end in a fall (``control``): it measures how
often the restore itself, or the reaction, fells a robot::

    L=/workspace/isaaclab/logs/surveillance
    python scripts/forge/surveillance/counterfactual.py $L/v3_hard/Robust/slip_field --falls_csv $L/cf/falls.csv \\
        --out $L/counterfactual_v1
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from collect import RUNNERS, configure_terrain  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

FIELDS = ["collection", "kind", "event", "class", "reaction", "fell", "t_fall_s", "cmd_vx", "pos_err_0.2s",
          "leg_contact", "tilt_end_deg", "tilt_max_deg", "travel_m"]  # fmt: skip
COMMANDS = {"continue": 1.0, "slow": 0.5, "stop": 0.0}
# joint offsets from the default pose (order of each leg: hip, thigh, calf; legs FL, FR, RL, RR)
CROUCH, WIDE = [0.0, 0.3, -0.5] * 4, [0.25, 0.0, 0.0, -0.25, 0.0, 0.0] * 2
POSTURES = {"stand": [0.0, 0.0, 0.0] * 4, "crouch": CROUCH, "wide": WIDE, "pause": [0.0, 0.0, 0.0] * 4,
            "crouch_1s": CROUCH, "wide_1s": WIDE}  # fmt: skip
HOLD_S = {"pause": 0.5, "crouch_1s": 1.0, "wide_1s": 1.0}  # posture only this long, then the policy again


def items(collection: str, falls_csv: str, class_column: str, lead: int, controls: int) -> list[dict]:
    """Falls of ``collection`` (restore frame = balance loss - ``lead``) and up to ``controls`` event windows without
    a fall (restore frame = trigger - ``lead``)."""
    out = []

    def key(path):  # campaign/policy/terrain: the csv holds the local paths
        return path.rstrip("/").split("/")[-3:]

    with open(falls_csv) as f:
        for r in csv.DictReader(f):
            if key(r["collection"]) == key(collection):
                path = os.path.join(collection, "events", f"{int(r['event']):06d}.npz")
                out.append({"kind": "fall", "event": int(r["event"]), "class": r.get(class_column, ""),
                            "path": path, "frame": int(r["onset"]) - lead})  # fmt: skip
    index = os.path.join(collection, "triggers", "index.csv")
    if controls and os.path.isfile(index):
        with open(index) as f:
            rows = [r for r in csv.DictReader(f) if r["terminal"] == "False" and int(r["num_frames"]) == 250]
        for r in rows[:controls]:
            path = os.path.join(collection, "triggers", "events", f"{int(r['event']):06d}.npz")
            out.append({"kind": "control", "event": int(r["event"]), "class": "", "path": path, "frame": 124 - lead})
    return [it for it in out if it["frame"] >= 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collection", help="A collection of collect.py (Robust).")
    parser.add_argument("--falls_csv", required=True, help="sequences.py falls.csv (paths are matched by name).")
    parser.add_argument("--class_column", default="label_ward_mech")
    parser.add_argument("--lead_s", type=float, default=0.5)
    parser.add_argument("--horizon_s", type=float, default=3.0)
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--reactions", nargs="+", default=["continue", "stop"],
                        choices=[*COMMANDS, "back", *POSTURES], help="See the module docstring.")  # fmt: skip
    parser.add_argument("--repeats", type=int, default=1, help="Restores per state (the terrain events are random).")
    parser.add_argument("--num_envs", type=int, default=None, help="Parallel restores (default: the collection's).")
    parser.add_argument("--out", required=True)
    cf_args, argv = parser.parse_known_args(sys.argv[1:])
    with open(os.path.join(cf_args.collection, "meta.json")) as f:
        meta = json.load(f)
    dt = meta["step_dt"]
    todo = items(cf_args.collection, cf_args.falls_csv, cf_args.class_column, round(cf_args.lead_s / dt),
                 cf_args.controls)  # fmt: skip
    todo = [it for it in todo for _ in range(cf_args.repeats)]
    print(f"[cf] {len(todo)} restores from {cf_args.collection}", flush=True)
    if not todo:
        return
    num_envs = cf_args.num_envs or meta["num_envs"]
    args_cli = _parse_args(["--task", meta["task"], "--checkpoint", meta["checkpoint"], "--num_envs", str(num_envs),
                            "--seed", str(meta["seed"]), *argv])  # fmt: skip
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = meta["seed"]
        configure_terrain(env_cfg, meta["seed"], meta["args"].get("sub_terrains"))
        # no terminations (the outcome is read from the contact sensor), no pushes, episodes longer than the horizon
        for name, term in vars(env_cfg.terminations).items():
            if term is not None and not term.time_out:
                setattr(env_cfg.terminations, name, None)
        for name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        env_cfg.episode_length_s = 1.0e4
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        try:
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            device = env.unwrapped.device
            runner = RUNNERS[agent_cfg.class_name](env, to_dict(agent_cfg), log_dir=None, device=device)
            runner.load(resume_path, map_location=device)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            base_env = env.unwrapped
            robot, contact = base_env.scene["robot"], base_env.scene["contact_forces"]
            base_id = contact.find_bodies(meta["body_names"][0])[0][0]
            leg_ids = contact.find_bodies(".*_(thigh|calf)")[0]  # knees / shins on the ground: kneeling, not standing
            command = base_env.command_manager.get_term("base_velocity")
            command.compute = lambda dt: None  # hold the written command
            cpg = base_env.action_manager.get_term("joint_pos")
            # posture reactions: overwrite the joint targets of the envs in ``hold`` after the action term wrote them
            default = robot.data.default_joint_pos.torch
            hold = {"ids": None, "target": None}
            apply_actions = cpg.apply_actions

            def apply_with_posture():
                apply_actions()
                if hold["ids"] is not None:
                    robot.set_joint_position_target_index(target=hold["target"], env_ids=hold["ids"])

            cpg.apply_actions = apply_with_posture
            active = [name for names in base_env.event_manager.active_terms.values() for name in names]
            terrain = base_env.event_manager.get_term_cfg("terrain").func if "terrain" in active else None
            dev = base_env.device
            os.makedirs(cf_args.out, exist_ok=True)
            out_csv = os.path.join(cf_args.out, "outcomes.csv")
            new = not os.path.isfile(out_csv)
            with open(out_csv, "a", newline="") as f, torch.inference_mode():
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    writer.writeheader()
                for start in range(0, len(todo), num_envs):
                    batch = todo[start : start + num_envs]
                    data = [np.load(it["path"]) for it in batch]
                    ids = torch.arange(len(batch), device=dev)

                    def logged(key, offset=0):
                        return torch.as_tensor(np.stack([d[key][it["frame"] + offset] for d, it in zip(data, batch)]),
                                               device=dev, dtype=torch.float32)  # fmt: skip

                    for reaction in cf_args.reactions:
                        env.reset()
                        robot.write_root_link_pose_to_sim_index(
                            root_pose=torch.cat([logged("root_pos_w"), logged("root_quat_w")], 1), env_ids=ids
                        )
                        robot.write_root_link_velocity_to_sim_index(
                            root_velocity=torch.cat([logged("root_lin_vel_w"), logged("root_ang_vel_w")], 1),
                            env_ids=ids)  # fmt: skip
                        robot.write_joint_position_to_sim_index(position=logged("joint_pos"), env_ids=ids)
                        robot.write_joint_velocity_to_sim_index(velocity=logged("joint_vel"), env_ids=ids)
                        # private access: the last actions are part of the observation
                        base_env.action_manager._action[ids] = logged("action")
                        base_env.action_manager._prev_action[ids] = logged("action", -1)
                        # private access: the CPG term keeps its phase and target history as state
                        if hasattr(cpg, "_phase") and "action_joint_pos_phase" in data[0]:
                            cpg._phase[ids] = logged("action_joint_pos_phase")
                            cpg._joint_targets[ids] = logged("joint_pos_target")
                            cpg._prev_joint_targets[ids] = logged("joint_pos_target", -1)
                            cpg._prev_prev_joint_targets[ids] = logged("joint_pos_target", -2)
                        # private access: the terrain event's per-foot friction, drawn at touchdown; a stance foot
                        # on ice would otherwise keep the nominal friction until its next touchdown
                        if terrain is not None and "terrain_foot_mu" in data[0]:
                            terrain.foot_mu[ids] = logged("terrain_foot_mu")
                            terrain.low_mu[ids] = logged("terrain_low_mu").bool()
                            terrain.on_ice[ids] = logged("terrain_on_ice").bool()
                            terrain._in_slip[ids] = True  # no re-draw now; feet off the slip zone get "leave"
                            terrain._write_foot_mu(ids)
                        cmd = logged("command_base_velocity")
                        if reaction in COMMANDS:
                            cmd = cmd * COMMANDS[reaction]
                        elif reaction == "back":
                            cmd = torch.zeros_like(cmd)
                            cmd[:, 0] = -0.3
                        command.vel_command_b[ids] = cmd
                        if reaction in POSTURES:
                            hold["ids"] = ids
                            hold["target"] = default[ids] + torch.tensor(POSTURES[reaction], device=dev)
                        else:
                            hold["ids"] = None
                        base_env.observation_manager.reset(ids)
                        obs = env.get_observations()
                        policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=dev))
                        t_fall = torch.full((len(batch),), float("nan"), device=dev)
                        leg_hit = torch.zeros(len(batch), dtype=torch.bool, device=dev)
                        tilt_max = torch.zeros(len(batch), device=dev)
                        start_xy = logged("root_pos_w")[:, :2]
                        check = round(0.2 / dt)  # restore check: base position vs the log, 0.2 s later
                        for step in range(round(cf_args.horizon_s / dt)):
                            if reaction in HOLD_S and step == round(HOLD_S[reaction] / dt):
                                hold["ids"] = None
                            obs, _, _, _ = env.step(policy(obs))
                            forces = contact.data.net_forces_w.torch[ids]
                            hit = forces[:, base_id].norm(dim=-1) > 1.0
                            t_fall = torch.where(hit & t_fall.isnan(), (step + 1) * dt, t_fall)
                            leg_hit |= (forces[:, leg_ids].norm(dim=-1) > 1.0).any(1)
                            up = (-robot.data.projected_gravity_b.torch[ids, 2]).clamp(-1, 1)
                            tilt_max = torch.maximum(tilt_max, torch.rad2deg(torch.acos(up)))
                            if step + 1 == check:
                                err = (robot.data.root_link_pos_w.torch[ids] - logged("root_pos_w", check)).norm(dim=-1)
                        tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b.torch[ids, 2]).clamp(-1, 1)))
                        travel = (robot.data.root_link_pos_w.torch[ids, :2] - start_xy).norm(dim=-1)
                        for i, (it, tf) in enumerate(zip(batch, t_fall.tolist())):
                            writer.writerow({"collection": cf_args.collection, "kind": it["kind"], "event": it["event"],
                                             "class": it["class"], "reaction": reaction, "fell": not np.isnan(tf),
                                             "t_fall_s": tf, "cmd_vx": round(cmd[i, 0].item(), 3),
                                             "pos_err_0.2s": round(err[i].item(), 4),
                                             "leg_contact": bool(leg_hit[i]), "tilt_end_deg": round(tilt[i].item(), 1),
                                             "tilt_max_deg": round(tilt_max[i].item(), 1),
                                             "travel_m": round(travel[i].item(), 3)})  # fmt: skip
                        f.flush()
                        fell = (~t_fall.isnan()).float()
                        kinds = np.array([it["kind"] for it in batch])
                        print(
                            f"[cf] batch {start // num_envs} {reaction}: fell falls "
                            f"{fell[torch.as_tensor(kinds == 'fall', device=dev)].mean().item():.2f}, controls "
                            f"{fell[torch.as_tensor(kinds == 'control', device=dev)].mean().item():.2f}",
                            flush=True,
                        )
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
