# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the foot-stuck injector (``forge/fault_injector/foot_stuck.py``) with a frozen walking policy.

Every env gets a stuck event at ``--onset`` s; the first episode of each env is measured until ``--seconds``:
anchor drift (cm), rotation drift (deg), slide travel (cm), anchor force, falls, escapes and the robot's force on
the anchor at each escape (it must lie in the cone and above the threshold). ``--pull`` adds an external force on the
stuck foot, ramped at ``--pull_rate`` N/s from the onset: ``calf`` (out along the calf axis, the default cone axis)
must free it, ``side`` (horizontal, across the calf) is only reported.

Momentum balance (every run): over each physics step with the anchor on, the whole robot's momentum change must equal
anchor + pull + ground contact + gravity, so whatever the leg pushes against the anchor reaches the base::

    CKPT=/workspace/isaaclab/logs/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt
    python scripts/forge/fault_injector/test_foot_stuck.py --task IsaacContrib-Velocity-A2Study-RobustTeacher \\
        --checkpoint $CKPT --num_envs 300 --escape_force inf inf          # anchor only: drift and stability
    ... --escape_force 150 150 --pull calf                              # pulled out along the leg: the feet come free
    ... --escape_force 150 150 --pull side                              # pulled across the leg: outside the cone
    ... --escape_force inf inf --sweep_deg 5 10 20 30                   # escape rate per cone angle, one run

The sweep needs ``F_esc = inf``: nothing escapes by force, so up to an env's first escape at a given cone angle the
run is the one with that angle, and a hold counter per angle (F_esc drawn from ``--sweep_force``) gives that escape.

Exits with status 1 when a check fails.
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
from isaaclab.utils.math import quat_box_minus  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.forge.fault_injector.foot_stuck import (  # noqa: E402
    ESCAPED,
    STUCK,
    STUCK_MODES,
    FootStuckInjectorCfg,
    add_foot_stuck,
)
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the foot-stuck injector with a frozen policy.")
    parser.add_argument("--onset", type=float, default=2.0, help="Onset time [s] of every env's event.")
    parser.add_argument("--seconds", type=float, default=8.0, help="Measured time per env [s].")
    parser.add_argument("--escape_force", type=float, nargs=2, default=[100.0, 300.0], help="F_esc range [N].")
    parser.add_argument(
        "--pull", nargs="+", default=None, help="External pull: calf (out along the calf), side, or x y z (world)."
    )
    parser.add_argument("--pull_rate", type=float, default=400.0, help="Pull ramp [N/s] from the onset.")
    parser.add_argument("--pull_max", type=float, default=1500.0, help="Pull cap [N].")
    parser.add_argument("--p95_drift_cm", type=float, default=3.0, help="Check: 95th percentile anchor drift [cm].")
    parser.add_argument("--cone_deg", type=float, default=None, help="Escape cone half-angle [deg] (default: cfg).")
    parser.add_argument(
        "--sweep_deg", type=float, nargs="+", default=None, help="Cone half-angles [deg] for the escape-rate sweep."
    )
    parser.add_argument(
        "--sweep_force", type=float, nargs=2, default=[100.0, 300.0], help="F_esc range [N] of the sweep."
    )
    parser.add_argument("--save", default=None, help="Also save per-step series (n_envs, steps) to this .npz.")
    st_args, argv = parser.parse_known_args(sys.argv[1:])
    args_cli = _parse_args(argv)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        # easiest terrain rows, no pushes: the stuck foot is the only disturbance
        env_cfg.curriculum = None
        env_cfg.scene.terrain.max_init_terrain_level = 0
        env_cfg.events.push_robot = None
        env_cfg.events.base_external_force_torque = None
        if st_args.sweep_deg and not math.isinf(st_args.escape_force[0]):
            raise ValueError("--sweep_deg needs --escape_force inf inf: an escape would change the trajectory.")
        cfg = FootStuckInjectorCfg(
            p_stuck=1.0, onset_time_s=(st_args.onset, st_args.onset), escape_force=tuple(st_args.escape_force)
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
            ok = run(env, policy, st_args)
        finally:
            close_env(env)
    sys.exit(0 if ok else 1)


def run(env, policy, st_args) -> bool:
    base = env.unwrapped
    inj = base.event_manager.get_term_cfg("foot_stuck").func
    robot, n, dev = inj.robot, base.num_envs, base.device
    rows = torch.arange(n, device=dev)

    # per physics step: optional pull, then the anchor; record the anchor force of every substep
    sub_force_max = torch.zeros(n, device=dev)
    sub_flips = torch.zeros(n, device=dev)  # substeps whose anchor force reverses (> 100 N each): oscillation
    last_force = torch.zeros(n, 3, device=dev)
    up = torch.tensor([0.0, 0.0, 1.0], device=dev)
    live = torch.ones(n, dtype=torch.bool, device=dev)  # still in the first episode
    streak = torch.zeros(n, dtype=torch.long, device=dev)
    # sweep: with F_esc = inf nothing escapes by force, so up to an env's first escape at angle q the run is the one
    # with that cone; the injector's hold counter per angle gives that first escape exactly
    sweep_deg = torch.tensor(st_args.sweep_deg or [], device=dev)
    sweep_cos = torch.cos(torch.deg2rad(sweep_deg))
    lo, hi = st_args.sweep_force
    sweep_force = lo + (hi - lo) * torch.rand(n, 1, device=dev)
    sweep_hold = torch.zeros(n, len(sweep_deg), dtype=torch.long, device=dev)
    sweep_step = torch.full((n, len(sweep_deg)), -1, dtype=torch.long, device=dev)  # policy step of the escape

    def pull_direction() -> torch.Tensor:
        if st_args.pull == ["calf"]:
            return inj.axis
        if st_args.pull == ["side"]:
            return torch.nn.functional.normalize(torch.linalg.cross(inj.axis, up.expand(n, 3)), dim=-1)
        return torch.nn.functional.normalize(torch.tensor([float(v) for v in st_args.pull], device=dev), dim=0)

    # momentum balance: d/dt sum(m_i v_i) = anchor + pull + contact + m g over each physics step
    mass = robot.data.body_mass.torch
    weight = mass.sum(1, keepdim=True) * torch.tensor(base.cfg.sim.gravity, device=dev)
    prev = {"momentum": None}
    balance = {
        "steps": 0,
        "residual": 0.0,
        "anchor": 0.0,
        "free_steps": 0,
        "free_residual": 0.0,
        "num": 0.0,
        "den": 0.0,
    }
    write_data_to_sim = robot.write_data_to_sim

    def write_with_probe():
        momentum = (mass.unsqueeze(-1) * robot.data.body_com_lin_vel_w.torch).sum(1)
        if prev["momentum"] is not None:
            # the contact sensor holds the forces of the physics step that just ran
            contact = inj.sensor.data.net_forces_w.torch.sum(1)
            rate = (momentum - prev["momentum"]) / base.physics_dt
            residual = rate - (prev["anchor"] + prev["pull"] + contact + weight)
            # anchored in that step, and no reset in between; the other steps give the balance's own error
            same = base.episode_length_buf >= prev["length"]
            sel, free = prev["held"] & same, ~prev["held"] & same
            balance["steps"] += int(sel.sum())
            balance["residual"] += float(residual[sel].norm(dim=-1).sum())
            balance["anchor"] += float(prev["anchor"][sel].norm(dim=-1).sum())
            balance["free_steps"] += int(free.sum())
            balance["free_residual"] += float(residual[free].norm(dim=-1).sum())
            # least-squares share of the anchor force in the balance without it: 1 if it all reaches the robot
            without = residual + prev["anchor"]
            balance["num"] += float((without * prev["anchor"])[sel].sum())
            balance["den"] += float((prev["anchor"] * prev["anchor"])[sel].sum())
        pull = torch.zeros(n, 3, device=dev)
        if st_args.pull is not None:
            # external pull on the stuck foot, ramped from the stuck step; added before the anchor's wrench
            since = ((base.episode_length_buf - inj.stuck_step).float() * base.step_dt).clamp(min=0.0)
            pulled = (inj.state == STUCK) & (inj.stuck_step >= 0)
            magnitude = (st_args.pull_rate * since).clamp(max=st_args.pull_max) * pulled
            pull = magnitude.unsqueeze(1) * pull_direction()
            forces = torch.zeros(n, len(inj.foot_body_ids), 3, device=dev)
            forces[rows, inj.foot] = pull
            robot.instantaneous_wrench_composer.add_forces_and_torques_index(
                forces=forces, body_ids=inj.foot_body_ids, is_global=True
            )
        write_data_to_sim()
        # longest time in the cone above F_esc (the injector's hold counter), first episode only
        streak.copy_(torch.where(live, torch.maximum(streak, inj.hold), streak))
        f = inj.force
        if len(sweep_deg):
            push, stuck_now = -f, (inj.state == STUCK).unsqueeze(1)
            magnitude = push.norm(dim=-1, keepdim=True)
            along = (push * inj.cone_axis).sum(-1, keepdim=True)
            inside = stuck_now & (along >= sweep_cos * magnitude) & (magnitude >= sweep_force)
            sweep_hold.copy_(torch.where(inside, sweep_hold + 1, 0))
            first = live.unsqueeze(1) & (sweep_step < 0) & (sweep_hold >= inj.hold_steps)
            sweep_step.copy_(torch.where(first, base.episode_length_buf.unsqueeze(1), sweep_step))
        held = f.norm(dim=-1) > 0
        prev.update(momentum=momentum, anchor=f.clone(), pull=pull, held=held, length=base.episode_length_buf.clone())
        sub_force_max.copy_(torch.maximum(sub_force_max, f.norm(dim=-1)))
        big = (f.norm(dim=-1) > 100.0) & (last_force.norm(dim=-1) > 100.0)
        sub_flips.add_((big & ((f * last_force).sum(-1) < 0)).float())
        last_force.copy_(f)

    robot.write_data_to_sim = write_with_probe
    series = {
        k: []
        for k in (
            "drift",
            "rot",
            "travel",
            "force_max",
            "flips",
            "state",
            "live",
            "base_z",
            "contact",
            "force_z",
            "force_h",
            "axis_z",
            "foot_z",
        )
    }

    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    drift_max = torch.zeros(n, device=dev)
    drift_sum = torch.zeros(n, device=dev)
    rot_max = torch.zeros(n, device=dev)
    knee_max = torch.zeros(n, device=dev)  # calf modes: the knee end's deviation across the calf
    knee_sum = torch.zeros(n, device=dev)
    travel_max = torch.zeros(n, device=dev)
    force_max = torch.zeros(n, device=dev)
    samples = torch.zeros(n, device=dev)
    drift_samples, flip_count = [], 0.0  # drift of every held env-step; anchor force reversals over all of them
    free_move = torch.zeros(n, device=dev)  # foot distance from the anchor 1 s after the escape
    # the injector re-samples at a reset, so keep the first episode's event
    names = ("mode", "stuck_step", "escape_step", "escape_push", "escape_force", "cone_axis")
    rec = {k: getattr(inj, k).clone() for k in names}
    nan = False
    obs = env.get_observations()
    steps = round(st_args.seconds / base.step_dt)
    with torch.inference_mode():
        for t in range(steps):
            if t == round((st_args.onset + 0.5) / base.step_dt):
                live_late = live.clone()  # alive 0.5 s after the onset: must be stuck by then
            obs, _, dones, _ = env.step(policy(obs))
            policy.reset(dones)
            d = robot.data
            nan |= bool(torch.isnan(d.root_pos_w.torch).any())
            fell |= live & dones.bool() & ~base.termination_manager.time_outs
            live &= ~dones.bool()
            for k, v in rec.items():
                v[live] = getattr(inj, k)[live]
            body = inj.foot_ids[inj.foot]
            e = d.body_link_pos_w.torch[rows, body] - inj.anchor_pos
            travel = (e * inj.axis).sum(-1)
            # calf_slide: the slide out is free, a slide deeper counts as drift
            e = torch.where(((inj.mode == 3) & (travel > 0)).unsqueeze(1), e - travel.unsqueeze(1) * inj.axis, e)
            rot = quat_box_minus(d.body_link_quat_w.torch[rows, body], inj.anchor_quat).norm(dim=-1)
            held = live & (inj.state == STUCK)
            drift_max = torch.where(held, torch.maximum(drift_max, e.norm(dim=-1)), drift_max)
            drift_sum += torch.where(held, e.norm(dim=-1), 0.0)
            drift_samples.append(e.norm(dim=-1)[held])
            flip_count += float(sub_flips[held].sum())
            samples += held.float()
            rot_max = torch.where(held & (inj.mode >= 2), torch.maximum(rot_max, rot), rot_max)
            knee = d.body_link_pos_w.torch[rows, inj.calf_ids[inj.foot]]
            knee_dev = knee - (inj.anchor_pos + inj.calf_length.unsqueeze(1) * inj.axis)
            knee_dev = (knee_dev - (knee_dev * inj.axis).sum(-1, keepdim=True) * inj.axis).norm(dim=-1)
            knee_max = torch.where(held & (inj.mode >= 2), torch.maximum(knee_max, knee_dev), knee_max)
            knee_sum += torch.where(held & (inj.mode >= 2), knee_dev, 0.0)
            travel_max = torch.where(held & (inj.mode == 3), torch.maximum(travel_max, travel), travel_max)
            force_max = torch.where(held, torch.maximum(force_max, sub_force_max), force_max)
            if st_args.save:
                values = (
                    e.norm(dim=-1),
                    rot,
                    travel,
                    sub_force_max,
                    sub_flips,
                    inj.state,
                    live,
                    d.root_pos_w.torch[:, 2],
                    inj.sensor.data.net_forces_w.torch[rows, inj.sensor_ids[inj.foot]].norm(dim=-1),
                    inj.force[:, 2],
                    inj.force[:, :2].norm(dim=-1),
                    inj.axis[:, 2],
                    d.body_link_pos_w.torch[rows, body, 2],
                )
                for k, v in zip(series, values):
                    series[k].append(v.float().cpu().clone())
            sub_force_max.zero_()
            sub_flips.zero_()
            after = (
                live & (inj.state == ESCAPED) & (base.episode_length_buf - inj.escape_step == round(1 / base.step_dt))
            )
            free_move = torch.where(
                after, (d.body_link_pos_w.torch[rows, body] - inj.anchor_pos).norm(dim=-1), free_move
            )

    if st_args.save:
        os.makedirs(os.path.dirname(os.path.abspath(st_args.save)), exist_ok=True)
        arrays = {k: torch.stack(v, dim=1).numpy() for k, v in series.items()}
        arrays.update({k: v.cpu().numpy() for k, v in rec.items()})
        arrays.update(sweep_deg=sweep_deg.cpu().numpy(), sweep_step=sweep_step.cpu().numpy())
        arrays.update(sweep_force=sweep_force.squeeze(1).cpu().numpy())
        np.savez_compressed(st_args.save, fell=fell.cpu().numpy(), foot=inj.foot.cpu().numpy(), **arrays)
    mode, stuck_step, escape_step = rec["mode"], rec["stuck_step"], rec["escape_step"]
    stuck = stuck_step >= 0
    escaped = escape_step >= 0
    push = rec["escape_push"]
    magnitude = push.norm(dim=-1)
    along = (push * rec["cone_axis"]).sum(-1)
    angle = torch.rad2deg(torch.acos((along / magnitude.clamp(min=1e-9)).clamp(-1, 1)))
    by_force = escaped & (magnitude > 0)  # escape_push is 0 for a slide-out
    cone_ok = (angle <= inj.stuck_cfg.escape_cone_angle_deg + 1e-3) & (magnitude >= rec["escape_force"] - 1e-3)
    ok = not nan
    print(
        f"[stuck] {n} envs, onset {st_args.onset} s, {st_args.seconds} s, escape_force {st_args.escape_force} N,"
        f" pull {st_args.pull} at {st_args.pull_rate} N/s, hold {inj.hold_steps} physics steps",
        flush=True,
    )
    print("mode        envs stuck  fell escaped(force/slide) t_esc[s]  drift mean/max[cm]  rot max[deg]  "
          "travel max[cm]  |F| max[N]  |f_esc| mean[N]  angle max[deg]  free move[cm]")  # fmt: skip
    for m in range(1, len(STUCK_MODES)):
        sel = mode == m
        s, es = sel & stuck, sel & escaped
        ef = es & by_force
        t_esc = ((escape_step - stuck_step).float() * base.step_dt)[es]
        mean = lambda x, mask: float(x[mask].mean()) if mask.any() else float("nan")  # noqa: E731
        maxi = lambda x, mask: float(x[mask].max()) if mask.any() else float("nan")  # noqa: E731
        drift_mean = float((drift_sum[s] / samples[s].clamp(min=1)).mean()) if s.any() else float("nan")
        print(
            f"{STUCK_MODES[m]:<11} {int(sel.sum()):>4} {int(s.sum()):>5} {int((sel & fell).sum()):>5}"
            f" {int(es.sum()):>4} ({int(ef.sum())}/{int((es & ~by_force).sum())})"
            f"        {float(t_esc.mean()) if len(t_esc) else float('nan'):>6.2f}"
            f"   {100 * drift_mean:>6.2f} / {100 * maxi(drift_max, s):>6.2f}"
            f"      {math.degrees(maxi(rot_max, s)) if m >= 2 else float('nan'):>7.2f}"
            f"       {100 * maxi(travel_max, s) if m == 3 else float('nan'):>7.2f}"
            f"   {maxi(force_max, s):>9.1f}   {mean(magnitude, ef):>9.1f}       {maxi(angle, ef):>7.2f}"
            f"      {100 * mean(free_move, es & (free_move > 0)):>7.2f}",
            flush=True,
        )
        if m >= 2 and s.any():
            knee_mean = float((knee_sum[s] / samples[s].clamp(min=1)).mean())
            print(
                f"            knee across the calf: mean {100 * knee_mean:.2f} cm,"
                f" max {100 * float(knee_max[s].max()):.2f} cm",
                flush=True,
            )
        held_s = streak[s].float() * base.physics_dt
        if len(held_s):
            q = torch.quantile(held_s, torch.tensor([0.5, 0.9], device=dev)).tolist()
            print(
                f"            longest time in the cone above F_esc: p50 {q[0]:.3f} s, p90 {q[1]:.3f} s, max"
                f" {float(held_s.max()):.3f} s (needed {inj.stuck_cfg.escape_hold_s} s)",
                flush=True,
            )
    if len(sweep_deg):
        print(
            f"[sweep] escapes of the stuck feet per cone half-angle, F_esc ~ U[{lo:g}, {hi:g}] N, hold"
            f" {inj.stuck_cfg.escape_hold_s} s; calf_slide: by force before the slide-out + slide-outs",
            flush=True,
        )
        print("deg   mode        stuck  by force   slide-out  escaped   t_esc p50/p90[s]")
        t_sweep = (sweep_step - stuck_step.unsqueeze(1)).float() * base.step_dt
        for j, deg in enumerate(st_args.sweep_deg):
            for m in range(1, len(STUCK_MODES)):
                s = (mode == m) & stuck
                f_esc = s & (sweep_step[:, j] >= 0)
                # with F_esc = inf every escape is a slide-out; it ends the stuck state, so a force escape came first
                slide = s & escaped & ~f_esc
                t = torch.cat((t_sweep[f_esc, j], ((escape_step - stuck_step).float() * base.step_dt)[slide]))
                q = torch.quantile(t, torch.tensor([0.5, 0.9], device=dev)).tolist() if len(t) else [math.nan] * 2
                n_s, n_e = int(s.sum()), int((f_esc | slide).sum())
                print(
                    f"{deg:<5g} {STUCK_MODES[m]:<11} {n_s:>5} {int(f_esc.sum()):>9} {int(slide.sum()):>11}"
                    f" {n_e:>5} ({100 * n_e / max(n_s, 1):3.0f} %)   {q[0]:.2f} / {q[1]:.2f}",
                    flush=True,
                )
    drift_all = torch.cat(drift_samples)
    drift_p95 = float(torch.quantile(drift_all, 0.95)) if len(drift_all) else 0.0
    print(f"[stuck] all modes: drift p95 {100 * drift_p95:.2f} cm, {flip_count:.0f} force reversals", flush=True)
    residual = balance["residual"] / max(balance["steps"], 1)
    anchor = balance["anchor"] / max(balance["steps"], 1)
    free_residual = balance["free_residual"] / max(balance["free_steps"], 1)
    share = balance["num"] / max(balance["den"], 1e-9)
    print(
        f"[stuck] momentum balance: anchored steps {balance['steps']}, |residual| mean {residual:.1f} N"
        f" (|anchor force| mean {anchor:.1f} N); unanchored steps {balance['free_steps']}, |residual| mean"
        f" {free_residual:.1f} N; share of the anchor force in the balance {share:.3f}",
        flush=True,
    )
    checks = {
        "no NaN in the root state": not nan,
        # a foot that stays off the ground (robot down on its belly, leg raised) never sticks
        "95 % of the envs alive 0.5 s after the onset got stuck": bool(stuck[live_late].float().mean() >= 0.95),
        "every force escape inside the cone and above F_esc": bool(cone_ok[by_force].all()),
        # with a pull the deflection is the pull's (1500 N / k = 3 cm), so only checked without one
        f"95th percentile anchor drift below {st_args.p95_drift_cm} cm": (
            st_args.pull is not None or drift_p95 < st_args.p95_drift_cm / 100
        ),
        # an explicit spring-damper reversed its force in about 2 of 4 substeps (README, Decisions)
        "no chatter (< 0.1 anchor force reversals per held policy step)": flip_count < 0.1 * len(drift_all),
        # the anchor force reaches the whole robot: its share in the balance is 1 (0 if it stopped at the foot). The
        # residual (~100 N anchored, ~60 N not) does not grow with the anchor force (~110 N under a 1500 N pull): it is
        # the contact sensor's error, larger while a foot is pressed into the ground
        "anchor force fully in the robot's momentum (share 0.9-1.1)": balance["steps"] > 0 and 0.9 <= share <= 1.1,
    }
    if math.isinf(st_args.escape_force[0]):
        checks["no force escape with F_esc = inf"] = not bool(by_force.any())
    if st_args.pull == ["calf"] and inj.stuck_cfg.escape_cone_axis is None:
        # a pull across the leg is compared with the run without pull instead (the policy may pull the foot free);
        # a foot wedged into the terrain (8.5 kN contact before the onset, 2026-11-02) is held by the terrain
        checks["pull along the calf frees 98 % of the surviving feet"] = bool(
            escaped[stuck & ~fell].float().mean() >= 0.98
        )
    for name, passed in checks.items():
        print(f"[stuck] {'PASS' if passed else 'FAIL'}: {name}", flush=True)
        ok &= passed
    return ok


if __name__ == "__main__":
    main()
