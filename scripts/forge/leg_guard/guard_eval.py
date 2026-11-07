# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Randomized closed-loop test of the leg flag on a surveillance map: robots are split at random into arms that differ
only in where the flag comes from, so all arms share the map, the policy and the fault distribution.

An arm is ``<source>[@<p>][/<hold_s>][:slow|:stop]``. Sources: ``oracle`` (the leg of a fault or terrain event,
``--latency_s`` late), ``fault`` (the same for motor faults and stuck feet only), ``wrong`` (same alarm, another leg),
``random`` (a random leg at the rate of the first oracle or detector arm), ``detector`` (the exported TCN and per-leg
net, ``--detector``). Every alarm is logged; it triggers the reaction with probability ``p`` (default 1), so
``oracle@0`` is the no-reaction arm and ``@0.5`` gives a randomized reaction per alarm: what ``recoverable`` is
estimated from. The reaction sets the flag for ``hold_s`` (default ``--hold_s``; a motor fault or stuck foot keeps it
on for ``oracle``, ``fault`` and ``wrong`` unless the arm gives its own hold), or with ``:slow`` / ``:stop`` halves /
zeroes the velocity command for ``hold_s`` instead and sets no flag.

A fall is base contact or tilt above 60 deg. Writes ``result.json`` (per arm), ``alarms.npz``, ``falls.csv``::

    python scripts/forge/leg_guard/guard_eval.py --task IsaacContrib-Surveillance-MixedField-Robust-Hard-Faults \
        --checkpoint <model.pt> --flag_dim 4 --arms oracle@0 oracle wrong random --out <dir>
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "surveillance"))
from collect import RUNNERS, configure_terrain, left_tile  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import to_dict  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args  # noqa: E402
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args, _resolve_checkpoint  # noqa: E402
from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env, create_isaaclab_env  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.forge.velocity.a2_fault_aware.mdp.fault_mode import fault_mode  # noqa: E402
from isaaclab_tasks.forge.velocity.a2_leg_guard.mdp import GuardCPGActionCfg, external_leg_flag  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
FEET = [f"{leg}_foot" for leg in LEGS]
CAUSES = ("motor", "stuck", "low_mu", "sand", "collapse")
TILT_LIMIT = math.radians(60.0)
LAGS = (0, 5, 10)
"""Frames before the alarm whose detector input is saved with it."""


class LegNet(nn.Module):
    """Copy of the per-leg detector (``scripts/legnet.py`` of the report): one TCN scores each leg on the leg-rotated
    input with a one-hot leg id."""

    def __init__(self, num_channels: int, rotate: torch.Tensor, d: int = 64):
        super().__init__()
        layers, ch = [], num_channels + 4
        for dil in (1, 2, 4, 8):
            layers += [nn.Conv1d(ch, d, 3, dilation=dil, padding=2 * dil), nn.ELU()]
            ch = d
        self.conv = nn.Sequential(*layers)
        self.out = nn.Linear(2 * d, 1)
        self.register_buffer("rotate", rotate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (N, H, C) -> leg logits (N, 4)."""
        n, h = len(x), x.shape[1]
        leg = torch.arange(4, device=x.device).repeat_interleave(n)
        xr = torch.einsum("nhc,ncd->nhd", x.repeat(4, 1, 1), self.rotate[leg])
        ids = nn.functional.one_hot(leg, 4).float()[:, None].expand(-1, h, -1)
        f = self.conv(torch.cat([xr, ids], -1).transpose(1, 2))[:, :, :h]
        return self.out(torch.cat([f[:, :, -1], f.mean(-1)], 1)).view(4, n).T


def tilt_over(env, limit: float) -> torch.Tensor:
    g = env.scene["robot"].data.projected_gravity_b
    g = g.torch if hasattr(g, "torch") else g
    return g[:, 2] > -math.cos(limit)


def zero_input(env, dim: int) -> torch.Tensor:
    """The mode input of a policy trained without information (``a2_fault_aware`` Blind)."""
    return torch.zeros(env.num_envs, dim, device=env.device)


def parse_arm(text: str) -> tuple[str, float, float | None, float | None]:
    """``<source>[@<p>][/<hold_s>][:slow|:stop]`` -> source, p, hold or None, command scale or None."""
    head, _, kind = text.partition(":")
    head, _, hold = head.partition("/")
    source, _, p = head.partition("@")
    assert source in ("oracle", "fault", "wrong", "random", "detector") and kind in ("", "slow", "stop"), text
    scale = {"": None, "slow": 0.5, "stop": 0.0}[kind]
    return source, float(p) if p else 1.0, float(hold) if hold else None, scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=["oracle@0"])
    parser.add_argument("--flag_dim", type=int, default=4, help="4: leg flag; 12 or other: zeros; 0: no extra input.")
    parser.add_argument("--tuck", action="store_true", help="The action term pulls the flagged foot back.")
    parser.add_argument("--detector", default=None, help="Exported detector (needed by detector arms).")
    parser.add_argument("--latency_s", type=float, default=0.2, help="Oracle alarm delay (detector's median).")
    parser.add_argument("--hold_s", type=float, default=1.5, help="Minimum flag time.")
    parser.add_argument("--refractory_s", type=float, default=0.5, help="No alarm this long after a flag ends.")
    parser.add_argument("--horizon_s", type=float, default=3.0, help="A fall this soon after an alarm counts for it.")
    parser.add_argument("--minutes", type=float, default=5.0, help="Simulated minutes per robot.")
    parser.add_argument("--out", required=True)
    ga, argv = parser.parse_known_args(sys.argv[1:])
    arms = [parse_arm(a) for a in ga.arms]
    args_cli = _parse_args(argv)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        os.makedirs(ga.out, exist_ok=True)
        # as collect.configure_env, with the tilt fall
        configure_terrain(env_cfg, env_cfg.seed, None)
        for name, term in vars(env_cfg.terminations).items():
            if term is not None and not term.time_out and name != "base_contact":
                setattr(env_cfg.terminations, name, None)
        env_cfg.terminations.tilt_over = DoneTerm(func=tilt_over, params={"limit": TILT_LIMIT})
        world = env_cfg.terrain_world
        env_cfg.scene.terrain.meta_path = os.path.join(ga.out, "terrain_map.yaml")
        env_cfg.terminations.left_tile = DoneTerm(func=left_tile, params={"half_size": world.tile_size / 2},
                                                  time_out=True)  # fmt: skip
        for name in ("push_robot", "base_external_force_torque"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        if ga.flag_dim == 4:
            env_cfg.observations.policy.leg_flag = ObsTerm(func=external_leg_flag)
        elif ga.flag_dim:
            env_cfg.observations.policy.mode_input = ObsTerm(func=zero_input, params={"dim": ga.flag_dim})
        if ga.tuck:
            base = env_cfg.actions.joint_pos
            env_cfg.actions.joint_pos = GuardCPGActionCfg(**{k: getattr(base, k) for k in vars(base)
                                                             if k != "class_type"})  # fmt: skip
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
            motor = base.event_manager.get_term_cfg("motor_fault").func
            stuck = base.event_manager.get_term_cfg("foot_stuck").func
            terrain = base.event_manager.get_term_cfg("terrain").func
            sensor = base.scene.sensors["contact_forces"]
            foot_ids = sensor.find_bodies(FEET, preserve_order=True)[0]
            command = base.command_manager.get_term("base_velocity")
            num_j = motor.num_joints
            joint_leg = torch.zeros(num_j, 4, device=dev)
            joint_leg[torch.arange(num_j), torch.tensor([LEGS.index(j[:2]) for j in robot.joint_names])] = 1.0
            base.leg_flag = torch.zeros(n, 4, device=dev)

            detector = None
            if ga.detector:
                cfg = ObsTerm(func=fault_mode, params={"source": "detector", "detector_path": ga.detector})
                detector = fault_mode(cfg, base)
                detector.command_cols = []  # this detector's data has the same commands as this env
                ck = torch.load(ga.detector, map_location="cpu", weights_only=True)
                any_th = float(ck["any_threshold"])
                leg_net = None
                if "leg_state_dict" in ck:
                    leg_net = LegNet(len(ck["channels"]), ck["leg_M"]).to(dev).eval()
                    leg_net.load_state_dict({**ck["leg_state_dict"], "rotate": ck["leg_M"]})

            # random split of the robots into arms
            gen = torch.Generator().manual_seed(agent_cfg.seed)
            arm = (torch.randperm(n, generator=gen) % len(arms)).to(dev)
            source = [a[0] for a in arms]
            is_src = {s: torch.tensor([x == s for x in source], device=dev)[arm]
                      for s in set(source) | {"oracle", "fault", "wrong"}}  # fmt: skip
            fire_p = torch.tensor([a[1] for a in arms], device=dev)[arm]
            arm_hold = torch.tensor([round((a[2] or ga.hold_s) / dt) for a in arms], device=dev)[arm]
            own_hold = torch.tensor([a[2] is not None for a in arms], device=dev)[arm]
            slow_arm = torch.tensor([a[3] is not None for a in arms], device=dev)[arm]
            speed = torch.tensor([1.0 if a[3] is None else a[3] for a in arms], device=dev)[arm]
            truth_arm = is_src["oracle"] | is_src["fault"] | is_src["wrong"]
            # the arm whose alarm rate the random arm matches
            first_oracle = next((i for i, x in enumerate(source) if x in ("oracle", "detector")), None)

            refractory = round(ga.refractory_s / dt)
            latency, horizon = round(ga.latency_s / dt), round(ga.horizon_s / dt)
            rows = torch.arange(n, device=dev)
            never = -(10**9)
            prev_truth = torch.zeros(n, 4, dtype=torch.bool, device=dev)
            prev_persist = torch.zeros(n, 4, dtype=torch.bool, device=dev)
            pending_step = torch.full((n,), never, dtype=torch.long, device=dev)
            pending_leg = torch.zeros(n, dtype=torch.long, device=dev)
            flag_until = torch.full((n,), never, dtype=torch.long, device=dev)
            flag_leg = torch.zeros(n, dtype=torch.long, device=dev)
            quiet_until = torch.full((n,), never, dtype=torch.long, device=dev)
            last_cause = torch.full((n, len(CAUSES)), never, dtype=torch.long, device=dev)
            ring = torch.zeros(n, max(LAGS) + 1, 96, device=dev) if detector else None

            acc = {k: torch.zeros(len(arms), device=dev) for k in (
                "falls", "steps", "progress", "alarms", "fired", "flag_steps", "load_flagged", "load_free",
                "free_steps", "episodes")}  # fmt: skip
            # [arm, none / power out / power loss / stuck / power out + stuck / power loss + stuck, episodes / falls]
            cond = torch.zeros(len(arms), 6, 2, device=dev)
            seen = torch.zeros(n, 2, dtype=torch.long, device=dev)  # motor fault type (1 out, 2 loss), foot stuck
            alarms: dict[str, list] = {k: [] for k in ("env", "step", "leg", "fired", "cause", "true_leg", "x", "head",
                                                       "state", "x_lag", "fault")}  # fmt: skip
            fall_log: list[tuple] = []
            fall_steps: list[list[int]] = [[] for _ in range(n)]
            own = torch.zeros(n, 2, device=dev)
            slowing = torch.zeros(n, dtype=torch.bool, device=dev)
            compute = command.compute
            own_cmd = command.vel_command_b.clone()

            def compute_with_slow(step_dt):
                # the command term keeps its buffer between resamples: restore it, or the scale compounds every step
                command.vel_command_b[:] = own_cmd
                compute(step_dt)
                own_cmd[:] = command.vel_command_b
                command.vel_command_b[slowing] *= speed[slowing, None]

            command.compute = compute_with_slow
            obs = env.get_observations()
            steps = round(ga.minutes * 60 / dt)
            head = torch.zeros(n, 12, device=dev)
            with torch.inference_mode():
                for step in range(steps):
                    own = own_cmd[:, :2] / own_cmd[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-3)
                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    done = dones.bool()
                    fell = base.reset_terminated
                    v = robot.data.root_link_lin_vel_b.torch[:, :2]
                    acc["steps"].index_add_(0, arm, torch.ones(n, device=dev))
                    acc["progress"].index_add_(0, arm, (v * own).sum(-1) * dt)
                    acc["falls"].index_add_(0, arm, fell.float())
                    acc["episodes"].index_add_(0, arm, done.float())
                    kind = seen[:, 0] + 3 * seen[:, 1]
                    cond[:, :, 0].index_put_((arm[done], kind[done]), torch.ones(int(done.sum()), device=dev),
                                             accumulate=True)  # fmt: skip
                    cond[:, :, 1].index_put_((arm[fell], kind[fell]), torch.ones(int(fell.sum()), device=dev),
                                             accumulate=True)  # fmt: skip
                    for i in fell.nonzero().flatten().tolist():
                        fall_steps[i].append(step)
                        age = (step - last_cause[i]).tolist()
                        fall_log.append((int(arm[i]), i, step, int(base.leg_flag[i].sum() > 0), *age))
                    seen[done] = 0
                    last_cause[done] = never
                    pending_step[done] = never
                    flag_until[done] = never
                    quiet_until[done] = never

                    # per-leg truth of this step
                    m_labels = motor.labels()
                    faulty_joint = m_labels[:, :num_j] > 0
                    faulty = faulty_joint.float() @ joint_leg > 0
                    held = stuck.labels()[:, :4] > 0
                    seen[:, 0] = torch.where(faulty.any(1), m_labels[:, num_j].long(), seen[:, 0])
                    seen[:, 1] |= held.any(1).long()
                    parts = (faulty, held, terrain.low_mu, terrain.in_sand, terrain.collapse_event)
                    for c, part in enumerate(parts):
                        last_cause[part.any(1), c] = step
                    truth = faulty | held | terrain.low_mu | terrain.in_sand | terrain.collapse_event
                    persist = faulty | held
                    edge = torch.where(is_src["fault"][:, None], persist & ~prev_persist, truth & ~prev_truth)
                    prev_truth = truth & ~done[:, None]
                    prev_persist = persist & ~done[:, None]

                    if detector is not None:
                        ids = done.nonzero().flatten()
                        if len(ids):
                            detector.reset(ids)
                        head = detector(base, "detector", "robot", ga.detector)
                        if leg_net is not None:
                            head[:, 8:12] = torch.sigmoid(leg_net(detector.buffer))
                        ring = torch.cat([ring[:, 1:], detector.buffer[:, -1:]], 1)

                    # alarms
                    free = (step >= quiet_until) & (pending_step == never)
                    start = truth_arm & free & edge.any(1)
                    weight = edge.float() * (1.0 + persist.float())  # a motor or stuck leg before a terrain event
                    pending_step = torch.where(start, step + latency, pending_step)
                    pending_leg = torch.where(start, weight.argmax(1), pending_leg)
                    alarm = pending_step == step
                    leg = pending_leg.clone()
                    pending_step = torch.where(alarm, never, pending_step)
                    if first_oracle is not None and "random" in is_src:
                        rate = acc["alarms"][first_oracle] / acc["steps"][first_oracle].clamp(min=1.0)
                        hit = is_src["random"] & (step >= quiet_until) & (torch.rand(n, device=dev) < rate)
                        alarm |= hit
                        leg = torch.where(hit, torch.randint(4, (n,), device=dev), leg)
                    if is_src["wrong"].any():
                        shifted = (leg + torch.randint(1, 4, (n,), device=dev)) % 4
                        leg = torch.where(is_src["wrong"], shifted, leg)
                    if detector is not None and "detector" in is_src:
                        hit = is_src["detector"] & (step >= quiet_until) & (head[:, 0] > any_th)
                        alarm |= hit
                        leg = torch.where(hit, head[:, 8:12].argmax(1), leg)

                    fire = alarm & (torch.rand(n, device=dev) < fire_p)
                    flag_until = torch.where(fire, step + arm_hold, flag_until)
                    flag_leg = torch.where(fire, leg, flag_leg)
                    quiet_until = torch.where(alarm, step + arm_hold + refractory, quiet_until)
                    # an oracle flag lasts as long as the motor fault or the stuck foot it reports
                    keep = truth_arm & ~own_hold & (step < flag_until) & persist.any(1)
                    flag_until = torch.where(keep, flag_until.clamp(min=step + 2), flag_until)
                    quiet_until = torch.where(keep, flag_until + refractory, quiet_until)
                    on = step < flag_until
                    slowing[:] = on & slow_arm
                    base.leg_flag.zero_()
                    flagged = on & ~slow_arm
                    base.leg_flag[rows[flagged], flag_leg[flagged]] = 1.0

                    load = torch.linalg.norm(sensor.data.net_forces_w.torch[:, foot_ids], dim=-1)
                    acc["alarms"].index_add_(0, arm, alarm.float())
                    acc["fired"].index_add_(0, arm, fire.float())
                    acc["flag_steps"].index_add_(0, arm, on.float())
                    acc["load_flagged"].index_add_(0, arm, torch.where(on, load[rows, flag_leg], 0.0))
                    acc["load_free"].index_add_(0, arm, torch.where(on, 0.0, load.mean(1)))
                    acc["free_steps"].index_add_(0, arm, (~on).float())

                    if alarm.any():
                        i = alarm.nonzero().flatten()
                        d = robot.data
                        state = torch.cat([d.projected_gravity_b.torch[i], d.root_link_ang_vel_b.torch[i],
                                           d.root_link_lin_vel_b.torch[i], d.root_link_pos_w.torch[i, 2:3], load[i],
                                           d.joint_pos.torch[i], d.joint_vel.torch[i]], 1)  # fmt: skip
                        on_leg = torch.stack([p[i, leg[i]] for p in parts], 1)
                        any_leg = torch.stack([p[i].any(1) for p in parts], 1)
                        alarms["env"].append(i.cpu())
                        alarms["step"].append(torch.full_like(i, step).cpu())
                        alarms["leg"].append(leg[i].cpu())
                        alarms["fired"].append(fire[i].cpu())
                        alarms["cause"].append(torch.cat([on_leg, any_leg], 1).cpu())
                        alarms["true_leg"].append(truth[i].cpu())
                        alarms["state"].append(state.cpu())
                        alarms["fault"].append(torch.stack([seen[i, 0], faulty_joint[i].float().argmax(1)], 1).cpu())
                        alarms["head"].append(head[i].cpu())
                        if ring is not None:
                            lag = [ring[i, -1 - k] for k in LAGS]
                            alarms["x_lag"].append(torch.stack(lag, 1).half().cpu())
                    if (step + 1) % 1500 == 0:
                        print(f"[guard] step {step + 1}/{steps}: falls {acc['falls'].tolist()}, "
                              f"alarms {acc['alarms'].tolist()}", flush=True)  # fmt: skip

            minutes = acc["steps"] * dt / 60
            result = {"task": args_cli.task, "checkpoint": resume_path, "args": vars(ga), "seed": agent_cfg.seed,
                      "num_envs": n, "arms": {}}  # fmt: skip
            for a, name in enumerate(ga.arms):
                m = float(minutes[a])
                result["arms"][name] = {
                    "robots": int((arm == a).sum()),
                    "robot_minutes": m,
                    "falls": float(acc["falls"][a]),
                    "falls_per_min": float(acc["falls"][a]) / m,
                    "progress_m_per_min": float(acc["progress"][a]) / m,
                    "alarms_per_min": float(acc["alarms"][a]) / m,
                    "fired_per_min": float(acc["fired"][a]) / m,
                    "flag_time_fraction": float(acc["flag_steps"][a] / acc["steps"][a]),
                    "load_flagged_N": float(acc["load_flagged"][a] / acc["flag_steps"][a].clamp(min=1.0)),
                    "load_free_N": float(acc["load_free"][a] / acc["free_steps"][a].clamp(min=1.0)),
                    "episodes": cond[a, :, 0].tolist(),
                    "episode_falls": cond[a, :, 1].tolist(),
                }
            with open(os.path.join(ga.out, "result.json"), "w") as f:
                json.dump(result, f, indent=1)
            print("[guard] " + json.dumps(result["arms"]), flush=True)
            save = {k: torch.cat(v).numpy() for k, v in alarms.items() if v}
            if save:
                env_id, at = save["env"], save["step"]
                save["fell"] = np.array([any(s < f <= s + horizon for f in fall_steps[e]) for e, s in zip(env_id, at)])
                save["lead"] = np.array([min([f - s for f in fall_steps[e] if f > s], default=-1)
                                         for e, s in zip(env_id, at)])  # fmt: skip
            np.savez_compressed(os.path.join(ga.out, "alarms.npz"), arm=arm.cpu().numpy(), **save)
            with open(os.path.join(ga.out, "falls.csv"), "w") as f:
                f.write("arm,env,step,flagged," + ",".join(f"age_{c}" for c in CAUSES) + "\n")
                f.writelines(",".join(map(str, r)) + "\n" for r in fall_log)
        finally:
            close_env(env)


if __name__ == "__main__":
    main()
