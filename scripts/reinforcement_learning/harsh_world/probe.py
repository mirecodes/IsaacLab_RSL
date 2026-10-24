# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator probes of the Harsh World acceptance tests. Each probe prints ``[T<n>] ... PASS/FAIL`` lines.

    python scripts/reinforcement_learning/harsh_world/probe.py ground [--mesh]  # terrain collision check
    python scripts/reinforcement_learning/harsh_world/probe.py sand      # T2: sinkage, stiffness, scanner, drag
    python scripts/reinforcement_learning/harsh_world/probe.py collapse [--collapse_mode disable]  # T6: crust
    python scripts/reinforcement_learning/harsh_world/probe.py rocks     # T6: settling, penetration, rock return
    python scripts/reinforcement_learning/harsh_world/probe.py slip --checkpoint <robust model.pt>   # T3
    python scripts/reinforcement_learning/harsh_world/probe.py smoke --checkpoint <model.pt> [--full]  # T4, T6
    python scripts/reinforcement_learning/harsh_world/probe.py faults <collect.py output dir>  # T4 fault coverage
    python scripts/reinforcement_learning/harsh_world/probe.py view [<out dir>] [--full] [--checkpoint <model.pt>]

``sand``, ``collapse``, and ``rocks`` need no policy and use small worlds; ``slip`` and ``smoke`` play a trained
robust checkpoint (``--task`` picks another Harsh World task). Add ``--device cpu`` to run without a GPU.
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import ast  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool]] = []


def check(tag: str, ok: bool, message: str) -> None:
    RESULTS.append((tag, bool(ok)))
    print(f"[{tag}] {message}  {'PASS' if ok else 'FAIL'}", flush=True)


def parse():
    parser = argparse.ArgumentParser(description="Harsh World simulator probes.")
    parser.add_argument("probe", choices=["ground", "sand", "collapse", "rocks", "slip", "smoke", "view", "faults"])
    parser.add_argument("collection", nargs="?", default=None,
                        help="faults: collect.py output directory; view: output directory.")  # fmt: skip
    parser.add_argument("--full", action="store_true", help="smoke, view: all 10 properties.")
    parser.add_argument("--collapse_mode", choices=["soften", "disable"], default="soften")
    parser.add_argument("--mesh", action="store_true", help="Collide against the triangle mesh, not heightfields.")
    parser.add_argument("--set", nargs="*", default=[], help="HarshWorldCfg overrides, e.g. rocks.dynamic_rocks=0.")
    parser.add_argument("--seconds", type=float, default=None, help="Simulated seconds (smoke 300, slip 60).")
    parser.add_argument("--robots_per_cell", type=int, default=None)
    parser.add_argument("--horizontal_scale", type=float, default=None)
    parser.add_argument("--region_size", type=float, default=None)
    args, argv = parser.parse_known_args()
    if args.probe == "faults":
        return args, None
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args

    if "--task" not in argv:
        variant = "Robust-Full" if args.full or args.probe in ("collapse", "rocks") else "Robust"
        argv += ["--task", f"IsaacContrib-Velocity-A2HarshWorld-{variant}"]
    return args, _parse_args(argv)


def world(args, keep: list[str], **overrides):
    """A HarshWorldCfg with only the ``keep`` properties (the default world if empty), and CLI overrides."""
    from isaaclab_tasks.forge.velocity.a2_harsh_world.harsh_world_cfg import HarshWorldCfg

    hw = HarshWorldCfg()
    if keep:
        hw.disabled = [name for name in hw.properties if name not in keep]
    for key in ("robots_per_cell", "horizontal_scale", "region_size"):
        if getattr(args, key) is not None:
            setattr(hw, key, getattr(args, key))
    for key, value in overrides.items():
        setattr(hw, key, value)
    for item in args.set:
        path, value = item.split("=", 1)
        *parents, leaf = path.split(".")
        target = hw
        for name in parents:  # e.g. properties.mesa_drop.scree_prob
            target = target[name] if isinstance(target, dict) else getattr(target, name)
        setattr(target, leaf, ast.literal_eval(value))
    if args.mesh:
        for cfg in hw.properties.values():
            cfg.convert_to_heightfield = False
    return hw


def make_env(args_cli, hw, configure=None):
    """Create the task's env with ``hw`` applied; ``configure(env_cfg)`` may add scene entities."""
    from isaaclab_rl.entrypoints.common import apply_env_overrides, create_isaaclab_env

    from isaaclab_tasks.forge.velocity.a2_harsh_world.env_cfg import apply_harsh_world
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    apply_harsh_world(env_cfg, hw, robust_commands="Robust" in args_cli.task)
    apply_env_overrides(args_cli, env_cfg)
    if args_cli.device is not None:
        # launch_simulation applies --device to the cfg it was given; this is a fresh one
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = hw.num_envs
    env_cfg.seed = hw.seed
    env_cfg.scene.terrain.terrain_generator.seed = hw.seed
    if configure is not None:
        configure(env_cfg)
    env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
    return env, agent_cfg


def load_policy(env, agent_cfg, args_cli):
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab.utils import to_dict

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=env.unwrapped.device)
    runner.load(args_cli.checkpoint, map_location=env.unwrapped.device)
    return wrapped, runner.get_inference_policy(device=env.unwrapped.device)


def add_probe_spheres(radius: float, mass: float):
    """Scene hook: one free sphere of the foot's radius per environment (``/Probe``), parked high."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    def configure(env_cfg):
        env_cfg.scene.probe = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Probe",
            spawn=sim_utils.SphereCfg(
                radius=radius,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 30.0)),
        )

    return configure


def place(asset, pos: torch.Tensor) -> None:
    """Put each env's single rigid body at ``pos`` (N, 3), upright and at rest."""
    pose = torch.cat([pos, torch.tensor([0.0, 0.0, 0.0, 1.0], device=pos.device).expand(len(pos), 4)], dim=-1)
    ids = torch.arange(len(pos), device=pos.device)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(pos), 6, device=pos.device), env_ids=ids)


def park_robots(env) -> None:
    """Move the robots onto the flat border, out of the cells' way."""
    robot = env.scene["robot"]
    zones = env.scene.terrain.zone_map
    pos = torch.zeros(env.num_envs, 3, device=env.device)
    pos[:, 0] = zones.corner[0] - 3.0
    pos[:, 1] = zones.corner[1] + 2.0 * torch.arange(env.num_envs, device=env.device)
    pos[:, 2] = 0.45
    place(robot, pos)


def run(env, seconds: float, actions=None) -> None:
    steps = round(seconds / env.step_dt)
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(steps):
        env.step(zero if actions is None else actions())


##
# Probes.
##


def probe_ground(args, args_cli):
    """Terrain collision: spheres dropped all over two cells, robots standing (``--mesh`` for the triangle mesh
    instead of Newton's heightfields)."""
    hw = world(args, ["gravel", "rough_slope"], num_rows=1, robots_per_cell=8)
    radius = hw.sand.foot_radius
    env, _ = make_env(args_cli, hw, add_probe_spheres(radius, 10.0))
    env = env.unwrapped
    env.reset()
    zones = env.scene.terrain.zone_map
    xy = zones.corner + torch.rand(env.num_envs, 2, device=env.device) * zones.size * torch.tensor(
        [zones.num_rows, zones.num_cols], device=env.device)  # fmt: skip
    ground = zones.height(xy)
    place(env.scene["probe"], torch.cat([xy, (ground + 0.1 + radius)[:, None]], -1))
    start = time.time()
    run(env, 2.0)
    wall = time.time() - start
    # spheres on the slopes roll and bounce away: check that none is below the ground where it ended up
    pos = env.scene["probe"].data.root_link_pos_w.torch
    depth = zones.height(pos[:, :2]) - (pos[:, 2] - radius)
    base = env.scene["robot"].data.root_link_pos_w.torch[:, 2] - env.scene.terrain.env_origins[:, 2]
    kind = "mesh" if args.mesh else "heightfield"
    check("T0", (depth < 0.02).all().item(), f"{kind}: no sphere below the terrain (max depth {depth.max() * 100:.1f} "
          f"cm, {int((depth.abs() < 0.02).sum())}/{env.num_envs} resting on it), 2 s in {wall:.1f} s wall")  # fmt: skip
    check("T0", (base > 0.15).all().item(), f"{kind}: robots stay above the ground (min base {base.min():.2f} m)")
    env.close()


def probe_sand(args, args_cli):
    """T2: slab sinkage (and its stiffness dependence), rigid sinkage, scanner on the sand surface, drag."""
    hw = world(args, ["sand_field"], num_rows=1, robots_per_cell=6, region_size=args.region_size or 16.0)
    radius = hw.sand.foot_radius
    env, _ = make_env(args_cli, hw, add_probe_spheres(radius, 10.0))
    env = env.unwrapped
    env.reset()
    zones, term = env.scene.terrain.zone_map, env.event_manager.get_term_cfg("harsh_world").func
    patches = sorted(zones.patches, key=lambda p: -min(p["size"]))[:2]
    # patch A soft (800 N/m), patch B stiff (3000 N/m)
    for p, k in zip(patches, (800.0, 3000.0)):
        shape = term.slab_shape[p["id"]].item()
        term._ke[shape], term._kd[shape] = k, hw.sand.damping_ratio * k
    term._newton.add_model_change(term._shape_change)
    # envs 0-1: patch A, 2-3: patch B, 4-5: rigid ground (the flat platform; the hardpan beside it is rough)
    xy = torch.tensor([[p["x"], p["y"]] for p in (patches[0], patches[0], patches[1], patches[1])], device=env.device)
    xy[1::2] += 0.1
    origin = env.scene.terrain.terrain_origins[0, 0, :2]
    xy = torch.cat([xy, (origin + torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=env.device))])
    surface = torch.tensor([patches[0]["surface"]] * 2 + [patches[1]["surface"]] * 2, device=env.device)
    surface = torch.cat([surface, zones.height(xy[4:])])
    park_robots(env)
    place(env.scene["probe"], torch.cat([xy, (surface + 0.10 + radius)[:, None]], dim=-1))
    run(env, 3.0)
    z = env.scene["probe"].data.root_link_pos_w.torch[:, 2]
    sink = (surface - (z - radius)).cpu().numpy()
    soft, stiff, rigid = sink[:2].mean(), sink[2:4].mean(), np.abs(sink[4:]).max()
    deepest = max(patches[0]["depth"], patches[1]["depth"])
    print(
        f"[T2] sinkage of a 10 kg foot sphere: 800 N/m {soft * 100:.2f} cm, 3000 N/m {stiff * 100:.2f} cm, "
        f"rigid {rigid * 1000:.2f} mm (expected ~ m g / k: {98.1 / 800 * 100:.1f} / {98.1 / 3000 * 100:.1f} cm)"
    )
    check("T2", 0.01 <= soft <= patches[0]["depth"] + 0.03 and 0.01 <= stiff <= deepest + 0.03,
          "sand sinkage between 1 cm and the sand depth")  # fmt: skip
    check("T2", rigid < 0.002, "rigid sinkage below 2 mm")
    check("T2", soft > stiff, "sinkage decreases with stiffness")

    # scanner: base over the biggest patch, rays inside the patch footprint see the surface, not the hardpan
    robot = env.scene["robot"]
    p = patches[0]
    pos = torch.tensor([[p["x"], p["y"], p["surface"] + 0.6]], device=env.device).expand(env.num_envs, 3).clone()
    place(robot, pos)
    run(env, env.step_dt)
    hits = env.scene["height_scanner"].data.ray_hits_w.torch[0]
    inside = ((hits[:, :2] - torch.tensor([p["x"], p["y"]], device=env.device)).abs()
              < torch.tensor(p["size"], device=env.device) / 2 - 0.05).all(-1)  # fmt: skip
    err = (hits[inside, 2] - p["surface"]).abs().max().item() if inside.any() else float("nan")
    check("T2", inside.any() and err <= 0.01, f"scanner hits over the patch within {err * 100:.2f} cm of the surface")

    # drag on synthetic feet: in sand moving at 0.5 m/s horizontally, and out of the sand
    pos = torch.zeros(env.num_envs, 4, 3, device=env.device)
    pos[..., 0], pos[..., 1] = p["x"], p["y"]
    pos[..., 2] = p["surface"] - 0.03
    pos[:, 2:, :2] = origin + 2.5  # feet 2-3 on the hardpan
    vel = torch.zeros_like(pos)
    vel[..., 0] = 0.5
    zone = zones.lookup(pos[..., :2])
    patch, surf = zones.patch(pos[..., :2])
    term._sand(zone, patch, surf, pos, vel)
    expected = (hw.sand.c_drag[0] + zone["difficulty"][0, 0].item() * (hw.sand.c_drag[1] - hw.sand.c_drag[0])) * 0.5
    got = term._forces[0, 0].norm().item()
    check("T2", abs(got - expected) <= 0.05 * expected and term._forces[:, 2:].abs().max().item() == 0.0,
          f"drag {got:.2f} N for c_drag * 0.5 m/s = {expected:.2f} N; feet outside the sand exactly 0 N")  # fmt: skip
    env.close()


def probe_collapse(args, args_cli):
    """T6: a crust holds a load below the threshold, collapses within one step above it, recovers later."""
    from isaaclab_tasks.forge.velocity.a2_harsh_world.harsh_world_cfg import default_properties

    radius = 0.032
    # one mode per process: a second env in the same process fails the cfg validation (recursion)
    for mode in (args.collapse_mode,):
        hw = world(args, ["crust_collapse"], num_rows=1, robots_per_cell=4, region_size=args.region_size or 16.0)
        hw.properties = default_properties()
        hw.properties["crust_collapse"].crust_prob = 1.0
        hw.collapse.recover_time = 3.0
        hw.collapse.collapse_mode = mode
        env, _ = make_env(args_cli, hw, add_probe_spheres(radius, 10.0))
        env = env.unwrapped
        env.reset()
        zones, term = env.scene.terrain.zone_map, env.event_manager.get_term_cfg("harsh_world").func
        patch = sorted(zones.patches, key=lambda p: -min(p["size"]))[0]
        park_robots(env)
        xy = torch.tensor([[patch["x"], patch["y"]]], device=env.device).expand(env.num_envs, 2)
        place(env.scene["probe"], torch.cat([xy, torch.full((env.num_envs, 1), patch["surface"] + radius + 0.01,
                                                            device=env.device)], -1))  # fmt: skip
        sphere = env.scene["probe"]
        limit = hw.collapse.collapse_force[0] + zones.difficulty[0, 0].item() * (
            hw.collapse.collapse_force[1] - hw.collapse.collapse_force[0]
        )

        def step_with_load(scale: float):
            """One step; the sphere's weight x ``scale`` stands in for a robot's feet."""
            env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
            # the sphere as one "foot" (the other three columns carry no load); its weight, not the contact
            # sensor, so that the trigger is checked against a known load
            force = torch.zeros(env.num_envs, 4, device=env.device)
            force[:, 0] = 98.1 * scale
            pid, _ = zones.patch(sphere.data.root_link_pos_w.torch[:, None, :2].expand(-1, 4, 2))
            term._collapse(env, pid, force)

        def mean_sinkage(scale: float, settle: float, window: float) -> float:
            """Steps with a load for ``settle + window`` seconds; the sphere's sinkage averaged over the window
            (a sphere on a slab keeps oscillating by a few millimeters, MuJoCo-Warp has no sleeping)."""
            for _ in range(round(settle / env.step_dt)):
                step_with_load(scale)
            z = []
            for _ in range(round(window / env.step_dt)):
                step_with_load(scale)
                z.append(sphere.data.root_link_pos_w.torch[:, 2].mean().item())
            return patch["surface"] - (float(np.mean(z)) - radius)

        sink0 = mean_sinkage(1.0, 1.0, 1.0)  # 98 N, below the threshold
        held = not term.collapsed[patch["id"]].item()
        shape = term.slab_shape[patch["id"]].item()
        ke0 = term._ke[shape].item()
        step_with_load(limit / 98.1 * 1.2)
        collapsed = bool(term.collapsed[patch["id"]].item())
        print(f"[T6] {mode}: slab ke {ke0:.0f} -> {term._ke[shape].item():.1f} N/m")
        sink1 = mean_sinkage(0.0, 0.5, 1.0)
        final = patch["surface"] - (sphere.data.root_link_pos_w.torch[:, 2].mean().item() - radius)
        hardpan = patch["surface"] - zones.height(sphere.data.root_link_pos_w.torch[:1, :2]).item()
        if mode == "disable":
            check("T6", final >= hardpan - 0.01, f"disable: sphere down to the hardpan ({final * 1000:.0f} mm, "
                  f"hardpan {hardpan * 1000:.0f} mm below the surface)")  # fmt: skip
        for _ in range(round((hw.collapse.recover_time - 1.4) / env.step_dt)):
            step_with_load(0.0)
        recovered = not term.collapsed[patch["id"]].item()
        print(
            f"[T6] collapse ({mode}): limit {limit:.0f} N, sinkage before {sink0 * 1000:.1f} mm, after "
            f"{sink1 * 1000:.1f} mm, hardpan at {(patch['surface'] - patch['bottom']) * 1000:.0f} mm below"
        )
        check("T6", held, f"{mode}: crust holds 98 N (below {limit:.0f} N)")
        check("T6", collapsed, f"{mode}: crust collapses in the step the load exceeds the threshold")
        check("T6", sink1 >= 3 * max(sink0, 1e-3), f"{mode}: sinkage x{sink1 / max(sink0, 1e-3):.1f} after collapse")
        check("T6", recovered, f"{mode}: patch restored after recover_time")
        env.close()


def probe_rocks(args, args_cli):
    """T6: dynamic rocks settle without penetration, and only drifted rocks return to their slots."""
    hw = world(args, ["loose_rocks"], num_rows=1, robots_per_cell=args.robots_per_cell or 8)
    env, _ = make_env(args_cli, hw)
    env = env.unwrapped
    env.reset()
    zones, rocks = env.scene.terrain.zone_map, env.scene["rocks"]
    term = env.event_manager.get_term_cfg("harsh_world").func
    park_robots(env)
    data = rocks.data
    speeds = []
    for i in range(round(10.0 / env.step_dt)):
        run(env, env.step_dt)
        speeds.append(data.body_com_lin_vel_w.torch.norm(dim=-1).cpu())
        if i == round(5.0 / env.step_dt):
            pos5 = data.body_link_pos_w.torch.clone()
        if i == round(9.0 / env.step_dt) - 1:
            pos9 = data.body_link_pos_w.torch.clone()
    speed = torch.stack(speeds)  # (T, N, K)
    late = speed[round(5.0 / env.step_dt) :]
    rest = (late < 0.01).float().mean().item()
    pos = data.body_link_pos_w.torch
    ground = zones.height(pos[..., :2])
    sunk = (pos[..., 2] < ground - 0.02).sum().item()
    total = pos.shape[0] * pos.shape[1]
    # at rest = net speed over the last second; without sleeping a few resting rocks jitter in place at ~1 cm/s
    net, moved = (pos - pos9).norm(dim=-1), (pos - pos5).norm(dim=-1).max() * 1000
    check("T6", (net < 0.01).all().item(), f"all {total} rocks at rest after 10 s (net speed over the last 1 s: max "
          f"{net.max():.4f} m/s; instantaneous: max {speed[-1].max():.4f} m/s, {int((speed[-1] >= 0.01).sum())} rocks "
          f">= 0.01 m/s; largest displacement over the last 5 s {moved:.1f} mm)")  # fmt: skip
    check("T6", sunk == 0, f"{sunk} rocks below the terrain")
    check("T6", rest >= 0.95, f"{rest * 100:.1f} % of rock-steps at rest after 5 s (no sleeping in MuJoCo-Warp)")
    # return: push half of env 0's rocks 2 m away, nudge the rest by 0.3 m
    k = pos.shape[1]
    pose = data.body_link_pose_w.torch.clone()
    before = pose.clone()
    pose[0, : k // 2, 0] += 2.0
    pose[0, k // 2 :, 0] += 0.3
    first = torch.tensor([0], device=env.device)
    rocks.write_body_link_pose_to_sim_index(body_poses=pose[:1], body_ids=list(range(k)), env_ids=first)
    moved = term.return_rocks(first, only_drifted=True)
    after = data.body_link_pose_w.torch
    back = (after[0, : k // 2, :2] - zones.rock_slots[env.scene.terrain.terrain_levels[0], : k // 2, :2]).norm(dim=-1)
    kept = (after[0, k // 2 :, :3] - pose[0, k // 2 :, :3]).norm(dim=-1)
    check("T6", moved == k // 2 and back.max() < 1e-3 and kept.max() < 1e-4,
          f"rock return moved {moved}/{k} rocks (the {k // 2} drifted ones)")  # fmt: skip
    del before
    env.close()


def probe_slip(args, args_cli):
    """T3: friction written on entry and restored on exit, sample distribution, slide vs. the rigid twin."""
    from scipy import stats

    from isaaclab_tasks.forge.velocity.a2_harsh_world.harsh_world_cfg import default_properties
    from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp.terrains import RoughSlopeCfg

    props = default_properties()
    twin = props["slip_slope"].replace(zone="rigid")
    hw = world(args, [], num_rows=1, robots_per_cell=args.robots_per_cell or 16)
    hw.properties = {"slip_slope": props["slip_slope"], "slip_slope_rigid": twin}
    hw.disabled = []
    assert isinstance(twin, RoughSlopeCfg)
    env, agent_cfg = make_env(args_cli, hw)
    wrapped, policy = load_policy(env, agent_cfg, args_cli)
    env = env.unwrapped
    term = env.event_manager.get_term_cfg("harsh_world").func
    zones = env.scene.terrain.zone_map
    robot, contact = env.scene["robot"], env.scene["contact_forces"]
    obs = wrapped.get_observations()
    slide = torch.zeros(env.num_envs, device=env.device)
    worst = 0.0
    changes = 0
    with torch.inference_mode():
        for _ in range(round((args.seconds or 60.0) / env.step_dt)):
            before = term.foot_mu.clone()
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
            if not torch.equal(before, term.foot_mu):
                changes += 1
                view = wp.to_torch(term._foot_mu_binding)
                written = torch.stack([view[:, s.to(view.device)].mean(-1) for s in term._foot_shapes], -1)
                worst = max(worst, (written - term.foot_mu).abs().max().item())
            vel = robot.data.body_com_lin_vel_w.torch[:, term.foot_ids, :2].norm(dim=-1)
            touching = contact.data.net_forces_w.torch[:, term.contact_foot_ids].norm(dim=-1) > 1.0
            slide += (vel * touching).sum(-1) * env.step_dt
    check("T3", changes > 0 and worst < 1e-6, f"model friction equals the sampled friction after {changes} changes")
    col = env.scene.terrain.terrain_types
    slip, rigid = slide[col == 0].mean().item(), slide[col == 1].mean().item()
    ratio = slip / max(rigid, 1e-6)
    check(
        "T3",
        slip > 0 and ratio >= 3,
        f"foot slide per robot: slip {slip:.2f} m, rigid twin {rigid:.2f} m (x{ratio:.1f})",
    )
    # entry samples: 1000 synthetic entries at difficulty d
    d = zones.difficulty[0, 0].item()
    n = term.foot_mu.shape[0]
    samples, lows = [], []
    for i in range(math.ceil(1000 / (4 * n)) * 2):
        # even i: every foot enters the slip zone; odd i: every foot leaves it
        inside = (i + 1) % 2
        zone = {"zone": torch.full((n, 4), inside, device=env.device), "difficulty": torch.full((n, 4), d)}
        zone["difficulty"] = zone["difficulty"].to(env.device)
        term._slip(env, zone)
        if i % 2 == 0:
            samples.append(term.foot_mu.flatten().cpu())
            lows.append(term.low_mu.flatten().cpu())
    mu, low = torch.cat(samples)[:1000].numpy(), torch.cat(lows)[:1000].numpy()
    lo = hw.slip.mu_slip_easy[0] + d * (hw.slip.mu_slip_hard[0] - hw.slip.mu_slip_easy[0])
    hi = hw.slip.mu_slip_easy[1] + d * (hw.slip.mu_slip_hard[1] - hw.slip.mu_slip_easy[1])
    p = stats.kstest(mu[low], stats.uniform(loc=lo, scale=hi - lo).cdf).pvalue
    check("T3", p > 0.05 and abs(low.mean() - hw.slip.slip_prob) <= 0.05 and (mu[~low] == hw.slip.mu_nominal).all(),
          f"1000 entries: low share {low.mean():.3f} (p={hw.slip.slip_prob}), KS p={p:.3f} vs "
          f"U({lo:.3f}, {hi:.3f})")  # fmt: skip
    env.close()


def probe_smoke(args, args_cli):
    """T4/T6: spawn, labels, cell tracking, flights, rock contacts, stability, and speed under the policy."""
    hw = world(args, [])
    if args.full:
        hw.disabled = []
    env, agent_cfg = make_env(args_cli, hw)
    wrapped, policy = load_policy(env, agent_cfg, args_cli)
    env = env.unwrapped
    zones, terrain = env.scene.terrain.zone_map, env.scene.terrain
    robot, contact = env.scene["robot"], env.scene["contact_forces"]
    term = env.event_manager.get_term_cfg("harsh_world").func
    names = env.scene.terrain.meta["properties"]
    obs = wrapped.get_observations()
    # spawn: base inside its own cell's platform
    base = robot.data.root_link_pos_w.torch
    offset = (base[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=-1).values
    check("T4", (offset <= hw.platform_width / 2).all().item(), f"all {env.num_envs} robots spawn on their platform "
          f"(max offset {offset.max():.2f} m)")  # fmt: skip
    seconds = args.seconds or 300.0  # the spec's 5 min smoke run
    steps = round(seconds / env.step_dt)
    early_falls = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    left_falls, stumbles, left = (torch.zeros_like(early_falls) for _ in range(3))
    flight = torch.zeros(env.num_envs, device=env.device)
    flights = torch.zeros(len(names), device=env.device)
    rock_contacts = torch.zeros(len(names), device=env.device)
    label_errors, crossings = 0, 0
    nan_count, fast_count = 0, torch.zeros(len(names), device=env.device)
    prev_cell = None
    start = time.time()
    trigger = [i for i, n in enumerate(contact.body_names) if not n.endswith("_foot")]
    with torch.inference_mode():
        for step in range(steps):
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
            zone = env.obs_buf["zone"]
            pos = robot.data.root_link_pos_w.torch
            if step == 0:
                for key, value in zone.items():
                    print(f"[T4] zone field {key}: shape {tuple(value.shape[1:])}, dtype {value.dtype}")
            # labels: base_cell from the stored label vs. recomputed from the position (10 robots)
            expected = torch.floor((pos[:10, :2] - zones.corner) / zones.size).long()
            cell = zone["base_cell"][:10]
            grid = torch.tensor([zones.num_rows, zones.num_cols], device=env.device)
            valid = ((expected >= 0) & (expected < grid)).all(-1)  # robots on the border outside the grid get -1
            label_errors += int((cell[valid] != expected[valid]).any(-1).sum())
            if prev_cell is not None:
                moved = (zone["base_cell"] != prev_cell).any(-1) & ~dones.bool()
                crossings += int(moved.sum())
            prev_cell = zone["base_cell"].clone()
            # first second: falls = the trained task's terminations (base contact, tilt > 1 rad), split by whether
            # the robot had left its platform (robots start at up to 1.5 m/s); stumbles = any other body touching
            if step * env.step_dt < 1.0:
                fell = env.termination_manager.terminated
                early_falls |= fell & ~left
                left_falls |= fell & left
                stumbles |= (contact.data.net_forces_w.torch[:, trigger].norm(dim=-1) > 1.0).any(-1) & ~fell
                off = (pos[:, :2] - terrain.env_origins[:, :2]).abs().max(-1).values > hw.platform_width / 2
                left |= off & ~fell  # a fallen robot's position is already its reset position
            # flights: all feet off the ground > 0.3 s, then a touchdown
            air = (contact.data.net_forces_w.torch[:, term.contact_foot_ids].norm(dim=-1) < 1.0).all(-1)
            landed = ~air & (flight > 0.3)
            prop = zone["base_property"].clamp(min=0)
            flights.index_add_(0, prop[landed], torch.ones(int(landed.sum()), device=env.device))
            flight = torch.where(air, flight + env.step_dt, 0.0)
            moving = zone["rock_moving"].any(-1)
            rock_contacts.index_add_(0, prop[moving], torch.ones(int(moving.sum()), device=env.device))
            vel = robot.data.body_com_lin_vel_w.torch
            nan_count += int(torch.isnan(vel).any(-1).any(-1).sum())
            fast = (vel.norm(dim=-1) > 20.0).any(-1)
            fast_count.index_add_(0, prop[fast], torch.ones(int(fast.sum()), device=env.device))
        wall = time.time() - start
        # speed without this probe's bookkeeping (its per-step host syncs cost more than the world itself)
        torch.cuda.synchronize()
        bench = time.time()
        for _ in range(round(10.0 / env.step_dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
        torch.cuda.synchronize()
        bench = time.time() - bench
    rtf = 10.0 / bench

    def where(mask):
        rows, cols = terrain.terrain_levels[mask], terrain.terrain_types[mask]
        return [f"{names[p]} d={d:.1f}" for p, d in zip(zones.property_id[rows, cols].tolist(),
                                                         zones.difficulty[rows, cols].tolist())]  # fmt: skip

    check("T4", not early_falls.any().item(), f"{int(early_falls.sum())} robots fell on their platform within 1 s "
          f"of spawning {where(early_falls)}")  # fmt: skip
    print(f"[T4] {int(left_falls.sum())} more fell within 1 s after leaving the platform {where(left_falls)}; "
          f"{int(stumbles.sum())} touched the ground with a leg without falling {where(stumbles)}")  # fmt: skip
    check("T4", label_errors == 0, f"base_cell matched the position for 10 robots x {steps} steps")
    print(f"[T4] cell crossings detected in the step they happened: {crossings} (labels are computed per step)")
    print(f"[T4] real-time factor: {rtf:.2f} ({env.num_envs} robots, 10 s sim in {bench:.1f} s wall; with the "
          f"probe's checks {seconds / wall:.2f})")  # fmt: skip
    check("T4", rtf >= 5.0, f"real-time factor {rtf:.2f} >= 5")
    check("T6", nan_count == 0, f"NaN states {nan_count}")
    per_minute = 60.0 / seconds
    for i, name in enumerate(names):
        print(f"[T6] {name}: flights {int(flights[i])}, moving-rock contacts {rock_contacts[i] * per_minute:.1f}/min, "
              f"robot-steps with a body faster than 20 m/s {int(fast_count[i])}")  # fmt: skip
    if "mesa_drop" in names:
        rate = flights[names.index("mesa_drop")] * per_minute
        check("T6", rate >= 1, f"mesa_drop: {rate:.1f} flights and landings per minute (>= 1)")
    if "loose_rocks" in names and hw.rocks.dynamic_rocks > 0:
        i = names.index("loose_rocks")
        others = rock_contacts.sum() - rock_contacts[i]
        check("T6", rock_contacts[i] * per_minute >= 1 and others == 0, f"moving-rock contacts only in loose_rocks "
              f"({rock_contacts[i] * per_minute:.1f}/min there, {int(others)} elsewhere)")  # fmt: skip
    if hw.collapse.enabled and "crust_collapse" in names:
        # the spec's collapse stability condition (tumbles down the 10 m slopes can exceed 20 m/s elsewhere)
        fast = int(fast_count[names.index("crust_collapse")])
        check("T6", fast == 0, f"crust_collapse: {term.num_collapses} collapses, {fast} robot-steps faster than 20 m/s")
    env.close()


def probe_view(args, args_cli):
    """Render the world to PNGs: one overview, one oblique view per cell, one robot close-up per property."""
    from PIL import Image

    from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES

    hw = world(args, [])
    if args.full:
        hw.disabled = []

    def configure(env_cfg):
        # the visualizer launch_simulation set up on its own cfg (this one is fresh)
        env_cfg.sim.visualizer_cfgs = args.launch_cfg.sim.visualizer_cfgs
        for term in vars(env_cfg.commands).values():
            if hasattr(term, "debug_vis"):
                term.debug_vis = False

    env, agent_cfg = make_env(args_cli, hw, configure)
    if args_cli.checkpoint:
        wrapped, policy = load_policy(env, agent_cfg, args_cli)
        obs = wrapped.get_observations()
        env = env.unwrapped
        with torch.inference_mode():
            for _ in range(round((args.seconds or 5.0) / env.step_dt)):
                obs, _, dones, _ = wrapped.step(policy(obs))
                policy.reset(dones)
    else:
        env = env.unwrapped
        env.reset()
        run(env, args.seconds or 2.0)  # rocks settle, robots stand
    viz = next(v for v in env.sim.visualizers if v.cfg.backend.visualizer_type in CAPTURE_VISUALIZER_TYPES)
    zones, meta = env.scene.terrain.zone_map, env.scene.terrain.meta
    out = os.path.abspath(args.collection or "logs/harsh_world/view")
    os.makedirs(out, exist_ok=True)

    def shot(name: str, eye, target) -> None:
        env.sim.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))
        for _ in range(2):  # the first frame after a camera jump can be stale
            env.sim.render()
        Image.fromarray(viz.render_rgb_array()).save(os.path.join(out, f"{name}.png"))
        print(f"[view] {out}/{name}.png", flush=True)

    size, corner = zones.size.cpu().numpy(), zones.corner.cpu().numpy()
    extent = size * np.array([zones.num_rows, zones.num_cols])
    center = np.array([*(corner + extent / 2), 0.0])
    shot("overview", center + [-0.75 * extent.max(), 0.0, 0.45 * extent.max()], center)
    base = env.scene["robot"].data.root_link_pos_w.torch.cpu().numpy()
    for cell in meta["cells"]:
        c = np.array([*(corner + (np.array([cell["row"], cell["col"]]) + 0.5) * size), 0.0])
        c[2] = zones.height(torch.tensor(c[None, :2], device=env.device, dtype=torch.float32)).item()
        tag = f"r{cell['row']}_c{cell['col']}_{cell['property']}"
        shot(f"cell_{tag}", c + [-0.6 * size[0], -0.6 * size[1], 0.45 * size.max()], c)
        if cell["row"] == zones.num_rows - 1:  # close-up of a robot in the hardest row
            near = base[np.linalg.norm(base[:, :2] - c[:2], axis=-1).argmin()]
            shot(f"robot_{tag}", near + [-2.5, -2.5, 1.5], near)
    env.close()


def probe_faults(args):
    """T4: faults of every property in a collect.py collection, each with zone labels and >= 2 s of history."""
    import csv

    from isaaclab_tasks.forge.velocity.a2_harsh_world.harsh_world_cfg import HarshWorldCfg

    c = args.collection
    with open(os.path.join(c, "meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(c, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    names = None
    counts, labelled, history = {}, 0, 0
    fields = [k for k in meta["fields"] if k.startswith("obs_zone_")]
    for row in rows:
        e = np.load(os.path.join(c, "events", f"{int(row['event']):06d}.npz"))
        prop = int(e["obs_zone_base_property"][-1]) if "obs_zone_base_property" in e else -1
        counts[prop] = counts.get(prop, 0) + 1
        labelled += all(k in e for k in fields)
        history += len(e["root_pos_w"]) >= round(2.0 / meta["step_dt"]) or int(row["episode_step"]) < 100
    hw = HarshWorldCfg(disabled=[] if "Full" in meta["task"] else HarshWorldCfg().disabled)
    names = list(hw.enabled)
    print(f"[T4] {len(rows)} faults; label fields: {[f[9:] for f in fields]}")
    for i, name in enumerate(names):
        print(f"[T4] {name}: {counts.get(i, 0)} faults")
    check("T4", all(counts.get(i, 0) > 0 for i in range(len(names))), "faults in every property")
    check("T4", labelled == len(rows) and history == len(rows), "every fault has zone labels and >= 2 s of history")


def main():
    args, args_cli = parse()
    if args.probe == "faults":
        probe_faults(args)
    else:
        from isaaclab.app import launch_simulation

        from isaaclab_tasks.utils import resolve_task_config

        env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
        if args.probe == "view":
            from isaaclab_rl.entrypoints.common import pre_launch_video_config

            # a capture-capable visualizer (headless Newton GL unless --viz picks one), as in fault_analysis/render.py
            args_cli.video = args_cli.video or "viz"
            pre_launch_video_config(env_cfg, args_cli)
            args.launch_cfg = env_cfg
        with launch_simulation(env_cfg, args_cli):
            {"ground": probe_ground, "sand": probe_sand, "collapse": probe_collapse, "rocks": probe_rocks,
             "slip": probe_slip, "smoke": probe_smoke, "view": probe_view}[args.probe](args, args_cli)  # fmt: skip
    failed = [tag for tag, ok in RESULTS if not ok]
    print(f"[probe] {args.probe}: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
