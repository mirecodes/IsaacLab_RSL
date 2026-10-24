# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulator probes of the terrain maps. Each check prints ``[T<n>] ... PASS/FAIL`` (the spec's test ids).

    python scripts/reinforcement_learning/terrain/probe.py ground [--mesh]   # terrain collision
    python scripts/reinforcement_learning/terrain/probe.py sand              # T2: sinkage, stiffness, scanner, drag
    python scripts/reinforcement_learning/terrain/probe.py collapse [--collapse_mode disable]   # T6: crust
    python scripts/reinforcement_learning/terrain/probe.py rocks             # T6: settling, penetration, rock return
    python scripts/reinforcement_learning/terrain/probe.py slip --checkpoint <robust model.pt>  # T3
    python scripts/reinforcement_learning/terrain/probe.py smoke --terrain <type> --checkpoint <model.pt>  # T4, T6
    python scripts/reinforcement_learning/terrain/probe.py view [<out dir>] --terrain <type> [--checkpoint <model.pt>]

Each probe builds the map of one terrain type (``--terrain``, a default per probe) on the matching
``IsaacContrib-Terrain-<Type>-A2Robust`` task (``--task`` picks another). ``sand``, ``collapse``, and ``rocks`` need
no policy and use small maps; ``slip`` and ``smoke`` play a trained checkpoint. ``--device cpu`` runs without a GPU,
but Newton's CPU contacts are unreliable, so that only checks that the code runs.
"""

# set before importing anything that defines Warp kernels (see play.py)
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import ast  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RESULTS: list[tuple[str, bool]] = []
DEFAULT_TERRAIN = {"ground": "rough_slope", "sand": "sand_field", "collapse": "crust_collapse", "rocks": "loose_rocks",
                   "slip": "slip_slope", "smoke": "gravel", "view": "gravel"}  # fmt: skip
EVENT = "terrain"


def check(tag: str, ok: bool, message: str) -> None:
    RESULTS.append((tag, bool(ok)))
    print(f"[{tag}] {message}  {'PASS' if ok else 'FAIL'}", flush=True)


def parse():
    from isaaclab_tasks.forge.terrain import TERRAINS, task_id

    parser = argparse.ArgumentParser(description="Terrain-map simulator probes.")
    parser.add_argument("probe", choices=list(DEFAULT_TERRAIN))
    parser.add_argument("out", nargs="?", default=None, help="view: output directory.")
    parser.add_argument("--terrain", choices=TERRAINS, default=None, help="Terrain type (default: one per probe).")
    parser.add_argument("--collapse_mode", choices=["soften", "disable"], default="soften")
    parser.add_argument("--mesh", action="store_true", help="Collide against the triangle mesh, not heightfields.")
    parser.add_argument("--set", nargs="*", default=[], help="TerrainWorldCfg overrides, e.g. rocks.dynamic_rocks=0.")
    parser.add_argument("--seconds", type=float, default=None, help="Simulated seconds (smoke 300, slip 60).")
    parser.add_argument("--robots_per_tile", type=int, default=None)
    parser.add_argument("--num_levels", type=int, default=None)
    parser.add_argument("--tiles_per_level", type=int, default=None)
    parser.add_argument("--tile_size", type=float, default=None)
    parser.add_argument("--horizontal_scale", type=float, default=None)
    args, argv = parser.parse_known_args()
    args.terrain = args.terrain or DEFAULT_TERRAIN[args.probe]
    from isaaclab_rl.entrypoints.backends.play_rsl_rl import _parse_args

    if "--task" not in argv:
        argv += ["--task", task_id(args.terrain, "A2Robust")]
    return args, _parse_args(argv)


def world(args, **overrides):
    """The probe's TerrainWorldCfg: its terrain, ``overrides``, then the CLI options."""
    from isaaclab_tasks.forge.terrain.world_cfg import TerrainWorldCfg

    cfg = TerrainWorldCfg(terrain=args.terrain)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    for key in ("robots_per_tile", "num_levels", "tiles_per_level", "tile_size", "horizontal_scale"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    for item in args.set:
        path, value = item.split("=", 1)
        *parents, leaf = path.split(".")
        target = cfg
        for name in parents:  # e.g. terrains.mesa_drop.scree_prob
            target = target[name] if isinstance(target, dict) else getattr(target, name)
        setattr(target, leaf, ast.literal_eval(value))
    if args.mesh:
        cfg.terrains[cfg.terrain].convert_to_heightfield = False
    return cfg


def make_env(args_cli, cfg, configure=None):
    """Create the task's env with ``cfg`` applied; ``configure(env_cfg)`` may add scene entities."""
    from isaaclab_rl.entrypoints.common import apply_env_overrides, create_isaaclab_env

    from isaaclab_tasks.forge.terrain.env_cfg import apply_terrain
    from isaaclab_tasks.utils import resolve_task_config

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=False)
    apply_terrain(env_cfg, cfg, robust_commands="Robust" in args_cli.task)
    apply_env_overrides(args_cli, env_cfg)
    if args_cli.device is not None:
        # launch_simulation applies --device to the cfg it was given; this is a fresh one
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.scene.terrain.terrain_generator.seed = cfg.seed
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
    """Move the robots onto the flat border, out of the tiles' way."""
    tmap = env.scene.terrain.terrain_map
    pos = torch.zeros(env.num_envs, 3, device=env.device)
    pos[:, 0] = tmap.corner[0] - 3.0
    pos[:, 1] = tmap.corner[1] + 2.0 * torch.arange(env.num_envs, device=env.device)
    pos[:, 2] = 0.45
    place(env.scene["robot"], pos)


def run(env, seconds: float, actions=None) -> None:
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(round(seconds / env.step_dt)):
        env.step(zero if actions is None else actions())


def term_of(env):
    return env.event_manager.get_term_cfg(EVENT).func


##
# Probes.
##


def probe_ground(args, args_cli):
    """Terrain collision: spheres dropped all over the map, robots standing (``--mesh``: the triangle mesh instead
    of Newton's heightfields)."""
    cfg = world(args, num_levels=1, tiles_per_level=2, robots_per_tile=8)
    radius = cfg.sand.foot_radius
    env, _ = make_env(args_cli, cfg, add_probe_spheres(radius, 10.0))
    env = env.unwrapped
    env.reset()
    tmap = env.scene.terrain.terrain_map
    extent = tmap.size * torch.tensor([tmap.num_rows, tmap.num_cols], device=env.device)
    xy = tmap.corner + torch.rand(env.num_envs, 2, device=env.device) * extent
    place(env.scene["probe"], torch.cat([xy, (tmap.height(xy) + 0.1 + radius)[:, None]], -1))
    start = time.time()
    run(env, 2.0)
    wall = time.time() - start
    # spheres on the slopes roll and bounce away: check that none is below the ground where it ended up
    pos = env.scene["probe"].data.root_link_pos_w.torch
    depth = tmap.height(pos[:, :2]) - (pos[:, 2] - radius)
    base = env.scene["robot"].data.root_link_pos_w.torch[:, 2] - env.scene.terrain.env_origins[:, 2]
    kind = f"{args.terrain}, {'mesh' if args.mesh else 'heightfield'}"
    check("T0", (depth < 0.02).all().item(), f"{kind}: no sphere below the terrain (max depth {depth.max() * 100:.1f} "
          f"cm, {int((depth.abs() < 0.02).sum())}/{env.num_envs} resting on it), 2 s in {wall:.1f} s wall")  # fmt: skip
    check("T0", (base > 0.15).all().item(), f"{kind}: robots stay above the ground (min base {base.min():.2f} m)")
    env.close()


def probe_sand(args, args_cli):
    """T2: slab sinkage (and its stiffness dependence), rigid sinkage, scanner on the sand surface, drag."""
    cfg = world(args, num_levels=1, tiles_per_level=1, robots_per_tile=6, tile_size=args.tile_size or 16.0)
    radius = cfg.sand.foot_radius
    env, _ = make_env(args_cli, cfg, add_probe_spheres(radius, 10.0))
    env = env.unwrapped
    env.reset()
    tmap, term = env.scene.terrain.terrain_map, term_of(env)
    patches = sorted(tmap.patches, key=lambda p: -min(p["size"]))[:2]
    # patch A soft (800 N/m), patch B stiff (3000 N/m)
    for p, k in zip(patches, (800.0, 3000.0)):
        shape = term.slab_shape[p["id"]].item()
        term._ke[shape], term._kd[shape] = k, cfg.sand.damping_ratio * k
    term._newton.add_model_change(term._shape_change)
    # envs 0-1: patch A, 2-3: patch B, 4-5: rigid ground (the flat platform; the hardpan beside it is rough)
    xy = torch.tensor([[p["x"], p["y"]] for p in (patches[0], patches[0], patches[1], patches[1])], device=env.device)
    xy[1::2] += 0.1
    origin = env.scene.terrain.terrain_origins[0, 0, :2]
    xy = torch.cat([xy, (origin + torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=env.device))])
    surface = torch.tensor([patches[0]["surface"]] * 2 + [patches[1]["surface"]] * 2, device=env.device)
    surface = torch.cat([surface, tmap.height(xy[4:])])
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
    _, difficulty = tmap.lookup(pos[..., :2])
    patch, surf = tmap.patch(pos[..., :2])
    term._sand(difficulty, patch, surf, pos, vel)
    expected = (cfg.sand.c_drag[0] + difficulty[0, 0].item() * (cfg.sand.c_drag[1] - cfg.sand.c_drag[0])) * 0.5
    got = term._forces[0, 0].norm().item()
    check("T2", abs(got - expected) <= 0.05 * expected and term._forces[:, 2:].abs().max().item() == 0.0,
          f"drag {got:.2f} N for c_drag * 0.5 m/s = {expected:.2f} N; feet outside the sand exactly 0 N")  # fmt: skip
    env.close()


def probe_collapse(args, args_cli):
    """T6: a crust holds a load below the threshold, collapses within one step above it, recovers later."""
    radius, mode = 0.032, args.collapse_mode
    cfg = world(args, num_levels=1, tiles_per_level=1, robots_per_tile=4, tile_size=args.tile_size or 16.0)
    cfg.terrains[cfg.terrain].crust_prob = 1.0
    cfg.collapse.recover_time = 3.0
    cfg.collapse.collapse_mode = mode
    env, _ = make_env(args_cli, cfg, add_probe_spheres(radius, 10.0))
    env = env.unwrapped
    env.reset()
    tmap, term = env.scene.terrain.terrain_map, term_of(env)
    patch = sorted(tmap.patches, key=lambda p: -min(p["size"]))[0]
    park_robots(env)
    xy = torch.tensor([[patch["x"], patch["y"]]], device=env.device).expand(env.num_envs, 2)
    place(env.scene["probe"], torch.cat([xy, torch.full((env.num_envs, 1), patch["surface"] + radius + 0.01,
                                                        device=env.device)], -1))  # fmt: skip
    sphere = env.scene["probe"]
    d = tmap.difficulty[0, 0].item()
    limit = cfg.collapse.collapse_force[0] + d * (cfg.collapse.collapse_force[1] - cfg.collapse.collapse_force[0])

    def step_with_load(scale: float):
        """One step; the sphere's weight x ``scale`` stands in for a robot's feet."""
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
        # the sphere as one "foot" (the other three columns carry no load); its weight, not the contact
        # sensor, so that the trigger is checked against a known load
        force = torch.zeros(env.num_envs, 4, device=env.device)
        force[:, 0] = 98.1 * scale
        pid, _ = tmap.patch(sphere.data.root_link_pos_w.torch[:, None, :2].expand(-1, 4, 2))
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
    hardpan = patch["surface"] - tmap.height(sphere.data.root_link_pos_w.torch[:1, :2]).item()
    if mode == "disable":
        check("T6", final >= hardpan - 0.01, f"disable: sphere down to the hardpan ({final * 1000:.0f} mm, "
              f"hardpan {hardpan * 1000:.0f} mm below the surface)")  # fmt: skip
    for _ in range(round((cfg.collapse.recover_time - 1.4) / env.step_dt)):
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
    cfg = world(args, num_levels=1, tiles_per_level=2, robots_per_tile=args.robots_per_tile or 4)
    env, _ = make_env(args_cli, cfg)
    env = env.unwrapped
    env.reset()
    tmap, rocks, term = env.scene.terrain.terrain_map, env.scene["rocks"], term_of(env)
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
    rest = (speed[round(5.0 / env.step_dt) :] < 0.01).float().mean().item()
    pos = data.body_link_pos_w.torch
    sunk = (pos[..., 2] < tmap.height(pos[..., :2]) - 0.02).sum().item()
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
    pose[0, : k // 2, 0] += 2.0
    pose[0, k // 2 :, 0] += 0.3
    first = torch.tensor([0], device=env.device)
    rocks.write_body_link_pose_to_sim_index(body_poses=pose[:1], body_ids=list(range(k)), env_ids=first)
    moved = term.return_rocks(first, only_drifted=True)
    after = data.body_link_pose_w.torch
    terrain = env.scene.terrain
    slots = tmap.rock_slots[terrain.terrain_levels[0], terrain.terrain_types[0]]
    back = (after[0, : k // 2, :2] - slots[: k // 2, :2]).norm(dim=-1)
    kept = (after[0, k // 2 :, :3] - pose[0, k // 2 :, :3]).norm(dim=-1)
    check("T6", moved == k // 2 and back.max() < 1e-3 and kept.max() < 1e-4,
          f"rock return moved {moved}/{k} rocks (the {k // 2} drifted ones)")  # fmt: skip
    env.close()


def probe_slip(args, args_cli):
    """T3: friction written on entry and restored on exit, sample distribution, slide vs. a rigid twin.

    The twin is every second robot on the same map with slip switched off, so both walk the same tiles.
    """
    from scipy import stats

    cfg = world(args, num_levels=1, tiles_per_level=2, robots_per_tile=args.robots_per_tile or 16)
    env, agent_cfg = make_env(args_cli, cfg)
    wrapped, policy = load_policy(env, agent_cfg, args_cli)
    env = env.unwrapped
    term, tmap = term_of(env), env.scene.terrain.terrain_map
    twin = (torch.arange(env.num_envs, device=env.device) % 2 == 1)[:, None]
    slip = term._slip
    term._slip = lambda env_, in_slip, difficulty: slip(env_, in_slip & ~twin, difficulty)
    robot, contact = env.scene["robot"], env.scene["contact_forces"]
    obs = wrapped.get_observations()
    slide = torch.zeros(env.num_envs, device=env.device)
    worst, changes = 0.0, 0
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
    term._slip = slip
    check("T3", changes > 0 and worst < 1e-6, f"model friction equals the sampled friction after {changes} changes")
    slipping, rigid = slide[~twin[:, 0]].mean().item(), slide[twin[:, 0]].mean().item()
    ratio = slipping / max(rigid, 1e-6)
    check("T3", slipping > 0 and ratio >= 3,
          f"foot slide per robot: slip {slipping:.2f} m, rigid twin {rigid:.2f} m (x{ratio:.1f})")  # fmt: skip
    # entry samples: 1000 synthetic entries at difficulty d
    d = tmap.difficulty[0, 0].item()
    n = term.foot_mu.shape[0]
    samples, lows = [], []
    for i in range(math.ceil(1000 / (4 * n)) * 2):
        # even i: every foot enters the slip terrain; odd i: every foot leaves it
        inside = torch.full((n, 4), i % 2 == 0, device=env.device)
        term._slip(env, inside, torch.full((n, 4), d, device=env.device))
        if i % 2 == 0:
            samples.append(term.foot_mu.flatten().cpu())
            lows.append(term.low_mu.flatten().cpu())
    mu, low = torch.cat(samples)[:1000].numpy(), torch.cat(lows)[:1000].numpy()
    s = cfg.slip
    lo = s.mu_slip_easy[0] + d * (s.mu_slip_hard[0] - s.mu_slip_easy[0])
    hi = s.mu_slip_easy[1] + d * (s.mu_slip_hard[1] - s.mu_slip_easy[1])
    p = stats.kstest(mu[low], stats.uniform(loc=lo, scale=hi - lo).cdf).pvalue
    ok = p > 0.05 and abs(low.mean() - s.slip_prob) <= 0.05 and (mu[~low] == s.mu_nominal).all()
    check("T3", ok, f"1000 entries: low share {low.mean():.3f} (p={s.slip_prob}), KS p={p:.3f} vs U({lo:.3f},{hi:.3f})")
    env.close()


def probe_smoke(args, args_cli):
    """T4/T6: spawn, early falls, stability, flights, moving-rock contacts, collapses, and speed under the policy."""
    cfg = world(args)
    env, agent_cfg = make_env(args_cli, cfg)
    wrapped, policy = load_policy(env, agent_cfg, args_cli)
    env = env.unwrapped
    terrain, term = env.scene.terrain, term_of(env)
    robot, contact = env.scene["robot"], env.scene["contact_forces"]
    rocks = env.scene.rigid_object_collections.get("rocks")
    rock_sizes = torch.tensor([s["size"] for s in cfg.dynamic_rock_specs()], device=env.device)
    obs = wrapped.get_observations()
    tag = f"{args.terrain}, {env.num_envs} robots"
    # spawn: base inside its own tile's platform
    base = robot.data.root_link_pos_w.torch
    offset = (base[:, :2] - terrain.env_origins[:, :2]).abs().max(dim=-1).values
    check("T4", (offset <= cfg.platform_width / 2).all().item(), f"{tag}: all robots spawn on their platform "
          f"(max offset {offset.max():.2f} m)")  # fmt: skip
    seconds = args.seconds or 300.0  # the spec's 5 min smoke run
    early_falls = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    left_falls, stumbles, left = (torch.zeros_like(early_falls) for _ in range(3))
    flight = torch.zeros(env.num_envs, device=env.device)
    flights, rock_contacts, nan_count, fast_count, falls = 0, 0, 0, 0, 0
    trigger = [i for i, n in enumerate(contact.body_names) if not n.endswith("_foot")]
    with torch.inference_mode():
        for step in range(round(seconds / env.step_dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
            pos = robot.data.root_link_pos_w.torch
            fell = env.termination_manager.terminated
            falls += int(fell.sum())
            # first second: falls = the trained task's terminations (base contact, tilt > 1 rad), split by whether
            # the robot had left its platform (robots start at up to 1.5 m/s); stumbles = any other body touching
            if step * env.step_dt < 1.0:
                early_falls |= fell & ~left
                left_falls |= fell & left
                stumbles |= (contact.data.net_forces_w.torch[:, trigger].norm(dim=-1) > 1.0).any(-1) & ~fell
                off = (pos[:, :2] - terrain.env_origins[:, :2]).abs().max(-1).values > cfg.platform_width / 2
                left |= off & ~fell  # a fallen robot's position is already its reset position
            # flights: all feet off the ground > 0.3 s, then a touchdown
            foot_force = contact.data.net_forces_w.torch[:, term.contact_foot_ids].norm(dim=-1)
            air = (foot_force < 1.0).all(-1)
            flights += int((~air & (flight > 0.3)).sum())
            flight = torch.where(air, flight + env.step_dt, 0.0)
            if rocks is not None:
                # a foot in contact within a moving dynamic rock's bounding radius (origin at the rock's bottom)
                feet = robot.data.body_link_pos_w.torch[:, term.foot_ids]
                center = rocks.data.body_link_pos_w.torch.clone()
                center[..., 2] += rock_sizes / 4
                touch = (feet[:, :, None] - center[:, None]).norm(dim=-1) < rock_sizes / 2 + 0.032 + 0.02
                moving = rocks.data.body_com_lin_vel_w.torch.norm(dim=-1)[:, None] > 0.05
                rock_contacts += int(((touch & moving).any(-1) & (foot_force > 1.0)).any(-1).sum())
            vel = robot.data.body_com_lin_vel_w.torch
            nan_count += int(torch.isnan(vel).any(-1).any(-1).sum())
            fast_count += int((vel.norm(dim=-1) > 20.0).any(-1).sum())
        # speed without this probe's bookkeeping (its per-step host syncs cost more than the terrain itself)
        torch.cuda.synchronize()
        bench = time.time()
        for _ in range(round(10.0 / env.step_dt)):
            obs, _, dones, _ = wrapped.step(policy(obs))
            policy.reset(dones)
        torch.cuda.synchronize()
        bench = time.time() - bench
    per_minute = 60.0 / seconds
    check("T4", not early_falls.any().item(), f"{tag}: {int(early_falls.sum())} robots fell on their platform within "
          f"1 s of spawning (levels {terrain.terrain_levels[early_falls].tolist()})")  # fmt: skip
    print(f"[T4] {tag}: {int(left_falls.sum())} more fell within 1 s after leaving the platform; "
          f"{int(stumbles.sum())} touched the ground with a leg without falling; "
          f"{falls * per_minute:.1f} terminations per minute")  # fmt: skip
    print(f"[T4] {tag}: real-time factor {10.0 / bench:.2f} (10 s sim in {bench:.1f} s wall)")
    # the spec's stability condition is for crust_collapse; tumbles down the 10 m slopes can exceed 20 m/s elsewhere
    stable = nan_count == 0 and (fast_count == 0 or args.terrain != "crust_collapse")
    check("T6", stable, f"{tag}: {nan_count} NaN states, {fast_count} robot-steps with a body faster than 20 m/s, "
          f"{term.num_collapses} collapses")  # fmt: skip
    if args.terrain == "mesa_drop":
        check("T6", flights * per_minute >= 1, f"{tag}: {flights * per_minute:.1f} flights and landings per minute")
    if rocks is not None:
        check("T6", rock_contacts * per_minute >= 1, f"{tag}: {rock_contacts * per_minute:.1f} moving-rock foot "
              "contacts per minute")  # fmt: skip
    env.close()


def probe_view(args, args_cli):
    """Render the map to PNGs: two overviews, one oblique view per difficulty row, one robot close-up per row."""
    from PIL import Image

    from isaaclab.envs.utils.video_recorder_cfg import CAPTURE_VISUALIZER_TYPES

    cfg = world(args)

    def configure(env_cfg):
        # the visualizer launch_simulation set up on its own cfg (this one is fresh)
        env_cfg.sim.visualizer_cfgs = args.launch_cfg.sim.visualizer_cfgs
        for command in vars(env_cfg.commands).values():
            if hasattr(command, "debug_vis"):
                command.debug_vis = False

    env, agent_cfg = make_env(args_cli, cfg, configure)
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
    tmap = env.scene.terrain.terrain_map
    out = os.path.abspath(args.out or f"logs/terrain/view/{args.terrain}")
    os.makedirs(out, exist_ok=True)

    def shot(name: str, eye, target) -> None:
        env.sim.set_camera_view(tuple(float(v) for v in eye), tuple(float(v) for v in target))
        for _ in range(2):  # the first frame after a camera jump can be stale
            env.sim.render()
        Image.fromarray(viz.render_rgb_array()).save(os.path.join(out, f"{name}.png"))
        print(f"[view] {out}/{name}.png", flush=True)

    size, corner = tmap.size.cpu().numpy(), tmap.corner.cpu().numpy()
    extent = size * np.array([tmap.num_rows, tmap.num_cols])
    # the map spans 120 x 120 m by default: from far away it is a flat sheet, so look across it from two corners
    near, far = np.array([*corner, 0.0]), np.array([*(corner + extent), 0.0])
    for name, eye, target in (("overview_a", near, far), ("overview_b", far, near)):
        direction = np.sign(target - eye)
        shot(name, eye - 20.0 * direction + [0.0, 0.0, 30.0], eye + direction * [extent[0] / 2, extent[1] / 3, 0.0])
    base = env.scene["robot"].data.root_link_pos_w.torch.cpu().numpy()
    col = tmap.num_cols // 2
    for row in range(tmap.num_rows):
        c = np.array([*(corner + (np.array([row, col]) + 0.5) * size), 0.0])
        c[2] = tmap.height(torch.tensor(c[None, :2], device=env.device, dtype=torch.float32)).item()
        tag = f"level{row}_d{tmap.difficulty[row, col].item():.2f}"
        shot(f"tile_{tag}", c + [-0.6 * size[0], -0.6 * size[1], 0.45 * size.max()], c)
        robot = base[np.linalg.norm(base[:, :2] - c[:2], axis=-1).argmin()]
        shot(f"robot_{tag}", robot + [-2.5, -2.5, 1.5], robot)
    env.close()


def main():
    args, args_cli = parse()
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
        globals()[f"probe_{args.probe}"](args, args_cli)
    failed = [tag for tag, ok in RESULTS if not ok]
    print(f"[probe] {args.probe} ({args.terrain}): {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
