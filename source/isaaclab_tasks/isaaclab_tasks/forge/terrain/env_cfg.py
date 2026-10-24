# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain-map environments: the trained A2 robust and energy tasks with their terrain replaced by one terrain map.

The policy's observations and actions are untouched, so the tasks' trained checkpoints play as they are. What
changes (all derived from :class:`~.world_cfg.TerrainWorldCfg`): terrain, robot spread and spawn, commands (robust
only), episode length, and the terrain event (slip, sand, collapse, rocks). Disturbances (pushes, external wrench)
and the curriculum are removed: failures should come from the terrain, and the sand drag owns the feet's external
wrench.
"""

import math

from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, to_dict

from isaaclab_tasks.forge.velocity.a2_velocity.energy_env_cfg import A2EnergyTeacherRoughEnvCfg
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import A2RobustTeacherEnvCfg

from .mdp import events
from .mdp.generator import TerrainGridImporterCfg
from .templates.gravel_field import rock_mesh
from .world_cfg import TerrainWorldCfg

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


ROCK_COLOR = (0.45, 0.43, 0.40)


def rocks_cfg(world: TerrainWorldCfg) -> RigidObjectCollectionCfg:
    """The dynamic rocks: one copy per environment (MuJoCo-Warp simulates each robot in its own world)."""
    rocks = {}
    for i, spec in enumerate(world.dynamic_rock_specs()):
        # plain arrays, not the trimesh object: cfg validation recurses into the trimesh caches
        mesh = rock_mesh(world.seed, spec["template"], spec["size"], spec["axis_ratio"], 0.0)
        rocks[f"rock_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Rock_{i}",
            spawn=sim_utils.MeshFileCfg(
                mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()),
                rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg()],
                mass_props=sim_utils.MassCfg(density=spec["density"]),
                # condim 6: torsional and rolling friction (set by the terrain event) damp a resting rock's rocking
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    MujocoCollisionCfg(condim=6),
                ],
                mesh_collision_props=[sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROCK_COLOR),
            ),
            # parked high above the robot until the terrain event places them at the first reset
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 50.0 + i)),
        )
    return RigidObjectCollectionCfg(rigid_objects=rocks)


def apply_terrain(env_cfg, world: TerrainWorldCfg, robust_commands: bool) -> None:
    """Turn a trained A2 task's env cfg into a terrain map (again, with another ``world``, if called twice)."""
    env_cfg.terrain_world = world
    env_cfg.seed = world.seed
    env_cfg.scene.num_envs = world.num_envs
    env_cfg.episode_length_s = world.episode_length_s
    env_cfg.curriculum = None

    # -- terrain: the tile grid; robots spread evenly over the tiles
    old = env_cfg.scene.terrain
    env_cfg.scene.terrain = TerrainGridImporterCfg(
        prim_path=old.prim_path,
        terrain_type="generator",
        terrain_generator=world.terrain_generator(),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=old.physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.48)),
        settings=to_dict(world),
        meta_path=world.meta_path or None,
    )
    env_cfg.scene.rocks = rocks_cfg(world) if world.has_dynamic_rocks else None

    # -- spawn: on the platform, any heading, at rest
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, world.spawn_height)
    half = world.spawn_radius / math.sqrt(2.0)
    pose_range = {"x": (-half, half), "y": (-half, half), "yaw": (-math.pi, math.pi)}
    env_cfg.events.reset_base.params["pose_range"] = pose_range
    env_cfg.events.reset_base.params["velocity_range"] = {}

    # -- commands
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = world.command_resampling_s
    if robust_commands:
        command.heading_command = False
        command.ranges.lin_vel_x = world.lin_vel_x
        command.ranges.lin_vel_y = world.lin_vel_y
        command.ranges.ang_vel_z = world.ang_vel_z

    # -- events: no disturbances; the terrain event owns the foot friction and the feet's external wrench
    for name in ("foot_friction", "push_robot", "base_external_force_torque"):
        setattr(env_cfg.events, name, None)
    setattr(
        env_cfg.events,
        events.EVENT,
        EventTerm(
            func=events.terrain_step,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={
                "world": world,
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
            },
        ),
    )

    # -- observations: the policy's privileged friction now comes from the terrain event
    policy = env_cfg.observations.policy
    for name in ("foot_friction", "friction"):
        term = getattr(policy, name, None)
        if term is not None:
            term.params["event_name"] = events.EVENT


@configclass
class A2RobustTerrainEnvCfg(A2RobustTeacherEnvCfg):
    """Robust A2 teacher (Miki et al. 2022) on a terrain map."""

    terrain_world: TerrainWorldCfg = TerrainWorldCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_terrain(self, self.terrain_world, robust_commands=True)


@configclass
class A2EnergyTerrainEnvCfg(A2EnergyTeacherRoughEnvCfg):
    """Energy-efficient A2 policy (Fu et al. 2021, rough-terrain variant) on a terrain map.

    The policy has no velocity command input (fixed 0.9 m/s forward), so its commands are kept.
    """

    terrain_world: TerrainWorldCfg = TerrainWorldCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_terrain(self, self.terrain_world, robust_commands=False)


POLICIES = {"A2Robust": A2RobustTerrainEnvCfg, "A2Energy": A2EnergyTerrainEnvCfg}


def make_env_cfg(terrain: str, policy: str):
    """The env cfg of ``IsaacContrib-Terrain-<terrain>-<policy>``."""
    return POLICIES[policy](terrain_world=TerrainWorldCfg(terrain=terrain))
