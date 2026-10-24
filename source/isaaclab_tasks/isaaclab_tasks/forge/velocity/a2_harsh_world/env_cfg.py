# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harsh World environments: the A2 robust and energy tasks with their terrain replaced by the Harsh World.

The policy's observations and actions are untouched, so the tasks' trained checkpoints play as they are. What
changes (all derived from :class:`~.harsh_world_cfg.HarshWorldCfg`): terrain, robot spread and spawn, commands
(robust only), episode length, the zone event (slip, sand, collapse, rocks), and the ``zone`` label group.
Disturbances (pushes, external wrench) and the curriculum are removed: failures should come from the terrain,
and the sand drag owns the feet's external wrench.
"""

import math

from isaaclab_newton.sim.schemas import MujocoCollisionCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, to_dict

from isaaclab_tasks.forge.velocity.a2_velocity.energy_env_cfg import A2EnergyTeacherRoughEnvCfg
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import A2RobustTeacherEnvCfg

from .harsh_world_cfg import HarshWorldCfg
from .mdp import events, observations
from .mdp.world import ROCK_COLOR, HarshWorldImporterCfg, rock_mesh

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class ZoneObsCfg(ObsGroup):
    """The spec's ``zone_logger`` fields, one term each; recorded by ``fault_analysis/collect.py`` as
    ``obs_zone_<field>``."""

    base_cell = ObsTerm(func=observations.base_cell)
    base_property = ObsTerm(func=observations.base_property)
    base_zone = ObsTerm(func=observations.base_zone)
    base_difficulty = ObsTerm(func=observations.base_difficulty)
    boundary = ObsTerm(func=observations.boundary)
    foot_cell = ObsTerm(func=observations.foot_cell)
    foot_property = ObsTerm(func=observations.foot_property)
    foot_zone = ObsTerm(func=observations.foot_zone)
    foot_sand_patch = ObsTerm(func=observations.foot_sand_patch)
    spawn_cell = ObsTerm(func=observations.spawn_cell)
    terrain_height_under_foot = ObsTerm(func=observations.terrain_height_under_foot)
    # condition fields
    foot_in_sand = ObsTerm(func=observations.foot_in_sand)
    foot_mu = ObsTerm(func=observations.foot_mu)
    foot_low_mu = ObsTerm(func=observations.foot_low_mu)
    # event fields
    foot_on_rock = ObsTerm(func=observations.foot_on_rock)
    rock_moving = ObsTerm(func=observations.rock_moving)
    collapse_event = ObsTerm(func=observations.collapse_event)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


def rocks_cfg(hw: HarshWorldCfg) -> RigidObjectCollectionCfg:
    """The dynamic rocks: one copy per environment (MuJoCo-Warp simulates each robot in its own world)."""
    rocks = {}
    for i, spec in enumerate(hw.dynamic_rock_specs()):
        # plain arrays, not the trimesh object: cfg validation recurses into the trimesh caches
        mesh = rock_mesh(hw.seed, spec["template"], spec["size"], spec["axis_ratio"], 0.0)
        rocks[f"rock_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Rock_{i}",
            spawn=sim_utils.MeshFileCfg(
                mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()),
                rigid_props=[sim_utils.UsdPhysicsRigidBodyCfg()],
                mass_props=sim_utils.MassCfg(density=spec["density"]),
                # condim 6: torsional and rolling friction (set by the zone event) damp a resting rock's rocking
                collision_props=[
                    sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                    MujocoCollisionCfg(condim=6),
                ],
                mesh_collision_props=[sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=ROCK_COLOR),
            ),
            # parked high above the robot until the zone event places them at the first reset
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 50.0 + i)),
        )
    return RigidObjectCollectionCfg(rigid_objects=rocks)


def apply_harsh_world(env_cfg, hw: HarshWorldCfg, robust_commands: bool) -> None:
    """Turn a trained A2 task's env cfg into the Harsh World (again, with another ``hw``, if called twice)."""
    env_cfg.harsh_world = hw
    env_cfg.seed = hw.seed
    env_cfg.scene.num_envs = hw.num_envs
    env_cfg.episode_length_s = hw.episode_length_s
    env_cfg.curriculum = None

    # -- terrain: the world grid; robots spread evenly over the cells and stay in theirs
    old = env_cfg.scene.terrain
    env_cfg.scene.terrain = HarshWorldImporterCfg(
        prim_path=old.prim_path,
        terrain_type="generator",
        terrain_generator=hw.terrain_generator(),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=old.physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.48)),
        settings=to_dict(hw),
        meta_path=hw.meta_path or None,
    )
    env_cfg.scene.rocks = rocks_cfg(hw) if "loose_rocks" in hw.enabled and hw.rocks.dynamic_rocks > 0 else None

    # -- spawn: on the platform, any heading, at rest
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, hw.spawn_height)
    half = hw.spawn_radius / math.sqrt(2.0)
    pose_range = {"x": (-half, half), "y": (-half, half), "yaw": (-math.pi, math.pi)}
    env_cfg.events.reset_base.params["pose_range"] = pose_range
    env_cfg.events.reset_base.params["velocity_range"] = {}

    # -- commands
    command = env_cfg.commands.base_velocity
    command.resampling_time_range = hw.command_resampling_s
    if robust_commands:
        command.heading_command = False
        command.ranges.lin_vel_x = hw.lin_vel_x
        command.ranges.lin_vel_y = hw.lin_vel_y
        command.ranges.ang_vel_z = hw.ang_vel_z

    # -- events: no disturbances; the zone event owns the foot friction and the feet's external wrench
    for name in ("foot_friction", "push_robot", "base_external_force_torque"):
        setattr(env_cfg.events, name, None)
    env_cfg.events.harsh_world = EventTerm(
        func=events.harsh_world_step,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "world": hw,
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True),
        },
    )

    # -- observations: the policy's privileged friction now comes from the zone event; labels in group "zone"
    policy = env_cfg.observations.policy
    for name in ("foot_friction", "friction"):
        term = getattr(policy, name, None)
        if term is not None:
            term.params["event_name"] = observations.EVENT
    env_cfg.observations.zone = ZoneObsCfg()


@configclass
class A2HarshWorldRobustEnvCfg(A2RobustTeacherEnvCfg):
    """Robust A2 teacher (Miki et al. 2022) in the Harsh World, 7 properties."""

    harsh_world: HarshWorldCfg = HarshWorldCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_harsh_world(self, self.harsh_world, robust_commands=True)


@configclass
class A2HarshWorldEnergyEnvCfg(A2EnergyTeacherRoughEnvCfg):
    """Energy-efficient A2 policy (Fu et al. 2021, rough-terrain variant) in the Harsh World, 7 properties.

    The policy has no velocity command input (fixed 0.9 m/s forward), so its commands are kept.
    """

    harsh_world: HarshWorldCfg = HarshWorldCfg()

    def __post_init__(self):
        super().__post_init__()
        apply_harsh_world(self, self.harsh_world, robust_commands=False)


@configclass
class A2HarshWorldRobustFullEnvCfg(A2RobustTeacherEnvCfg):
    """Robust A2 teacher in the Harsh World with all 10 properties (drops, loose rocks, collapsing crust)."""

    harsh_world: HarshWorldCfg = HarshWorldCfg(disabled=[])

    def __post_init__(self):
        super().__post_init__()
        apply_harsh_world(self, self.harsh_world, robust_commands=True)


@configclass
class A2HarshWorldEnergyFullEnvCfg(A2EnergyTeacherRoughEnvCfg):
    """Energy-efficient A2 policy in the Harsh World with all 10 properties."""

    harsh_world: HarshWorldCfg = HarshWorldCfg(disabled=[])

    def __post_init__(self):
        super().__post_init__()
        apply_harsh_world(self, self.harsh_world, robust_commands=False)
