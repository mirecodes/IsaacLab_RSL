# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment of the trained A2 policies on a terrain map: difficulty rows 1-3, 20 robots on random tiles, steering
(on the inclines: straight up or down the slope instead), one extra terrain type (``slip_field``).

Built on the terrain-map play tasks (``forge/terrain``), whose map, physics and policy wiring stay as they are; see
``REPORT.md``.
"""

import math
from collections.abc import Callable

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from isaaclab_tasks.forge.terrain.env_cfg import A2EnergyTerrainEnvCfg, A2RobustTerrainEnvCfg
from isaaclab_tasks.forge.terrain.templates.rough_field import RoughFieldCfg
from isaaclab_tasks.forge.terrain.templates.sand import SandFieldCfg
from isaaclab_tasks.forge.terrain.world_cfg import A2_NOMINAL_FOOT_LOAD, TerrainWorldCfg

from . import mdp

NUM_ROBOTS = 20
LEVELS = 3
"""The first difficulty rows of the full map that are kept."""
FULL_LEVELS = 5
"""Rows of the full terrain map (``TerrainWorldCfg.num_levels``), whose difficulties rows 1-3 keep."""
INCLINES = ("rough_slope", "rough_stairs", "slip_slope")
"""One-direction inclines rising along +x: robots face up or down them and are not turned."""
INCLINE_YAW = math.radians(15.0)
"""Largest spawn yaw [rad] off the incline's +x / -x direction (the user's request: within about 15 deg)."""
TRIGGERS = {
    "crust_collapse": mdp.crust_collapse,
    "sand_field": mdp.sand_sink,
    "gravel_field": mdp.stone_step,
    "slip_slope": mdp.foot_slip,
    "slip_field": mdp.foot_slip,
    "mixed_field": mdp.any_event,
}
"""Event trigger of each terrain type (``mdp/triggers.py``); the others sample normal walking (``random_walk``)."""


def extra_templates() -> dict:
    """Terrain types added to ``forge/terrain``'s catalog; keep the keys equal to ``EXTRA_TERRAINS`` in
    ``__init__.py``."""
    return {
        # random foot slips on generic uneven ground: rough_field's template on the slip zone
        "slip_field": RoughFieldCfg(zone="slip"),
        # several events on one map (sand patches, collapsing crusts, ice patches between them; see MIXED), so that
        # which event acted cannot be read off the terrain type, as it can on the one-event maps
        "mixed_field": SandFieldCfg(crust_prob=0.3),
    }


MIXED_ICE = 0.15
"""Share of the ice cells of ``mixed_field`` (``SlipCfg.ice_frac``); it has no random single-foot slips."""


def deploy_world(terrain: str, hard: bool = False) -> TerrainWorldCfg:
    """The terrain map of ``terrain`` (including :func:`extra_templates`) cut to its first ``LEVELS`` rows, at the
    full map's difficulties; ``hard``: its last ``LEVELS`` rows and the stronger events of :func:`apply_hard`."""
    world = TerrainWorldCfg(num_levels=LEVELS, meta_path="")
    world.terrains.update(extra_templates())
    world.terrain = terrain
    template = world.terrains[terrain]
    lo, hi = template.difficulty_range or world.difficulty_range
    # row i of the full map: d_i = lo + (hi - lo) i / (FULL_LEVELS - 1); keep i < LEVELS (hard: i >= FULL - LEVELS)
    first = FULL_LEVELS - LEVELS if hard else 0
    template.difficulty_range = (lo + (hi - lo) * first / (FULL_LEVELS - 1),
                                 lo + (hi - lo) * (first + LEVELS - 1) / (FULL_LEVELS - 1))  # fmt: skip
    if terrain == "mixed_field":
        world.slip.ice_frac, world.slip.slip_prob = MIXED_ICE, 0.0
    if hard:
        apply_hard(world)
    return world


def apply_hard(world: TerrainWorldCfg) -> None:
    """Events strong enough to fail the policies (``REPORT.md`` §3.5; v2 had 25 falls in 66.7 robot-h)."""
    # icy patches on the slip terrains: every foot stepping on one slips (v2: one random foot per 6 s, no fall)
    if world.terrains[world.terrain].zone == "slip":
        world.slip.ice_frac = 0.15
    # sand: soft enough to sink in one stance, deeper patches, and a lift-off suction that holds a foot (v2: <= 60 N
    # against legs that pull a few hundred N). Sinkage within a stance grows ~ 1 / sqrt(stiffness): 3.1 cm median at
    # 800-3000 N/m (v2), 6.0 cm at 200-800 N/m (smoke_hard), so 60-240 N/m for ~11 cm
    for name in ("sand_field", "crust_collapse", "mixed_field"):
        world.terrains[name].stiffness = (60.0, 240.0)
        world.terrains[name].sand_depth = (0.12, 0.25)
    world.sand.c_suction = (150.0, 400.0)
    world.sand.force_limit = 120.0
    # crust: gives way under one stance foot and drops it to the hardpan (v2 "soften" = ke 3000 N/m, the stiffest
    # sand, 2.3 cm median sinkage after a collapse)
    world.collapse.collapse_mode = "disable"
    world.collapse.collapse_force = (0.8 * A2_NOMINAL_FOOT_LOAD, 1.5 * A2_NOMINAL_FOOT_LOAD)
    # stubs and roots up to a swing height (v2: <= 6 cm, never a fall); still narrower than the 0.1 m scanner grid
    world.terrains["stub_field"].stub_height = (0.04, 0.12)
    world.terrains["stub_field"].root_height = (0.03, 0.08)


@configclass
class DeployCurriculumCfg:
    """Each episode on a uniformly drawn tile."""

    spawn_tile = CurrTerm(func=mdp.random_tile)


def apply_deploy(env_cfg) -> None:
    """20 robots, each episode on a random tile of the map; on an incline facing up or down it; the terrain's event
    trigger."""
    env_cfg.scene.num_envs = NUM_ROBOTS
    env_cfg.curriculum = DeployCurriculumCfg()
    env_cfg.event_trigger = TRIGGERS.get(env_cfg.terrain_world.terrain, mdp.random_walk)
    if env_cfg.terrain_world.terrain in INCLINES:
        env_cfg.events.reset_base.func = mdp.reset_root_up_or_down
        env_cfg.events.reset_base.params["pose_range"]["yaw"] = (-INCLINE_YAW, INCLINE_YAW)


@configclass
class A2RobustDeployEnvCfg(A2RobustTerrainEnvCfg):
    """Robust A2 teacher with forward-only steering commands (no turns on the inclines)."""

    event_trigger: Callable | None = None
    """``env -> bool [num_envs]``: the terrain event acts on the robot (``mdp/triggers.py``); set by the terrain."""

    def __post_init__(self):
        super().__post_init__()
        apply_deploy(self)
        base = self.commands.base_velocity
        self.commands.base_velocity = mdp.SteeringCommandCfg(
            asset_name=base.asset_name,
            resampling_time_range=self.terrain_world.command_resampling_s,
            rel_standing_envs=0.0,
            heading_command=False,
            debug_vis=base.debug_vis,
            ranges=mdp.SteeringCommandCfg.Ranges(
                # chosen here: forward only, inside the trained range (-0.5, 1.5) m/s
                lin_vel_x=(0.5, 1.0),
                lin_vel_y=(0.0, 0.0),
                # chosen here: at most 0.6 rad/s, so a 45 deg turn takes about 2 s
                ang_vel_z=(-0.6, 0.6),
            ),
            # the user's request: turns within +-45 deg; none on the inclines, which the robot walks up or down
            max_turn=0.0 if self.terrain_world.terrain in INCLINES else math.pi / 4,
        )


@configclass
class A2EnergyDeployEnvCfg(A2EnergyTerrainEnvCfg):
    """Energy A2 policy; it has no command input (fixed 0.9 m/s forward), so it cannot be steered."""

    event_trigger: Callable | None = None
    """As :attr:`A2RobustDeployEnvCfg.event_trigger`."""

    def __post_init__(self):
        super().__post_init__()
        apply_deploy(self)


POLICIES = {"Robust": A2RobustDeployEnvCfg, "Energy": A2EnergyDeployEnvCfg}


def make_env_cfg(terrain: str, policy: str, hard: bool = False):
    """The env cfg of ``IsaacContrib-Surveillance-<terrain>-<policy>`` (``hard``: ``...-Hard``)."""
    return POLICIES[policy](terrain_world=deploy_world(terrain, hard))
