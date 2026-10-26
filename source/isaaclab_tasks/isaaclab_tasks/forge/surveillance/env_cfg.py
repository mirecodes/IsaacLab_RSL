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
from isaaclab_tasks.forge.terrain.world_cfg import TerrainWorldCfg

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
}
"""Event trigger of each terrain type (``mdp/triggers.py``); the others sample normal walking (``random_walk``)."""


def extra_templates() -> dict:
    """Terrain types added to ``forge/terrain``'s catalog; keep the keys equal to ``EXTRA_TERRAINS`` in
    ``__init__.py``."""
    return {
        # random foot slips on generic uneven ground: rough_field's template on the slip zone
        "slip_field": RoughFieldCfg(zone="slip"),
    }


def deploy_world(terrain: str) -> TerrainWorldCfg:
    """The terrain map of ``terrain`` (including :func:`extra_templates`) cut to its first ``LEVELS`` rows, at the
    full map's difficulties."""
    world = TerrainWorldCfg(num_levels=LEVELS, meta_path="")
    world.terrains.update(extra_templates())
    world.terrain = terrain
    template = world.terrains[terrain]
    lo, hi = template.difficulty_range or world.difficulty_range
    # row i of the full map: d_i = lo + (hi - lo) i / (FULL_LEVELS - 1); keep i < LEVELS
    template.difficulty_range = (lo, lo + (hi - lo) * (LEVELS - 1) / (FULL_LEVELS - 1))
    return world


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


def make_env_cfg(terrain: str, policy: str):
    """The env cfg of ``IsaacContrib-Surveillance-<terrain>-<policy>``."""
    return POLICIES[policy](terrain_world=deploy_world(terrain))
