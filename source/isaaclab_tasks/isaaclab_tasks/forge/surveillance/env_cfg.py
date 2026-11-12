# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployment of the trained A2 policies on a terrain map: difficulty rows 1-3, 20 robots on random tiles, steering
(on the inclines: straight up or down the slope instead), one extra terrain type (``mixed_field``).

Built on the terrain-map play tasks (``forge/terrain``), whose map, physics and policy wiring stay as they are; see
``REPORT.md``.
"""

import math
from collections.abc import Callable

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.terrain.env_cfg import A2EnergyTerrainEnvCfg, A2RobustTerrainEnvCfg
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
    "ice_field": mdp.foot_slip,
    "mixed_field": mdp.any_event,
}
"""Event trigger of each terrain type (``mdp/triggers.py``); the others sample normal walking (``random_walk``)."""


def extra_templates() -> dict:
    """Terrain types added to ``forge/terrain``'s catalog; keep the keys equal to ``EXTRA_TERRAINS`` in
    ``__init__.py``."""
    return {
        # several events on one map (sand patches, collapsing crusts, ice patches between them; see MIXED), so that
        # which event acted cannot be read off the terrain type, as it can on the one-event maps
        "mixed_field": SandFieldCfg(crust_prob=0.3),
    }


MIXED_ICE = 0.15
"""Share of the ice cells of ``mixed_field`` (``SlipCfg.ice_frac``); it has no random single-foot slips."""


_PATCH_SHARE = ((0.40, 0.60), (0.55, 0.75), (0.70, 0.90)) * 2
_PATCH_DEPTH = ((0.20, 0.35), (0.30, 0.45), (0.40, 0.60)) * 2
_PATCH_CELLS = ((1.2, 4.5),) * 3 + ((3.0, 8.0),) * 3
"""Patchwork cells [m] of tiers 1-3 (as ``-Hard``) and 4-6, which are tiers 1-3 with patches that hold all four feet."""
_PATCH_JITTER = (0.6,) * 3 + (0.2,) * 3


def _then_last(values: tuple) -> tuple:
    """``values`` (tiers 1-6) followed by its last value for tiers 7-9, which change only the sand's stiffness."""
    return values + (values[-1],) * 3


STRESS = {
    "rough_field": {
        "amplitude": ((0.04, 0.10), (0.06, 0.15), (0.08, 0.20)),
        "fold_amplitude": ((0.2, 0.7), (0.3, 1.0), (0.4, 1.4)),
        "fold_size": ((8.0, 4.0), (6.0, 3.0), (5.0, 2.5)),
    },
    "pit_field": {
        "pit_density": ((1.0, 3.0), (2.0, 5.0), (3.0, 8.0)),
        "pit_radius": ((0.08, 0.20), (0.10, 0.25), (0.12, 0.30)),
        "pit_depth": ((0.15, 0.35), (0.20, 0.45), (0.30, 0.60)),
        "trench_count": ((4.0, 15.0), (10.0, 40.0), (20.0, 80.0)),
        "trench_width": ((0.10, 0.25), (0.12, 0.30), (0.15, 0.40)),
        "trench_depth": ((0.15, 0.35), (0.20, 0.45), (0.30, 0.60)),
    },
    # tiers 4-6: many stubs and roots, kept below the belly (tiers 2-3 mostly touch the belly of a robot still upright)
    "stub_field": {
        "stub_density": ((2.0, 6.0), (4.0, 10.0), (8.0, 16.0), (8.0, 16.0), (16.0, 30.0), (16.0, 30.0)),
        "stub_radius": ((0.03, 0.06), (0.03, 0.06), (0.04, 0.08), (0.03, 0.06), (0.03, 0.06), (0.02, 0.04)),
        "stub_height": ((0.08, 0.18), (0.10, 0.25), (0.15, 0.35), (0.08, 0.18), (0.10, 0.18), (0.12, 0.20)),
        "root_count": ((20.0, 80.0), (60.0, 200.0), (120.0, 400.0), (120.0, 400.0), (200.0, 600.0), (200.0, 600.0)),
        "root_width": ((0.03, 0.05), (0.04, 0.08), (0.05, 0.10), (0.03, 0.05), (0.03, 0.05), (0.03, 0.05)),
        "root_height": ((0.06, 0.12), (0.08, 0.16), (0.10, 0.22), (0.06, 0.12), (0.08, 0.14), (0.08, 0.14)),
        "root_length": ((1.0, 4.0), (1.5, 5.0), (2.0, 6.0), (1.0, 4.0), (1.0, 4.0), (1.0, 4.0)),
    },
    # tiers 1-6 change the geometry; 7-9 are tier 6 with softer sand (at 60-240 N/m the feet sink 3 cm at any depth)
    "sand_field": {
        "stiffness": ((60.0, 240.0),) * 6 + ((15.0, 40.0), (5.0, 15.0), (1.0, 5.0)),
        "sand_prob": _then_last(_PATCH_SHARE),
        "sand_depth": _then_last(_PATCH_DEPTH),
        "cell_size": _then_last(_PATCH_CELLS),
        "inset_jitter": _then_last(_PATCH_JITTER),
        "sand.c_drag": _then_last(((20.0, 80.0), (40.0, 120.0), (60.0, 160.0)) * 2),
        "sand.c_suction": _then_last(((300.0, 600.0), (400.0, 800.0), (600.0, 1200.0)) * 2),
        "sand.force_limit": _then_last((200.0, 250.0, 300.0) * 2),
    },
    "crust_collapse": {
        "crust_prob": (0.6, 0.8, 0.95) * 2,
        "sand_prob": _PATCH_SHARE,
        "sand_depth": _PATCH_DEPTH,
        "cell_size": _PATCH_CELLS,
        "inset_jitter": _PATCH_JITTER,
        "collapse.collapse_force": (
            (0.6 * A2_NOMINAL_FOOT_LOAD, 1.0 * A2_NOMINAL_FOOT_LOAD),
            (0.5 * A2_NOMINAL_FOOT_LOAD, 0.8 * A2_NOMINAL_FOOT_LOAD),
            (0.4 * A2_NOMINAL_FOOT_LOAD, 0.6 * A2_NOMINAL_FOOT_LOAD),
        )
        * 2,
    },
    # tier 1: larger stones only; tier 2: stones that move (same sizes as -Hard); tier 3: both; tiers 4-6: small mobile
    # stones that cannot reach the belly even on edge (<= 0.3 m; tiers 1-3 mostly touch the belly of an upright robot):
    # round cobbles (4), lighter and more of them (5), flat stones that slide (6)
    "gravel_field": {
        "rocks.size": ((0.20, 0.70), (0.10, 0.45), (0.20, 0.70), (0.10, 0.28), (0.10, 0.28), (0.15, 0.30)),
        "rocks.dynamic_rocks": (12, 16, 16, 16, 24, 24),
        "rocks.density": (
            (2000.0, 3000.0),
            (800.0, 1500.0),
            (1000.0, 1800.0),
            (800.0, 1500.0),
            (500.0, 1000.0),
            (800.0, 1500.0),
        ),
        "rocks.friction": ((0.8, 1.0), (0.1, 0.3), (0.1, 0.3), (0.1, 0.3), (0.1, 0.3), (0.05, 0.1)),
        "rocks.rolling_friction": (0.04, 0.005, 0.005, 0.005, 0.005, 0.005),
        "rocks.torsional_friction": (0.05, 0.01, 0.01, 0.01, 0.01, 0.01),
        "rocks.sector": (0.7, 0.7, 0.6, 0.7, 0.7, 0.7),
        "rocks.ahead": ((1.2, 3.0), (1.2, 3.0), (0.9, 2.2), (1.2, 3.0), (1.2, 3.0), (1.2, 3.0)),
        "flat_frac": (0.8, 0.2, 0.2, 0.2, 0.2, 1.0),
        "round_max": (0.2, 0.45, 0.45, 0.28, 0.28, 0.2),
    },
}
"""Test tiers on top of :func:`apply_hard`, to find which terrain settings fail the robust policy (chosen here, not
tuned; ``-Hard`` gave it 0-6 falls in 200 robot-min on these maps). Knob -> its value at each tier; a knob without a dot
is the map's template, ``section.knob`` a section of ``TerrainWorldCfg``. Keep the keys and tier counts equal to
``STRESS_TIERS`` in ``__init__.py``."""


def apply_stress(world: TerrainWorldCfg, level: int) -> None:
    """Tier ``level`` (from 1) of :data:`STRESS` on the map of ``world.terrain``."""
    for knob, values in STRESS[world.terrain].items():
        section, _, name = knob.rpartition(".")
        setattr(getattr(world, section) if section else world.terrains[world.terrain], name, values[level - 1])


def deploy_world(terrain: str, hard: bool = False, stress: int = 0) -> TerrainWorldCfg:
    """The terrain map of ``terrain`` (including :func:`extra_templates`) cut to its first ``LEVELS`` rows, at the
    full map's difficulties; ``hard``: its last ``LEVELS`` rows and the stronger events of :func:`apply_hard`;
    ``stress``: then that tier of :data:`STRESS`."""
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
    if stress:
        apply_stress(world, stress)
    return world


def apply_hard(world: TerrainWorldCfg) -> None:
    """Events strong enough to fail the policies (``REPORT.md`` §3.5; v2 had 25 falls in 66.7 robot-h)."""
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

FAULT_ONSET_S = (3.0, 12.0)
"""Onset range of the injected faults [s after the reset]: at least half a 5 s trigger window of history before it
(``collect.py`` needs 2.5 s), and within the shortest episodes (robots leave their tile after ~30 s)."""


FOOT_STUCK_SEED = 7919
"""Seed of the foot-stuck schedule (chosen here; any value other than the env seed)."""


MILD_POWER_LOSS_H = (0.6, 0.85)
"""Health range of the ``-FaultsMild`` variant: above the default power-loss range U[0.1, 0.6] (out of a detector's
training range, and harder to see)."""


def apply_faults(env_cfg, mild: bool = False) -> None:
    """The ``-Faults`` variant: the motor fault and the foot-stuck injector of ``forge/fault_injector`` (each in 40 %
    of the episodes, drawn independently, so 16 % have both), and the trigger :func:`mdp.fault_or_event`. Both
    injectors run the actuators in Isaac Lab (``use_newton_actuators = False``)."""
    if mild:
        # chosen here: a mild power loss in 70 % of the episodes, no stuck feet
        mild_cfg = MotorFaultInjectorCfg(
            p_fault=0.7,
            fault_types={"power_loss": 1.0},
            power_loss_health=MILD_POWER_LOSS_H,
            onset_time_s=FAULT_ONSET_S,
        )
        add_motor_faults(env_cfg, mild_cfg)
        add_foot_stuck(env_cfg, FootStuckInjectorCfg(p_stuck=0.0, onset_time_s=FAULT_ONSET_S, seed=FOOT_STUCK_SEED))
    else:
        # chosen here: 40 % each, so that faults, terrain events and nominal walking all have many windows
        add_motor_faults(env_cfg, MotorFaultInjectorCfg(p_fault=0.4, onset_time_s=FAULT_ONSET_S))
        # its own seed: with the env seed of the motor injector, both draw the same numbers (always together)
        add_foot_stuck(env_cfg, FootStuckInjectorCfg(p_stuck=0.4, onset_time_s=FAULT_ONSET_S, seed=FOOT_STUCK_SEED))
    env_cfg.event_trigger = mdp.fault_or_event
    # the energy policy's PD-gain randomization needs Newton actuators, which the injectors turn off: drop it and feed
    # its motor-strength extrinsics the nominal 1
    if getattr(env_cfg.events, "pd_gains", None) is not None:
        env_cfg.events.pd_gains = None
        for group in vars(env_cfg.observations).values():
            if getattr(group, "motor_strength", None) is not None:
                group.motor_strength = ObsTerm(func=mdp.nominal_motor_strength)


def make_env_cfg(terrain: str, policy: str, hard: bool = False, faults: str = "", stress: int = 0):
    """The env cfg of ``IsaacContrib-Surveillance-<terrain>-<policy>`` (``hard``: ``...-Hard``, ``faults``:
    ``"Faults"`` or ``"FaultsMild"`` for ``...-Faults`` / ``...-FaultsMild``, ``stress``: ``...-Stress<N>``)."""
    env_cfg = POLICIES[policy](terrain_world=deploy_world(terrain, hard, stress))
    if faults:
        apply_faults(env_cfg, mild=faults == "FaultsMild")
    return env_cfg
