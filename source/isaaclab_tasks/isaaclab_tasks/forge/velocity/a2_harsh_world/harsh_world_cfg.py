# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every knob of the Harsh World, in one place. Edit this file only; everything else is derived from it.

* Property knobs are ``(min, max)`` pairs interpolated by the cell difficulty (a scalar means ``(v, v)``).
  Densities and amplitudes take the interpolated value, per-instance sizes are drawn up to it.
* Columns are the enabled properties in :attr:`HarshWorldCfg.properties` order, rows the difficulties.
* Defaults follow the spec (``REPORT.md`` lists every value that differs and why). They are starting points for
  observing failures, not tuned values.
"""

from __future__ import annotations

import numpy as np

from isaaclab.utils import configclass

from .mdp.terrains import (
    GravelCfg,
    LooseRocksCfg,
    MesaDropCfg,
    MixedCfg,
    RoughSlopeCfg,
    RoughStairsCfg,
    SandFieldCfg,
    StubFieldCfg,
)
from .mdp.world import HarshWorldGeneratorCfg

A2_FOOT_RADIUS = 0.032
"""A2 foot collider: a sphere of 0.032 m (``a2.urdf``)."""

A2_NOMINAL_FOOT_LOAD = 40.07 * 9.81 / 4
"""Static load [N] per foot of A2 (40.07 kg) standing on four feet."""


def default_properties() -> dict:
    """The property catalog. Keys are the property names (column order)."""
    return {
        # 1: foot traps, trips, generic rough ground
        "gravel": GravelCfg(),
        # 2: toe stubbing on bumps below the scanner resolution
        "stub_field": StubFieldCfg(),
        # 3: steep climb / descent failures
        "rough_slope": RoughSlopeCfg(),
        # 4: missed steps, edge catches
        "rough_stairs": RoughStairsCfg(),
        # 5: random slips on a gentle slope
        "slip_slope": RoughSlopeCfg(zone="slip", slope=(0.25, 0.6), amplitude=(0.0, 0.03), ledge_count=(0.0, 0.0)),
        # 6: sinking, lift-off failure, whole-body sinkage
        "sand_field": SandFieldCfg(),
        # 7: compound causes, feature densities halved
        "mixed": MixedCfg(),
        # 8: falls from terraces, hard landings
        "mesa_drop": MesaDropCfg(),
        # 9: support moving under the foot, ankle twists, body jamming
        "loose_rocks": LooseRocksCfg(),
        # 10: support lost right after touchdown (sand variant with collapsing crust patches)
        "crust_collapse": SandFieldCfg(crust_prob=0.4),
    }


@configclass
class SlipCfg:
    """Foot friction in ``slip`` zones (sampled once when a foot enters the zone)."""

    mu_nominal: float = 0.8
    """Foot friction outside slip zones and after leaving one: A2's material value in the base task."""
    mu_slip_easy: tuple[float, float] = (0.05, 0.35)
    """Low-friction range at difficulty 0."""
    mu_slip_hard: tuple[float, float] = (0.03, 0.25)
    """Low-friction range at difficulty 1 (bounds interpolated in between)."""
    slip_prob: float = 0.75
    """Probability that an entering foot gets a low-friction sample (spec 0.5: the slide on ``slip_slope`` was only
    2.6x the rigid twin's, below the spec's 3x)."""
    resample_on_touchdown: bool = False
    """Re-draw at every touchdown inside the zone instead of once per entry."""
    observe_slip: bool = False
    """Whether the policy's privileged friction input reports the applied friction (True) or the nominal one."""
    ground_mu: float = 0.02
    """Friction of the terrain itself. MuJoCo-Warp uses ``max(mu_foot, mu_ground)``, so it must stay below the
    slip range for the foot value to decide the contact friction."""


@configclass
class SandCfg:
    """Sand patches: compliant slabs plus velocity-dependent drag on feet inside the sand."""

    c_drag: tuple[float, float] = (20.0, 80.0)
    """Horizontal drag [N s/m], interpolated by difficulty: ``F_xy = -c_drag * v_xy``."""
    c_suction: tuple[float, float] = (30.0, 150.0)
    """Lift-off suction [N s/m]: ``F_z = -c_suction * max(v_z, 0)``."""
    force_limit: float = 60.0
    """Cap [N] on the drag force of one foot: the force is applied for a whole 20 ms policy step, and an
    uncapped 80 N s/m on a ~1 kg leg would overshoot the velocity it damps."""
    foot_radius: float = A2_FOOT_RADIUS
    damping_ratio: float = 0.1
    """Slab contact damping as a fraction of its stiffness, ``kd = damping_ratio * ke`` [N s/m]."""
    friction: float = 0.6
    """Foot-slab friction (the slab's value; it has contact priority over the foot)."""


@configclass
class CollapseCfg:
    """Crust patches of ``crust_collapse``: stiff until loaded, then they give way and recover later."""

    enabled: bool = True
    crust_stiffness: float = 3.0e4
    """Stiffness [N/m] of an intact crust (about 3 mm sinkage under one foot)."""
    collapse_force: tuple[float, float] = (1.5 * A2_NOMINAL_FOOT_LOAD, 3.0 * A2_NOMINAL_FOOT_LOAD)
    """Load [N] on one patch (summed over one robot's feet) that collapses it, interpolated by difficulty."""
    collapse_mode: str = "soften"
    """``"soften"``: stiffness / 10. ``"disable"``: stiffness x 1e-4, the foot drops to the hardpan."""
    recover_time: float = 30.0
    """Seconds until a collapsed patch is restored."""


@configclass
class RockCfg:
    """Dynamic rocks of ``loose_rocks`` (static rocks are part of the terrain)."""

    dynamic_rocks: int = 4
    """Dynamic rocks per robot. MuJoCo-Warp simulates every robot in its own world, so each robot gets its own
    copy of the dynamic rocks of its row's ``loose_rocks`` cell. The spec's ~1500 rocks (16 per robot) cut the
    real-time factor to 42 % of the rock-free world; 4 keep 77-79 % (6: 71-73 %, at the spec's 70 % limit)."""
    size: tuple[float, float] = (0.06, 0.25)
    """Largest extent [m] of the dynamic rocks, power law with the property's exponent."""
    density: tuple[float, float] = (2000.0, 3000.0)
    """Density [kg/m^3], uniform per rock."""
    friction: tuple[float, float] = (0.4, 0.9)
    """Rock friction, uniform per rock."""
    torsional_friction: float = 0.02
    rolling_friction: float = 0.005
    """MuJoCo torsional [m] and rolling [m] friction of the rocks (``condim`` 6). Without sleeping, small resting
    rocks otherwise keep rocking at about 1-2 cm/s."""
    reset_interval: float = 300.0
    """Seconds between returns of drifted rocks to their slots."""
    drift_max: float = 1.0
    """Rocks farther than this [m] from their slot are returned."""
    moving_speed: float = 0.05
    """A rock under a foot faster than this [m/s] counts as moving."""


@configclass
class HarshWorldCfg:
    """The Harsh World: a grid of property cells walked by a trained policy, with zone labels for every fault."""

    seed: int = 42
    """Seed of the terrain, spawns, and commands. ``collect.py --seed`` overrides it."""
    region_size: float = 24.0
    """Side [m] of one square cell."""
    num_rows: int = 3
    """Difficulty rows."""
    difficulty_range: tuple[float, float] = (0.4, 1.0)
    """Difficulty of the first and the last row."""
    horizontal_scale: float = 0.03
    vertical_scale: float = 0.005
    border_width: float = 5.0
    """Flat border [m] around the grid."""
    platform_width: float = 3.0
    """Side [m] of the flat spawn square (spec 2.0: robots standing on it touched the stairs and slopes starting at
    its edge within their first second)."""
    spawn_radius: float = 0.8
    """Spawn offset [m] from the cell center (a square of half side ``spawn_radius / sqrt(2)``), yaw uniform."""
    spawn_height: float = 0.45
    """Base height [m] above the platform at spawn."""
    robots_per_cell: int = 4
    """Robots per cell; ``num_envs = num_rows x num_cols x robots_per_cell`` (84 for 7 properties)."""
    episode_length_s: float = 60.0
    command_resampling_s: tuple[float, float] = (5.0, 10.0)
    lin_vel_x: tuple[float, float] = (-0.5, 1.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    meta_path: str = "logs/harsh_world/world_meta_seed{seed}.yaml"
    """Where the importer writes ``world_meta.yaml`` (``{seed}`` is the terrain seed); empty to skip."""

    properties: dict = default_properties()
    """Property name -> cfg, in column order."""
    disabled: list = ["mesa_drop", "loose_rocks", "crust_collapse"]
    """Properties left out of the grid. The default world is the spec's 7 static properties; the ``-Full`` tasks
    empty this list for all 10."""

    slip: SlipCfg = SlipCfg()
    sand: SandCfg = SandCfg()
    collapse: CollapseCfg = CollapseCfg()
    rocks: RockCfg = RockCfg()

    @property
    def enabled(self) -> dict:
        """Enabled properties with their names filled in."""
        props = {k: v for k, v in self.properties.items() if k not in self.disabled}
        for name, cfg in props.items():
            cfg.name = name
            cfg.platform_width = self.platform_width
            if isinstance(cfg, LooseRocksCfg):
                cfg.dynamic_sizes = [spec["size"] for spec in self.dynamic_rock_specs()]
        return props

    @property
    def num_envs(self) -> int:
        return self.num_rows * len(self.enabled) * self.robots_per_cell

    def terrain_generator(self) -> HarshWorldGeneratorCfg:
        """The generator: one column per enabled property, one row per difficulty."""
        props = self.enabled
        return HarshWorldGeneratorCfg(
            seed=self.seed,
            size=(self.region_size, self.region_size),
            border_width=self.border_width,
            num_rows=self.num_rows,
            num_cols=len(props),
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=0.75,
            difficulty_range=self.difficulty_range,
            use_cache=False,
            sub_terrains=props,
        )

    def dynamic_rock_specs(self) -> list[dict]:
        """Shape and material of each dynamic rock, shared by all environments and rows."""
        rng = np.random.default_rng([self.seed, 9])
        lo, hi, a = self.rocks.size[0], self.rocks.size[1], LooseRocksCfg().size_exponent + 1.0
        specs = []
        for i in range(self.rocks.dynamic_rocks):
            specs.append({
                "template": i % LooseRocksCfg().template_count,
                "size": float((lo**a + rng.uniform() * (hi**a - lo**a)) ** (1.0 / a)),
                "axis_ratio": rng.uniform(*LooseRocksCfg().axis_ratio, 2).tolist(),
                "density": float(rng.uniform(*self.rocks.density)),
                "friction": float(rng.uniform(*self.rocks.friction)),
            })  # fmt: skip
        return specs
