# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Every knob of a terrain map, in one place: which template, the grid, spawning, commands, and runtime physics.

* A map holds one terrain type: ``num_levels`` difficulty rows x ``tiles_per_level`` tiles of that type.
* Template knobs are ``(min, max)`` pairs interpolated by the tile difficulty (see ``templates/common.py``).
* Defaults follow the Harsh World spec; ``REPORT.md`` lists every value that differs and why.
"""

from __future__ import annotations

import numpy as np

from isaaclab.utils import configclass

from .mdp.generator import TerrainGridGeneratorCfg
from .templates.common import TerrainTemplateCfg
from .templates.gravel import GravelCfg
from .templates.loose_rocks import LooseRocksCfg
from .templates.mesa import MesaDropCfg
from .templates.mixed import MixedCfg
from .templates.sand import SandFieldCfg
from .templates.slope import RoughSlopeCfg
from .templates.stairs import RoughStairsCfg
from .templates.stub_field import StubFieldCfg

A2_FOOT_RADIUS = 0.032
"""A2 foot collider: a sphere of 0.032 m (``a2.urdf``)."""

A2_NOMINAL_FOOT_LOAD = 40.07 * 9.81 / 4
"""Static load [N] per foot of A2 (40.07 kg) standing on four feet."""


def default_terrains() -> dict:
    """The terrain catalog: name -> template cfg. Keep the keys equal to ``TERRAINS`` in ``__init__.py``."""
    return {
        # foot traps, trips, generic rough ground
        "gravel": GravelCfg(),
        # toe stubbing on bumps below the scanner resolution
        "stub_field": StubFieldCfg(),
        # steep climb / descent failures
        "rough_slope": RoughSlopeCfg(),
        # missed steps, edge catches
        "rough_stairs": RoughStairsCfg(),
        # random slips on a gentle slope
        "slip_slope": RoughSlopeCfg(zone="slip", slope=(0.25, 0.6), amplitude=(0.0, 0.03), ledge_count=(0.0, 0.0)),
        # sinking, lift-off failure, whole-body sinkage
        "sand_field": SandFieldCfg(),
        # compound causes, feature densities halved
        "mixed": MixedCfg(),
        # falls from terraces, hard landings
        "mesa_drop": MesaDropCfg(),
        # support moving under the foot, ankle twists, body jamming
        "loose_rocks": LooseRocksCfg(),
        # support lost right after touchdown (sand with collapsing crust patches)
        "crust_collapse": SandFieldCfg(crust_prob=0.4),
    }


@configclass
class SlipCfg:
    """Foot friction on ``slip`` terrain (sampled once when a foot enters the map, i.e. at spawn)."""

    mu_nominal: float = 0.8
    """Foot friction off slip terrain and after leaving it: A2's material value in the base task."""
    mu_slip_easy: tuple[float, float] = (0.05, 0.35)
    """Low-friction range at difficulty 0."""
    mu_slip_hard: tuple[float, float] = (0.03, 0.25)
    """Low-friction range at difficulty 1 (bounds interpolated in between)."""
    slip_prob: float = 0.75
    """Probability that an entering foot gets a low-friction sample (spec 0.5: the slide on ``slip_slope`` was only
    2.6x the rigid twin's, below the spec's 3x)."""
    resample_on_touchdown: bool = False
    """Re-draw at every touchdown instead of once per entry."""
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
    copy of the dynamic rocks of its tile. The spec's ~1500 rocks (16 per robot) cut the real-time factor to 42 % of
    the rock-free world; 4 keep 77-79 % (6: 71-73 %, at the spec's 70 % limit)."""
    size: tuple[float, float] = (0.06, 0.25)
    """Largest extent [m] of the dynamic rocks, power law with the template's exponent."""
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


@configclass
class TerrainWorldCfg:
    """One terrain map: ``num_levels`` difficulty rows x ``tiles_per_level`` tiles of the terrain type ``terrain``."""

    terrain: str = "gravel"
    """Terrain type: a key of :attr:`terrains`."""
    terrains: dict = default_terrains()
    """Name -> template cfg (edit a template's knobs here)."""

    seed: int = 42
    """Seed of the terrain, spawns, and commands. ``collect.py --seed`` overrides it."""
    tile_size: float = 24.0
    """Side [m] of one square tile."""
    num_levels: int = 5
    """Difficulty rows (0.4, 0.55, 0.7, 0.85, 1.0 with the default range)."""
    tiles_per_level: int = 5
    """Tiles per difficulty row; each has its own random layout."""
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
    """Spawn offset [m] from the tile center (a square of half side ``spawn_radius / sqrt(2)``), yaw uniform."""
    spawn_height: float = 0.45
    """Base height [m] above the platform at spawn."""
    robots_per_tile: int = 4
    """Robots per tile; ``num_envs = num_levels x tiles_per_level x robots_per_tile`` (100 by default)."""
    episode_length_s: float = 60.0
    command_resampling_s: tuple[float, float] = (5.0, 10.0)
    lin_vel_x: tuple[float, float] = (-0.5, 1.5)
    lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    meta_path: str = "logs/terrain/{terrain}_seed{seed}.yaml"
    """Where the importer writes the map description (``{terrain}``, ``{seed}`` filled in); empty to skip."""

    slip: SlipCfg = SlipCfg()
    sand: SandCfg = SandCfg()
    collapse: CollapseCfg = CollapseCfg()
    rocks: RockCfg = RockCfg()

    @property
    def template(self) -> TerrainTemplateCfg:
        """The selected template's cfg with the world's values filled in."""
        cfg = self.terrains[self.terrain]
        cfg.name = self.terrain
        cfg.platform_width = self.platform_width
        if isinstance(cfg, LooseRocksCfg):
            cfg.dynamic_sizes = [spec["size"] for spec in self.dynamic_rock_specs()]
        return cfg

    @property
    def num_envs(self) -> int:
        return self.num_levels * self.tiles_per_level * self.robots_per_tile

    @property
    def has_dynamic_rocks(self) -> bool:
        return isinstance(self.terrains[self.terrain], LooseRocksCfg) and self.rocks.dynamic_rocks > 0

    def terrain_generator(self) -> TerrainGridGeneratorCfg:
        """The generator: one row per difficulty, ``tiles_per_level`` columns of the selected template."""
        return TerrainGridGeneratorCfg(
            seed=self.seed,
            size=(self.tile_size, self.tile_size),
            border_width=self.border_width,
            num_rows=self.num_levels,
            num_cols=self.tiles_per_level,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=0.75,
            difficulty_range=self.difficulty_range,
            use_cache=False,
            sub_terrains={self.terrain: self.template},
        )

    def dynamic_rock_specs(self) -> list[dict]:
        """Shape and material of each dynamic rock, shared by all environments and tiles."""
        rng = np.random.default_rng([self.seed, 9])
        template = LooseRocksCfg()
        lo, hi, a = self.rocks.size[0], self.rocks.size[1], template.size_exponent + 1.0
        specs = []
        for i in range(self.rocks.dynamic_rocks):
            specs.append({
                "template": i % template.template_count,
                "size": float((lo**a + rng.uniform() * (hi**a - lo**a)) ** (1.0 / a)),
                "axis_ratio": rng.uniform(*template.axis_ratio, 2).tolist(),
                "density": float(rng.uniform(*self.rocks.density)),
                "friction": float(rng.uniform(*self.rocks.friction)),
            })  # fmt: skip
        return specs
