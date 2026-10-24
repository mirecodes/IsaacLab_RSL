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
from .templates.gravel_field import GravelFieldCfg, stone_shape
from .templates.pit_field import PitFieldCfg
from .templates.rough_field import RoughFieldCfg
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
        # generic uneven ground: fine bumps on rolling folds
        "rough_field": RoughFieldCfg(),
        # foot traps, trips
        "pit_field": PitFieldCfg(),
        # toe stubbing on bumps below the scanner resolution
        "stub_field": StubFieldCfg(),
        # climb / descent failures on a 0-45 deg incline, plain or rough
        "rough_slope": RoughSlopeCfg(),
        # missed steps, edge catches on 0-45 deg stairs, clean or uneven
        "rough_stairs": RoughStairsCfg(),
        # random slips on a gentler incline (difficulty rows 0.4-1.0: 18-30 deg), fine roughness only
        "slip_slope": RoughSlopeCfg(
            zone="slip", difficulty_range=None, slope_deg=(10.0, 30.0), amplitude=(0.0, 0.03), fold_amplitude=(0.0, 0.0)
        ),
        # sinking, lift-off failure, whole-body sinkage
        "sand_field": SandFieldCfg(),
        # support tilting under the foot (loose, mostly flat stones), kicked stones
        "gravel_field": GravelFieldCfg(),
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
    """The loose stones of ``gravel_field``: every stone is a dynamic body, and the stones follow their robot.

    MuJoCo-Warp simulates every robot in its own world, so a stone cannot be shared: each robot gets its own
    ``dynamic_rocks`` stones, and the cost grows with stones per robot, not with the field's size. So instead of
    strewing a whole tile, every ``recycle_interval`` the stones left farther than ``recycle_radius`` behind their
    robot are moved to free spots ahead of it (``ahead``, within ``sector`` of its walking direction), where the robot
    walks into them. The robust policy's foot scanners see only the global terrain, so a moved stone never pops
    into its view.
    """

    dynamic_rocks: int = 8
    """Stones per robot. Measured on ``gravel_field`` with 100 robots (RTX 4090): real-time factor 3.44 without
    stones, 0.91 with 8, 0.76 with 12, 0.57 with 16. A stone collides with every height-field cell under it, so
    large flat stones cost far more than the earlier small rocks."""
    size: tuple[float, float] = (0.10, 0.45)
    """Length [m] of the stones: log-uniform and stratified, so the stones of a robot always run from kickable ones
    to slabs it can stand on. Shapes (mostly flat) come from the ``gravel_field`` template's knobs."""
    density: tuple[float, float] = (2000.0, 3000.0)
    """Density [kg/m^3], uniform per stone."""
    friction: tuple[float, float] = (0.8, 1.0)
    """Stone friction, uniform per stone (it decides the stone-ground contact). High, so a stone stepped on near its
    edge tilts and rocks on its support rather than sliding away."""
    torsional_friction: float = 0.02
    rolling_friction: float = 0.005
    """MuJoCo torsional [m] and rolling [m] friction of the rocks (``condim`` 6). Without sleeping, small resting
    rocks otherwise keep rocking at about 1-2 cm/s."""
    recycle_interval: float = 1.0
    """Seconds between moves of the stones left behind."""
    recycle_radius: float = 3.0
    """Stones farther than this [m] from their robot's base (horizontally) are moved ahead."""
    ahead: tuple[float, float] = (1.2, 3.0)
    """Distance [m] from the base at which moved stones are placed (beyond the foot scanners' 0.5 m)."""
    sector: float = 1.0
    """Half angle [rad] around the walking direction (the heading when the robot stands) for moved stones."""
    candidates: int = 8
    """Random spots tried per moved stone; the one with the most clearance to the robot's other stones wins."""


@configclass
class TerrainWorldCfg:
    """One terrain map: ``num_levels`` difficulty rows x ``tiles_per_level`` tiles of the terrain type ``terrain``."""

    terrain: str = "rough_field"
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
    """Difficulty of the first and the last row (a template's own ``difficulty_range`` takes precedence)."""
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
        return cfg

    @property
    def num_envs(self) -> int:
        return self.num_levels * self.tiles_per_level * self.robots_per_tile

    @property
    def has_dynamic_rocks(self) -> bool:
        return isinstance(self.terrains[self.terrain], GravelFieldCfg) and self.rocks.dynamic_rocks > 0

    def terrain_generator(self) -> TerrainGridGeneratorCfg:
        """The generator: one row per difficulty, ``tiles_per_level`` columns of the selected template."""
        template = self.template
        return TerrainGridGeneratorCfg(
            seed=self.seed,
            size=(self.tile_size, self.tile_size),
            border_width=self.border_width,
            num_rows=self.num_levels,
            num_cols=self.tiles_per_level,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            # no vertical correction: the mesh vertices stay on the height-field grid, so the heightfield collider
            # equals the template, and a 45 deg incline is not turned into one-pixel steps
            slope_threshold=None,
            difficulty_range=template.difficulty_range or self.difficulty_range,
            use_cache=False,
            sub_terrains={self.terrain: template},
        )

    def dynamic_rock_specs(self) -> list[dict]:
        """Shape and material of each stone, shared by all environments."""
        rng = np.random.default_rng([self.seed, 9])
        template = self.terrains[self.terrain] if self.has_dynamic_rocks else GravelFieldCfg()
        lo, hi, k = self.rocks.size[0], self.rocks.size[1], self.rocks.dynamic_rocks
        specs = []
        for i in range(k):
            size = float(lo * (hi / lo) ** ((i + rng.uniform()) / k))
            specs.append({
                "template": i % template.template_count,
                "size": size,
                "axis_ratio": stone_shape(rng, template, size),
                "density": float(rng.uniform(*self.rocks.density)),
                "friction": float(rng.uniform(*self.rocks.friction)),
            })  # fmt: skip
        return specs
