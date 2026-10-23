# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Energy-efficient A2 policy: the Go2 base policy of :mod:`..go2_study.energy_env_cfg` (Fu et al. 2021),
adapted to A2.

The method is unchanged: reward, observations, joint-delta action clipping, fractal terrain, randomization,
and early terminations. The robot is swapped as in the A2 robust teacher (:func:`.robust_env_cfg.use_a2_asset`),
and the paper's A1 values are scaled to A2 by the rules the Go2 policy uses:

* Lengths by :data:`A2_LENGTH_SCALE` (thigh and calf 0.275 m vs. A1's 0.2 m), payload by :data:`A2_MASS_SCALE`;
  the target speed and joint-delta limits are kept.

Deviations from the Go2 energy policy, forced by A2's size:

* The energy weight is divided by :data:`A2_MASS_SCALE`. Power at a fixed speed grows with the robot's weight,
  so the paper's 0.04 would weigh energy ~3.3x heavier against r_forward and r_alive than on A1. On Go2 the
  factor is 1.25 and is not applied.
* Height limit: Go2's limit is 90% of its measured zero-action stance. A2's stance is not measured yet; its
  kinematic default-pose height is :data:`~.robust_env_cfg.A2_STANDING_HEIGHT`, and Unitree's stiff gains
  should sag it by a few cm. :data:`A2_MIN_BASE_HEIGHT` takes 80% of the kinematic height to leave that margin.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp

from isaaclab_assets.robots.unitree import UNITREE_A2_CFG

from ..go2_study.energy_env_cfg import JOINT_DELTA_LIMITS, Go2EnergyTeacherEnvCfg
from .robust_env_cfg import (
    A2_MASS,
    A2_STANDING_HEIGHT,
    BASE,
    GO2_TO_A2,
    teacher_terrain_generator,
    use_a2_asset,
)

A2_LENGTH_SCALE = 0.275 / 0.2
"""A2 / A1 length scale, from the thigh and calf lengths."""

A2_MASS_SCALE = A2_MASS / 12.0
"""A2 / A1 mass scale."""

A2_MIN_BASE_HEIGHT = 0.8 * A2_STANDING_HEIGHT
"""Termination height [m] of the base above the terrain. Calibrate to 90% of A2's measured zero-action stance."""


@configclass
class A2EnergyTeacherEnvCfg(Go2EnergyTeacherEnvCfg):
    """A2 energy-efficient base policy after Fu et al. 2021."""

    def __post_init__(self):
        super().__post_init__()

        # -- robot and sensors on the imported asset's prim layout; this also restores A2's 0.4 m spawn height
        use_a2_asset(self)
        self.events.add_payload.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)
        self.observations.policy.payload_and_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)

        # -- actions: joint target deltas around A2's standing pose
        self.actions.joint_pos.clip = {
            name: (pos - JOINT_DELTA_LIMITS[joint], pos + JOINT_DELTA_LIMITS[joint])
            for name, pos in UNITREE_A2_CFG.init_state.joint_pos.items()
            for joint in JOINT_DELTA_LIMITS
            if joint in name
        }

        # -- terrain: fractal amplitude and wavelength scaled to A2
        fractal = self.scene.terrain.terrain_generator.sub_terrains["fractal"]
        fractal.amplitude = 0.23 * A2_LENGTH_SCALE
        fractal.base_wavelength = 2.0 * A2_LENGTH_SCALE

        # -- domain randomization
        self.events.add_payload.params["mass_distribution_params"] = (0.0, 0.5 * A2_MASS_SCALE)

        # -- terminations
        self.terminations.base_height.params["minimum_height"] = A2_MIN_BASE_HEIGHT

        # -- rewards
        # Source: Fu et al. 2021, Eq. 3-5, via the Go2 energy policy (see Go2EnergyTeacherEnvCfg). Inherited unchanged:
        # r_forward -20 |v_x error|, -1 v_y^2, -1 w_yaw^2 and r_alive 20 x 0.9 m/s, since the target speed is kept.
        # r_energy: the paper's 0.04 / A2_MASS_SCALE (~3.3), since power at a fixed speed grows with the robot's
        # weight (see module docstring); the Go2 policy skips its 1.25 factor
        self.rewards.energy.weight = -0.04 / A2_MASS_SCALE


@configclass
class A2EnergyTeacherRoughEnvCfg(A2EnergyTeacherEnvCfg):
    """Fu et al. 2021 energy policy on the A2 robust teacher's terrains, as
    :class:`..go2_study.energy_env_cfg.Go2EnergyTeacherRoughEnvCfg` for Go2."""

    def __post_init__(self):
        super().__post_init__()

        # -- terrain: the robust teacher's families scaled to A2, sorted into difficulty levels
        self.scene.terrain.terrain_generator = teacher_terrain_generator(length_scale=GO2_TO_A2)
        self.scene.terrain.max_init_terrain_level = 5
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # -- terminations: as in the robust teacher
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE), "threshold": 1.0},
        )
        self.terminations.base_height = None
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
