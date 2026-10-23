# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust A2 policy: the Go2 teacher of :mod:`..go2_study.robust_env_cfg` (Miki et al. 2022), adapted to A2.

The method is unchanged: CPG action space, observations, reward terms, terrain families, and curricula.
The robot's own values (gains, limits, armature, default pose) are Unitree's, see
:obj:`~isaaclab_assets.robots.unitree.UNITREE_A2_CFG`. The task is scaled from the paper's ANYmal to A2 by
the same rules the Go2 teacher uses with :data:`~..go2_study.robust_env_cfg.GO2_SCALE`, with
:data:`A2_SCALE` in its place:

* Lengths, heights, and the swing height x :data:`A2_SCALE`; the base frequency by Froude similarity; the
  tracking std by sqrt(:data:`A2_SCALE`); the torque weight by the squared ratio of peak joint torques.
* Values the Go2 teacher chose for Go2 rather than taking from the paper (hill and rough-terrain noise,
  box widths and minimum height) are scaled by ``A2_SCALE / GO2_SCALE``, i.e. treated as Go2-sized.
* The external force is about 0.2 x the robot's weight, as for Go2.
* Commands, pushes, and the remaining randomization keep the shared velocity-task defaults, as for Go2.

Deviations from the Go2 teacher, forced by the A2 asset or robot:

* The importer nests the links (``Robot/Geometry/base_link/FL_hip/...``) and names the base ``base_link``,
  so prim paths and body names differ from ``go2.usd``.
* The base-height target is the base height of Unitree's default pose instead of Go2's spawn height.
* The large and grid steps reach 0.35 m at the top difficulty level, above A2's rated 0.30 m step height.
  They are kept as the curriculum ceiling, as the paper's 0.4 m is for ANYmal.
"""

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_A2_CFG

from ..go2_study.base_env_cfg import Go2StudyBaseEnvCfg
from ..go2_study.robust_env_cfg import (
    FEET,
    GO2_SCALE,
    STEP_DT,
    Go2RobustTeacherEnvCfg,
    TeacherSceneCfg,
    foot_scanner,
    teacher_terrain_generator,
)

A2_SCALE = (0.275 + 0.275) / (0.285 + 0.349)
"""A2 / ANYmal length scale, from the leg lengths (hip-flexion joint to foot) of A2 and ANYmal C.

The same ratio for Go2 (0.426 m) is 0.672, which matches :data:`GO2_SCALE` (0.667, from base heights).
"""

GO2_TO_A2 = A2_SCALE / GO2_SCALE
"""Length scale from the Go2 teacher's values to A2."""

A2_MASS = 40.07
"""Total link mass [kg] of the A2 URDF."""

A2_STANDING_HEIGHT = 0.385
"""Base height [m] of Unitree's default pose (hip -/+0.1, thigh 0.9, calf -1.8 rad) from the URDF kinematics,
including the 0.032 m foot radius."""

A2_BASE_FREQUENCY = 1.25 / math.sqrt(A2_SCALE)
"""CPG base frequency [Hz]: 1.25 Hz (Lee et al. 2020, ANYmal) scaled by Froude similarity."""

BASE = "base_link"
"""Base body name in the imported A2 asset (``base`` on Go2)."""

FOOT_PATHS = {foot: f"Geometry/{BASE}/{foot[:2]}_hip/{foot[:2]}_thigh/{foot[:2]}_calf/{foot}" for foot in FEET}
"""Foot prim paths below the robot prim."""


def use_a2_asset(cfg: Go2StudyBaseEnvCfg):
    """Swap a Go2 study config's robot for A2 and point its base-body references at A2's prim layout.

    Terms a scenario removed (set to ``None``) are skipped.
    """
    cfg.scene.robot = replace(UNITREE_A2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.height_scanner.prim_path = f"{{ENV_REGEX_NS}}/Robot/Geometry/{BASE}"
    # body-level sensing only matches rigid bodies, so this skips the collision and visual prims
    cfg.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/Geometry/.*"
    for event in ("add_base_mass", "base_com", "base_external_force_torque"):
        if (term := getattr(cfg.events, event)) is not None:
            term.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)
    if cfg.terminations.base_contact is not None:
        cfg.terminations.base_contact.params["sensor_cfg"].body_names = BASE
    if cfg.rewards.base_height_l2 is not None:
        cfg.rewards.base_height_l2.params["target_height"] = A2_STANDING_HEIGHT


@configclass
class A2TeacherSceneCfg(TeacherSceneCfg):
    """Go2 teacher scene with the foot scanners on A2's nested feet and radii scaled to A2."""

    foot_scanner_FL = foot_scanner(FOOT_PATHS["FL_foot"], length_scale=GO2_TO_A2)
    foot_scanner_FR = foot_scanner(FOOT_PATHS["FR_foot"], length_scale=GO2_TO_A2)
    foot_scanner_RL = foot_scanner(FOOT_PATHS["RL_foot"], length_scale=GO2_TO_A2)
    foot_scanner_RR = foot_scanner(FOOT_PATHS["RR_foot"], length_scale=GO2_TO_A2)


@configclass
class A2RobustTeacherEnvCfg(Go2RobustTeacherEnvCfg):
    """A2 teacher policy after Miki et al. 2022."""

    scene: A2TeacherSceneCfg = A2TeacherSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        # -- robot and sensors on the imported asset's prim layout
        use_a2_asset(self)
        self.observations.policy.external_wrench.params["asset_cfg"] = SceneEntityCfg("robot", body_names=BASE)

        # -- actions: A2 leg geometry, swing height and base frequency scaled from ANYmal
        cpg = self.actions.joint_pos
        cpg.thigh_length = 0.275
        cpg.calf_length = 0.275
        cpg.base_frequency = A2_BASE_FREQUENCY
        cpg.swing_height = 0.2 * A2_SCALE
        cpg.phase_offset_scale = 0.5 * 2.0 * math.pi * A2_BASE_FREQUENCY * STEP_DT

        # -- terrain: the paper's families, heights scaled to A2
        self.scene.terrain.terrain_generator = teacher_terrain_generator(length_scale=GO2_TO_A2)

        # -- domain randomization: external wrench about 0.2 x A2's weight; the torque keeps Go2's
        # 0.1 m lever arm, scaled to A2
        force = 0.2 * A2_MASS * 9.81
        torque = 0.1 * GO2_TO_A2 * force
        self.events.base_external_force_torque.params["force_range"] = (-force, force)
        self.events.base_external_force_torque.params["torque_range"] = (-torque, torque)

        # -- rewards: Go2's weights with A2's scale
        # Source: Miki et al. 2022, Sec. S7, via the Go2 teacher: every weight is inherited as 2 x the paper's
        # coefficient (see Go2RobustTeacherEnvCfg). Weights the Go2 teacher leaves unscaled (tracking, r_b, r_fc,
        # r_bc, r_jm, r_jc, r_s, r_slip) stay the same; only the robot-size-dependent values are re-scaled from
        # ANYmal with A2_SCALE in place of GO2_SCALE:
        # velocity std: sqrt(A2_SCALE) by Froude similarity (1 on ANYmal)
        velocity_std = math.sqrt(A2_SCALE)
        rewards = self.rewards
        rewards.track_lin_vel.params["std"] = velocity_std
        rewards.track_ang_vel.params["std"] = velocity_std
        rewards.lin_vel_orthogonal.params["std"] = velocity_std
        # r_fc threshold: the paper's 0.2 m x A2_SCALE
        rewards.foot_clearance.params["threshold"] = 0.2 * A2_SCALE
        # r_tau: paper 1e-6 x (ANYmal's 80 N*m / A2's ~140 N*m peak torque)^2 (120 N*m hip and thigh, 180 N*m calf;
        # Go2's ~30 N*m is the same joint average of 23.7 and 45.4 N*m)
        rewards.dof_torques_l2.weight = -2 * 1.0e-6 * (80.0 / 140.0) ** 2
        # base_height_l2 (not in the paper): weight -30 inherited; target A2_STANDING_HEIGHT, set in use_a2_asset
