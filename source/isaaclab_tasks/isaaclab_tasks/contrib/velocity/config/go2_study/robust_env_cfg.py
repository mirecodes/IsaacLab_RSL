# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robust Go2 policy: the teacher stage of Miki et al. 2022, "Learning robust perceptive locomotion
for quadrupedal robots in the wild" (Science Robotics), adapted to Go2.

Follows the paper's teacher: CPG action space, proprioception + noiseless foot-centered height samples +
privileged state, the Sec. S7 reward, an exponential curriculum on selected rewards and disturbances,
and the paper's terrain families and domain randomization.

Deviations, either forced by Isaac Lab or not stated in the paper:

* Lengths, heights, and the swing height are scaled from ANYmal to Go2 by :data:`GO2_SCALE` (nominal
  base heights 0.4 m vs. 0.6 m); torque weight by the squared ratio of peak joint torques; the base
  frequency by Froude similarity.
* Reward weights are the paper's coefficients times 2 so the tracking terms match the stock Go2 task's
  per-second scale (Isaac Lab multiplies weights by the 0.02 s step).
* The joint-acceleration penalty keeps Isaac Lab's weight: the paper's 0.001 on raw accelerations
  [rad^2/s^4] would dominate every other term.
* Terrain curriculum: Isaac Lab's distance-based terrain levels instead of the particle-filter curriculum.
  Perlin hills are approximated by smoothed uniform noise.
* No torque-limit termination: Go2's DC-motor model saturates routinely, and Isaac Lab clips the torque.
* Leg frequencies are bounded to [0.5, 1.5] x f_0, so the legs always step. With the full [0, 2] x f_0
  range the policy froze its CPG and stood still (iteration 300: phase action -0.91, 0.03 m/s forward).
* Ranges the paper leaves unstated (friction, masses, external wrench, knee threshold, c_0) are chosen
  here and marked below.
* The stock Go2 base-height penalty is kept: without it Go2 learns to rest on its legs.
"""

import math

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2Rewards
from isaaclab_tasks.core.velocity.velocity_env_cfg import CurriculumCfg, EventsCfg, MySceneCfg, TerminationsCfg

from . import mdp as study_mdp
from .base_env_cfg import Go2StudyBaseEnvCfg

GO2_SCALE = 0.4 / 0.6
"""Go2 / ANYmal length scale, from the nominal standing base heights."""

FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
LEGS = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh", "FL_calf", "FR_calf", "RL_calf", "RR_calf"]
LEG_JOINTS = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]
FOOT_SCANNERS = [f"foot_scanner_{foot[:2]}" for foot in FEET]

BASE_FREQUENCY = 1.25 / math.sqrt(GO2_SCALE)
"""CPG base frequency [Hz]: 1.25 Hz (Lee et al. 2020, ANYmal) scaled by Froude similarity."""

STEP_DT = 0.02
"""Policy step [s] (sim dt 0.005 s x decimation 4), i.e. the paper's 50 Hz."""

FEET_CFG = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)
LEGS_CFG = SceneEntityCfg("contact_forces", body_names=LEGS, preserve_order=True)


def _foot_scanner(foot: str) -> RayCasterCfg:
    """Foot-centered height samples: 52 rays on 5 circles, radii scaled to Go2."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        ray_alignment="yaw",
        pattern_cfg=study_mdp.ConcentricCirclesPatternCfg(
            radii=tuple(r * GO2_SCALE for r in (0.08, 0.16, 0.26, 0.36, 0.48)), num_points=(6, 8, 10, 12, 16)
        ),
        max_distance=2.0,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )


def teacher_terrain_generator() -> terrain_gen.TerrainGeneratorCfg:
    """The paper's terrain families with heights scaled to Go2, sorted into difficulty levels."""
    return terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=10,
        num_cols=20,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            # "rough": Perlin hills, approximated by smoothed uniform noise
            "hills": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.15, noise_range=(0.02, 0.15), noise_step=0.01, downsampled_scale=0.5, border_width=0.25
            ),
            # "rough discrete": quantized rough terrain
            "rough_discrete": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.15, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            # "large steps": two height levels, h in [0, 0.4] m
            "step_up": terrain_gen.MeshPitTerrainCfg(
                proportion=0.075, pit_depth_range=(0.0, 0.4 * GO2_SCALE), platform_width=3.0
            ),
            "step_down": terrain_gen.MeshBoxTerrainCfg(
                proportion=0.075, box_height_range=(0.0, 0.4 * GO2_SCALE), platform_width=3.0
            ),
            # "grid steps": h in [0.05, 0.4] m, d in [0.2, 0.7] m (the paper's evaluation ranges)
            "grid_steps": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.15,
                grid_width=0.45 * GO2_SCALE,
                grid_height_range=(0.05 * GO2_SCALE, 0.4 * GO2_SCALE),
                platform_width=2.0,
            ),
            # "stairs": h in [0.01, 0.22] m, d in [0.25, 1.0] m
            "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15,
                step_height_range=(0.01 * GO2_SCALE, 0.22 * GO2_SCALE),
                step_width=0.4 * GO2_SCALE,
                platform_width=3.0,
                border_width=1.0,
            ),
            "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.1,
                step_height_range=(0.01 * GO2_SCALE, 0.22 * GO2_SCALE),
                step_width=0.4 * GO2_SCALE,
                platform_width=3.0,
                border_width=1.0,
            ),
            # "boxes": max height 0.25 m, random position and yaw
            "boxes": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                proportion=0.15,
                obstacle_height_mode="choice",
                obstacle_width_range=(0.3, 1.0),
                obstacle_height_range=(0.03, 0.25 * GO2_SCALE),
                num_obstacles=40,
                platform_width=2.0,
                border_width=0.25,
            ),
        },
    )


@configclass
class TeacherSceneCfg(MySceneCfg):
    """Stock velocity scene plus one height scanner per foot."""

    foot_scanner_FL = _foot_scanner("FL_foot")
    foot_scanner_FR = _foot_scanner("FR_foot")
    foot_scanner_RL = _foot_scanner("RL_foot")
    foot_scanner_RR = _foot_scanner("RR_foot")


@configclass
class TeacherObservationsCfg:
    """Teacher observations (Table S3). The teacher sees noiseless state."""

    @configclass
    class PolicyCfg(ObsGroup):
        # -- proprioception (133)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_pos_error = ObsTerm(func=study_mdp.joint_pos_error, history_length=3)
        # current joint velocity plus a 2-step history
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=3)
        joint_target = ObsTerm(func=study_mdp.joint_target_rel, history_length=2)
        cpg_phase = ObsTerm(func=study_mdp.cpg_phase)
        # -- exteroception (208)
        foot_height_scan = ObsTerm(
            func=study_mdp.foot_height_scan, params={"sensor_names": FOOT_SCANNERS}, clip=(-1.0, 1.0)
        )
        # -- privileged information (50)
        foot_contact = ObsTerm(func=study_mdp.contact_states, params={"sensor_cfg": FEET_CFG})
        foot_contact_forces = ObsTerm(func=study_mdp.contact_forces, params={"sensor_cfg": FEET_CFG}, scale=0.01)
        foot_contact_normals = ObsTerm(func=study_mdp.contact_normals, params={"sensor_cfg": FEET_CFG})
        foot_friction = ObsTerm(func=study_mdp.foot_friction, params={"event_name": "foot_friction"})
        leg_contact = ObsTerm(func=study_mdp.contact_states, params={"sensor_cfg": LEGS_CFG})
        external_wrench = ObsTerm(
            func=study_mdp.external_wrench_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            scale=0.05,
        )
        foot_air_time = ObsTerm(func=study_mdp.air_time, params={"sensor_cfg": FEET_CFG})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TeacherEventsCfg(EventsCfg):
    """Stock events plus per-foot friction and leg-mass randomization."""

    foot_friction = EventTerm(
        func=study_mdp.randomize_foot_friction,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
            # not stated in the paper
            "friction_range": (0.4, 1.0),
            "low_friction_range": (0.05, 0.2),
            "low_friction_prob": 0.1,
        },
    )

    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_(hip|thigh|calf)"),
            # not stated in the paper
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


@configclass
class TeacherRewardsCfg(UnitreeGo2Rewards):
    """Miki et al. 2022 Sec. S7 reward. Weights are set in :class:`Go2RobustTeacherEnvCfg`."""

    track_lin_vel = RewTerm(
        func=study_mdp.track_lin_vel_saturated, weight=0.0, params={"command_name": "base_velocity"}
    )
    track_ang_vel = RewTerm(
        func=study_mdp.track_ang_vel_saturated, weight=0.0, params={"command_name": "base_velocity"}
    )
    lin_vel_orthogonal = RewTerm(
        func=study_mdp.lin_vel_orthogonal_exp, weight=0.0, params={"command_name": "base_velocity"}
    )
    body_motion = RewTerm(func=study_mdp.body_motion, weight=0.0)
    foot_clearance = RewTerm(
        func=study_mdp.foot_clearance, weight=0.0, params={"sensor_names": FOOT_SCANNERS, "threshold": 0.2 * GO2_SCALE}
    )
    leg_collision = RewTerm(func=mdp.undesired_contacts, weight=0.0, params={"sensor_cfg": LEGS_CFG, "threshold": 1.0})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0)
    knee_constraint = RewTerm(
        func=study_mdp.joint_upper_threshold_l2,
        weight=0.0,
        # not stated in the paper: keep Go2's knee (calf range [-2.72, -0.84] rad) away from full extension
        params={"threshold": -1.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint")},
    )
    target_smoothness = RewTerm(func=study_mdp.joint_target_smoothness, weight=0.0)
    foot_slip = RewTerm(
        func=study_mdp.feet_slip_l2,
        weight=0.0,
        params={"sensor_cfg": FEET_CFG, "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)},
    )


@configclass
class TeacherTerminationsCfg(TerminationsCfg):
    """Base contact plus a large-tilt termination."""

    # threshold not stated in the paper
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class TeacherCurriculumCfg(CurriculumCfg):
    """Terrain levels plus the paper's exponential curriculum factor."""

    curriculum_factor = CurrTerm(
        func=study_mdp.exponential_curriculum_factor,
        params={
            # c_0 not stated in the paper
            "initial_factor": 0.3,
            "decay": 0.98,
            "reward_terms": [
                "leg_collision",
                "joint_vel_l2",
                "dof_acc_l2",
                "target_smoothness",
                "dof_torques_l2",
                "foot_slip",
            ],
            "event_ranges": {"base_external_force_torque": ["force_range", "torque_range"]},
        },
    )


@configclass
class Go2RobustTeacherEnvCfg(Go2StudyBaseEnvCfg):
    """Go2 teacher policy after Miki et al. 2022."""

    scene: TeacherSceneCfg = TeacherSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    events: TeacherEventsCfg = TeacherEventsCfg()
    rewards: TeacherRewardsCfg = TeacherRewardsCfg()
    terminations: TeacherTerminationsCfg = TeacherTerminationsCfg()
    curriculum: TeacherCurriculumCfg = TeacherCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- sensors
        for name in FOOT_SCANNERS:
            getattr(self.scene, name).update_period = self.decimation * self.sim.dt

        # -- actions: CPG phase offsets (4) + residual joint targets (12)
        self.actions.joint_pos = study_mdp.CPGJointPositionActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINTS,
            thigh_length=0.213,
            calf_length=0.213,
            base_frequency=BASE_FREQUENCY,
            swing_height=0.2 * GO2_SCALE,
            initial_phase=(0.0, math.pi, math.pi, 0.0),
            # a unit action spans leg frequencies [0.5 * f_0, 1.5 * f_0]; with [0, 2 * f_0] the policy froze
            # its legs (phase action -> -1) and stood still instead of learning to walk
            phase_offset_scale=0.5 * 2.0 * math.pi * BASE_FREQUENCY * STEP_DT,
            residual_scale=0.25,
        )

        # -- terrain: the paper's families, heights scaled to Go2
        self.scene.terrain.terrain_generator = teacher_terrain_generator()

        # -- domain randomization
        # MuJoCo-Warp uses max(mu_foot, mu_ground); a low ground value lets foot_friction set the contact friction
        self.scene.terrain.physics_material.static_friction = 0.05
        self.scene.terrain.physics_material.dynamic_friction = 0.05
        self.sim.physics_material = self.scene.terrain.physics_material
        self.events.reset_base.params["pose_range"].update({"roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)})
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={"position_range": (-0.25, 0.25), "velocity_range": (-1.0, 1.0)},
        )
        # external wrench on the base, resampled during the episode (magnitudes not stated in the paper;
        # 30 N is about 0.2 x Go2's weight) and scaled by the curriculum factor
        self.events.base_external_force_torque.mode = "interval"
        self.events.base_external_force_torque.interval_range_s = (2.0, 6.0)
        self.events.base_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.base_external_force_torque.params["torque_range"] = (-3.0, 3.0)

        # -- rewards: 2 x the paper's coefficients (see module docstring)
        velocity_std = math.sqrt(GO2_SCALE)
        rewards = self.rewards
        for name in ("track_lin_vel", "track_ang_vel", "lin_vel_orthogonal"):
            getattr(rewards, name).weight = 2 * 0.75
        rewards.track_lin_vel.params["std"] = velocity_std
        rewards.track_ang_vel.params["std"] = velocity_std
        rewards.lin_vel_orthogonal.params["std"] = velocity_std
        rewards.body_motion.weight = -2 * 1.0
        rewards.foot_clearance.weight = -2 * 0.003
        rewards.leg_collision.weight = -2 * 0.1
        rewards.joint_vel_l2.weight = -2 * 0.001 * 0.01
        rewards.dof_acc_l2.weight = -2.5e-7
        rewards.knee_constraint.weight = -2 * 0.08
        rewards.target_smoothness.weight = -2 * 0.003
        # ANYmal's 80 N*m peak torque vs. Go2's ~30 N*m
        rewards.dof_torques_l2.weight = -2 * 1.0e-6 * (80.0 / 30.0) ** 2
        rewards.foot_slip.weight = -2 * 0.003
        # replaced by the terms above
        rewards.track_lin_vel_xy_exp = None
        rewards.track_ang_vel_z_exp = None
        rewards.lin_vel_z_l2 = None
        rewards.ang_vel_xy_l2 = None
        rewards.action_rate_l2 = None
        rewards.feet_air_time = None
        rewards.flat_orientation_l2 = None
        rewards.dof_pos_limits = None
