# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor-fault data collection for the detector: the trained A2 robust teacher's task with the motor-fault
injector (``forge/fault_injector``) and three recorded observation groups. See ``REPORT.md``.

Play-only: the policy is frozen and sees its usual ``policy`` group; the other groups are recorded, never fed back.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity import mdp
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults
from isaaclab_tasks.forge.velocity.a2_velocity.robust_env_cfg import FEET, A2RobustTeacherEnvCfg

from . import mdp as fd_mdp

FEET_CFG = SceneEntityCfg("contact_forces", body_names=FEET, preserve_order=True)


def _group(**terms) -> ObsGroup:
    """Noiseless, concatenated observation group of the given terms (noise is added offline, see ``REPORT.md``)."""

    @configclass
    class Group(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    group = Group()
    for name, term in terms.items():
        setattr(group, name, term)
    return group


@configclass
class MotorFaultA2RobustEnvCfg(A2RobustTeacherEnvCfg):
    """A2 robust teacher on its training terrain with one random motor fault per episode (70 % of episodes).

    The fault distribution is ``events.motor_fault.params["cfg"]`` (:class:`MotorFaultInjectorCfg`).
    """

    def __post_init__(self):
        super().__post_init__()

        # -- terrain: the training grid, robots spread uniformly over all difficulty levels and kept there
        self.curriculum = None
        self.scene.terrain.max_init_terrain_level = None
        # -- no pushes or external wrench: abnormal motion comes from the policy, the terrain and the fault
        self.events.push_robot = None
        self.events.base_external_force_torque = None

        # -- detector inputs, in spec order (step dim 3 + 3 + 12 + 12 + 16 + 3 = 49)
        self.observations.detector = _group(
            base_ang_vel=ObsTerm(func=mdp.base_ang_vel),
            projected_gravity=ObsTerm(func=mdp.projected_gravity),
            joint_pos_rel=ObsTerm(func=mdp.joint_pos_rel),
            joint_vel=ObsTerm(func=mdp.joint_vel_rel),
            # the CPG action: 4 phase offsets + 12 residual joint targets
            last_action=ObsTerm(func=mdp.last_action),
            velocity_command=ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        )
        # -- privileged inputs (sim ground truth): base height above the terrain, base linear velocity (x, y, z)
        self.observations.privileged = _group(
            base_height=ObsTerm(func=fd_mdp.base_height, params={"sensor_cfg": SceneEntityCfg("height_scanner")}),
            base_lin_vel=ObsTerm(func=mdp.base_lin_vel),
        )
        # -- analysis only, never a detector input
        self.observations.analysis = _group(
            joint_pos_target=ObsTerm(func=fd_mdp.joint_pos_target_rel),
            imu_lin_acc=ObsTerm(func=fd_mdp.imu_lin_acc),
            foot_contact=ObsTerm(func=fd_mdp.foot_contact, params={"sensor_cfg": FEET_CFG}),
            foot_contact_force=ObsTerm(func=fd_mdp.foot_contact_force, params={"sensor_cfg": FEET_CFG}),
            applied_torque=ObsTerm(func=fd_mdp.applied_torque),
        )
        # -- fault injector and its ``fault`` label group
        add_motor_faults(self, MotorFaultInjectorCfg())
