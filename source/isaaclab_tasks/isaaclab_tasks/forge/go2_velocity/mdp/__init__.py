# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Go2 robust (Miki et al. 2022 teacher) and energy (Fu et al. 2021) environments."""

from .actions import CPGJointPositionAction, CPGJointPositionActionCfg
from .curriculums import exponential_curriculum_factor
from .events import randomize_foot_friction, randomize_pd_gains
from .observations import (
    air_time,
    base_roll_pitch,
    contact_forces,
    contact_normals,
    contact_states,
    cpg_phase,
    external_wrench_b,
    foot_friction,
    foot_height_scan,
    ground_friction,
    joint_pos_error,
    joint_target_rel,
    motor_strength,
    payload_and_com,
    planar_base_velocity,
)
from .patterns import ConcentricCirclesPatternCfg, concentric_circles_pattern
from .rewards import (
    ang_vel_z_l2,
    body_motion,
    feet_slip_l2,
    foot_clearance,
    joint_target_smoothness,
    joint_upper_threshold_l2,
    lin_vel_orthogonal_exp,
    lin_vel_x_error_l1,
    lin_vel_y_l2,
    mechanical_power,
    track_ang_vel_saturated,
    track_lin_vel_saturated,
)
from .terminations import bad_roll_pitch, base_height_below_terrain
from .terrains import FractalTerrainCfg, fractal_terrain
