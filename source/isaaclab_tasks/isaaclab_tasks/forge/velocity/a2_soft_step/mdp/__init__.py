# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This task's terms: the soft-step walk gait and its load reward, the leg signal, the stop command."""

from .gait import SoftStepCPGAction, SoftStepCPGActionCfg, load_ramp  # noqa: F401
from .signal import (  # noqa: F401
    StopOnFlagVelocityCommand,
    StopOnFlagVelocityCommandCfg,
    flagged_foot_contact,
    get_flag,
    leg_flag,
    leg_signal,
)
from .terminations import base_below_terrain_height  # noqa: F401
