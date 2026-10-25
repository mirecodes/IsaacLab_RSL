# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fault-injection templates: random mid-episode faults that a task adds to its env cfg (see ``README.md``).

Shared infrastructure like ``forge/terrain``: tasks import ``injector`` directly. Nothing is registered here, and
this file imports nothing, since every forge package is imported at startup.
"""
