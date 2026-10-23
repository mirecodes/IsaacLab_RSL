# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared base configuration of the A2 study: :class:`..go2_study.base_env_cfg.Go2StudyBaseEnvCfg` on A2.

The robot is swapped as in the A2 robust teacher (:func:`.robust_env_cfg.use_a2_asset`) and the task is
scaled by the same rules:

* Terrain heights the Go2 base chose for Go2 x :data:`~.robust_env_cfg.GO2_TO_A2`; slopes are angles and kept.
* The torque weight by the squared ratio of peak joint torques (Go2 ~30 N*m, A2 ~140 N*m).
* Commands, actions, pushes, and the remaining randomization keep the shared velocity-task defaults, as for Go2.
"""

from isaaclab.utils import configclass

from ..go2_study.base_env_cfg import Go2StudyBaseEnvCfg
from .robust_env_cfg import GO2_TO_A2, use_a2_asset


@configclass
class A2StudyBaseEnvCfg(Go2StudyBaseEnvCfg):
    """Go2 study base task on A2."""

    def __post_init__(self):
        super().__post_init__()

        # -- robot and sensors on the imported asset's prim layout
        use_a2_asset(self)

        # -- terrain: Go2's heights scaled to A2
        s = GO2_TO_A2
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["pyramid_stairs"].step_height_range = (0.025 * s, 0.12 * s)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025 * s, 0.12 * s)
        terrains["boxes"].grid_height_range = (0.025 * s, 0.1 * s)
        terrains["random_rough"].noise_range = (0.01 * s, 0.06 * s)

        # -- rewards
        # Source: the Go2 base's weights (Isaac Lab velocity defaults + stock Go2 overrides, see
        # go2_study/base_env_cfg.py), all inherited unchanged except:
        # dof_torques_l2: Go2's -0.0002 x (Go2's ~30 N*m / A2's ~140 N*m peak torque)^2, so the same fraction of
        # the peak torque costs the same as on Go2
        self.rewards.dof_torques_l2.weight = -0.0002 * (30.0 / 140.0) ** 2
        # base_height_l2: weight -30 inherited; target 0.4 m -> A2_STANDING_HEIGHT, set in use_a2_asset
