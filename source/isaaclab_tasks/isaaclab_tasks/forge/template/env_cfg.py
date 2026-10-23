# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Template forge task: an empty manager-based RL environment.

Start either way:

* Inherit: replace ``ManagerBasedRLEnvCfg`` with an existing task's env cfg (core, contrib or forge), delete
  the empty manager configs below, and override in ``__post_init__``.
* From scratch: fill every manager config below with terms.

Every change made to a copy of this file is recorded in the task's ``REPORT.md``.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Terrain, robots, objects, sensors and lights."""


@configclass
class CommandsCfg:
    """Command generators."""


@configclass
class ActionsCfg:
    """Action terms."""


@configclass
class ObservationsCfg:
    """Observation groups (``policy``, ``critic``, ...)."""


@configclass
class EventsCfg:
    """Event terms (``startup``, ``reset`` and ``interval`` modes)."""


@configclass
class RewardsCfg:
    """Reward terms."""


@configclass
class TerminationsCfg:
    """Termination terms."""


@configclass
class CurriculumCfg:
    """Curriculum terms."""


@configclass
class TemplateEnvCfg(ManagerBasedRLEnvCfg):
    """Empty manager-based RL environment."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # control: policy at 1 / (dt * decimation) = 50 Hz
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
