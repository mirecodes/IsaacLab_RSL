# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The leg signal: which foot must be unloaded now (``REPORT.md`` §3.2).

``env.leg_flag`` (N, 4), legs FL, FR, RL, RR, is the one source of truth. In training :class:`leg_signal` (an event
run every step) writes it; in deployment an outside script writes it and the event is removed. The observation,
the action term, the command and the rewards only read it. Adapted from ``velocity/a2_leg_guard/mdp/leg_flag.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

LEGS = ("FL", "FR", "RL", "RR")


def _t(x):
    return getattr(x, "torch", x)


def get_flag(env) -> torch.Tensor:
    """``env.leg_flag``, created with zeros on first use."""
    if not hasattr(env, "leg_flag"):
        env.leg_flag = torch.zeros(env.num_envs, len(LEGS), device=env.device)
    return env.leg_flag


def leg_flag(env: ManagerBasedEnv) -> torch.Tensor:
    """Observation: the signal, one-hot over legs (all zero when no leg is signalled)."""
    return get_flag(env)


class leg_signal(ManagerTermBase):
    """Training signal, written to ``env.leg_flag`` every step (``mode="interval"``, global, zero interval).

    * Random: at each touchdown of a foot, with probability ``p_touchdown``, that foot is signalled after
      ``U[0, load_s]`` (inside its loading time) for ``U[hold_s]``; one signal at a time.
    * Injectors (if the env has them): a motor fault or a stuck foot signals its leg ``latency`` after the onset
      (missed in a ``p_miss`` share of the episodes), for as long as the fault lasts; it overrides a random one.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.rows = torch.arange(n, device=dev)
        self.leg = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.on = torch.zeros(n, dtype=torch.long, device=dev)
        self.off = torch.zeros(n, dtype=torch.long, device=dev)
        self.latency = torch.zeros(n, dtype=torch.long, device=dev)
        self.miss = torch.zeros(n, dtype=torch.bool, device=dev)
        self.joint_leg = None
        # which injectors the env has: known only once the event manager exists (first call)
        self.motor = self.stuck = None
        self.reset()

    def _steps(self, lo_hi, n):
        lo, hi = lo_hi
        return ((lo + (hi - lo) * torch.rand(n, device=self.rows.device)) / self._env.step_dt).long()

    def reset(self, env_ids=None) -> None:
        ids = self.rows if env_ids is None else self.rows[env_ids]
        p = self.cfg.params
        self.leg[ids] = -1
        self.latency[ids] = self._steps(p["latency_s"], len(ids))
        self.miss[ids] = torch.rand(len(ids), device=self.rows.device) < p["p_miss"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        sensor_cfg: SceneEntityCfg,
        p_touchdown: float,
        load_s: float,
        hold_s: tuple[float, float],
        latency_s: tuple[float, float],
        p_miss: float,
    ):
        flag = get_flag(env)
        step = env.episode_length_buf
        n, dt = env.num_envs, env.step_dt
        if self.motor is None:
            terms = [name for names in env.event_manager.active_terms.values() for name in names]
            self.motor, self.stuck = "motor_fault" in terms, "foot_stuck" in terms
        # random signals: a new one at a touchdown when none is scheduled or on
        self.leg[(self.leg >= 0) & (step >= self.off)] = -1
        first = _t(env.scene.sensors[sensor_cfg.name].compute_first_contact(dt))[:, sensor_cfg.body_ids]
        touch = first.bool() & (torch.rand(n, len(LEGS), device=env.device) < p_touchdown)
        new = touch.any(1) & (self.leg < 0) & ~flag.any(1)
        if new.any():
            k = int(new.sum())
            self.leg[new] = touch.float().argmax(1)[new]
            self.on[new] = step[new] + self._steps((0.0, load_s), k)
            self.off[new] = self.on[new] + self._steps(hold_s, k)
        flag.zero_()
        active = (self.leg >= 0) & (step >= self.on)
        flag[self.rows[active], self.leg[active]] = 1.0
        # injector signals override
        if self.motor:
            injector = env.event_manager.get_term_cfg("motor_fault").func
            if self.joint_leg is None:
                names = env.scene["robot"].joint_names
                self.joint_leg = torch.tensor([LEGS.index(name[:2]) for name in names], device=env.device)
            num_j = injector.num_joints
            labels = injector.labels()
            faulty = labels[:, :num_j] > 0
            on = faulty.any(1) & (labels[:, 2 * num_j + 2] >= self.latency) & ~self.miss
            flag[on] = 0.0
            flag[self.rows[on], self.joint_leg[faulty.float().argmax(1)][on]] = 1.0
        if self.stuck:
            labels = env.event_manager.get_term_cfg("foot_stuck").func.labels()
            stuck = labels[:, : len(LEGS)] > 0
            on = stuck.any(1) & (labels[:, len(LEGS) + 2] >= self.latency) & ~self.miss
            flag[on] = 0.0
            flag[self.rows[on], stuck.float().argmax(1)[on]] = 1.0


class StopOnFlagVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that reads zero while any leg is signalled: standing still then earns the tracking
    rewards, and the policy sees the stop in its command input."""

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b * (get_flag(self._env).sum(1, keepdim=True) == 0)


@configclass
class StopOnFlagVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = StopOnFlagVelocityCommand


def flagged_foot_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, nominal_load: float, threshold: float = 5.0
) -> torch.Tensor:
    """Discrete cost of a signalled foot on the ground: ``1 + min(F_z / F_0, 2)`` once its vertical force ``F_z``
    exceeds ``threshold`` [N], else 0."""
    fz = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    cost = (fz > threshold).float() * (1.0 + (fz / nominal_load).clamp(max=2.0))
    return (cost * get_flag(env)).sum(1)
