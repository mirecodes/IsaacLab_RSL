# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The leg flag: which leg the policy must stop relying on (``REPORT.md`` §3).

``env.leg_flag`` (N, 4) is the one source of truth, in leg order FL, FR, RL, RR. In training :class:`leg_flag` writes
it from the injectors' schedules, as a detector would report them (latency, misses, phantom alarms); in deployment
``scripts/forge/leg_guard/guard_eval.py`` writes it and :func:`external_leg_flag` only reads it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.forge.velocity.a2_velocity.mdp.actions import CPGJointPositionAction, CPGJointPositionActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

LEGS = ("FL", "FR", "RL", "RR")


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def get_flag(env) -> torch.Tensor:
    """``env.leg_flag``, created with zeros on first use."""
    if not hasattr(env, "leg_flag"):
        env.leg_flag = torch.zeros(env.num_envs, len(LEGS), device=env.device)
    return env.leg_flag


def external_leg_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The flag as written from outside the env (deployment)."""
    return get_flag(env)


class leg_flag(ManagerTermBase):
    """Training flag, a function of the episode step only, so repeated calls in a step agree.

    A motor fault or a stuck foot sets the flag of its leg ``latency`` after the onset (miss: never, per episode with
    probability ``p_miss``), for as long as the fault lasts. A phantom alarm flags a random leg for a random time,
    with or without a fault (a false alarm, and practice for causes the injectors do not cover).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n, dev = env.num_envs, env.device
        self.joint_leg = None
        self.latency = torch.zeros(n, dtype=torch.long, device=dev)
        self.miss = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ph_start = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.ph_end = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.ph_leg = torch.zeros(n, dtype=torch.long, device=dev)
        self.rows = torch.arange(n, device=dev)
        self.reset()

    def reset(self, env_ids=None) -> None:
        ids = self.rows if env_ids is None else self.rows[env_ids]
        p, dt, n = self.cfg.params, self._env.step_dt, len(ids)
        dev = self.rows.device

        def uniform(lo_hi):
            return ((lo_hi[0] + (lo_hi[1] - lo_hi[0]) * torch.rand(n, device=dev)) / dt).long()

        self.latency[ids] = uniform(p["latency_s"])
        self.miss[ids] = torch.rand(n, device=dev) < p["p_miss"]
        start = uniform(p["phantom_start_s"])
        self.ph_start[ids] = torch.where(torch.rand(n, device=dev) < p["p_phantom"], start, -1)
        self.ph_end[ids] = start + uniform(p["phantom_hold_s"])
        self.ph_leg[ids] = torch.randint(len(LEGS), (n,), device=dev)

    def __call__(self, env: ManagerBasedRLEnv, latency_s, p_miss, p_phantom, phantom_start_s, phantom_hold_s):
        flag = get_flag(env)
        flag.zero_()
        step = env.episode_length_buf
        robot = env.scene["robot"]
        if self.joint_leg is None:
            self.joint_leg = torch.tensor([LEGS.index(n[:2]) for n in robot.joint_names], device=env.device)
        # phantom first: a true alarm overrides it
        phantom = (self.ph_start >= 0) & (step >= self.ph_start) & (step < self.ph_end)
        flag[self.rows[phantom], self.ph_leg[phantom]] = 1.0
        # without injectors (the Stand variant) every flag is a scheduled one
        if not any("motor_fault" in names for names in env.event_manager.active_terms.values()):
            return flag

        injector = env.event_manager.get_term_cfg("motor_fault").func
        num_j = injector.num_joints
        labels = injector.labels()
        faulty = labels[:, :num_j] > 0
        since = labels[:, 2 * num_j + 2]
        on = faulty.any(1) & (since >= self.latency) & ~self.miss
        flag[self.rows[on], self.joint_leg[faulty.float().argmax(1)][on]] = 1.0

        labels = env.event_manager.get_term_cfg("foot_stuck").func.labels()
        stuck = labels[:, : len(LEGS)] > 0
        since = labels[:, len(LEGS) + 2]
        on = stuck.any(1) & (since >= self.latency) & ~self.miss
        flag[self.rows[on]] = 0.0
        flag[self.rows[on], stuck.float().argmax(1)[on]] = 1.0
        return flag


def flagged_foot_load(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, nominal_load: float) -> torch.Tensor:
    """Contact force on the flagged feet in units of the nominal foot load, each at most 2 (a trot stance foot carries
    about 2): the cost of relying on a flagged leg."""
    forces = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids]
    load = (torch.linalg.norm(forces, dim=-1) / nominal_load).clamp(max=2.0)
    return (load * get_flag(env)).sum(1)


def flagged_foot_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, nominal_load: float, threshold: float = 5.0
) -> torch.Tensor:
    """Discrete cost of a flagged foot on the ground: ``1 + F_z / F_0`` once its vertical force ``F_z`` exceeds
    ``threshold`` [N], else 0. ``F_z / F_0`` is capped at 2 so touchdown impulses do not dominate the return."""
    fz = _t(env.scene.sensors[sensor_cfg.name].data.net_forces_w)[:, sensor_cfg.body_ids, 2].clamp(min=0.0)
    cost = (fz > threshold).float() * (1.0 + (fz / nominal_load).clamp(max=2.0))
    return (cost * get_flag(env)).sum(1)


class GuardCPGAction(CPGJointPositionAction):
    """CPG action that pulls the flagged foot back: ``tuck_height`` above its nominal stance position, the hip at its
    default angle, reached in ``tuck_ramp_s``. The policy keeps the other legs; its action on this leg is ignored."""

    cfg: GuardCPGActionCfg

    def __init__(self, cfg: GuardCPGActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._blend = torch.zeros(self.num_envs, self._num_legs, device=self.device)
        thigh, calf = self._sagittal_ik(self._foot_x, self._foot_z + cfg.tuck_height)
        self._tuck = torch.stack((self._default_hip, thigh, calf), dim=-1)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        step = self._env.step_dt / self.cfg.tuck_ramp_s
        self._blend = (self._blend + torch.where(get_flag(self._env) > 0, step, -step)).clamp(0.0, 1.0)
        blend = self._blend[..., None]
        targets = self._joint_targets.view(self.num_envs, self._num_legs, 3)
        self._joint_targets = ((1.0 - blend) * targets + blend * self._tuck).view(self.num_envs, -1)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._blend[slice(None) if env_ids is None else env_ids] = 0.0


@configclass
class GuardCPGActionCfg(CPGJointPositionActionCfg):
    class_type: type = GuardCPGAction

    tuck_height: float = 0.12
    """Foot lift [m] while flagged: above the CPG's swing height 0.2 * A2_SCALE, chosen here."""
    tuck_ramp_s: float = 0.1
    """Time [s] to reach the tuck pose and to return, chosen here: about one swing's rise."""
