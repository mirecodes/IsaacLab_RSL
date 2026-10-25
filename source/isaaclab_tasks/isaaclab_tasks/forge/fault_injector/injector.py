# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor power-out / power-loss injector: one random single-joint fault per episode, from a random onset step on.

* :class:`FaultyDCMotor` scales the DC-motor torque of every joint by a health ``h`` in [0, 1].
* :class:`MotorFaultInjector` (an ``interval`` event run every policy step) samples each episode's fault at reset
  and sets ``h``; it owns the schedule and the labels.
* :func:`motor_fault_labels` exposes the labels as an observation term, so recorders see them next to the inputs.
* :func:`add_motor_faults` wires the three into an env cfg whose robot uses a :class:`DCMotorCfg` actuator group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FAULT_TYPES = ("none", "power_out", "power_loss")
"""Fault-type labels, by index."""


@configclass
class MotorFaultInjectorCfg:
    """Per-episode fault distribution (sampled at every reset)."""

    p_fault: float = 0.7
    """Probability that an episode gets a fault; the others stay nominal to the end."""

    fault_types: dict[str, float] = {"power_out": 0.5, "power_loss": 0.5}
    """Relative probability of each fault type (keys from :data:`FAULT_TYPES`)."""

    joint_names: list[str] = [".*"]
    """Candidate joints (regex over the actuator group's joints); the faulty one is drawn uniformly."""

    onset_time_s: tuple[float, float] = (2.0, 8.0)
    """Onset time after the reset, uniform [s]; quantized to policy steps."""

    power_loss_health: tuple[float, float] = (0.1, 0.6)
    """Health ``h`` of a power-loss joint, uniform. Power out is ``h = 0``."""

    passive_damping: float = 0.0
    """Damping ``d`` [N*m*s/rad] of a joint without power (gear friction, back-EMF): ``tau = -d * qdot``."""

    ramp_steps: int = 0
    """Linear ramp from ``h = 1`` to the fault's ``h`` over this many policy steps (0: immediate)."""

    seed: int | None = None
    """Seed of the fault schedule; None uses the env seed."""


class FaultyDCMotor(DCMotor):
    r"""DC motor with a per-env, per-joint health :math:`h \in [0, 1]`, written by :class:`MotorFaultInjector`.

    .. math:: \tau = h \cdot \mathrm{clip}_{DC}(\tau_{PD}) - [h = 0]\, d\, \dot q

    Scaling the output (not the torque limit) leaves a tracking error whenever ``h < 1``, so every faulty sample is
    observable in principle; a reduced limit would act only while the demand exceeds it.
    """

    def __init__(self, cfg: DCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.health = torch.ones(self._num_envs, self.num_joints, device=self._device)
        self.passive_damping = 0.0
        self.probe = None
        """Optional callable run at every actuator (PD) step, before the torque is computed: a tap for recorders
        that sample at the PD rate. The articulation data is then that of the state the PD law acts on."""

    def compute(self, control_action, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        if self.probe is not None:
            self.probe()
        control_action = super().compute(control_action, joint_pos, joint_vel)
        effort = self.health * self.applied_effort - self.passive_damping * (self.health == 0.0) * joint_vel
        self.applied_effort = effort
        control_action.joint_efforts = effort
        return control_action


class MotorFaultInjector(ManagerTermBase):
    """Samples one fault per episode at reset and applies it from its onset step on (see module doc).

    Run as an ``interval`` event every policy step, after the resets: it records the health the finished physics
    step used (the labels of the observation computed next), then sets the health of the next step. The first
    faulty label is therefore at episode step ``onset_step``, the first observation that saw faulty physics.

    Params: ``cfg`` (:class:`MotorFaultInjectorCfg`), ``actuator`` (actuator group name), ``asset_name``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.fault_cfg: MotorFaultInjectorCfg = cfg.params["cfg"]
        robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.actuator: FaultyDCMotor = robot.actuators[cfg.params["actuator"]]
        if not isinstance(self.actuator, FaultyDCMotor):
            raise TypeError(f"Actuator '{cfg.params['actuator']}' is not a FaultyDCMotor; use add_motor_faults().")
        self.actuator.passive_damping = self.fault_cfg.passive_damping
        # actuator column -> articulation joint index (labels are in articulation order)
        self.columns = torch.tensor(
            robot.find_joints(self.actuator.joint_names, preserve_order=True)[0], device=self.device
        )
        self.num_joints = robot.num_joints
        candidates, _ = robot.find_joints(self.fault_cfg.joint_names)
        self.candidates = torch.tensor(
            [self.actuator.joint_names.index(robot.joint_names[j]) for j in candidates], device=self.device
        )
        self.type_probs = torch.tensor(
            [self.fault_cfg.fault_types.get(name, 0.0) for name in FAULT_TYPES[1:]], device=self.device
        )
        seed = self.fault_cfg.seed if self.fault_cfg.seed is not None else (env.cfg.seed or 0)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.onset_range = [round(t / env.step_dt) for t in self.fault_cfg.onset_time_s]
        n = self.num_envs
        self.fault_type = torch.zeros(n, dtype=torch.long, device=self.device)
        """0 none, 1 power out, 2 power loss."""
        self.fault_column = torch.zeros(n, dtype=torch.long, device=self.device)
        """Faulty joint as an actuator column."""
        self.onset_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.severity = torch.ones(n, device=self.device)
        """The fault's final health ``h``."""
        self.observed_health = torch.ones(n, self.num_joints, device=self.device)
        """Health during the last physics step, in articulation order: the labels of the current observation."""

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids if env_ids is not None else slice(None)]
        k, g, cfg = len(ids), self.generator, self.fault_cfg
        rand = lambda: torch.rand(k, device=self.device, generator=g)  # noqa: E731
        has_fault = rand() < cfg.p_fault
        kind = torch.multinomial(self.type_probs, k, replacement=True, generator=g) + 1
        self.fault_type[ids] = torch.where(has_fault, kind, 0)
        pick = torch.randint(len(self.candidates), (k,), device=self.device, generator=g)
        self.fault_column[ids] = self.candidates[pick]
        lo, hi = self.onset_range
        onset = torch.randint(lo, hi + 1, (k,), device=self.device, generator=g)
        self.onset_step[ids] = torch.where(has_fault, onset, -1)
        lo, hi = cfg.power_loss_health
        h = lo + (hi - lo) * rand()
        self.severity[ids] = torch.where(kind == 1, 0.0, h)
        self.actuator.health[ids] = 1.0
        self.observed_health[ids] = 1.0

    def __call__(self, env: ManagerBasedRLEnv, env_ids, cfg, actuator: str, asset_name: str = "robot"):
        self.observed_health[:, self.columns] = self.actuator.health
        # steps since onset as seen by the next observation
        since = env.episode_length_buf + 1 - self.onset_step
        active = (self.fault_type > 0) & (since >= 0)
        ramp = 1.0 if cfg.ramp_steps <= 0 else ((since + 1).float() / cfg.ramp_steps).clamp(max=1.0)
        h = 1.0 - (1.0 - self.severity) * ramp
        health = torch.ones_like(self.actuator.health)
        rows = active.nonzero().squeeze(-1)
        health[rows, self.fault_column[rows]] = h[rows]
        self.actuator.health[:] = health

    def labels(self) -> torch.Tensor:
        """``[fault_mask (J), fault_type, health (J), onset_step, steps_since_onset, severity]`` per env.

        ``fault_mask`` and ``fault_type`` are nonzero from ``onset_step`` on; ``steps_since_onset`` is -1 before.
        """
        mask = (self.observed_health < 1.0).float()
        active = mask.any(dim=1)
        step = self._env.episode_length_buf
        since = torch.where(active, step - self.onset_step, -1)
        columns = [
            mask,
            torch.where(active, self.fault_type, 0).unsqueeze(1).float(),
            self.observed_health,
            self.onset_step.unsqueeze(1).float(),
            since.unsqueeze(1).float(),
            torch.where(self.fault_type > 0, self.severity, 1.0).unsqueeze(1),
        ]
        return torch.cat(columns, dim=1)


def motor_fault_labels(env: ManagerBasedRLEnv, event_name: str = "motor_fault") -> torch.Tensor:
    """Labels of :class:`MotorFaultInjector` (see :meth:`MotorFaultInjector.labels`), shape ``(N, 2 J + 4)``."""
    return env.event_manager.get_term_cfg(event_name).func.labels()


def add_motor_faults(
    env_cfg, cfg: MotorFaultInjectorCfg, actuator: str = "base_legs", event_name: str = "motor_fault"
) -> None:
    """Add the injector to ``env_cfg`` in place: faulty actuator, injector event, and a ``fault`` label group.

    The actuator runs in Isaac Lab (``use_newton_actuators = False``): Newton's native actuators accept only the
    stock actuator classes. Both execute the same DC-motor model.
    """
    env_cfg.sim.use_newton_actuators = False
    actuators = env_cfg.scene.robot.actuators
    actuators[actuator] = actuators[actuator].replace(class_type=FaultyDCMotor)
    setattr(
        env_cfg.events,
        event_name,
        EventTermCfg(
            func=MotorFaultInjector,
            mode="interval",
            # every policy step: the timer starts at 0 and is due again after each step
            interval_range_s=(0.0, 0.0),
            is_global_time=True,
            params={"cfg": cfg, "actuator": actuator},
        ),
    )

    @configclass
    class FaultLabelsCfg(ObservationGroupCfg):
        labels = ObservationTermCfg(func=motor_fault_labels, params={"event_name": event_name})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    env_cfg.observations.fault = FaultLabelsCfg()
