# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foot-stuck injector: once per episode, one random foot gets stuck where it stands, until the robot pulls it free
with enough force inside an escape cone.

* :class:`FootStuckInjector` (a ``reset`` event term) samples each episode's event at reset and runs a stiff
  spring-damper anchor on the foot at every physics step, written as an instantaneous wrench before the robot's
  :meth:`write_data_to_sim`. Three modes: the foot point is pinned (``foot_point``), the calf is welded
  (``calf_weld``), or the calf is welded but slides freely out along its own axis (``calf_slide``).
* :func:`foot_stuck_labels` exposes the labels as an observation term.
* :func:`add_foot_stuck` wires both into an env cfg.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_box_minus, skew_symmetric_matrix

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

STUCK_MODES = ("none", "foot_point", "calf_weld", "calf_slide")
"""Stuck-mode labels, by index. ``foot_point``: the foot point is pinned, the calf turns freely about it (3 DoF
locked). ``calf_weld``: the calf (and the foot on it) is locked in place (6 DoF). ``calf_slide``: as ``calf_weld``,
but the calf slides freely out along its own axis, from the foot toward the knee (5 DoF; not deeper than where it
stuck)."""

WAITING, STUCK, ESCAPED = 0, 1, 2
"""Event state of an env: not (yet) stuck, stuck, escaped."""


@configclass
class FootStuckInjectorCfg:
    """Per-episode stuck event (sampled at every reset) and its anchor and escape law."""

    p_stuck: float = 0.7
    """Probability that an episode gets a stuck event; the others stay nominal to the end."""

    modes: dict[str, float] = {"foot_point": 1.0, "calf_weld": 1.0, "calf_slide": 1.0}
    """Relative probability of each mode (keys from :data:`STUCK_MODES`)."""

    feet: list[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    """Candidate foot bodies; the stuck one is drawn uniformly. The anchor point is the foot body's origin."""

    calves: list[str] = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    """Calf body of each foot (same order); the calf axis runs from the foot origin to the calf origin (knee)."""

    onset_time_s: tuple[float, float] = (2.0, 8.0)
    """Onset time after the reset, uniform [s]; quantized to policy steps. The foot sticks at the first physics step
    from the onset on at which it is in contact."""

    contact_threshold: float = 1.0
    """Net contact force [N] above which the foot counts as in contact."""

    stiffness: float = 5.0e4
    """Point stiffness ``k`` [N/m] of the foot (``foot_point``) and along the calf axis (calf modes): 4 mm per 200 N."""

    damping: float = 400.0
    """Point damping ``d`` [N*s/m]: about critical for a 1 kg effective mass."""

    calf_stiffness: float = 2.0e5
    """Calf modes: total stiffness ``k_c`` [N/m] of the crevice across the calf, spread evenly from the foot to the
    knee. A load across the calf at its end deflects it as ``k_c / 4`` = ``k`` would; a load at its middle, 4x less."""

    calf_damping: float = 1600.0
    """Calf modes: total damping ``d_c`` [N*s/m] across the calf, spread like ``k_c``."""

    twist_stiffness: float = 2000.0
    """Calf modes: stiffness ``k_t`` [N*m/rad] against twisting the calf about its own axis: 3 deg per 100 N*m."""

    twist_damping: float = 20.0
    """Calf modes: damping ``d_t`` [N*m*s/rad] against twisting the calf about its own axis."""

    escape_force: tuple[float, float] = (100.0, 300.0)
    """Escape threshold ``F_esc`` [N], uniform per episode: the robot's force on the anchor must reach it."""

    escape_cone_axis: tuple[float, float, float] | None = None
    """Axis of the escape cone: the direction the foot must be pulled to come free. None: the calf axis at the onset
    (foot to knee), as for a foot caught along the leg; else a world-frame vector (normalized)."""

    escape_cone_angle_deg: float = 5.0
    """Half-angle of the escape cone [deg]."""

    escape_hold_s: float = 0.2
    """The force must stay in the cone and above ``F_esc`` this long [s], in physics steps (at least one)."""

    slide_length: float = 0.05
    """``calf_slide``: the foot also comes free once pulled out this far along the calf axis [m]."""

    seed: int | None = None
    """Seed of the event schedule; None uses the env seed."""


class FootStuckInjector(ManagerTermBase):
    r"""Samples one stuck event per episode at reset and runs its anchor at every physics step (see module doc).

    At the first physics step from ``onset_step`` on with the chosen foot in contact, the anchor stores the foot
    position :math:`p_0`, its orientation :math:`q_0` and the calf axis :math:`a` (foot to knee). From then on the
    foot body gets a wrench :math:`W = [F; \tau]` at its origin :math:`p`, from a spring-damper on the deviation
    :math:`x = [p - p_0;\ \log(q q_0^{-1})]` and the velocity :math:`u = [v;\ \omega]`, integrated by backward Euler:

    .. math::

        W = -K x - B u', \quad u' = u + \Delta t\, (a_o + A W), \quad B = D + \Delta t K
        \quad\Rightarrow\quad (I + \Delta t\, B A)\, W = -K x - B (u + \Delta t\, a_o)

    with the foot's inverse effective mass :math:`A = J M^{-1} J^T` (Jacobian at the foot origin, joint-space mass
    matrix) and the acceleration of all other forces (actuators, contacts, gravity) as seen in the last physics step,
    :math:`a_o = (u - u_{prev}) / \Delta t - A W_{prev}` (0 at the first anchored step). Without :math:`a_o` the
    anchor would hold a static load with the stiffness :math:`(I + \Delta t B A)^{-1} K`, several times below
    :math:`K` for a light foot; with it, a static load deflects the anchor as :math:`K` alone would. An explicit
    spring-damper chatters: the foot's effective mass is about 0.5 kg in some directions, so
    :math:`(k \Delta t^2 + 2 d \Delta t) / m` nears 4.

    ``foot_point``: :math:`K = \mathrm{diag}(k I, 0)`. Calf modes: the calf sits in a crevice that pushes back on
    every point :math:`p + s a`, :math:`s \in [0, L]` (foot to knee, :math:`L` the calf length at the onset), across
    the calf with :math:`k_c / L` per unit length. The point's deviation is :math:`e - s [a]_\times \theta`
    (:math:`x = [e; \theta]`), so with :math:`P_n = I - a a^T` and :math:`S = [a]_\times`:

    .. math::

        K = \int_0^L \frac{k_c}{L} \begin{bmatrix} I \\ s S \end{bmatrix} P_n \begin{bmatrix} I & -s S \end{bmatrix} ds
          + \begin{bmatrix} k_a a a^T & 0 \\ 0 & k_t a a^T \end{bmatrix}
          = \begin{bmatrix} k_c P_n + k_a a a^T & -\tfrac{L}{2} k_c S \\ \tfrac{L}{2} k_c S &
            \tfrac{L^2}{3} k_c P_n + k_t a a^T \end{bmatrix}

    The crevice's reaction is spread evenly along the calf, so what the leg's actuators push across the calf comes back
    to the robot as that distributed load (its resultant wrench at the foot origin). Along the calf, :math:`k_a = k`
    holds the foot (``calf_weld``; ``calf_slide`` only when pushed deeper than :math:`p_0`, free when pulled out);
    :math:`k_t` holds the twist about the calf. :math:`D` has the same form with :math:`d, d_c, d_t`. The foot comes
    free when the robot's force on the anchor, :math:`f = -F`, has stayed for ``escape_hold_s`` inside the cone
    :math:`f \cdot \hat c \ge |f| \cos\theta` with :math:`|f| \ge F_{esc}`, or (``calf_slide``) once
    :math:`a \cdot (p - p_0) \ge` ``slide_length``. The cone axis :math:`\hat c` is the calf axis :math:`a` of the
    onset by default. In ``calf_slide`` the anchor exerts no force along :math:`a` once the foot is pulled out, so the
    pull along the leg frees it by the slide-out.

    The anchor is an external wrench on the foot body, so whatever the leg's actuators push against it reaches the
    base through the joints, as the reaction of a foot held in the ground would.

    The anchor runs inside a wrapper of the robot's ``write_data_to_sim``, so it needs the Python decimation loop
    (``use_newton_actuators = False``, set by :func:`add_foot_stuck`): a stiff spring held for a whole policy step
    would be unstable.

    Params: ``cfg`` (:class:`FootStuckInjectorCfg`), ``asset_name``, ``sensor_name`` (contact sensor on the feet).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        c: FootStuckInjectorCfg = cfg.params["cfg"]
        self.stuck_cfg = c
        self.robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.sensor = env.scene.sensors[cfg.params.get("sensor_name", "contact_forces")]
        self.foot_body_ids = self.robot.find_bodies(c.feet, preserve_order=True)[0]
        self.foot_ids = torch.tensor(self.foot_body_ids, device=self.device)
        self.calf_ids = torch.tensor(self.robot.find_bodies(c.calves, preserve_order=True)[0], device=self.device)
        self.sensor_ids = torch.tensor(self.sensor.find_bodies(c.feet, preserve_order=True)[0], device=self.device)
        self.mode_probs = torch.tensor([c.modes.get(name, 0.0) for name in STUCK_MODES[1:]], device=self.device)
        seed = c.seed if c.seed is not None else (env.cfg.seed or 0)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.onset_range = [round(t / env.step_dt) for t in c.onset_time_s]
        self.hold_steps = max(1, round(c.escape_hold_s / env.physics_dt))
        self.cone_cos = math.cos(math.radians(c.escape_cone_angle_deg))
        n = self.num_envs
        self.rows = torch.arange(n, device=self.device)
        self.mode = torch.zeros(n, dtype=torch.long, device=self.device)
        """Index into :data:`STUCK_MODES` (0: no event this episode)."""
        self.foot = torch.zeros(n, dtype=torch.long, device=self.device)
        """Stuck foot, as an index into ``cfg.feet``."""
        self.onset_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.escape_force = torch.full((n,), math.inf, device=self.device)
        self.state = torch.zeros(n, dtype=torch.long, device=self.device)
        """:data:`WAITING`, :data:`STUCK` or :data:`ESCAPED`."""
        self.stuck_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        """Episode step of the first observation that saw the stuck foot."""
        self.escape_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        """Episode step of the first observation after the escape."""
        self.hold = torch.zeros(n, dtype=torch.long, device=self.device)
        self.anchor_pos = torch.zeros(n, 3, device=self.device)
        self.anchor_quat = torch.zeros(n, 4, device=self.device)
        self.axis = torch.zeros(n, 3, device=self.device)
        """Calf axis at the onset, foot to knee (world frame)."""
        self.calf_length = torch.zeros(n, device=self.device)
        """Foot-to-knee distance ``L`` at the onset [m]."""
        self.cone_axis = torch.zeros(n, 3, device=self.device)
        """Escape-cone axis of the event (world frame): the calf axis at the onset or ``cfg.escape_cone_axis``."""
        self.force = torch.zeros(n, 3, device=self.device)
        """Anchor force on the robot ``F`` of the last physics step [N] (world frame, 0 when not stuck)."""
        self.prev_vel = torch.zeros(n, 6, device=self.device)
        """Foot velocity ``u`` at the last anchored physics step."""
        self.prev_anchor_accel = torch.zeros(n, 6, device=self.device)
        """Acceleration ``A W`` the anchor wrench of the last physics step produced."""
        self.escape_push = torch.zeros(n, 3, device=self.device)
        """The robot's force on the anchor ``-F`` at a force escape [N] (world frame; 0 for a slide-out)."""
        # per physics step, before the robot writes its wrenches and actuator commands
        write_data_to_sim = self.robot.write_data_to_sim

        def write_with_anchor():
            self.apply_anchor()
            write_data_to_sim()

        self.robot.write_data_to_sim = write_with_anchor

    def reset(self, env_ids=None):
        ids = self.rows[env_ids if env_ids is not None else slice(None)]
        k, g, c = len(ids), self.generator, self.stuck_cfg
        rand = lambda: torch.rand(k, device=self.device, generator=g)  # noqa: E731
        has_event = rand() < c.p_stuck
        mode = torch.multinomial(self.mode_probs, k, replacement=True, generator=g) + 1
        self.mode[ids] = torch.where(has_event, mode, 0)
        self.foot[ids] = torch.randint(len(c.feet), (k,), device=self.device, generator=g)
        lo, hi = self.onset_range
        self.onset_step[ids] = torch.where(
            has_event, torch.randint(lo, hi + 1, (k,), device=self.device, generator=g), -1
        )
        lo, hi = c.escape_force
        self.escape_force[ids] = lo + (hi - lo) * rand()
        self.state[ids] = WAITING
        self.stuck_step[ids] = -1
        self.escape_step[ids] = -1
        self.hold[ids] = 0
        self.force[ids] = 0.0
        self.escape_push[ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, env_ids, cfg, asset_name: str = "robot", sensor_name: str = ""):
        pass  # the event is sampled in reset() and runs in apply_anchor()

    def apply_anchor(self):
        """One physics step: start due events, add the anchor wrench of the stuck feet, release the escaped ones."""
        c, d, step = self.stuck_cfg, self.robot.data, self._env.episode_length_buf
        body = self.foot_ids[self.foot]
        pos = d.body_link_pos_w.torch[self.rows, body]
        quat = d.body_link_quat_w.torch[self.rows, body]
        due = (self.state == WAITING) & (self.mode > 0) & (step >= self.onset_step)
        if due.any():
            contact = self.sensor.data.net_forces_w.torch[self.rows, self.sensor_ids[self.foot]].norm(dim=-1)
            ids = (due & (contact > c.contact_threshold)).nonzero().squeeze(-1)
            knee = d.body_link_pos_w.torch[ids, self.calf_ids[self.foot[ids]]]
            self.anchor_pos[ids] = pos[ids]
            self.anchor_quat[ids] = quat[ids]
            self.axis[ids] = torch.nn.functional.normalize(knee - pos[ids], dim=-1)
            self.calf_length[ids] = (knee - pos[ids]).norm(dim=-1)
            if c.escape_cone_axis is None:
                self.cone_axis[ids] = self.axis[ids]
            else:
                self.cone_axis[ids] = torch.nn.functional.normalize(
                    torch.tensor(c.escape_cone_axis, device=self.device), dim=0
                )
            # no acceleration estimate yet at the first anchored step
            self.prev_vel[ids] = torch.cat(
                [d.body_link_lin_vel_w.torch[ids, body[ids]], d.body_link_ang_vel_w.torch[ids, body[ids]]], dim=-1
            )
            self.prev_anchor_accel[ids] = 0.0
            self.stuck_step[ids] = step[ids] + 1
            self.state[ids] = STUCK
        self.force.zero_()
        ids = (self.state == STUCK).nonzero().squeeze(-1)
        if len(ids) == 0:
            return
        b, a, mode = body[ids], self.axis[ids], self.mode[ids]
        e = pos[ids] - self.anchor_pos[ids]
        travel = (e * a).sum(-1)
        # deviation x and velocity u of the foot origin, [linear; angular] (world frame)
        x = torch.cat([e, quat_box_minus(quat[ids], self.anchor_quat[ids])], dim=-1)
        u = torch.cat([d.body_link_lin_vel_w.torch[ids, b], d.body_link_ang_vel_w.torch[ids, b]], dim=-1)
        # K and D (class doc): a point spring for foot_point; for the calf modes a crevice spread along the calf, plus
        # the axial lock and the twist. The calf only slides out: pushed deeper than the stick point, it is held (a
        # calf near horizontal would otherwise slide along the ground)
        eye = torch.eye(3, device=self.device).expand(len(ids), 3, 3)
        axial = a.unsqueeze(2) * a.unsqueeze(1)
        across = eye - axial
        skew = skew_symmetric_matrix(a)
        point = (mode == 1).view(-1, 1, 1).float()
        calf = (mode >= 2).view(-1, 1, 1).float()
        locked = ((mode == 2) | ((mode == 3) & (travel <= 0.0))).view(-1, 1, 1).float()
        length = self.calf_length[ids].view(-1, 1, 1)

        def law(k_point: float, k_calf: float, k_twist: float) -> torch.Tensor:
            m = torch.zeros(len(ids), 6, 6, device=self.device)
            m[:, :3, :3] = point * k_point * eye + calf * (k_calf * across + locked * k_point * axial)
            m[:, :3, 3:] = -calf * (length / 2) * k_calf * skew
            m[:, 3:, :3] = calf * (length / 2) * k_calf * skew
            m[:, 3:, 3:] = calf * (length**2 / 3 * k_calf * across + k_twist * axial)
            return m

        stiff = law(c.stiffness, c.calf_stiffness, c.twist_stiffness)
        damp = law(c.damping, c.calf_damping, c.twist_damping)
        # backward Euler (class doc): the spring-damper acts on the velocity it produces, with the foot's inverse
        # effective mass A = J M^-1 J^T and the other forces' acceleration seen in the last step
        jac = d.body_link_jacobian_w.torch[ids, b - int(self.robot.is_fixed_base)]
        inv_mass = jac @ torch.linalg.solve(d.mass_matrix.torch[ids], jac.transpose(1, 2))
        dt = self._env.physics_dt
        other = (u - self.prev_vel[ids]) / dt - self.prev_anchor_accel[ids]
        coupling = damp + dt * stiff
        lhs = torch.eye(6, device=self.device) + dt * coupling @ inv_mass
        rhs = -(stiff @ x.unsqueeze(2) + coupling @ (u + dt * other).unsqueeze(2))
        wrench = torch.linalg.solve(lhs, rhs).squeeze(2)
        self.prev_vel[ids] = u
        self.prev_anchor_accel[ids] = (inv_mass @ wrench.unsqueeze(2)).squeeze(2)
        force = wrench[:, :3]
        # the force acts at the foot origin; the composer takes forces at the CoM
        lever = pos[ids] - d.body_com_pos_w.torch[ids, b]
        torque = wrench[:, 3:] + torch.linalg.cross(lever, force)
        # escape: the robot's force on the anchor, held inside the cone above the threshold
        push = -force
        magnitude = push.norm(dim=-1)
        along = (push * self.cone_axis[ids]).sum(-1)
        in_cone = (along >= self.cone_cos * magnitude) & (magnitude >= self.escape_force[ids])
        self.hold[ids] = torch.where(in_cone, self.hold[ids] + 1, 0)
        by_force = self.hold[ids] >= self.hold_steps
        escaped = by_force | ((mode == 3) & (travel >= c.slide_length))
        out = ids[escaped]
        self.state[out] = ESCAPED
        self.escape_step[out] = step[out] + 1
        self.escape_push[out] = torch.where(by_force[escaped].unsqueeze(1), push[escaped], 0.0)
        # the escaped feet are free from this physics step on
        held = ~escaped
        forces = torch.zeros(self.num_envs, len(self.foot_body_ids), 3, device=self.device)
        torques = torch.zeros_like(forces)
        forces[ids[held], self.foot[ids[held]]] = force[held]
        torques[ids[held], self.foot[ids[held]]] = torque[held]
        self.force[ids[held]] = force[held]
        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=forces, torques=torques, body_ids=self.foot_body_ids, is_global=True
        )

    def labels(self) -> torch.Tensor:
        """``[stuck_mask (F), mode, stuck_step, steps_since_stuck, escape_step, anchor_force (3)]`` per env.

        ``stuck_mask`` (one-hot over ``cfg.feet``) and ``mode`` are nonzero while the foot is held, and
        ``steps_since_stuck`` is -1 otherwise; ``stuck_step`` and ``escape_step`` are -1 until they happen.
        ``anchor_force`` is the anchor's force on the robot in the last physics step (world frame) [N].
        """
        stuck = self.state == STUCK
        mask = torch.zeros(self.num_envs, len(self.foot_body_ids), device=self.device)
        mask[self.rows, self.foot] = stuck.float()
        since = torch.where(stuck, self._env.episode_length_buf - self.stuck_step, -1)
        columns = [
            mask,
            torch.where(stuck, self.mode, 0).unsqueeze(1).float(),
            self.stuck_step.unsqueeze(1).float(),
            since.unsqueeze(1).float(),
            self.escape_step.unsqueeze(1).float(),
            self.force,
        ]
        return torch.cat(columns, dim=1)


def foot_stuck_labels(env: ManagerBasedRLEnv, event_name: str = "foot_stuck") -> torch.Tensor:
    """Labels of :class:`FootStuckInjector` (see :meth:`FootStuckInjector.labels`), shape ``(N, F + 7)``."""
    return env.event_manager.get_term_cfg(event_name).func.labels()


def add_foot_stuck(
    env_cfg, cfg: FootStuckInjectorCfg, event_name: str = "foot_stuck", sensor_name: str = "contact_forces"
) -> None:
    """Add the injector to ``env_cfg`` in place: the injector event and a label group named ``event_name``.

    Sets ``use_newton_actuators = False``: the anchor runs at every physics step, which needs the Python decimation
    loop (see :class:`FootStuckInjector`).
    """
    env_cfg.sim.use_newton_actuators = False
    setattr(
        env_cfg.events,
        event_name,
        EventTermCfg(func=FootStuckInjector, mode="reset", params={"cfg": cfg, "sensor_name": sensor_name}),
    )

    @configclass
    class FootStuckLabelsCfg(ObservationGroupCfg):
        labels = ObservationTermCfg(func=foot_stuck_labels, params={"event_name": event_name})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    setattr(env_cfg.observations, event_name, FootStuckLabelsCfg())
