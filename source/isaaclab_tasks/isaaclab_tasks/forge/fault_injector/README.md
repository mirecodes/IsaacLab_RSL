# Fault injector templates

Random mid-episode faults that a task adds to its env cfg. Shared infrastructure like `forge/terrain`: tasks import
`injector.py` directly (not copied), so every study injects the same faults.

| Template | File | Fault |
| --- | --- | --- |
| Motor power out / power loss / torque-constant loss | [injector.py](injector.py) | one joint per episode, permanent from a random onset |
| Foot stuck (foot point / calf weld / calf slide) | [foot_stuck.py](foot_stuck.py) | one foot per episode, from a random onset until the robot pulls it free |

## Motor power out / power loss / torque-constant loss

```python
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults

add_motor_faults(env_cfg, MotorFaultInjectorCfg(p_fault=0.7, power_loss_health=(0.1, 0.6)))
```

`add_motor_faults` (robot with a `DCMotorCfg` actuator group, `base_legs` by default):

1. swaps the group's class for `FaultyDCMotor`, which scales the DC-motor output by a health $h_{e,j} \in [0, 1]$,
   so the gain and the torque limit (the DC torque-speed curve) both shrink by $h$:

   $$\tau_{e,j} = h_{e,j}\,\mathrm{clip}_{DC}\big(K_p (q^\ast - q) + K_d (\dot q^\ast - \dot q)\big) - [h_{e,j} = 0]\, d\, \dot q_{e,j}
   = \mathrm{clip}\big(h\,\tau_{PD},\ h\,\tau_{min}(\dot q),\ h\,\tau_{max}(\dot q)\big) - \dots$$

   It also reports the drive's motor-current reading in torque units, $c = K_{t,nom}\, i = c_{e,j}\,
   \mathrm{clip}_{DC}(\tau_{PD})$ (`FaultyDCMotor.motor_current`, observation term `motor_current`): the current
   loop drives $i$ to the command, so $c_{e,j} = 1$ for a torque-constant loss and $c_{e,j} = h_{e,j}$ for a power
   loss or power out (the supply limits the current).

2. adds the `motor_fault` event (`MotorFaultInjector`, `interval` mode, every policy step). At every reset it draws
   the episode's fault:

   | Quantity | Distribution (default) |
   | --- | --- |
   | fault or nominal | fault with `p_fault` = 0.7 |
   | type | `power_out` ($h = 0$) 0.5, `power_loss` 0.5 (`kt_loss` 0) |
   | joint | uniform over `joint_names` (all 12) |
   | onset | $U[2, 8]$ s after the reset, in policy steps |
   | power-loss health | $h \sim U[0.1, 0.6]$ |
   | torque-constant-loss health | $h \sim U[0, 0.8]$ (`kt_loss_health`; demagnetized magnets, worn gear) |
   | ramp | `ramp_steps` = 0 (immediate); else linear from 1 to $h$ |
   | passive damping $d$ | 0 N·m·s/rad |

3. adds the observation group `fault` (labels, never a policy input), per env, $J$ = 12:
   `[fault_mask (J), fault_type, health (J), onset_step, steps_since_onset, severity]`.
   `fault_type` 0 none / 1 power out / 2 power loss / 3 torque-constant loss; `steps_since_onset` = -1 before the onset.

**Timing.** The event runs after the step's resets and before its observations. It first stores the health the
finished physics step used (the labels of the observation computed next), then sets the health of the next step. So
the first faulty label is at episode step `onset_step`: the first observation that saw faulty physics. The schedule is
drawn from a `torch.Generator` seeded with `seed` (default: the env seed), so a seed gives the same faults.

## Decisions

- **Torque scaling, not a lower torque limit** (spec 3.1): a limit $h\,\tau_{max}$ leaves the joint unchanged whenever
  the demand is below it, so most "faulty" samples would be indistinguishable from nominal ones.
- **Actuator path** (spec 3.2): A2 uses an explicit `DCMotorCfg`, so option 1 (wrapper actuator) applies. Newton's
  native actuators (`use_newton_actuators = True`, used for training) accept only the stock actuator classes, so
  `add_motor_faults` sets `use_newton_actuators = False` and the DC motor runs in Isaac Lab. Both run the same
  DC-motor model; the nominal behaviour of the frozen policy under this path is checked in the detector study
  (`forge/fault_detector/motor_failure_detector/REPORT.md`).
- **The injector owns schedule, health and labels**; the actuator only reads `health`. One `interval` event at
  policy rate is enough: the fault starts at a policy step, as the labels are per policy step.
- **Torque-constant loss (`kt_loss`, 2026-10-26)**: the same torque law as a power loss (gain and limit scaled by
  $h$); only the current reading differs, which a detector with a motor-current input sees. Its $h$ comes from the
  same uniform draw as the power-loss one, so seeds without `kt_loss` give the schedules of earlier datasets.
- **Not implemented** (spec "cfg hook only"): recovery (`recover_prob`) and several faulty joints
  (`max_faulty_joints`). Add them in `MotorFaultInjector.reset` / `__call__` when a study needs them; the labels
  (`fault_mask` per joint) already allow several joints.

## Foot stuck

```python
from isaaclab_tasks.forge.fault_injector.foot_stuck import FootStuckInjectorCfg, add_foot_stuck

add_foot_stuck(env_cfg, FootStuckInjectorCfg(p_stuck=0.7, modes={"calf_slide": 1.0}))
```

`add_foot_stuck` (robot with foot and calf bodies and a contact sensor `contact_forces` on the feet) adds the
`foot_stuck` event (`FootStuckInjector`, `reset` mode) and the label group `foot_stuck`, and sets
`use_newton_actuators = False` (below). Both injectors can run in one env.

At every reset it draws the episode's event:

| Quantity | Distribution (default) |
| --- | --- |
| event or nominal | event with `p_stuck` = 0.7 |
| mode | `foot_point`, `calf_weld`, `calf_slide`, 1/3 each |
| foot | uniform over `feet` (FL, FR, RL, RR) |
| onset | $U[2, 8]$ s after the reset, in policy steps; the foot sticks at the first physics step after it with the foot in contact ($\lVert f_c \rVert > 1$ N) |
| escape threshold | $F_{esc} \sim U[100, 300]$ N |

**Anchor.** At the stick step it stores the foot origin $p_0$, its orientation $q_0$ and the calf axis
$a = (p_{knee} - p_0) / \lVert\cdot\rVert$ (world frame). From then on, at every physics step, the foot body gets a
wrench $W = [F; \tau]$ at its origin from a spring-damper on $x = [p - p_0;\ \log(q\, q_0^{-1})]$ and
$u = [v;\ \omega]$, integrated by backward Euler:

$$W = -K x - B u', \quad u' = u + \Delta t\, (a_o + A W), \quad B = D + \Delta t K
\;\Rightarrow\; (I + \Delta t\, B A)\, W = -K x - B (u + \Delta t\, a_o)$$

- $A = J M^{-1} J^T$: the foot's inverse effective mass (`body_link_jacobian_w`, `mass_matrix`).
- $a_o = (u - u_{prev}) / \Delta t - A W_{prev}$: the acceleration of all other forces (actuators, contact, gravity)
  in the last physics step (0 at the stick step), so a static load deflects the anchor as $K$ alone would.
- $K$ and $D$ per mode (below); $D$ has the form of $K$ with $d, d_c, d_t$.

`foot_point` pins the foot point (3 DoF) and lets the calf turn about it: $K = \mathrm{diag}(k I,\ 0)$.

The calf modes put the calf in a crevice. Every point $p + s a$, $s \in [0, L]$ (foot to knee, $L$ = calf length at
the onset), is pushed back across the calf with $k_c / L$ per unit length; that point's deviation is
$e - s S \theta$ with $x = [e; \theta]$, $S = [a]_\times$, $P_n = I - a a^T$. Integrated over the calf:

$$K = \int_0^L \frac{k_c}{L} \begin{bmatrix} I \\ s S \end{bmatrix} P_n \begin{bmatrix} I & -s S \end{bmatrix} ds
+ \begin{bmatrix} k_a\, a a^T & 0 \\ 0 & k_t\, a a^T \end{bmatrix}
= \begin{bmatrix} k_c P_n + k_a\, a a^T & -\tfrac{L}{2} k_c S \\ \tfrac{L}{2} k_c S & \tfrac{L^2}{3} k_c P_n + k_t\, a a^T \end{bmatrix}$$

The crevice pushes back evenly along the whole calf. Whatever the leg's actuators push across the calf (a force or a
turn of the calf) comes back as that distributed load, applied as its resultant wrench at the foot origin. A force of
100 N across the calf deflects it by 2 mm at either end and 0.5 mm at its middle. Along the calf, $k_a$ holds the
foot; about it, $k_t$ holds the twist.

| Mode | Locked | $k_a$ |
| --- | --- | --- |
| `calf_weld` | calf pose (6 DoF) | $k$ |
| `calf_slide` | calf pose except the slide out along $a$ (5 DoF) | $k$ when pushed deeper than $p_0$, 0 when pulled out |

Gains: $k = 5 \cdot 10^4$ N/m (4 mm per 200 N), $d = 400$ N·s/m; $k_c = 2 \cdot 10^5$ N/m (`calf_stiffness`, =
$4k$, so the calf's ends deflect as the foot point would), $d_c = 1600$ N·s/m; $k_t = 2000$ N·m/rad, $d_t = 20$
N·m·s/rad.

**Escape.** The foot is caught along the leg, so it comes out along the leg. The robot's force on the anchor is
$f = -F$; the foot comes free when $f$ stays for `escape_hold_s` (0.2 s = 40 physics steps) inside the cone around
$\hat c$ with half-angle $\theta$ (`escape_cone_angle_deg`, 5°), at or above the threshold:

$$f \cdot \hat c \ge \lVert f \rVert \cos\theta, \qquad \lVert f \rVert \ge F_{esc}$$

$\hat c$ is the calf axis $a$ at the onset (`escape_cone_axis = None`), or a fixed world vector. `calf_slide` exerts no
force along $a$ once pulled out, so a pull along the leg frees it by the slide-out, $a \cdot (p - p_0) \ge$
`slide_length` (5 cm). An escaped foot stays free for the rest of the episode.

**Reaction on the body.** The anchor is an external wrench on the foot body. Whatever the leg's actuators push
against it reaches the base through the joints, as the reaction of a foot held in the ground would. The check below
closes the whole robot's momentum balance with the anchor force at every anchored physics step.

**Labels** (group `foot_stuck`, per env, $F$ = 4 feet):
`[stuck_mask (F), mode, stuck_step, steps_since_stuck, escape_step, anchor_force (3)]`. `mode` 0 none / 1
foot_point / 2 calf_weld / 3 calf_slide, nonzero only while the foot is held; `stuck_step` is the first observation
that saw the anchor (as `onset_step` of the motor fault), `escape_step` the first after the escape (-1 until then);
`anchor_force` is $F$ of the last physics step.

### Decisions (foot stuck)

- **Soft anchor, not a joint.** A runtime weld would need an equality constraint per foot built into the model
  (MuJoCo `eq_active` toggling, anchor rewritten at the stick step) and reading the constraint force back for the
  escape; a spring-damper is a few torch lines, runs on any backend, and its force is the escape signal itself.
  The price is compliance: 4 mm per 200 N, a few cm under impacts.
- **Physics rate.** The anchor wraps the robot's `write_data_to_sim` and adds an instantaneous wrench before every
  physics step. Newton's native actuators fold the decimation loop into one call (`handles_decimation`), so a spring
  would be held for 20 ms; `add_foot_stuck` turns them off, as `add_motor_faults` does. Newton clears body forces after
  every substep, so an escaped foot gets no leftover force.
- **Implicit anchor** (measured with the check below, 2026-11-02). An explicit spring-damper
  ($F = -k e - d v$, $k = 2 \cdot 10^4$, $d = 100$) chattered: the foot's effective mass is about 0.5 kg in some
  directions, so $(k \Delta t^2 + 2 d \Delta t) / m \approx 3.75$, near the stability limit 4. Its force reversed in
  2 of every 4 physics steps and peaked at 13–37 kN, and a welded calf drifted 42 cm and threw 59 of 88 robots.
  Backward Euler with $A$ alone removed the chatter but held a static load with $(I + \Delta t B A)^{-1} K$, several
  times softer (foot point mean drift 4.3 cm). Adding $a_o$ gives the target stiffness: mean drift 0.5–0.7 cm.
  The cost is one $18 \times 18$ solve and a $6 \times 6$ solve per stuck env and physics step.
- **Crevice along the calf** (2026-11-02, user request). The calf modes first held the foot point with $k$ and the
  calf's rotation with a rotational spring $k_r = 2000$ N·m/rad. That holds the knee end about half as stiffly as the
  foot end ($k_r / L^2 \approx 2.6 \cdot 10^4$ vs $5 \cdot 10^4$ N/m), and the calf turned up to 10° (33° under a
  1500 N pull). The crevice spreads the hold evenly over the calf. As a rigid calf only feels the resultant, it is
  still one wrench at the foot origin, now with the stiffness of the distributed load.
- **Stick in contact only**: a foot caught mid-swing is not a stuck foot.
- **Not implemented**: several stuck events per episode, a stuck duration limit. Add them in `FootStuckInjector`
  when a study needs them.

### Check (foot stuck)

[scripts/forge/fault_injector/test_foot_stuck.py](../../../../../scripts/forge/fault_injector/test_foot_stuck.py):
frozen A2 robust teacher (`a2_robust_teacher_s0/model_2999.pt`) on the easiest terrain rows, no pushes, 300 envs,
every env stuck from 2 s, measured to 8 s (2026-11-02, Euler RTX 4090). Drift is the deviation in the locked
directions (for `calf_slide`, a slide deeper than $p_0$ counts) over all held policy steps; falls are terminations in
the first episode; the knee is the calf's far end, its deviation measured across the calf. Defaults (crevice along
the calf, calf-axis cone 5°, 0.2 s), run `foot_stuck_v12`:

| Run | Mode | Stuck / fell | Escaped (force / slide-out) | Foot drift mean / max | Knee across, mean / max | Longest in cone, p90 / max |
| --- | --- | --- | --- | --- | --- | --- |
| no pull, $F_{esc} \sim U[100, 300]$ N | foot_point | 102 / 20 | 0 | 0.71 / 7.3 cm | – | 0.015 / 0.040 s |
| | calf_weld | 88 / 4 | 0 | 0.58 / 4.8 cm | 0.49 / 2.6 cm | 0.011 / 0.040 s |
| | calf_slide | 101 / 2 | 69 (0 / 69), after 0.54 s | 0.46 / 2.7 cm | 0.44 / 2.4 cm | 0 / 0 s |
| pull along the calf to 1500 N, $F_{esc}$ = 150 N | foot_point | 102 / 22 | 26 (26 / 0), after 3.2 s | 2.0 / 11.2 cm | – | 0.20 / 0.20 s |
| | calf_weld | 88 / 4 | 15 (15 / 0), after 3.5 s | 2.1 / 8.4 cm | 0.51 / 2.9 cm | 0.20 / 0.20 s |
| | calf_slide | 101 / 1 | 101 (0 / 101), after 0.34 s | 0.47 / 4.2 cm | 0.46 / 1.8 cm | 0 / 0 s |
| pull across the calf to 1500 N | calf_weld | 88 / 4 | 0 | 2.2 / 8.2 cm | 1.17 / 3.6 cm | 0.005 / 0.025 s |

- With these defaults the frozen policy never frees `foot_point` or `calf_weld` by itself: its force on the anchor
  stays in the 5° cone above $F_{esc}$ for at most 0.04 s of the 0.2 s needed (with 0.5 s, `foot_stuck_v11`: at most
  0.06 s). A 1500 N pull along the leg frees 41 of 190 feet (0.5 s: 1); the leg's own forces across the axis keep
  the total out of the 5° cone (1500 N · tan 5° = 131 N) most of the time. `calf_slide` comes out only by the
  slide-out; it has no force along the leg to put in the cone. A pull across the leg frees nothing. Every force
  escape was inside the cone (max 4.7°) and above $F_{esc}$.
- Escape rate per half-angle (`test_foot_stuck.py --escape_force inf inf --sweep_deg ...`, `foot_stuck_v13`; a 20°
  run agrees): `foot_point` / `calf_weld` 0 % up to 10°, 1 % at 15–20°, 7 / 2 % at 25°, 10 / 3 % at 30°, 21 / 6 % at
  45°, 34 / 16 % at 60°; `calf_slide` 0 % by force at every angle. `render_foot_stuck.py` draws the cone translucent.
- Crevice: the knee end stays within 0.5 cm on average across the calf (max 2.6 cm; 3.6 cm under a 1500 N pull
  across). A 2.6 cm knee deviation is a 5° bend of the 27.5 cm calf, so most of the calf's rotation max (11°) is
  twist about its own axis.
- Reaction on the body: the anchor force's least-squares share in the whole robot's momentum balance is 1.014–1.015
  without a pull and 1.001–1.002 under the 1500 N pulls (0 would mean it stopped at the foot). The residual, ~105 N on
  anchored and ~60 N on other physics steps, does not grow with the anchor force (~106–111 N at ~1000 N): it is the
  contact sensor's error.
- All runs: drift p95 1.7 cm without a pull (3.7–3.9 cm under the 1500 N pulls, whose static deflection is 3 cm),
  about 0.01 anchor force reversals per held policy step (explicit law: about 2 in its 4 physics steps).
- Earlier defaults (45° cone around world up, 0.02 s hold; `foot_stuck_v8`): the policy freed 70–90 % of the feet in
  0.3–0.7 s; a pull up freed ≥ 98 % of the surviving feet. One foot wedged into the terrain (8.5 kN contact before the
  onset) stayed held by the terrain.
- `calf_slide` with a two-way slide (first version, all runs before `foot_stuck_v8`): the policy stands on a nearly
  flat calf (axis $a_z$ median 0.22, 5 % below 0.01), so the foot slid along the ground up to 1.5 m away from $p_0$.
  Holding the slide at $p_0$ when pushed deeper keeps it within 1.6 cm of $p_0$ on that side.
- 3 of 300 envs never stuck: their chosen foot had no contact after the onset (robot down, leg raised).
