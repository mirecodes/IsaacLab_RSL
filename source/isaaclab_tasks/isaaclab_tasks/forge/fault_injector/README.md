# Fault injector templates

Random mid-episode faults that a task adds to its env cfg. Shared infrastructure like `forge/terrain`: tasks import
`injector.py` directly (not copied), so every study injects the same faults.

| Template | File | Fault |
| --- | --- | --- |
| Motor power out / power loss | [injector.py](injector.py) | one joint per episode, permanent from a random onset |

## Motor power out / power loss

```python
from isaaclab_tasks.forge.fault_injector.injector import MotorFaultInjectorCfg, add_motor_faults

add_motor_faults(env_cfg, MotorFaultInjectorCfg(p_fault=0.7, power_loss_health=(0.1, 0.6)))
```

`add_motor_faults` (robot with a `DCMotorCfg` actuator group, `base_legs` by default):

1. swaps the group's class for `FaultyDCMotor`, which scales the DC-motor output by a health $h_{e,j} \in [0, 1]$:

   $$\tau_{e,j} = h_{e,j}\,\mathrm{clip}_{DC}\big(K_p (q^\ast - q) + K_d (\dot q^\ast - \dot q)\big) - [h_{e,j} = 0]\, d\, \dot q_{e,j}$$

2. adds the `motor_fault` event (`MotorFaultInjector`, `interval` mode, every policy step). At every reset it draws
   the episode's fault:

   | Quantity | Distribution (default) |
   | --- | --- |
   | fault or nominal | fault with `p_fault` = 0.7 |
   | type | `power_out` ($h = 0$) 0.5, `power_loss` 0.5 |
   | joint | uniform over `joint_names` (all 12) |
   | onset | $U[2, 8]$ s after the reset, in policy steps |
   | power-loss health | $h \sim U[0.1, 0.6]$ |
   | ramp | `ramp_steps` = 0 (immediate); else linear from 1 to $h$ |
   | passive damping $d$ | 0 N·m·s/rad |

3. adds the observation group `fault` (labels, never a policy input), per env, $J$ = 12:
   `[fault_mask (J), fault_type, health (J), onset_step, steps_since_onset, severity]`.
   `fault_type` 0 none / 1 power out / 2 power loss; `steps_since_onset` = -1 before the onset.

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
- **Not implemented** (spec "cfg hook only"): recovery (`recover_prob`) and several faulty joints
  (`max_faulty_joints`). Add them in `MotorFaultInjector.reset` / `__call__` when a study needs them; the labels
  (`fault_mask` per joint) already allow several joints.
