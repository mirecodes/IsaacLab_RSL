# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks of the balance-loss alignment and the failure-mode rules of ``taxonomy.py`` (``python test_taxonomy.py``
or pytest)."""

import numpy as np
from taxonomy import LABELS, PRIVILEGED, balance_loss, failure_mode


def tilt_gravity(deg: list[float]) -> dict:
    """Projected gravity tilted about the body y axis by each angle."""
    rad = np.radians(deg)
    return {"projected_gravity_b": np.stack([np.sin(rad), np.zeros_like(rad), -np.cos(rad)], 1)}


def test_balance_loss():
    # upright, a 10 deg wobble, then 30 deg from frame 6 on: the fall starts at 6, not at the wobble
    e = tilt_gravity([0, 0, 25, 0, 0, 0, 30, 40, 60, 90])
    assert balance_loss(e, 9, dt=0.02) == 6
    # never past 20 deg: an upright base hit, aligned at the contact itself
    assert balance_loss(tilt_gravity([0, 5, 10, 15]), 3, dt=0.02) == 3
    # tilted for longer than the span: clipped to it
    assert balance_loss(tilt_gravity([45] * 200), 199, dt=0.02, span_s=1.0) == 199 - round(1.0 / 0.02)


def test_failure_mode():
    n, dt = 200, 0.02
    blank = {name: np.zeros(n, dtype=bool) for name in LABELS + PRIVILEGED}
    assert failure_mode(blank, n - 1, n - 1, dt) == "upright_hit"
    assert failure_mode(blank, 150, n - 1, dt) == "other"
    slide = blank | {"slip_2feet": np.arange(n) > 120}
    assert failure_mode(slide, 150, n - 1, dt) == "slide"
    # a single leg touch is common in normal walking; a sustained one (>= 20 % of 0.5 s) is a collision
    touch = blank | {"A_leg_contact": np.arange(n) == 140}
    assert failure_mode(touch, 150, n - 1, dt) == "other"
    sustained = blank | {"A_leg_contact": (np.arange(n) > 140) & (np.arange(n) <= 150)}
    assert failure_mode(sustained, 150, n - 1, dt) == "leg_collision"
    # priority: a sliding fall with a sunk foot is a slide
    both = slide | {"sink_5cm": np.arange(n) == 145}
    assert failure_mode(both, 150, n - 1, dt) == "slide"


if __name__ == "__main__":
    test_balance_loss()
    test_failure_mode()
    print("ok")
