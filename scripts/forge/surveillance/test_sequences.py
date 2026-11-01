# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks of the taxonomy-method helpers of ``sequences.py`` and the alarm logic of ``detect.py`` (``python
test_sequences.py`` or pytest; needs torch for the ``classify`` import)."""

import numpy as np
from detect import alarms
from sequences import GROUPS, STATE, first_violation, kmedoids


def test_kmedoids():
    # two tight groups on a line: k-medoids splits them, whatever the seed
    x = np.r_[np.zeros(10), np.full(10, 5.0)] + np.linspace(0, 0.1, 20)
    d = np.abs(x[:, None] - x[None])
    lab = kmedoids(d, 2, seed=3)
    assert len(set(lab[:10])) == 1 and len(set(lab[10:])) == 1 and lab[0] != lab[-1]


def test_first_violation():
    z = np.zeros((2, 100, len(STATE)))
    z[0, 40:, STATE.index("slip_hind")] = 6.0  # slip first ...
    z[0, 60:, STATE.index("roll")] = 9.0  # ... then the roll
    z[1, 70:72, STATE.index("roll")] = 9.0  # 2 frames only: not sustained
    assert list(first_violation(z, dt=0.02)) == ["slip", "none"]
    assert "slip_hind" in GROUPS["slip"]


def test_alarms():
    s = np.zeros(200)
    s[10:13] = 1.0  # 3 frames: one alarm at the third
    s[20:30] = 1.0  # within the refractory 50 frames: none
    s[100:110] = 1.0
    assert alarms(s, 0.5) == [12, 102]


if __name__ == "__main__":
    test_kmedoids()
    test_first_violation()
    test_alarms()
    print("ok")
