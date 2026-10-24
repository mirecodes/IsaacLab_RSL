# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks of the geometry and run-length helpers of ``analyze.py`` (``python test_analyze.py`` or pytest)."""

import numpy as np
from analyze import hull_margin, sustained


def test_hull_margin():
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert np.isclose(hull_margin(np.array([0.5, 0.25]), square), 0.25)  # inside: distance to the nearest edge
    assert np.isclose(hull_margin(np.array([2.0, 0.5]), square), -1.0)  # outside: negative distance
    assert np.isclose(hull_margin(np.array([0.5, 1.0]), square[:2]), -1.0)  # two feet: distance to the segment
    assert np.isclose(hull_margin(np.array([3.0, 4.0]), square[:1]), -5.0)  # one foot
    assert np.isnan(hull_margin(np.zeros(2), square[:0]))  # flight


def test_sustained():
    mask = np.array([1, 1, 0, 1, 1, 1, 0], dtype=bool)
    assert sustained(mask, 3).tolist() == [False, False, False, False, False, True, False]


if __name__ == "__main__":
    test_hull_margin()
    test_sustained()
    print("ok")
