# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check of the privileged per-frame labels of ``classify.py`` (``python test_classify.py`` or pytest; needs torch)."""

import numpy as np
from classify import frame_labels


def test_frame_labels():
    n, feet = 10, 4
    flags = {k: np.zeros((n, feet), dtype=bool) for k in ("terrain_low_mu", "terrain_in_sand", "terrain_collapse")}
    flags["terrain_low_mu"][1:4, 2] = True
    flags["terrain_in_sand"][3:6, 0] = True
    flags["terrain_collapse"][7, 1] = True
    # sand over slip, a collapse labels `label` frames from it
    assert list(frame_labels(flags, label=2)) == ["normal"] + ["foot_slip"] * 2 + ["sand_sink"] * 3 + ["normal"] + [
        "crust_collapse"
    ] * 2 + ["normal"]


if __name__ == "__main__":
    test_frame_labels()
    print("ok")
