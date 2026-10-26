# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check of the per-difficulty summary of ``campaign.py`` (``python test_campaign.py`` or pytest)."""

from campaign import summarize, terrains


def test_summarize():
    def episode(level, duration, outcome, failure):
        return {"terrain_level": level, "duration_s": duration, "outcome": outcome, "failure": failure}

    # level 0: a fall at 10 s, a time-out at 60 s, one still running; level 1 (read back from csv): one left its tile
    episodes = [
        episode(0, 10.0, "base_contact", True),
        episode(0, 60.0, "time_out", False),
        episode(0, 5.0, "running", False),
        episode("1", "20.0", "left_tile", "False"),
    ]
    s = summarize(episodes, (0.4, 1.0), 5)
    level0, level1 = s["levels"]
    assert (level0["episodes"], level0["successes"], level0["failures"], level0["running"]) == (2, 1, 1, 1)
    assert level0["success_rate"] == 0.5 and level0["mean_episode_s"] == 35.0 and level0["robot_s"] == 75.0
    assert level0["mean_time_to_failure_s"] == 10.0 and level0["difficulty"] == 0.4
    assert level1["success_rate"] == 1.0 and level1["mean_time_to_failure_s"] is None and level1["difficulty"] == 0.55
    assert s["total"]["episodes"] == 3 and s["total"]["failures"] == 1


def test_terrains():
    assert {"rough_field", "slip_field"} <= set(terrains()) and len(terrains()) == len(set(terrains())) == 10


if __name__ == "__main__":
    test_summarize()
    test_terrains()
    print("ok")
