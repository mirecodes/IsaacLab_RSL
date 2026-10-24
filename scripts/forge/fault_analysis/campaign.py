# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy trained A2 policies on every terrain type, collect their falls, render some, report.

One Euler job per (policy, terrain type) on ``IsaacContrib-Velocity-A2Deploy-<Type>-<Policy>`` (difficulty rows 1-3,
20 robots on random tiles, steering commands; ``forge/velocity/a2_deploy``): ``collect.py`` until ``--failures`` falls
or ``--minutes`` of sim time, then ``render.py --pick`` (videos evenly spread over the play). Standard library only,
so it runs on the Mac::

    python scripts/forge/fault_analysis/campaign.py enqueue --name terrain_v1        # queue 2 x 9 jobs
    euler fetch fault_analysis/terrain_v1
    python scripts/forge/fault_analysis/campaign.py report results/euler/isaaclab3/fault_analysis/terrain_v1
"""

import argparse
import ast
import json
import os
import shlex
import subprocess
from collections import Counter

SCRIPTS = "scripts/forge/fault_analysis"
LOGS = "/workspace/isaaclab/logs"
CHECKPOINTS = {
    "Robust": f"{LOGS}/rsl_rl/a2_study_robust_teacher/2026-10-23_11-54-58_a2_robust_teacher_s0/model_2999.pt",
    "Energy": (
        f"{LOGS}/rsl_rl/a2_study_energy_teacher_rough/2026-10-23_13-58-39_a2_energy_teacher_rough_s0/model_2999.pt"
    ),
}
TERRAIN_PACKAGE = os.path.join(
    os.path.dirname(__file__), "../../../source/isaaclab_tasks/isaaclab_tasks/forge/terrain/__init__.py"
)


def terrains() -> list[str]:
    """``TERRAINS`` of ``forge/terrain`` (read from its source: importing it needs Isaac Lab)."""
    with open(TERRAIN_PACKAGE) as f:
        tree = ast.parse(f.read())
    return next(
        list(ast.literal_eval(node.value))
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "TERRAINS"
    )


def task_id(terrain: str, policy: str) -> str:
    """As ``forge/velocity/a2_deploy/__init__.py:task_id``."""
    return f"IsaacContrib-Velocity-A2Deploy-{terrain.title().replace('_', '')}-{policy}"


def summarize(episodes: list[dict], difficulty_range=None, num_rows: int = 0) -> dict:
    """Per difficulty row and in total: finished episodes, successes (any end but a failure term: time-out or leaving
    the tile), failures, success rate, mean episode and time to failure [s], robot time [s]. ``episodes`` are the
    rows of ``episodes.csv`` (values as written or as read back as strings)."""

    def stats(rows):
        done = [r for r in rows if r["outcome"] != "running"]
        falls = [r for r in done if str(r["failure"]) == "True"]

        def mean(rs):
            return round(sum(float(r["duration_s"]) for r in rs) / len(rs), 2) if rs else None

        return {
            "episodes": len(done),
            "successes": len(done) - len(falls),
            "failures": len(falls),
            "running": len(rows) - len(done),
            "success_rate": round(1 - len(falls) / len(done), 4) if done else None,
            "mean_episode_s": mean(done),
            "mean_time_to_failure_s": mean(falls),
            "robot_s": round(sum(float(r["duration_s"]) for r in rows), 1),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
        }

    summary = {"levels": []}
    for level in sorted({int(r["terrain_level"]) for r in episodes}):
        row = {"level": level, **stats([r for r in episodes if int(r["terrain_level"]) == level])}
        if difficulty_range is not None and num_rows:
            lo, hi = difficulty_range
            row["difficulty"] = round(lo + (hi - lo) * level / max(num_rows - 1, 1), 4)
        summary["levels"].append(row)
    summary["total"] = stats(episodes)
    return summary


def enqueue(args: argparse.Namespace) -> None:
    reuse = []
    for policy in args.policies:
        for terrain in args.terrains:
            out = f"{LOGS}/fault_analysis/{args.name}/{policy}/{terrain}"
            collect = [
                "python", f"{SCRIPTS}/collect.py", "--task", task_id(terrain, policy),
                "--checkpoint", CHECKPOINTS[policy], "--output_dir", out,
                # A2 policy step 0.02 s
                "--max_events", str(args.failures), "--max_steps", str(round(args.minutes * 60 / 0.02)),
            ]  # fmt: skip
            render = ["python", f"{SCRIPTS}/render.py", out, "--pick", str(args.videos)]
            script = f"{shlex.join(collect)} && {shlex.join(render)}"
            cmd = ["euler", "enqueue", *reuse, f"{args.name}_{policy}_{terrain}", "--", "bash", "-c", script]
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
            # one code snapshot for the whole campaign
            reuse = ["--reuse-code"]


def report(args: argparse.Namespace) -> None:
    """``report.md`` (and stdout): one row per (policy, terrain, difficulty row), from each ``summary.json``."""
    lines = [
        "| policy | terrain | level | d | episodes | success | fail | success rate | mean episode s"
        " | mean time to fail s | play sim s | play wall s | stopped by | videos |",
        "|" + " --- |" * 14,
    ]
    for policy in sorted(os.listdir(args.root)):
        for terrain in sorted(os.listdir(os.path.join(args.root, policy))):
            path = os.path.join(args.root, policy, terrain)
            if not os.path.isfile(os.path.join(path, "summary.json")):
                lines.append(f"| {policy} | {terrain} | not finished |" + " |" * 11)
                continue
            with open(os.path.join(path, "summary.json")) as f:
                s = json.load(f)
            videos = len(os.listdir(os.path.join(path, "videos"))) if os.path.isdir(os.path.join(path, "videos")) else 0
            for row in [*s["levels"], {**s["total"], "level": "all"}]:
                level = row["level"] + 1 if isinstance(row["level"], int) else "all"
                lines.append(
                    f"| {policy} | {terrain} | {level} | {row.get('difficulty', '')} | {row['episodes']}"
                    f" | {row['successes']} | {row['failures']} | {row['success_rate']} | {row['mean_episode_s']}"
                    f" | {row['mean_time_to_failure_s']} | {s['sim_s']} | {s['wall_s']} | {s['stopped_by']}"
                    f" | {videos} |"
                )
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.root, "report.md"), "w") as f:
        f.write(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("enqueue", help="Queue one collect + render job per (policy, terrain) on Euler.")
    q.add_argument("--name", required=True, help="Campaign name: logs/fault_analysis/<name>/<policy>/<terrain>.")
    q.add_argument("--policies", nargs="+", default=list(CHECKPOINTS), choices=list(CHECKPOINTS))
    q.add_argument("--terrains", nargs="+", default=None, help="Default: all of forge/terrain.")
    q.add_argument("--failures", type=int, default=200, help="Stop a play after this many falls.")
    q.add_argument("--minutes", type=float, default=30.0, help="Stop a play after this much sim time.")
    q.add_argument("--videos", type=int, default=10, help="Falls rendered per play.")
    q.add_argument("--dry_run", action="store_true", help="Print the euler commands only.")
    r = sub.add_parser("report", help="Tabulate the summaries of a fetched campaign.")
    r.add_argument("root", help="Local campaign directory (results/euler/<profile>/fault_analysis/<name>).")
    args = parser.parse_args()
    if args.command == "enqueue":
        args.terrains = args.terrains or terrains()
        enqueue(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
