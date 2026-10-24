# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the Harsh World without a simulator: ``world_meta.yaml``, a top-down PNG, and the height map.

Runs on any machine with Isaac Lab installed (no GPU, no Kit), e.g. in the local container::

    python scripts/reinforcement_learning/harsh_world/build_world.py --out logs/harsh_world/build
    python scripts/reinforcement_learning/harsh_world/build_world.py --full --seed 3   # all 10 properties

``--plot <world_meta.yaml>`` only redraws an existing world (e.g. one written by the importer during a run).
"""

import argparse
import os
import time

import numpy as np
import yaml

ZONE_COLORS = {"rigid": "#7f7f7f", "slip": "#1f77b4", "sand": "#e3b04b"}


def plot_world(meta: dict, path: str, heights: np.ndarray | None = None, faults: np.ndarray | None = None) -> None:
    """Top-down map: cell borders, property names, zone colors, sand patches, rocks, and optional fault points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    rows, cols = meta["num_rows"], meta["num_cols"]
    sx, sy = meta["cell_size"]
    x0, y0 = meta["corner"]
    # world x (rows) is drawn vertically, y (cols) horizontally
    fig, ax = plt.subplots(figsize=(2.2 * cols + 2, 2.2 * rows + 1.5))
    if heights is not None:
        step = max(1, heights.shape[1] // 2000)
        h = heights[::step, ::step]
        im = ax.imshow(h, origin="lower", cmap="terrain", extent=(y0, y0 + cols * sy, x0, x0 + rows * sx))
        fig.colorbar(im, ax=ax, label="height [m]", fraction=0.02)
    for cell in meta["cells"]:
        cx, cy = x0 + cell["row"] * sx, y0 + cell["col"] * sy
        color = ZONE_COLORS[cell["zone"]]
        ax.add_patch(mpatches.Rectangle((cy, cx), sy, sx, fill=heights is None, alpha=0.25, ec=color, fc=color, lw=2))
        ax.text(cy + sy / 2, cx + sx - 1.5, f"{cell['property']}\nd={cell['difficulty']:.2f}", ha="center",
                va="top", fontsize=7, bbox={"fc": "white", "alpha": 0.7, "lw": 0})  # fmt: skip
    for p in meta["sand_patches"]:
        w, h = p["size"]
        ax.add_patch(mpatches.Rectangle((p["y"] - h / 2, p["x"] - w / 2), h, w, ec="saddlebrown" if p["crust"] else
                                        "orange", fc="none", lw=0.8))  # fmt: skip
    if meta["rocks"]:
        r = np.array([[q["y"], q["x"], q["size"], q["dynamic"]] for q in meta["rocks"]])
        ax.scatter(r[:, 0], r[:, 1], s=(r[:, 2] * 40) ** 2, c=np.where(r[:, 3] > 0, "red", "k"), lw=0)
    if faults is not None and len(faults):
        ax.scatter(faults[:, 1], faults[:, 0], marker="x", c="magenta", s=12, label="fault")
        ax.legend(loc="upper right")
    ax.set_xlim(y0, y0 + cols * sy)
    ax.set_ylim(x0, x0 + rows * sx)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"Harsh World, seed {meta['seed']}  (rigid gray, slip blue, sand yellow; red = dynamic rocks)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(seed: int, full: bool, overrides: dict | None = None):
    """Generate the world; returns (HarshWorldCfg, generator, meta, seconds)."""
    from isaaclab.utils import to_dict

    from isaaclab_tasks.forge.velocity.a2_harsh_world.harsh_world_cfg import HarshWorldCfg
    from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp.world import HarshWorldGenerator, world_meta

    hw = HarshWorldCfg(seed=seed)
    if full:
        hw.disabled = []
    for key, value in (overrides or {}).items():
        setattr(hw, key, value)
    start = time.time()
    generator = HarshWorldGenerator(hw.terrain_generator())
    meta = world_meta(generator, to_dict(hw))
    seconds = time.time() - start
    # the meshes (~3 GB for 30 cells) are only needed by the importer
    generator.terrain_mesh = generator.terrain_meshes = None
    return hw, generator, meta, seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="All 10 properties instead of the default 7.")
    parser.add_argument("--out", default="logs/harsh_world/build", help="Output directory.")
    parser.add_argument("--plot", default=None, help="Only redraw this world_meta.yaml.")
    args = parser.parse_args()

    if args.plot:
        with open(args.plot) as f:
            meta = yaml.safe_load(f)
        plot_world(meta, os.path.splitext(args.plot)[0] + ".png")
        return

    from isaaclab_tasks.forge.velocity.a2_harsh_world.mdp.world import write_world_meta

    _, generator, meta, seconds = build(args.seed, args.full)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, f"world_meta_seed{args.seed}{'_full' if args.full else ''}")
    write_world_meta(stem + ".yaml", meta)
    heights, corner = generator.world_heights()
    np.savez_compressed(stem + "_heights.npz", heights=heights, corner=corner)
    plot_world(meta, stem + ".png", heights)
    print(f"[harsh_world] generated {meta['num_rows']}x{meta['num_cols']} cells in {seconds:.1f} s -> {stem}.yaml")


if __name__ == "__main__":
    main()
