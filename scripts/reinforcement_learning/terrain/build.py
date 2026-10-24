# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate terrain maps without a simulator: the map description (yaml), a top-down PNG, and the height map.

Runs on any machine with Isaac Lab installed (no GPU, no Kit), e.g. in the local container::

    python scripts/reinforcement_learning/terrain/build.py gravel            # one terrain type
    python scripts/reinforcement_learning/terrain/build.py all --seed 3      # every terrain type
    python scripts/reinforcement_learning/terrain/build.py --plot logs/terrain/gravel_seed42.yaml   # redraw

``--plot`` only redraws an existing description (e.g. one written by the importer during a run).
"""

import argparse
import os
import time

import numpy as np
import yaml


def plot_map(meta: dict, path: str, heights: np.ndarray | None = None) -> None:
    """Top-down map: tile borders, difficulties, sand patches, and rocks (red: dynamic slots)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    rows, cols = meta["num_rows"], meta["num_cols"]
    sx, sy = meta["tile_size"]
    x0, y0 = meta["corner"]
    # world x (rows) is drawn vertically, y (cols) horizontally
    fig, ax = plt.subplots(figsize=(2.2 * cols + 2, 2.2 * rows + 1.5))
    if heights is not None:
        step = max(1, heights.shape[1] // 2000)
        im = ax.imshow(heights[::step, ::step], origin="lower", cmap="terrain",
                       extent=(y0, y0 + cols * sy, x0, x0 + rows * sx))  # fmt: skip
        fig.colorbar(im, ax=ax, label="height [m]", fraction=0.02)
    for tile in meta["tiles"]:
        tx, ty = x0 + tile["row"] * sx, y0 + tile["col"] * sy
        ax.add_patch(mpatches.Rectangle((ty, tx), sy, sx, fill=False, ec="gray", lw=1))
        ax.text(ty + sy / 2, tx + sx - 1.5, f"d={tile['difficulty']:.2f} #{tile['col']}", ha="center", va="top",
                fontsize=7, bbox={"fc": "white", "alpha": 0.7, "lw": 0})  # fmt: skip
    for p in meta["sand_patches"]:
        w, h = p["size"]
        ax.add_patch(mpatches.Rectangle((p["y"] - h / 2, p["x"] - w / 2), h, w, ec="saddlebrown" if p["crust"] else
                                        "orange", fc="none", lw=0.8))  # fmt: skip
    if meta["rocks"]:
        r = np.array([[q["y"], q["x"], q["size"], q["dynamic"]] for q in meta["rocks"]])
        ax.scatter(r[:, 0], r[:, 1], s=(r[:, 2] * 40) ** 2, c=np.where(r[:, 3] > 0, "red", "k"), lw=0)
    ax.set_xlim(y0, y0 + cols * sy)
    ax.set_ylim(x0, x0 + rows * sx)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(f"{meta['terrain']} ({meta['zone']}), seed {meta['seed']}: rows = difficulty, columns = tiles")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build(terrain: str, seed: int = 42, **overrides):
    """Generate one terrain map; returns (TerrainWorldCfg, generator, meta, seconds)."""
    from isaaclab.utils import to_dict

    from isaaclab_tasks.forge.terrain.mdp.generator import TerrainGridGenerator, map_meta
    from isaaclab_tasks.forge.terrain.world_cfg import TerrainWorldCfg

    world = TerrainWorldCfg(terrain=terrain, seed=seed)
    for key, value in overrides.items():
        setattr(world, key, value)
    start = time.time()
    generator = TerrainGridGenerator(world.terrain_generator())
    meta = map_meta(generator, to_dict(world))
    seconds = time.time() - start
    # the meshes are only needed by the importer
    generator.terrain_mesh = generator.terrain_meshes = None
    return world, generator, meta, seconds


def main():
    from isaaclab_tasks.forge.terrain import TERRAINS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("terrain", nargs="?", choices=[*TERRAINS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="logs/terrain/build", help="Output directory.")
    parser.add_argument("--plot", default=None, help="Only redraw this map description (yaml).")
    args = parser.parse_args()

    if args.plot:
        with open(args.plot) as f:
            meta = yaml.safe_load(f)
        plot_map(meta, os.path.splitext(args.plot)[0] + ".png")
        return

    from isaaclab_tasks.forge.terrain.mdp.generator import write_meta

    os.makedirs(args.out, exist_ok=True)
    for terrain in TERRAINS if args.terrain == "all" else [args.terrain]:
        _, generator, meta, seconds = build(terrain, args.seed)
        stem = os.path.join(args.out, f"{terrain}_seed{args.seed}")
        write_meta(stem + ".yaml", meta)
        heights, corner = generator.map_heights()
        np.savez_compressed(stem + "_heights.npz", heights=heights, corner=corner)
        plot_map(meta, stem + ".png", heights)
        print(f"[terrain] {terrain}: {meta['num_rows']}x{meta['num_cols']} tiles in {seconds:.1f} s -> {stem}.yaml")


if __name__ == "__main__":
    main()
