# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Harsh World generator and importer: one cell per (difficulty row, property column), plus the zone map.

* :class:`HarshWorldGenerator` generates the cells with a fixed difficulty per row and records what each cell
  holds (property, zone, difficulty, sand patches, rocks) and the full height field.
* :class:`HarshWorldImporter` imports the terrain, spawns the sand slabs and static rocks as global colliders,
  spreads the robots evenly over the cells, builds the :class:`ZoneMap`, and writes ``world_meta.yaml``.
* :class:`ZoneMap` turns world positions into cells, zones, sand patches, and terrain heights on the GPU.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import yaml

from pxr import Sdf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg, TerrainImporter, TerrainImporterCfg
from isaaclab.utils import clone, configclass

from .terrains import ZONES

if TYPE_CHECKING:
    from .terrains import HarshTerrainCfg

logger = logging.getLogger(__name__)

SAND_COLOR = (0.85, 0.72, 0.45)
CRUST_COLOR = (0.62, 0.52, 0.38)
ROCK_COLOR = (0.45, 0.43, 0.40)
SLAB_DEPTH = 0.6
"""Height [m] of the sand-slab boxes, from the sand surface down (sand is at most a few decimeters deep)."""


##
# Rocks.
##


def rock_mesh(seed: int, template: int, size: float, axis_ratio, yaw: float, flatten: float = 1.0) -> trimesh.Trimesh:
    """Random convex rock: hull of 12-30 noisy points on an ellipsoid, largest extent ``size``.

    The template index fixes the point cloud (16 templates per seed); ``axis_ratio`` (y/x, z/x), ``flatten``
    (extra z scale of buried rocks), and ``yaw`` shape the instance. The bottom is at ``z = 0``.
    """
    rng = np.random.default_rng([seed, 7, template])
    points = rng.normal(size=(int(rng.integers(12, 31)), 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= rng.uniform(0.8, 1.0, (len(points), 1))
    points *= np.array([1.0, axis_ratio[0], axis_ratio[1] * flatten]) * size / 2
    c, s = np.cos(yaw), np.sin(yaw)
    points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
    hull = trimesh.convex.convex_hull(points)
    hull.apply_translation([0.0, 0.0, -hull.bounds[0, 2]])
    return hull


##
# Generator.
##


class HarshWorldGenerator(TerrainGenerator):
    """Terrain generator that records the content of every cell.

    Columns follow :attr:`TerrainGeneratorCfg.sub_terrains` order (equal proportions), and the difficulty of row
    ``i`` is ``lower + (upper - lower) * i / (num_rows - 1)`` without Isaac Lab's random offset, so a cell's
    difficulty is exactly its row's.
    """

    last: HarshWorldGenerator | None = None
    """The most recent instance; :class:`TerrainImporter` creates the generator internally and drops it."""

    cfg: HarshWorldGeneratorCfg

    def __init__(self, cfg: HarshWorldGeneratorCfg, device: str = "cpu"):
        self.cells: list[list[dict]] = [[{} for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
        self.cell_heights: dict[tuple[int, int], np.ndarray] = {}
        super().__init__(cfg, device)
        HarshWorldGenerator.last = self

    def _generate_curriculum_terrains(self):
        names = list(self.cfg.sub_terrains)
        lower, upper = self.cfg.difficulty_range
        for col in range(self.cfg.num_cols):
            sub_cfg = self.cfg.sub_terrains[names[col % len(names)]]
            for row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * row / max(self.cfg.num_rows - 1, 1)
                self._cell = (row, col)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_cfg)

    def _get_terrain_mesh(self, difficulty: float, cfg: HarshTerrainCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        # the same steps as TerrainGenerator._get_terrain_mesh without the cache, keeping the cloned cfg so the
        # features the property function wrote into it survive
        cfg = clone(cfg)
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed
        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # the height field of the mesh: the function's heights inside the one-pixel zero border of the decorator
        self.cell_heights[self._cell] = np.pad(cfg.heights, 1)
        transform = np.eye(4)
        transform[0:2, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]
        row, col = self._cell
        self.cells[row][col] = {
            "property": cfg.name,
            "zone": cfg.zone,
            "difficulty": float(difficulty),
            "features": getattr(cfg, "features", {}),
        }
        return mesh, origin

    def world_heights(self) -> tuple[np.ndarray, np.ndarray]:
        """Terrain height [m] on the pixel grid of the whole cell grid, and its lower corner ``(x, y)`` [m].

        Cells share their border pixel. Heights outside the cell grid (the outer border) are zero.
        """
        (n, m), rows, cols = next(iter(self.cell_heights.values())).shape, self.cfg.num_rows, self.cfg.num_cols
        heights = np.zeros((rows * (n - 1) + 1, cols * (m - 1) + 1), dtype=np.float32)
        for (row, col), h in self.cell_heights.items():
            heights[row * (n - 1) : row * (n - 1) + n, col * (m - 1) : col * (m - 1) + m] = h
        corner = -0.5 * np.array([rows * self.cfg.size[0], cols * self.cfg.size[1]])
        return heights, corner


@configclass
class HarshWorldGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = HarshWorldGenerator
    curriculum: bool = True


##
# World description.
##


def world_meta(generator: HarshWorldGenerator, settings: dict) -> dict:
    """Plain-data description of the world (written to ``world_meta.yaml``).

    Positions are in the world frame. ``settings`` is the full :class:`HarshWorldCfg` as a dict.
    """
    from isaaclab import __version__ as isaaclab_version

    names = list(dict.fromkeys(cell["property"] for row in generator.cells for cell in row))
    cells, patches, rocks = [], [], []
    for row in range(generator.cfg.num_rows):
        for col in range(generator.cfg.num_cols):
            cell = generator.cells[row][col]
            ox, oy, oz = generator.terrain_origins[row, col]
            feats = cell["features"]
            cells.append({
                "row": row, "col": col, "property": cell["property"], "property_id": names.index(cell["property"]),
                "zone": cell["zone"], "zone_id": ZONES.index(cell["zone"]), "difficulty": cell["difficulty"],
                "origin": [ox, oy, oz],
                "features": {k: v for k, v in feats.items() if k not in ("patches", "rocks", "pits", "stubs",
                                                                       "trenches", "ledges")},
                "num_pits": len(feats.get("pits", [])), "num_stubs": len(feats.get("stubs", [])),
            })  # fmt: skip
            for i, p in enumerate(feats.get("patches", [])):
                patches.append({"id": len(patches), "row": row, "col": col, "index": i, **p,
                                "x": ox + p["x"], "y": oy + p["y"]})  # fmt: skip
            for r in feats.get("rocks", []):
                rocks.append({"row": row, "col": col, **r, "x": ox + r["x"], "y": oy + r["y"]})
    return _plain({
        "isaaclab_version": isaaclab_version,
        "seed": generator.cfg.seed,
        "num_rows": generator.cfg.num_rows,
        "num_cols": generator.cfg.num_cols,
        "cell_size": list(generator.cfg.size),
        "corner": (-0.5 * np.array([generator.cfg.num_rows, generator.cfg.num_cols]) * generator.cfg.size).tolist(),
        "properties": names,
        "zones": list(ZONES),
        "cells": cells,
        "sand_patches": patches,
        "rocks": rocks,
        "settings": settings,
    })  # fmt: skip


def _plain(value):
    """Nested plain Python data with floats rounded to 1e-6 (byte-identical yaml for a fixed seed)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    # str() also turns str subclasses (e.g. callables serialized by to_dict) into plain strings
    return None if value is None else str(value)


def write_world_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, width=120)


##
# Zone map.
##


class ZoneMap:
    """World position -> cell, property, zone, difficulty, sand patch, and terrain height (torch, vectorized)."""

    def __init__(self, meta: dict, heights: np.ndarray, corner: np.ndarray, horizontal_scale: float, device):
        self.device = device
        self.num_rows, self.num_cols = meta["num_rows"], meta["num_cols"]
        self.size = torch.tensor(meta["cell_size"], device=device)
        self.corner = torch.tensor(meta["corner"], device=device)
        shape = (self.num_rows, self.num_cols)
        self.property_id = torch.full(shape, -1, dtype=torch.long, device=device)
        self.zone_id = torch.zeros(shape, dtype=torch.long, device=device)
        self.difficulty = torch.zeros(shape, device=device)
        for cell in meta["cells"]:
            idx = (cell["row"], cell["col"])
            self.property_id[idx], self.zone_id[idx] = cell["property_id"], cell["zone_id"]
            self.difficulty[idx] = cell["difficulty"]
        # sand patches: per cell, slab footprint (center, half extents), surface, global id; padded with id -1
        self.patches = meta["sand_patches"]
        per_cell = max([sum(1 for p in self.patches if (p["row"], p["col"]) == (r, c))
                        for r in range(self.num_rows) for c in range(self.num_cols)] + [1])  # fmt: skip
        self.patch_table = torch.zeros(*shape, per_cell, 6, device=device)
        self.patch_table[..., 5] = -1
        count = torch.zeros(shape, dtype=torch.long)
        for p in self.patches:
            k = count[p["row"], p["col"]].item()
            half = [s / 2 + 0.05 for s in p["size"]]
            self.patch_table[p["row"], p["col"], k] = torch.tensor([p["x"], p["y"], *half, p["surface"], p["id"]])
            count[p["row"], p["col"]] += 1
        self.patch_surface = torch.tensor([p["surface"] for p in self.patches] + [0.0], device=device)
        # static rocks: (x, y, bottom, top, radius); dynamic rock slots of each row: (x, y, ground, yaw)
        static = []
        for r in meta["rocks"]:
            if not r["dynamic"]:
                bottom = r["ground"] - 0.01
                height = r["size"] * r["axis_ratio"][1] * (0.5 if r["buried"] else 1.0)
                static.append([r["x"], r["y"], bottom, bottom + height, r["size"] / 4 * (1 + r["axis_ratio"][0])])
        self.rocks_static = torch.tensor(static, device=device).reshape(-1, 5)
        slots = [[[r["x"], r["y"], r["ground"], r["yaw"]] for r in meta["rocks"] if r["dynamic"] and r["row"] == row]
                 for row in range(self.num_rows)]  # fmt: skip
        k = min(len(s) for s in slots)
        self.rock_slots = torch.tensor([s[:k] for s in slots], device=device).reshape(self.num_rows, k, 4)
        # terrain heights
        self.heights = torch.from_numpy(heights).to(device)
        self.heights_corner = torch.tensor(corner, device=device, dtype=torch.float32)
        self.hs = horizontal_scale

    def cell(self, xy: torch.Tensor) -> torch.Tensor:
        """(row, col) of each point, -1 outside the cell grid. ``xy`` (..., 2) -> (..., 2) long."""
        rc = torch.floor((xy - self.corner) / self.size).long()
        inside = (rc[..., 0] >= 0) & (rc[..., 0] < self.num_rows) & (rc[..., 1] >= 0) & (rc[..., 1] < self.num_cols)
        return torch.where(inside[..., None], rc, -1)

    def lookup(self, xy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Cell, property id (-1 outside), zone id (0 outside), and difficulty (0 outside) of each point."""
        rc = self.cell(xy)
        valid = rc[..., 0] >= 0
        r, c = rc[..., 0].clamp(min=0), rc[..., 1].clamp(min=0)
        return {
            "cell": rc,
            "property": torch.where(valid, self.property_id[r, c], -1),
            "zone": torch.where(valid, self.zone_id[r, c], 0),
            "difficulty": torch.where(valid, self.difficulty[r, c], 0.0),
        }

    def patch(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sand patch id under each point (-1 if none) and that patch's surface height (0 if none)."""
        rc = self.cell(xy)
        table = self.patch_table[rc[..., 0].clamp(min=0), rc[..., 1].clamp(min=0)]  # (..., P, 6)
        inside = ((xy[..., None, :] - table[..., :2]).abs() <= table[..., 2:4]).all(-1) & (table[..., 5] >= 0)
        inside &= (rc[..., 0] >= 0)[..., None]
        hit = inside.any(-1)
        pid = torch.where(hit, (table[..., 5] * inside).sum(-1).long(), -1)
        return pid, self.patch_surface[pid]

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height [m] at each point (nearest pixel; 0 outside the cell grid)."""
        ij = torch.round((xy - self.heights_corner) / self.hs).long()
        h, w = self.heights.shape
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < h) & (ij[..., 1] >= 0) & (ij[..., 1] < w)
        z = self.heights[ij[..., 0].clamp(0, h - 1), ij[..., 1].clamp(0, w - 1)]
        return torch.where(inside, z, 0.0)


##
# Importer.
##


class HarshWorldImporter(TerrainImporter):
    """Terrain importer for the Harsh World: sand slabs, static rocks, even robot spread, zone map, metadata."""

    cfg: HarshWorldImporterCfg

    def __init__(self, cfg: HarshWorldImporterCfg):
        super().__init__(cfg)
        generator = HarshWorldGenerator.last
        self.meta = world_meta(generator, cfg.settings)
        heights, corner = generator.world_heights()
        self.zone_map = ZoneMap(self.meta, heights, corner, generator.cfg.horizontal_scale, self.device)
        self._spawn_sand()
        self._spawn_static_rocks()
        if cfg.meta_path:
            self.meta_path = os.path.abspath(cfg.meta_path.format(seed=self.meta["seed"]))
            write_world_meta(self.meta_path, self.meta)
            logger.info(f"Wrote {self.meta_path}")

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        # Isaac Lab draws each robot's row at random; here the robots of a column cycle through the rows so that
        # every cell gets the same number of robots (num_envs = rows x cols x k)
        env_origins = super()._compute_env_origins_curriculum(num_envs, origins)
        columns = self.terrain_types
        first = torch.searchsorted(columns, columns, right=False)
        self.terrain_levels = (torch.arange(num_envs, device=self.device) - first) % origins.shape[0]
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    def _spawn_sand(self) -> None:
        """One static box collider per sand patch: top at the patch surface, reaching ``SLAB_DEPTH`` below it.

        A box primitive, not a triangle mesh: a sphere that sinks deeper than its radius into a mesh slab passes
        its top face and is pushed through, while a box pushes it back up until its center is closer to another
        face. The box reaches well below the hardpan, so a sinking foot stops on the hardpan first.
        """
        for p in self.meta["sand_patches"]:
            top = p["surface"]
            spawn = sim_utils.CuboidCfg(
                size=(p["size"][0] + 0.1, p["size"][1] + 0.1, SLAB_DEPTH),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CRUST_COLOR if p["crust"] else SAND_COLOR),
            )
            spawn.func(self.patch_path(p), spawn, translation=(p["x"], p["y"], top - SLAB_DEPTH / 2))
            # MuJoCo takes the contact parameters (stiffness, friction) of the higher-priority collider of a pair
            for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(self.patch_path(p))):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(1)

    def patch_path(self, patch: dict) -> str:
        return f"{self.cfg.prim_path}/sand/cell_{patch['row']}_{patch['col']}/patch_{patch['index']}"

    def _spawn_static_rocks(self) -> None:
        """One static convex collider per rock (buried ones flattened to half height).

        Convex, not one merged triangle mesh per cell: Newton finds no contacts between the dynamic rocks' convex
        hulls and a triangle mesh (the dynamic rocks fell through the mesh terrain), but does between convex shapes.
        """
        seed = self.meta["seed"] or 0
        material = f"{self.cfg.prim_path}/rocks/material"
        if any(not r["dynamic"] for r in self.meta["rocks"]):
            look = sim_utils.PreviewSurfaceCfg(diffuse_color=ROCK_COLOR)
            look.func(material, look)
        count: dict[tuple[int, int], int] = {}
        for r in self.meta["rocks"]:
            if r["dynamic"]:
                continue
            mesh = rock_mesh(seed, r["template"], r["size"], r["axis_ratio"], r["yaw"], 0.5 if r["buried"] else 1.0)
            i = count[(r["row"], r["col"])] = count.get((r["row"], r["col"]), -1) + 1
            spawn = sim_utils.MeshFileCfg(
                mesh=sim_utils.MeshFileCfg.TrimeshObjectCfg(mesh=mesh),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
                mesh_collision_props=[sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")],
            )
            path = f"{self.cfg.prim_path}/rocks/cell_{r['row']}_{r['col']}/rock_{i}"
            spawn.func(path, spawn, translation=(r["x"], r["y"], r["ground"] - 0.01))
            sim_utils.bind_visual_material(path, material)


@configclass
class HarshWorldImporterCfg(TerrainImporterCfg):
    class_type: type = HarshWorldImporter

    settings: dict = {}
    """The :class:`HarshWorldCfg` as a dict, copied into ``world_meta.yaml``."""

    meta_path: str | None = None
    """Where to write ``world_meta.yaml``; None to skip."""
