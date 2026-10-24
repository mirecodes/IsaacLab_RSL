# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain grid generator and importer: one template, one tile per (difficulty row, column), plus the terrain map.

* :class:`TerrainGridGenerator` generates the tiles with a fixed difficulty per row, chains the rows of an incline,
  and records what each tile holds (difficulty, sand patches) and the full height field.
* :class:`TerrainGridImporter` imports the terrain, spawns the sand slabs as global colliders,
  spreads the robots evenly over the tiles, builds the :class:`TerrainMap`, and writes the map description.
* :class:`TerrainMap` turns world positions into tiles, difficulties, sand patches, and terrain heights on the GPU.
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

from ..templates.common import ZONES

if TYPE_CHECKING:
    from ..templates.common import TerrainTemplateCfg

logger = logging.getLogger(__name__)

SAND_COLOR = (0.85, 0.72, 0.45)
CRUST_COLOR = (0.62, 0.52, 0.38)
SLAB_DEPTH = 0.6
"""Height [m] of the sand-slab boxes, from the sand surface down (sand is at most a few decimeters deep)."""


##
# Generator.
##


class TerrainGridGenerator(TerrainGenerator):
    """Terrain generator that records the content of every tile.

    Every column uses the single template of :attr:`TerrainGeneratorCfg.sub_terrains` (a tile index seeds each
    column differently), and the difficulty of row ``i`` is ``lower + (upper - lower) * i / (num_rows - 1)`` without
    Isaac Lab's random offset, so a tile's difficulty is exactly its row's.

    Rows are lifted so that each row's ``-x`` edge meets the previous row's ``+x`` edge (all zero, i.e. no lift,
    except for inclines), and the outer border continues the map's edge heights outward.
    """

    last: TerrainGridGenerator | None = None
    """The most recent instance; :class:`TerrainImporter` creates the generator internally and drops it."""

    cfg: TerrainGridGeneratorCfg

    def __init__(self, cfg: TerrainGridGeneratorCfg, device: str = "cpu"):
        if len(cfg.sub_terrains) != 1:
            raise ValueError(f"A terrain grid holds one template, got {list(cfg.sub_terrains)}.")
        self.tiles: list[list[dict]] = [[{} for _ in range(cfg.num_cols)] for _ in range(cfg.num_rows)]
        self.tile_heights: dict[tuple[int, int], np.ndarray] = {}
        self.row_lift = np.zeros(cfg.num_rows)
        """Height [m] each row is lifted by to meet the previous row."""
        super().__init__(cfg, device)
        TerrainGridGenerator.last = self

    def _generate_curriculum_terrains(self):
        sub_cfg = next(iter(self.cfg.sub_terrains.values()))
        lower, upper = self.cfg.difficulty_range
        # column 0 goes first, row by row, and fixes each row's lift; the other columns share the row's edge profile
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * row / max(self.cfg.num_rows - 1, 1)
                self._tile = (row, col)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, row, col, sub_cfg)

    def _get_terrain_mesh(self, difficulty: float, cfg: TerrainTemplateCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        # the same steps as TerrainGenerator._get_terrain_mesh without the cache, keeping the cloned cfg so the
        # features the template wrote into it survive
        row, col = self._tile
        cfg = clone(cfg)
        cfg.difficulty = float(difficulty)
        cfg.seed = self.cfg.seed or 0
        cfg.tile = col
        meshes, origin = cfg.function(difficulty, cfg)
        mesh = trimesh.util.concatenate(meshes)
        # the height field of the mesh: the template's heights inside the decorator's extrapolated border
        heights = np.pad(cfg.heights, 1, mode="reflect", reflect_type="odd")
        if col == 0 and row > 0:
            previous = self.tile_heights[(row - 1, 0)]
            self.row_lift[row] = previous[-1].mean() - heights[0].mean()
        self.tile_heights[self._tile] = heights + self.row_lift[row]
        transform = np.eye(4)
        transform[0:3, -1] = -cfg.size[0] * 0.5, -cfg.size[1] * 0.5, self.row_lift[row]
        mesh.apply_transform(transform)
        origin += transform[0:3, -1]
        self.tiles[row][col] = {"difficulty": float(difficulty), "features": getattr(cfg, "features", {})}
        return mesh, origin

    def _add_terrain_border(self):
        """A flat apron of ``border_width`` around the grid that continues each edge pixel's height outward (Isaac
        Lab's border sits at z = 0, a cliff next to a raised incline row)."""
        heights, _ = self.map_heights()
        hs, w = self.cfg.horizontal_scale, self.cfg.border_width
        x = np.arange(heights.shape[0]) * hs
        y = np.arange(heights.shape[1]) * hs
        # the two x edges also cover the corners
        y_long = np.concatenate([[-w], y, [y[-1] + w]])
        edges = [  # (points along the edge, outward offset)
            (np.stack([np.zeros_like(y_long), y_long, np.pad(heights[0], 1, mode="edge")], 1), (-w, 0.0)),
            (np.stack([np.full_like(y_long, x[-1]), y_long, np.pad(heights[-1], 1, mode="edge")], 1), (w, 0.0)),
            (np.stack([x, np.zeros_like(x), heights[:, 0]], 1), (0.0, -w)),
            (np.stack([x, np.full_like(x, y[-1]), heights[:, -1]], 1), (0.0, w)),
        ]
        for line, (dx, dy) in edges:
            n = len(line)
            vertices = np.vstack([line, line + [dx, dy, 0.0]])
            i = np.arange(n - 1)
            faces = np.concatenate([np.stack([i, i + 1, n + i + 1], 1), np.stack([i, n + i + 1, n + i], 1)])
            apron = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            down = apron.face_normals[:, 2] < 0
            faces[down] = faces[down][:, ::-1]
            self.terrain_meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))

    def map_heights(self) -> tuple[np.ndarray, np.ndarray]:
        """Terrain height [m] on the pixel grid of the whole tile grid, and its lower corner ``(x, y)`` [m].

        Tiles share their border pixel. The array covers the tile grid only (not the border).
        """
        (n, m), rows, cols = next(iter(self.tile_heights.values())).shape, self.cfg.num_rows, self.cfg.num_cols
        heights = np.zeros((rows * (n - 1) + 1, cols * (m - 1) + 1), dtype=np.float32)
        for (row, col), h in self.tile_heights.items():
            heights[row * (n - 1) : row * (n - 1) + n, col * (m - 1) : col * (m - 1) + m] = h
        corner = -0.5 * np.array([rows * self.cfg.size[0], cols * self.cfg.size[1]])
        return heights, corner


@configclass
class TerrainGridGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = TerrainGridGenerator
    curriculum: bool = True


##
# Map description.
##


def map_meta(generator: TerrainGridGenerator, settings: dict) -> dict:
    """Plain-data description of the map (written as yaml). Positions are in the world frame; ``settings`` is the
    full :class:`~..world_cfg.TerrainWorldCfg` as a dict."""
    from isaaclab import __version__ as isaaclab_version

    name, template = next(iter(generator.cfg.sub_terrains.items()))
    tiles, patches = [], []
    for row in range(generator.cfg.num_rows):
        for col in range(generator.cfg.num_cols):
            tile = generator.tiles[row][col]
            ox, oy, oz = generator.terrain_origins[row, col]
            feats = tile["features"]
            tiles.append({
                "row": row, "col": col, "difficulty": tile["difficulty"], "origin": [ox, oy, oz],
                "features": {k: v for k, v in feats.items() if k not in ("patches", "pits", "stubs", "trenches")},
                "num_pits": len(feats.get("pits", [])), "num_stubs": len(feats.get("stubs", [])),
            })  # fmt: skip
            for i, p in enumerate(feats.get("patches", [])):
                patches.append({"id": len(patches), "row": row, "col": col, "index": i, **p,
                                "x": ox + p["x"], "y": oy + p["y"]})  # fmt: skip
    return _plain({
        "isaaclab_version": isaaclab_version,
        "terrain": name,
        "zone": template.zone,
        "seed": generator.cfg.seed,
        "num_rows": generator.cfg.num_rows,
        "num_cols": generator.cfg.num_cols,
        "tile_size": list(generator.cfg.size),
        "corner": (-0.5 * np.array([generator.cfg.num_rows, generator.cfg.num_cols]) * generator.cfg.size).tolist(),
        "tiles": tiles,
        "sand_patches": patches,
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


def write_meta(path: str, meta: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, width=120)


##
# Terrain map.
##


class TerrainMap:
    """World position -> tile, difficulty, sand patch, and terrain height (torch, vectorized)."""

    def __init__(self, meta: dict, heights: np.ndarray, corner: np.ndarray, horizontal_scale: float, device):
        self.device = device
        self.num_rows, self.num_cols = meta["num_rows"], meta["num_cols"]
        self.zone = ZONES.index(meta["zone"])
        self.size = torch.tensor(meta["tile_size"], device=device)
        self.corner = torch.tensor(meta["corner"], device=device)
        shape = (self.num_rows, self.num_cols)
        self.difficulty = torch.zeros(shape, device=device)
        for tile in meta["tiles"]:
            self.difficulty[tile["row"], tile["col"]] = tile["difficulty"]
        # sand patches: per tile, slab footprint (center, half extents), surface, global id; padded with id -1
        self.patches = meta["sand_patches"]
        per_tile = max([sum(1 for p in self.patches if (p["row"], p["col"]) == (r, c))
                        for r in range(self.num_rows) for c in range(self.num_cols)] + [1])  # fmt: skip
        self.patch_table = torch.zeros(*shape, per_tile, 6, device=device)
        self.patch_table[..., 5] = -1
        count = torch.zeros(shape, dtype=torch.long)
        for p in self.patches:
            k = count[p["row"], p["col"]].item()
            half = [s / 2 + 0.05 for s in p["size"]]
            self.patch_table[p["row"], p["col"], k] = torch.tensor([p["x"], p["y"], *half, p["surface"], p["id"]])
            count[p["row"], p["col"]] += 1
        self.patch_surface = torch.tensor([p["surface"] for p in self.patches] + [0.0], device=device)
        # terrain heights
        self.heights = torch.from_numpy(heights).to(device)
        self.heights_corner = torch.tensor(corner, device=device, dtype=torch.float32)
        self.hs = horizontal_scale

    def tile(self, xy: torch.Tensor) -> torch.Tensor:
        """(row, col) of each point, -1 outside the tile grid. ``xy`` (..., 2) -> (..., 2) long."""
        rc = torch.floor((xy - self.corner) / self.size).long()
        inside = (rc[..., 0] >= 0) & (rc[..., 0] < self.num_rows) & (rc[..., 1] >= 0) & (rc[..., 1] < self.num_cols)
        return torch.where(inside[..., None], rc, -1)

    def lookup(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Whether each point is on the tile grid, and the tile's difficulty (0 outside)."""
        rc = self.tile(xy)
        inside = rc[..., 0] >= 0
        difficulty = self.difficulty[rc[..., 0].clamp(min=0), rc[..., 1].clamp(min=0)]
        return inside, torch.where(inside, difficulty, 0.0)

    def patch(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sand patch id under each point (-1 if none) and that patch's surface height (0 if none)."""
        rc = self.tile(xy)
        table = self.patch_table[rc[..., 0].clamp(min=0), rc[..., 1].clamp(min=0)]  # (..., P, 6)
        inside = ((xy[..., None, :] - table[..., :2]).abs() <= table[..., 2:4]).all(-1) & (table[..., 5] >= 0)
        inside &= (rc[..., 0] >= 0)[..., None]
        hit = inside.any(-1)
        pid = torch.where(hit, (table[..., 5] * inside).sum(-1).long(), -1)
        return pid, self.patch_surface[pid]

    def height(self, xy: torch.Tensor) -> torch.Tensor:
        """Terrain height [m] at each point (nearest pixel; 0 outside the tile grid)."""
        ij = torch.round((xy - self.heights_corner) / self.hs).long()
        h, w = self.heights.shape
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < h) & (ij[..., 1] >= 0) & (ij[..., 1] < w)
        z = self.heights[ij[..., 0].clamp(0, h - 1), ij[..., 1].clamp(0, w - 1)]
        return torch.where(inside, z, 0.0)


##
# Importer.
##


class TerrainGridImporter(TerrainImporter):
    """Terrain importer of a terrain grid: sand slabs, even robot spread, terrain map, description."""

    cfg: TerrainGridImporterCfg

    def __init__(self, cfg: TerrainGridImporterCfg):
        super().__init__(cfg)
        generator = TerrainGridGenerator.last
        self.meta = map_meta(generator, cfg.settings)
        heights, corner = generator.map_heights()
        self.terrain_map = TerrainMap(self.meta, heights, corner, generator.cfg.horizontal_scale, self.device)
        self._spawn_sand()
        if cfg.meta_path:
            self.meta_path = os.path.abspath(cfg.meta_path.format(terrain=self.meta["terrain"], seed=self.meta["seed"]))
            write_meta(self.meta_path, self.meta)
            logger.info(f"Wrote {self.meta_path}")

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        # Isaac Lab draws each robot's row at random; here the robots of a column cycle through the rows so that
        # every tile gets the same number of robots (num_envs = rows x cols x k)
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
        return f"{self.cfg.prim_path}/sand/tile_{patch['row']}_{patch['col']}/patch_{patch['index']}"


@configclass
class TerrainGridImporterCfg(TerrainImporterCfg):
    class_type: type = TerrainGridImporter

    settings: dict = {}
    """The :class:`~..world_cfg.TerrainWorldCfg` as a dict, copied into the map description."""

    meta_path: str | None = None
    """Where to write the map description (``{terrain}`` and ``{seed}`` are filled in); None to skip."""
