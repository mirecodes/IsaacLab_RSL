# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fractal height-field terrain approximating RaiSim's fractal terrain generator (Fu et al. 2021)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate

from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@height_field_to_mesh
def fractal_terrain(difficulty: float, cfg: FractalTerrainCfg) -> tuple[np.ndarray, np.ndarray]:
    """Sum of smooth value-noise octaves, tapered to zero height at the sub-terrain edges.

    Octave ``k`` samples uniform noise on a grid of spacing ``base_wavelength / lacunarity**k`` with
    amplitude ``gain**k``. The sum is normalized so the largest deviation is ``amplitude / 2`` (the
    peak-to-peak height is about ``amplitude``). The taper keeps neighboring sub-terrains seamless.

    The :obj:`difficulty` parameter is ignored for this terrain.
    """
    width_pixels = round(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = round(cfg.size[1] / cfg.horizontal_scale)
    x = np.linspace(0.0, cfg.size[0], width_pixels)
    y = np.linspace(0.0, cfg.size[1], length_pixels)

    heights = np.zeros((width_pixels, length_pixels))
    for octave in range(cfg.octaves):
        spacing = cfg.base_wavelength / cfg.lacunarity**octave
        grid_x = np.linspace(0.0, cfg.size[0], max(4, round(cfg.size[0] / spacing) + 1))
        grid_y = np.linspace(0.0, cfg.size[1], max(4, round(cfg.size[1] / spacing) + 1))
        samples = np.random.uniform(-1.0, 1.0, size=(len(grid_x), len(grid_y)))
        heights += cfg.gain**octave * interpolate.RectBivariateSpline(grid_x, grid_y, samples)(x, y)
    heights *= 0.5 * cfg.amplitude / max(np.abs(heights).max(), 1e-6)

    # smoothstep taper over the outer taper_width of the sub-terrain
    def taper(coords: np.ndarray, size: float) -> np.ndarray:
        s = np.clip(np.minimum(coords, size - coords) / cfg.taper_width, 0.0, 1.0)
        return 3.0 * s**2 - 2.0 * s**3

    heights *= taper(x, cfg.size[0])[:, None] * taper(y, cfg.size[1])[None, :]
    height_field = np.rint(heights / cfg.vertical_scale).astype(np.int16)

    # spawn at the center, on the highest point within 1 m
    center_x, center_y = width_pixels // 2, length_pixels // 2
    radius = max(1, round(1.0 / cfg.horizontal_scale))
    height = height_field[center_x - radius : center_x + radius, center_y - radius : center_y + radius].max()
    origin = np.array(
        [
            (width_pixels - 1) * cfg.horizontal_scale / 2,
            (length_pixels - 1) * cfg.horizontal_scale / 2,
            height * cfg.vertical_scale,
        ]
    )
    return height_field, origin


@configclass
class FractalTerrainCfg(HfTerrainBaseCfg):
    """Configuration for :func:`fractal_terrain`."""

    function: Callable = fractal_terrain

    amplitude: float = MISSING
    """Approximate peak-to-peak height [m]."""

    base_wavelength: float = MISSING
    """Grid spacing [m] of the first (coarsest) noise octave."""

    octaves: int = 2
    """Number of noise octaves."""

    lacunarity: float = 2.0
    """Spatial frequency ratio between successive octaves."""

    gain: float = 0.25
    """Amplitude ratio between successive octaves."""

    taper_width: float = 1.0
    """Width [m] over which the heights fade to zero at the sub-terrain edges."""
