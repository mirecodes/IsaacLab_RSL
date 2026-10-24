# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain templates: one module per terrain-generation algorithm.

Adding a terrain type (``rough_field.py`` is the smallest example):

1. Write ``<name>.py`` here: a ``@height_field_template`` function ``(difficulty, cfg) -> (height field, origin)``
   built from :mod:`.common` (start with ``Grid``, end with ``finish``) and its cfg class deriving from
   :class:`~.common.TerrainTemplateCfg`.
2. Add ``"<name>": <Name>Cfg()`` to :func:`~..world_cfg.default_terrains` and the name to ``TERRAINS`` in
   ``forge/terrain/__init__.py``; this registers ``IsaacContrib-Terrain-<Name>-A2{Robust,Energy}``.
3. Preview it without a simulator: ``scripts/reinforcement_learning/terrain/build.py <name>``.

This package is imported at startup (gym registration), so it imports nothing here.
"""
