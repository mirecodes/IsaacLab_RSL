# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the Go2 study environments (Miki et al. 2022 teacher, Fu et al. 2021 energy policy)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class randomize_foot_friction(ManagerTermBase):
    """Sample one friction coefficient per foot, occasionally from a low range to induce slippage.

    Each foot draws from ``low_friction_range`` with probability ``low_friction_prob`` and from
    ``friction_range`` otherwise. With ``shared`` set, all feet of an environment use the first foot's
    sample. The sampled values are kept in :attr:`foot_friction` for the privileged observation.

    Newton (MuJoCo-Warp) only: the contact friction is ``max(mu_foot, mu_ground)``, so the ground
    friction must be set below ``low_friction_range`` for the foot value to take effect.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        root_view = getattr(asset, "_root_view", None)
        if root_view is None or not hasattr(root_view, "body_shapes"):
            raise RuntimeError("'randomize_foot_friction' requires the Newton physics backend.")

        self._newton_manager = env.sim.backend_manager
        self._friction_binding = root_view.get_attribute("shape_material_mu", self._newton_manager.get_model())[:, 0]
        # shape indices of each selected foot, in asset_cfg.body_ids order
        backend_body_ids = asset.map_body_ids_to_backend(asset_cfg.body_ids)
        self._foot_shapes = [torch.tensor(root_view.body_shapes[i], dtype=torch.long) for i in backend_body_ids]
        self.foot_friction = torch.zeros(env.num_envs, len(self._foot_shapes), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        friction_range: tuple[float, float],
        low_friction_range: tuple[float, float],
        low_friction_prob: float,
        asset_cfg: SceneEntityCfg,
        shared: bool = False,
    ) -> None:
        from newton import ModelFlags  # noqa: PLC0415

        if env_ids is None:
            env_ids = slice(None)
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        size = (num_envs, len(self._foot_shapes))
        nominal = math_utils.sample_uniform(*friction_range, size, env.device)
        low = math_utils.sample_uniform(*low_friction_range, size, env.device)
        samples = torch.where(torch.rand(size, device=env.device) < low_friction_prob, low, nominal)
        if shared:
            samples = samples[:, :1].expand(size).contiguous()
        self.foot_friction[env_ids] = samples

        friction_view = wp.to_torch(self._friction_binding)
        env_rows = env_ids if isinstance(env_ids, slice) else env_ids[:, None]
        for foot, shapes in enumerate(self._foot_shapes):
            shapes = shapes.to(env.device)
            friction_view[env_rows, shapes] = samples[:, foot : foot + 1].expand(-1, len(shapes))
        self._newton_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


class randomize_pd_gains(ManagerTermBase):
    """Scale the PD gains of all joints by per-environment gain factors and per-joint motor strengths.

    Sets ``Kp = Kp_0 * s_p * m_j`` and ``Kd = Kd_0 * s_d * m_j``, where ``s_p`` and ``s_d`` are drawn once
    per environment and ``m_j`` once per joint (Fu et al. 2021, Table 1). A motor-strength factor scales
    the PD torque before the actuator's torque limit. The motor strengths are kept in
    :attr:`motor_strength` in articulation joint order for the privileged observation.

    Requires Newton-executed actuators (``sim.use_newton_actuators``), as used by the Go2 task.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        from isaaclab.actuators.newton import read_group_parameter  # noqa: PLC0415

        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        collection = self._asset.actuators
        group_names = sorted(getattr(collection, "_native_group_names", set()))
        if not group_names or len(group_names) != len(collection):
            raise RuntimeError("'randomize_pd_gains' requires all actuator groups to run as Newton actuators.")
        joint_indices = collection._group_joint_indices
        self._groups = {
            name: (
                torch.arange(self._asset.num_joints, device=env.device)
                if isinstance(joint_indices[name], slice)
                else joint_indices[name].to(env.device, dtype=torch.long),
                read_group_parameter(collection, name, "drive", "kp").clone(),
                read_group_parameter(collection, name, "drive", "kd").clone(),
            )
            for name in group_names
        }
        self.motor_strength = torch.ones(env.num_envs, self._asset.num_joints, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        stiffness_scale_range: tuple[float, float],
        damping_scale_range: tuple[float, float],
        motor_strength_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
    ) -> None:
        from isaaclab.actuators.newton import write_group_parameter  # noqa: PLC0415

        if env_ids is None:
            env_ids = slice(None)
        num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        stiffness_scale = math_utils.sample_uniform(*stiffness_scale_range, (num_envs, 1), env.device)
        damping_scale = math_utils.sample_uniform(*damping_scale_range, (num_envs, 1), env.device)
        strength = math_utils.sample_uniform(*motor_strength_range, (num_envs, self._asset.num_joints), env.device)
        self.motor_strength[env_ids] = strength

        for name, (joint_ids, kp, kd) in self._groups.items():
            group_strength = strength[:, joint_ids]
            for attr, default, scale in (("kp", kp, stiffness_scale), ("kd", kd, damping_scale)):
                write_group_parameter(
                    self._asset.actuators,
                    name,
                    "drive",
                    attr,
                    values=default[env_ids] * scale * group_strength,
                    env_ids=env_ids,
                )
