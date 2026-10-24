# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The terrain's runtime physics, as one event term run every policy step.

Newton (MuJoCo-Warp) only: shape properties are written into the Newton model, see ``REPORT.md`` for how each
interaction maps onto MuJoCo's contact model. Only the parts the terrain type needs run: slip on ``slip`` terrain,
sand drag where there are sand patches, collapse where there are crusts, stone upkeep where there are stones.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..templates.common import ZONES

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs import ManagerBasedEnv

    from ..world_cfg import TerrainWorldCfg
    from .generator import TerrainMap

SLIP = ZONES.index("slip")
EVENT = "terrain"
"""Name of the event term in the env cfg (the policy's privileged friction input reads :attr:`foot_friction`)."""


def interp(pair: tuple[float, float], difficulty: torch.Tensor) -> torch.Tensor:
    return pair[0] + difficulty * (pair[1] - pair[0])


class terrain_step(ManagerTermBase):
    """Slip friction, sand drag, crust collapse, and stone upkeep of the terrain map.

    Run as an ``interval`` event every step (``interval_range_s=(0, 0)``, global time) and on resets via
    :meth:`reset`. At construction it sets the static contact parameters: terrain friction, sand-slab stiffness
    and priority, foot shapes in force-space contact mode, rock friction.

    State, per foot (N, 4): :attr:`foot_friction` (what the policy's privileged friction input reports),
    :attr:`foot_mu` (friction applied), :attr:`low_mu` (a slip sample is active), :attr:`in_sand`,
    :attr:`collapse_event` (the foot's patch collapsed in the last step).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        from newton import ModelFlags  # noqa: PLC0415
        from newton._src.solvers.mujoco.constants import SOLREF_MODE_FORCE_SPACE  # noqa: PLC0415

        super().__init__(cfg, env)
        world: TerrainWorldCfg = cfg.params["world"]
        self.world = world
        self.map: TerrainMap = env.scene.terrain.terrain_map
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[asset_cfg.name]
        self.foot_ids = asset_cfg.body_ids
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.contact = env.scene[sensor_cfg.name]
        self.contact_foot_ids = sensor_cfg.body_ids
        n, f = env.num_envs, len(self.foot_ids)

        # -- Newton model handles
        root_view = getattr(self.robot, "_root_view", None)
        if root_view is None or not hasattr(root_view, "body_shapes"):
            raise RuntimeError("The terrain's runtime physics requires the Newton physics backend.")
        self._newton = env.sim.backend_manager
        self._shape_change = ModelFlags.SHAPE_PROPERTIES
        model = self._newton.get_model()
        self._foot_mu_binding = root_view.get_attribute("shape_material_mu", model)[:, 0]
        backend_body_ids = self.robot.map_body_ids_to_backend(self.foot_ids)
        self._foot_shapes = [torch.tensor(root_view.body_shapes[i], dtype=torch.long) for i in backend_body_ids]

        # -- static contact parameters, by shape label
        labels = list(model.shape_label)
        body = model.shape_body.numpy()
        mu = wp.to_torch(model.shape_material_mu)
        mu_torsional = wp.to_torch(model.shape_material_mu_torsional)
        mu_rolling = wp.to_torch(model.shape_material_mu_rolling)
        ke = wp.to_torch(model.shape_material_ke)
        kd = wp.to_torch(model.shape_material_kd)
        mode = wp.to_torch(model.mujoco.solref_mode)
        priority = model.mujoco.geom_priority.numpy()
        terrain_path = env.scene.terrain.cfg.prim_path
        patch_re = re.compile(rf"^{re.escape(terrain_path)}/sand/tile_(\d+)_(\d+)/patch_(\d+)/")
        patch_ids = {(p["row"], p["col"], p["index"]): p["id"] for p in self.map.patches}
        foot_names = [self.robot.body_names[i] for i in self.foot_ids]
        rock_specs = world.dynamic_rock_specs()
        self.slab_shape = torch.full((len(self.map.patches),), -1, dtype=torch.long)
        ground, foot = [], []
        for i, label in enumerate(labels):
            if body[i] >= 0:
                # feet, and the probe spheres of probe.py, contact the slabs in force space
                if any(f"/{name}" in label for name in foot_names) or "/Probe" in label:
                    foot.append(i)
                rock = re.search(r"/Rock_(\d+)", label)
                if rock:
                    mu[i] = rock_specs[int(rock.group(1))]["friction"]
                    mu_torsional[i] = world.rocks.torsional_friction
                    mu_rolling[i] = world.rocks.rolling_friction
                continue
            match = patch_re.match(label + "/")
            if match:
                self.slab_shape[patch_ids[tuple(int(g) for g in match.groups())]] = i
                if priority[i] < 1:
                    raise RuntimeError(f"Sand slab '{label}' lost its 'mjc:priority' on import.")
            else:
                ground.append(i)
        mu[ground] = world.slip.ground_mu
        if (self.slab_shape < 0).any():
            raise RuntimeError("Some sand slabs were not found among the Newton shapes.")
        # slabs: compliant contact in Newton's force space (stiffness [N/m]), with priority over the feet
        crust = torch.tensor([p["crust"] for p in self.map.patches], dtype=torch.bool)
        self.slab_ke = torch.tensor([p["stiffness"] for p in self.map.patches], dtype=torch.float32)
        self.slab_ke[crust] = world.collapse.crust_stiffness
        slabs = self.slab_shape.to(mu.device)
        mu[slabs] = world.sand.friction
        ke[slabs] = self.slab_ke.to(ke.device)
        kd[slabs] = world.sand.damping_ratio * self.slab_ke.to(kd.device)
        mode[slabs] = SOLREF_MODE_FORCE_SPACE
        mode[torch.tensor(foot, dtype=torch.long, device=mode.device)] = SOLREF_MODE_FORCE_SPACE
        self._newton.add_model_change(self._shape_change)
        self._ke = ke
        self._kd = kd

        # -- state
        dev = env.device
        nominal = world.slip.mu_nominal
        self.foot_mu = torch.full((n, f), nominal, device=dev)
        self.foot_friction = self.foot_mu.clone()
        self.low_mu = torch.zeros(n, f, dtype=torch.bool, device=dev)
        self._in_slip = torch.zeros(n, f, dtype=torch.bool, device=dev)
        self.in_sand = torch.zeros(n, f, dtype=torch.bool, device=dev)
        self.collapse_event = torch.zeros(n, f, dtype=torch.bool, device=dev)
        # crust / collapsed flags per patch, plus a trailing "no patch" entry so that patch id -1 indexes safely
        self.crust = torch.cat([crust, torch.zeros(1, dtype=torch.bool)]).to(dev)
        self.collapsed = torch.zeros(len(self.map.patches) + 1, dtype=torch.bool, device=dev)
        self._collapse_time = torch.zeros(len(self.map.patches) + 1, device=dev)
        self.num_collapses = 0
        difficulty = [self.map.difficulty[p["row"], p["col"]].item() for p in self.map.patches]
        self._patch_difficulty = torch.tensor(difficulty + [0.0], device=dev)
        self._forces = torch.zeros(n, f, 3, device=dev)
        self._rocks_placed = False
        self._last_recycle = 0.0
        self.rocks: RigidObjectCollection | None = env.scene.rigid_object_collections.get("rocks")
        self._rock_sizes = torch.tensor([spec["size"] for spec in rock_specs], device=dev)
        self._write_foot_mu(torch.arange(n, device=dev))

    def _write_foot_mu(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        view = wp.to_torch(self._foot_mu_binding)
        rows = env_ids[:, None]
        for foot, shapes in enumerate(self._foot_shapes):
            shapes = shapes.to(view.device)
            view[rows, shapes] = self.foot_mu[env_ids, foot : foot + 1].expand(-1, len(shapes))
        self._newton.add_model_change(self._shape_change)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.foot_mu[env_ids] = self.world.slip.mu_nominal
        self.foot_friction[env_ids] = self.world.slip.mu_nominal
        self.low_mu[env_ids] = False
        self._in_slip[env_ids] = False
        self.in_sand[env_ids] = False
        self.collapse_event[env_ids] = False
        self._forces[env_ids] = 0.0
        self._write_foot_mu(env_ids)
        if self.rocks is not None and not self._rocks_placed:
            # stones are placed around the robots once; afterwards the ones left behind follow their robot
            self.recycle_rocks(force=True)
            self._rocks_placed = True

    def __call__(self, env: ManagerBasedEnv, env_ids, world, asset_cfg, sensor_cfg) -> None:
        pos = self.robot.data.body_link_pos_w.torch[:, self.foot_ids]
        if self.map.zone == SLIP:
            self._slip(env, *self.map.lookup(pos[..., :2]))
        if self.map.patches:
            patch, surface = self.map.patch(pos[..., :2])
            self._collapse(env, patch, self.contact.data.net_forces_w.torch[:, self.contact_foot_ids, 2].abs())
            vel = self.robot.data.body_com_lin_vel_w.torch[:, self.foot_ids]
            self._sand(self.map.lookup(pos[..., :2])[1], patch, surface, pos, vel)
        if self.rocks is not None:
            now = env.common_step_counter * env.step_dt
            if now - self._last_recycle >= self.world.rocks.recycle_interval:
                self._last_recycle = now
                self.recycle_rocks()

    # -- slip

    def _slip(self, env: ManagerBasedEnv, in_slip: torch.Tensor, difficulty: torch.Tensor) -> None:
        """``in_slip``: foot on the slip terrain (the tile grid), ``difficulty``: its tile's difficulty (N, F)."""
        cfg = self.world.slip
        enter = in_slip & ~self._in_slip
        if cfg.resample_on_touchdown:
            touchdown = self.contact.compute_first_contact(env.step_dt).torch[:, self.contact_foot_ids].bool()
            enter |= in_slip & touchdown
        leave = ~in_slip & self._in_slip
        self._in_slip = in_slip
        if not (enter.any() or leave.any()):
            return
        lo = interp((cfg.mu_slip_easy[0], cfg.mu_slip_hard[0]), difficulty)
        hi = interp((cfg.mu_slip_easy[1], cfg.mu_slip_hard[1]), difficulty)
        sample = lo + torch.rand_like(lo) * (hi - lo)
        low = torch.rand_like(lo) < cfg.slip_prob
        self.foot_mu = torch.where(enter, torch.where(low, sample, cfg.mu_nominal), self.foot_mu)
        self.low_mu = torch.where(enter, low, self.low_mu)
        self.foot_mu[leave] = cfg.mu_nominal
        self.low_mu[leave] = False
        if cfg.observe_slip:
            self.foot_friction[:] = self.foot_mu
        self._write_foot_mu((enter | leave).any(dim=1).nonzero().squeeze(-1))

    # -- sand

    def _sand(self, difficulty: torch.Tensor, patch: torch.Tensor, surface: torch.Tensor, pos, vel) -> None:
        cfg = self.world.sand
        # intact crusts are hard; sand drag acts in ordinary patches and in collapsed crusts
        soft = (patch >= 0) & (~self.crust[patch] | self.collapsed[patch])
        self.in_sand = soft & (pos[..., 2] < surface + cfg.foot_radius)
        force = torch.zeros_like(vel)
        force[..., :2] = -interp(cfg.c_drag, difficulty)[..., None] * vel[..., :2]
        force[..., 2] = -interp(cfg.c_suction, difficulty) * vel[..., 2].clamp(min=0.0)
        norm = force.norm(dim=-1, keepdim=True)
        force = torch.where(norm > cfg.force_limit, force * cfg.force_limit / norm.clamp(min=1e-6), force)
        force[~self.in_sand] = 0.0
        if not (self.in_sand.any() or self._forces.any()):
            return
        self._forces = force
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force,
            torques=torch.zeros_like(force),
            body_ids=self.foot_ids,
            env_ids=torch.arange(self.num_envs, device=self.device),
            is_global=True,
        )

    # -- crust collapse

    def _collapse(self, env: ManagerBasedEnv, patch: torch.Tensor, force: torch.Tensor) -> None:
        """``patch``: patch id under each foot (N, F); ``force``: vertical contact force of each foot (N, F)."""
        cfg = self.world.collapse
        self.collapse_event[:] = False
        if not cfg.enabled or not self.crust.any():
            return
        now = env.common_step_counter * env.step_dt
        recovered = self.collapsed & (now - self._collapse_time >= cfg.recover_time)
        # load of each robot on each crust patch: the summed vertical force of its feet standing on it
        on_crust = self.crust[patch]
        load = torch.zeros(self.num_envs, len(self.map.patches) + 1, device=self.device)
        load.scatter_add_(1, torch.where(on_crust, patch, len(self.map.patches)), force * on_crust)
        limit = interp(cfg.collapse_force, self._patch_difficulty)
        new = (load > limit).any(dim=0) & self.crust & ~self.collapsed
        if not (new.any() or recovered.any()):
            return
        self.collapsed = (self.collapsed | new) & ~recovered
        self._collapse_time[new] = now
        self.num_collapses += int(new.sum())
        self.collapse_event = new[patch]
        scale = 0.1 if cfg.collapse_mode == "soften" else 1e-4
        changed = (new | recovered)[:-1].cpu()
        ke = self.slab_ke[changed] * torch.where(self.collapsed[:-1].cpu()[changed], scale, 1.0)
        shapes = self.slab_shape[changed].to(self._ke.device)
        self._ke[shapes] = ke.to(self._ke.device)
        self._kd[shapes] = self.world.sand.damping_ratio * ke.to(self._kd.device)
        self._newton.add_model_change(self._shape_change)

    # -- rocks

    def recycle_rocks(self, force: bool = False) -> int:
        """Move every robot's stones left behind (farther than ``recycle_radius``, or all with ``force``) to free spots
        ahead of it; returns the number moved.

        Spots: ``candidates`` random points at ``ahead`` distance within ``sector`` of the walking direction (the
        heading below 0.2 m/s), clamped to the tile grid; each stone takes the one with the most clearance to the
        robot's other stones (circle bounds of half their length). A stone is dropped 1 cm above the highest of five
        ground samples under it, so it settles onto the bumps.
        """
        cfg, dev = self.world.rocks, self.device
        data = self.rocks.data
        pose = data.body_link_pose_w.torch.clone()  # (N, K, 7)
        base = self.robot.data.root_link_pos_w.torch[:, :2]
        far = (pose[..., :2] - base[:, None]).norm(dim=-1) > cfg.recycle_radius
        if force:
            far[:] = True
        if not far.any():
            return 0
        n, k = far.shape
        vel = self.robot.data.root_lin_vel_w.torch[:, :2]
        q = self.robot.data.root_link_quat_w.torch  # (x, y, z, w)
        heading = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
        direction = torch.where(vel.norm(dim=-1) > 0.2, torch.atan2(vel[:, 1], vel[:, 0]), heading)
        xy = pose[..., :2]
        rows = torch.arange(n, device=dev)
        extent = self.map.size * torch.tensor([self.map.num_rows, self.map.num_cols], device=dev)
        grid_min, grid_max = self.map.corner + 0.3, self.map.corner + extent - 0.3
        for j in range(k):
            if not far[:, j].any():
                continue
            angle = direction[:, None] + cfg.sector * (2 * torch.rand(n, cfg.candidates, device=dev) - 1)
            radius = cfg.ahead[0] + (cfg.ahead[1] - cfg.ahead[0]) * torch.rand(n, cfg.candidates, device=dev)
            spots = base[:, None] + radius[..., None] * torch.stack([angle.cos(), angle.sin()], -1)  # (N, C, 2)
            spots = torch.maximum(torch.minimum(spots, grid_max), grid_min)
            clear = (spots[:, :, None] - xy[:, None]).norm(dim=-1) - (self._rock_sizes[j] + self._rock_sizes) / 2
            clear[..., j] = float("inf")
            best = spots[rows, clear.min(-1).values.argmax(-1)]
            xy[:, j] = torch.where(far[:, j, None], best, xy[:, j])
        offsets = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], device=dev) * 0.5
        under = xy[:, :, None] + offsets * self._rock_sizes[:, None, None]  # (N, K, 5, 2)
        z = self.map.height(under).max(-1).values + 0.01
        yaw = 2 * torch.pi * torch.rand(n, k, device=dev)
        new = torch.cat([xy, z[..., None], torch.zeros(n, k, 2, device=dev), torch.sin(yaw / 2)[..., None],
                         torch.cos(yaw / 2)[..., None]], -1)  # fmt: skip
        pose = torch.where(far[..., None], new, pose)
        velocity = data.body_com_vel_w.torch.clone()
        velocity[far] = 0.0
        self.rocks.write_body_link_pose_to_sim_index(body_poses=pose, body_ids=list(range(k)), env_ids=rows)
        self.rocks.write_body_com_velocity_to_sim_index(body_velocities=velocity, body_ids=list(range(k)),
                                                        env_ids=rows)  # fmt: skip
        return int(far.sum())
