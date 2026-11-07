# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The failure-mode observation: what a mechanism detector tells the policy (``REPORT.md`` §3.1).

12 values per env, ``[p_any, p_fall, p_mode (6), p_leg (4)]``, the heads of the detector of
``results/reports/clustering/20261102_mechanism_detection`` (modes: slip, support loss, sinkage, obstruction, body
contact, actuator; legs FL, FR, RL, RR), from one of three sources:

* ``blind``: zeros (the same network input size, no information);
* ``oracle``: the injectors' ground truth (motor fault -> actuator, stuck foot -> obstruction, with its leg);
* ``detector``: the exported TCN on the last 0.5 s of proprioception, computed as the detector's training data was
  (``scripts/build_dataset.py`` of that report): 96 channels, standardized by nominal walking.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import nn

from isaaclab.managers import ManagerTermBase, ObservationTermCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

LEGS = ("FL", "FR", "RL", "RR")
NUM_MODES = 6
OBSTRUCTION, ACTUATOR = 3, 5
"""Indices of the injected mechanisms in the detector's mode head."""
DIM = 2 + NUM_MODES + len(LEGS)
DETECTOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "detector", "tcn.pt")


class _Heads(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.out = nn.Linear(d, DIM)


class TCN(nn.Module):
    """Copy of the detector's TCN (``detect.py``): dilated causal 1-D convolutions over the history; the last frame's
    and the mean features feed one linear layer of all heads."""

    def __init__(self, num_channels: int, d: int = 64):
        super().__init__()
        layers, ch = [], num_channels
        for dil in (1, 2, 4, 8):
            layers += [nn.Conv1d(ch, d, 3, dilation=dil, padding=2 * dil), nn.ELU()]
            ch = d
        self.conv = nn.Sequential(*layers)
        self.heads = _Heads(2 * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (N, H, C) -> ``[p_any, p_fall, p_mode (6), p_leg (4)]`` (N, 12)."""
        h = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]]
        o = self.heads.out(torch.cat([h[:, :, -1], h.mean(-1)], 1))
        return torch.cat([torch.sigmoid(o[:, :2]), torch.softmax(o[:, 2:8], -1), torch.softmax(o[:, 8:], -1)], 1)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class fault_mode(ManagerTermBase):
    """``[p_any, p_fall, p_mode (6), p_leg (4)]`` from ``source`` (see module doc)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.source = cfg.params["source"]
        self.robot: Articulation = env.scene[cfg.params.get("asset_name", "robot")]
        if self.source == "oracle":
            self.joint_leg = torch.tensor([LEGS.index(n[:2]) for n in self.robot.joint_names], device=env.device)
        elif self.source == "detector":
            path = cfg.params.get("detector_path") or DETECTOR_PATH
            ckpt = torch.load(path, map_location=env.device, weights_only=True)
            self.model = TCN(len(ckpt["channels"])).to(env.device).eval()
            self.model.load_state_dict(ckpt["state_dict"])
            self.mu, self.sd = ckpt["mu"].to(env.device), ckpt["sd"].to(env.device)
            self.history = int(ckpt["H"])
            # the detector saw the surveillance steering commands only (v_x 0.5-1 m/s, v_y = 0): this task's
            # omnidirectional commands sit ~10 SDs off them, so the command channels are fed their nominal mean
            self.command_cols = [i for i, n in enumerate(ckpt["channels"]) if n.startswith("cmd_")]
            self.feet = self.robot.find_bodies([f"{leg}_foot" for leg in LEGS], preserve_order=True)[0]
            self.buffer = torch.zeros(env.num_envs, self.history, len(ckpt["channels"]), device=env.device)
            self.prev_feet = torch.zeros(env.num_envs, len(LEGS), 3, device=env.device)
            self.fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        elif self.source != "blind":
            raise ValueError(f"Unknown fault_mode source: {self.source}")

    def reset(self, env_ids=None) -> None:
        if self.source == "detector":
            self.fresh[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env: ManagerBasedRLEnv, source: str, asset_name: str = "robot", detector_path: str = ""):
        out = torch.zeros(env.num_envs, DIM, device=env.device)
        if self.source == "oracle":
            motor = env.event_manager.get_term_cfg("motor_fault").func.labels()[:, : len(self.joint_leg)] > 0
            stuck = env.event_manager.get_term_cfg("foot_stuck").func.labels()[:, : len(LEGS)] > 0
            has_motor, has_stuck = motor.any(1), stuck.any(1)
            rows = torch.arange(env.num_envs, device=env.device)
            out[:, 0] = (has_motor | has_stuck).float()
            out[:, 2 + ACTUATOR] = has_motor.float()
            out[:, 2 + OBSTRUCTION] = has_stuck.float()
            legs = 2 + NUM_MODES
            out[rows[has_motor], legs + self.joint_leg[motor.float().argmax(1)][has_motor]] = 1.0
            out[rows[has_stuck], legs + stuck.float().argmax(1)[has_stuck]] = 1.0
        elif self.source == "detector":
            x = (self._frame(env) - self.mu) / self.sd
            x = x.clamp(-20.0, 20.0)
            x[:, self.command_cols] = 0.0
            # a new episode starts with its first frame repeated, as the dataset pads short windows
            fresh = self.fresh
            self.buffer = torch.cat([self.buffer[:, 1:], x[:, None]], 1)
            self.buffer[fresh] = x[fresh, None].expand(-1, self.history, -1)
            self.fresh[:] = False
            with torch.no_grad():
                out = self.model(self.buffer)
        return out

    def _frame(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """The 96 proprio channels of ``build_dataset.py``: q, dq, measured torque, commanded (PD) torque, joint
        target, projected gravity, gyro, IMU specific force (base frame), command (3), foot positions in the base
        frame and their finite difference."""
        d = self.robot.data
        quat = _t(d.root_link_quat_w)
        g = torch.tensor([0.0, 0.0, -9.81], device=env.device)
        acc = math_utils.quat_apply_inverse(quat, _t(d.body_com_acc_w)[:, 0, :3] - g)
        feet_w = _t(d.body_link_pos_w)[:, self.feet] - _t(d.root_link_pos_w)[:, None]
        quat_feet = quat[:, None].expand(-1, len(LEGS), -1).reshape(-1, 4)
        feet = math_utils.quat_apply_inverse(quat_feet, feet_w.reshape(-1, 3)).view(-1, len(LEGS), 3)
        feet_v = torch.where(self.fresh[:, None, None], 0.0, (feet - self.prev_feet) / env.step_dt)
        self.prev_feet = feet
        cmd = env.command_manager.get_command("base_velocity")[:, :3]
        return torch.cat(
            [_t(d.joint_pos), _t(d.joint_vel), _t(d.applied_torque), _t(d.computed_torque), _t(d.joint_pos_target),
             _t(d.projected_gravity_b), _t(d.root_link_ang_vel_b), acc, cmd, feet.flatten(1), feet_v.flatten(1)],
            dim=1,
        )  # fmt: skip
