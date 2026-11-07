# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Append ``--extra`` zero input columns to the first layer of an RSL-RL actor and critic, so that a policy trained
without the mode observation acts the same with it appended (``forge/velocity/a2_fault_aware``). The optimizer state
is dropped (its moments have the old shapes): train with ``--no-resume_optimizer_state``.

    python scripts/forge/fault_aware/pad_checkpoint.py <in.pt> <out.pt> --extra 12
"""

import argparse
import os

import torch


def pad(state: dict, extra: int, drop: int = 0) -> dict:
    """The state dict with ``mlp.0.weight`` widened by ``extra`` zero columns, after removing its last ``drop``
    columns (they must be zero: an input a fine-tune only ever saw as zeros)."""
    state = dict(state)
    w = state["mlp.0.weight"]
    if drop:
        assert w[:, -drop:].abs().max() == 0, "dropped columns are not zero"
        w = w[:, :-drop]
    state["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra, dtype=w.dtype)], dim=1)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--extra", type=int, default=12)
    parser.add_argument("--drop", type=int, default=0, help="Remove this many (zero) last columns first.")
    args = parser.parse_args()
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    out = {k: v for k, v in ckpt.items() if k != "optimizer_state_dict"}
    for key in ("actor_state_dict", "critic_state_dict"):
        out[key] = pad(ckpt[key], args.extra, args.drop)
        x = torch.randn(4, ckpt[key]["mlp.0.weight"].shape[1] - args.drop)
        old = x @ ckpt[key]["mlp.0.weight"][:, : x.shape[1]].T
        new = torch.cat([x, torch.randn(4, args.extra)], 1) @ out[key]["mlp.0.weight"].T
        assert torch.allclose(old, new, atol=1e-5), key
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)
    print(f"{args.src} -> {args.dst}: first layer {tuple(ckpt['actor_state_dict']['mlp.0.weight'].shape)} -> "
          f"{tuple(out['actor_state_dict']['mlp.0.weight'].shape)}")  # fmt: skip


if __name__ == "__main__":
    main()
