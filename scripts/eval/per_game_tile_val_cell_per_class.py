#!/usr/bin/env python3
"""
Per **cell offset class** (0..8): validation cell-head accuracy for a saved ``PerGameTileCNN``.

Uses the same train/val split as ``train_per_game_action_tile_model.py`` with
``--holdout_last_seed``. Classes map to ``(Δgi, Δgj)`` in ``{-1,0,1}`` via
``per_game_tile_crops.class_to_cell_delta`` (row-major, ``i`` / column fast).

Example::

  python scripts/eval/per_game_tile_val_cell_per_class.py \\
    --checkpoint outputs/clean_up_20act_tile_cell_lr3e3_trainuntil.pt \\
    --val_dir datasets/rgb_npz_clean_up_20act_n100 \\
    --basename_contains clean_up \\
    --holdout_last_seed \\
    --precompute_crops
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import torch
from torch.utils.data import DataLoader

import per_game_tile_crops as ptc
from per_game_tile_model import PerGameTileCNN, PerGameTileNPZDataset, collect_npz_paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--val_dir", required=True, type=str)
    p.add_argument("--basename_contains", default=None)
    p.add_argument("--holdout_last_seed", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--precompute_crops", action="store_true", help="Faster eval on CPU/GPU.")
    return p.parse_args()


def _cell_label(c: int) -> str:
    dgi, dgj = ptc.class_to_cell_delta(c)
    return f"Δgi={dgi:+d},Δgj={dgj:+d}"


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    cell_px = int(ckpt["cell_px"])
    image_size = int(ckpt["image_size"])
    hidden = int(ckpt["hidden"])
    in_ch = int(ckpt["in_ch"])
    n_cls = int(ckpt["num_actions"])
    action_label_mode = str(ckpt.get("action_label_mode", "compact"))
    aux_meta = bool(ckpt.get("aux_meta_channels", False))
    input_noise_std = float(ckpt.get("input_noise_std", 0.0))
    input_mode = str(ckpt.get("input_mode", "rgb"))
    rgb_stack = str(ckpt.get("rgb_stack", "current"))
    pad_mode = str(ckpt.get("pad_mode", "constant"))
    pad_fill = ckpt.get("pad_fill")
    if pad_mode == "constant" and pad_fill is not None:
        pad_fill = tuple(float(x) for x in pad_fill)
    orig_ids: list[int] = list(ckpt.get("action_orig_ids") or [])

    paths = collect_npz_paths(args.val_dir, args.basename_contains)
    if not paths:
        raise SystemExit(f"No npz under {args.val_dir!r}")
    val_paths: list[str] | None = None
    train_paths = paths
    if args.holdout_last_seed:
        if len(paths) < 2:
            raise SystemExit("--holdout_last_seed needs ≥2 npz files")
        val_paths = [paths[-1]]
        train_paths = paths[:-1]

    if not val_paths:
        raise SystemExit("Need --holdout_last_seed (or extend script for full-dir val).")

    train_npz = [os.path.abspath(p) for p in train_paths]
    val_npz = [os.path.abspath(p) for p in val_paths]
    all_union = sorted(set(train_npz) | set(val_npz))
    aux_fi_ref = all_union if (aux_meta and val_npz) else None

    if action_label_mode != "compact":
        raise SystemExit("This script expects compact action labels for train-vocab alignment.")

    if orig_ids:
        compact_ref = list(orig_ids)
    else:
        ds_probe = PerGameTileNPZDataset(
            train_npz,
            cell_px=cell_px,
            image_size=image_size,
            num_actions=n_cls,
            pad_mode=pad_mode,
            pad_fill=pad_fill,
            action_label_mode=action_label_mode,
            aux_meta_channels=aux_meta,
            aux_fi_reference_paths=aux_fi_ref,
            input_noise_std=input_noise_std,
            input_mode=input_mode,
            rgb_stack=rgb_stack,
            precompute_crops=False,
        )
        compact_ref = list(ds_probe._action_orig_ids)

    ds_val = PerGameTileNPZDataset(
        val_npz,
        cell_px=cell_px,
        image_size=image_size,
        num_actions=n_cls,
        pad_mode=pad_mode,
        pad_fill=pad_fill,
        action_label_mode=action_label_mode,
        aux_meta_channels=aux_meta,
        compact_action_orig_ids=compact_ref,
        aux_fi_reference_paths=aux_fi_ref,
        input_noise_std=input_noise_std,
        input_mode=input_mode,
        rgb_stack=rgb_stack,
        precompute_crops=bool(args.precompute_crops),
    )
    n_val = len(ds_val)
    if n_val == 0:
        raise SystemExit("Val dataset empty after train-vocab filter.")

    ds_tr_index = PerGameTileNPZDataset(
        train_npz,
        cell_px=cell_px,
        image_size=image_size,
        num_actions=n_cls,
        pad_mode=pad_mode,
        pad_fill=pad_fill,
        action_label_mode=action_label_mode,
        aux_meta_channels=aux_meta,
        aux_fi_reference_paths=aux_fi_ref,
        input_noise_std=input_noise_std,
        input_mode=input_mode,
        rgb_stack=rgb_stack,
        precompute_crops=False,
    )
    train_cell_counts = torch.zeros(9, dtype=torch.long)
    for _fi, _t, _s, y_cell, _ya, _gi, _gj in ds_tr_index._index:
        train_cell_counts[int(y_cell)] += 1
    n_tr = len(ds_tr_index)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PerGameTileCNN(num_actions=n_cls, hidden=hidden, in_ch=in_ch).to(dev)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()

    dl = DataLoader(
        ds_val,
        batch_size=min(int(args.batch), max(1, n_val)),
        shuffle=False,
        num_workers=int(args.workers),
    )

    k = 9
    correct = torch.zeros(k, dtype=torch.long)
    total = torch.zeros(k, dtype=torch.long)
    tot_ok = tot_n = 0
    for xb, _ya, yc, _yo in dl:
        xb = xb.to(dev)
        yc = yc.to(dev)
        _la, lc, _lo = m(xb)
        pred = lc.argmax(-1)
        ok = pred == yc
        tot_ok += int(ok.sum().item())
        tot_n += len(yc)
        for c in range(k):
            mask = yc == c
            nc = int(mask.sum().item())
            if nc:
                total[c] += nc
                correct[c] += int((ok & mask).sum().item())

    print(f"checkpoint={args.checkpoint!r}", flush=True)
    print(f"val_npz={val_npz[0]!r}  n_val={n_val}  device={dev}", flush=True)
    print(f"overall_cell_acc%={100.0 * tot_ok / max(tot_n, 1):.2f}  (n={tot_n})", flush=True)
    print("", flush=True)
    print("train class counts (cell y, full train index):", train_cell_counts.tolist(), flush=True)
    print("train class %:", [round(100.0 * int(train_cell_counts[c]) / max(n_tr, 1), 2) for c in range(k)], flush=True)
    print("", flush=True)
    print("per_class (class_id, delta_label, n_val, acc%, correct)", flush=True)
    for c in range(k):
        n = int(total[c].item())
        if n == 0:
            print(f"  {c}  {_cell_label(c):16s}  n=0  (no val rows)", flush=True)
            continue
        acc = 100.0 * float(correct[c].item()) / float(n)
        print(
            f"  {c}  {_cell_label(c):16s}  n={n:4d}  acc={acc:6.2f}%  correct={int(correct[c].item())}",
            flush=True,
        )


if __name__ == "__main__":
    main()
