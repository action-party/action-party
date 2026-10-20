#!/usr/bin/env python3
"""
Per **compact** action class: validation action accuracy for a saved ``PerGameTileCNN``.

Rebuilds the same train/val file split as ``train_per_game_action_tile_model.py`` with
``--holdout_last_seed``, then evaluates **only** the held-out NPZ using
``action_orig_ids`` from the checkpoint (train vocabulary).

Example::

  python scripts/eval/per_game_tile_val_action_per_class.py \\
    --checkpoint outputs/clean_up_20act_tile_action_lr3e3_trainuntil.pt \\
    --val_dir datasets/rgb_npz_clean_up_20act_n100 \\
    --basename_contains clean_up \\
    --holdout_last_seed \\
    --dual_gt_cell_presence_gate_ckpt outputs/clean_up_20act_cell_presence_mlp.pt \\
    --dual_gt_cell_presence_gate_combine any
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import torch
from torch.utils.data import DataLoader

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
    p.add_argument(
        "--dual_gt_cell_presence_gate_ckpt",
        type=str,
        default=None,
        help="Optional CellPresenceMLP .pt: filter val rows by GT player cell at t and t+1.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_prob_min",
        type=float,
        default=0.5,
        help="Per-endpoint threshold with --dual_gt_cell_presence_gate_ckpt.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_gate_combine",
        type=str,
        choices=("all", "any"),
        default="all",
        help="With dual gate: 'all' = both cells must pass; 'any' = skip only when both fail.",
    )
    return p.parse_args()


@torch.no_grad()
def eval_holdout_val_action_metrics(
    *,
    checkpoint_path: str,
    train_npz_paths: list[str],
    val_npz_paths: list[str],
    batch: int = 256,
    workers: int = 0,
    precompute_crops: bool = True,
    dual_gt_cell_presence_gate_ckpt: str | None = None,
    dual_gt_cell_presence_prob_min: float = 0.5,
    dual_gt_cell_presence_gate_combine: str = "all",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Same split semantics as the CLI: ``train_npz_paths`` = train seeds,
    ``val_npz_paths`` = held-out NPZ list (usually one file). Compact labels only.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cell_px = int(ckpt["cell_px"])
    image_size = int(ckpt["image_size"])
    hidden = int(ckpt["hidden"])
    in_ch = int(ckpt["in_ch"])
    n_cls = int(ckpt["num_actions"])
    action_label_mode = str(ckpt.get("action_label_mode", "compact"))
    if action_label_mode != "compact":
        raise ValueError("eval_holdout_val_action_metrics requires action_label_mode=compact")
    aux_meta = bool(ckpt.get("aux_meta_channels", False))
    input_noise_std = float(ckpt.get("input_noise_std", 0.0))
    input_mode = str(ckpt.get("input_mode", "rgb"))
    rgb_stack = str(ckpt.get("rgb_stack", "current"))
    pad_mode = str(ckpt.get("pad_mode", "constant"))
    pad_fill = ckpt.get("pad_fill")
    if pad_mode == "constant" and pad_fill is not None:
        pad_fill = tuple(float(x) for x in pad_fill)
    orig_ids: list[int] = list(ckpt.get("action_orig_ids") or [])

    train_npz = [os.path.abspath(p) for p in train_npz_paths]
    val_npz = [os.path.abspath(p) for p in val_npz_paths]
    if not val_npz:
        raise ValueError("val_npz_paths must be non-empty")
    all_union = sorted(set(train_npz) | set(val_npz))
    aux_fi_ref = all_union if (aux_meta and val_npz) else None

    if orig_ids:
        compact_ref = list(orig_ids)
    else:
        ds_tr = PerGameTileNPZDataset(
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
            precompute_crops=bool(precompute_crops),
        )
        compact_ref = list(ds_tr._action_orig_ids)

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
        precompute_crops=bool(precompute_crops),
        dual_gt_cell_presence_gate_ckpt=dual_gt_cell_presence_gate_ckpt,
        dual_gt_cell_presence_prob_min=float(dual_gt_cell_presence_prob_min),
        dual_gt_cell_presence_gate_combine=str(dual_gt_cell_presence_gate_combine),
    )
    n_val = len(ds_val)
    if n_val == 0:
        raise RuntimeError("Val dataset empty after train-vocab filter.")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PerGameTileCNN(num_actions=n_cls, hidden=hidden, in_ch=in_ch).to(dev)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()

    dl = DataLoader(
        ds_val,
        batch_size=min(int(batch), max(1, n_val)),
        shuffle=False,
        num_workers=int(workers),
    )

    k = n_cls
    correct = torch.zeros(k, dtype=torch.long)
    total = torch.zeros(k, dtype=torch.long)
    tot_ok = tot_n = 0
    for xb, ya, _, _ in dl:
        xb = xb.to(dev)
        ya = ya.to(dev)
        la, _, _ = m(xb)
        pred = la.argmax(-1)
        ok = pred == ya
        tot_ok += int(ok.sum().item())
        tot_n += len(ya)
        for c in range(k):
            mask = ya == c
            nc = int(mask.sum().item())
            if nc:
                total[c] += nc
                correct[c] += int((ok & mask).sum().item())

    dual_stats = getattr(ds_val, "_dual_gt_cell_presence_gate_stats", None)
    per_class: list[dict[str, Any]] = []
    for c in range(k):
        oid = compact_ref[c] if c < len(compact_ref) else -1
        n = int(total[c].item())
        per_class.append(
            {
                "compact_idx": int(c),
                "mp_action_id": int(oid),
                "n": int(n),
                "correct": int(correct[c].item()) if n else 0,
                "acc_pct": round(100.0 * float(correct[c].item()) / float(n), 4) if n else 0.0,
            }
        )

    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "val_npz": val_npz[0],
        "n_val_samples": int(n_val),
        "tot_correct": int(tot_ok),
        "tot_n": int(tot_n),
        "overall_action_acc_pct": round(100.0 * tot_ok / max(tot_n, 1), 4),
        "dual_gt_cell_presence_gate_ckpt": os.path.abspath(dual_gt_cell_presence_gate_ckpt)
        if dual_gt_cell_presence_gate_ckpt
        else None,
        "dual_gt_cell_presence_prob_min": float(dual_gt_cell_presence_prob_min),
        "dual_gt_cell_presence_gate_combine": str(dual_gt_cell_presence_gate_combine),
        "dual_gt_cell_presence_gate_stats": dual_stats,
        "per_class": per_class,
        "device": str(dev),
    }


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    print(
        f"ckpt action_loss={str(ckpt.get('action_loss', 'ce'))!r} "
        f"action_decomposition={str(ckpt.get('action_decomposition', 'joint'))!r}",
        flush=True,
    )

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

    out = eval_holdout_val_action_metrics(
        checkpoint_path=args.checkpoint,
        train_npz_paths=[os.path.abspath(p) for p in train_paths],
        val_npz_paths=[os.path.abspath(p) for p in val_paths],
        batch=int(args.batch),
        workers=int(args.workers),
        precompute_crops=bool(args.precompute_crops),
        dual_gt_cell_presence_gate_ckpt=args.dual_gt_cell_presence_gate_ckpt,
        dual_gt_cell_presence_prob_min=float(args.dual_gt_cell_presence_prob_min),
        dual_gt_cell_presence_gate_combine=str(args.dual_gt_cell_presence_gate_combine),
    )

    print(f"checkpoint={args.checkpoint!r}", flush=True)
    if args.dual_gt_cell_presence_gate_ckpt:
        print(
            f"dual_gt_presence_gate ckpt={args.dual_gt_cell_presence_gate_ckpt!r} "
            f"combine={args.dual_gt_cell_presence_gate_combine!r} "
            f"prob_min={args.dual_gt_cell_presence_prob_min}  stats={out['dual_gt_cell_presence_gate_stats']!r}",
            flush=True,
        )
    print(f"val_npz={out['val_npz']!r}  n_val={out['n_val_samples']}  device={out['device']}", flush=True)
    print(f"overall_action_acc%={out['overall_action_acc_pct']:.2f}  (n={out['tot_n']})", flush=True)
    print("", flush=True)
    print("per_class (compact_idx, mp_action_id, n, acc%, correct)", flush=True)
    for row in out["per_class"]:
        c = row["compact_idx"]
        oid = row["mp_action_id"]
        n = row["n"]
        if n == 0:
            print(f"  {c:3d}  orig_id={oid:4d}  n=0  (no val rows)", flush=True)
            continue
        print(
            f"  {c:3d}  orig_id={oid:4d}  n={n:4d}  acc={row['acc_pct']:6.2f}%  correct={row['correct']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
