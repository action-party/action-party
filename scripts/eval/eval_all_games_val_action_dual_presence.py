#!/usr/bin/env python3
"""
Batch **held-out val** action accuracy for every game under ``--npz_parent``.

Each immediate subdirectory is a **game key** (same layout as
``train_tile_dual_presence_per_game_subdirs.py``). For each game with ≥2 ``*.npz`` files,
uses the last sorted file as validation and the rest as train (``holdout_last_seed``),
then evaluates ``--action_ckpt_template`` with optional **dual GT cell presence** filtering.

Default **``--dual_gt_cell_presence_gate_combine any``** matches the recipe where we
**skip** timesteps only when the player is **not** detected at **both** logged cells
(``t`` and ``t+1``); pass ``all`` for the stricter both-must-pass filter.

Example (tile sweep output + per-game presence checkpoints)::

  python3 scripts/eval/eval_all_games_val_action_dual_presence.py \\
    --npz_parent datasets/rgb_npz_20act_by_game \\
    --action_ckpt_template 'outputs/tile_dual/{game}/action.pt' \\
    --dual_presence_ckpt_template 'outputs/presence/{game}_cell_presence_mlp.pt' \\
    --output outputs/tile_dual/val_action_dual_presence_any.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from typing import Any

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

from per_game_tile_val_action_per_class import eval_holdout_val_action_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz_parent", required=True, type=str, help="Parent: one subdir per game with *.npz")
    p.add_argument(
        "--action_ckpt_template",
        required=True,
        type=str,
        help="Path with {game} to PerGameTileCNN action .pt (e.g. out_root/<game>/action.pt).",
    )
    p.add_argument(
        "--dual_presence_ckpt_template",
        type=str,
        default=None,
        help="Optional {game} template for CellPresenceMLP (dual GT gate). Omit for unfiltered val.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_gate_combine",
        type=str,
        choices=("all", "any"),
        default="any",
        help="Dual gate combine mode (default any = skip only when both endpoints fail).",
    )
    p.add_argument("--dual_gt_cell_presence_prob_min", type=float, default=0.5)
    p.add_argument("--games", nargs="*", default=[], help="If set, only these subdirectory names.")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--precompute_crops", action="store_true", help="Recommended on GPU.")
    p.add_argument("--output", type=str, required=True, help="Write summary JSON here.")
    return p.parse_args()


def _template_path(tpl: str, game: str) -> str:
    return tpl.replace("{game}", game)


def _discover_games(npz_parent: str, only: set[str] | None) -> list[str]:
    parent = os.path.abspath(npz_parent)
    if not os.path.isdir(parent):
        raise SystemExit(f"--npz_parent is not a directory: {parent!r}")
    names: list[str] = []
    for name in sorted(os.listdir(parent)):
        sub = os.path.join(parent, name)
        if not os.path.isdir(sub):
            continue
        if only is not None and name not in only:
            continue
        names.append(name)
    return names


def _npz_paths_for_game(npz_parent: str, game: str) -> list[str]:
    sub = os.path.join(os.path.abspath(npz_parent), game)
    paths = sorted(glob.glob(os.path.join(sub, "*.npz")))
    return [os.path.abspath(p) for p in paths]


def main() -> None:
    args = _parse_args()
    only = set(args.games) if args.games else None
    games = _discover_games(args.npz_parent, only)
    if not games:
        raise SystemExit(f"No game subdirs under {args.npz_parent!r}")

    rows: list[dict[str, Any]] = []
    sum_ok = sum_n = 0
    for game in games:
        paths = _npz_paths_for_game(args.npz_parent, game)
        rec: dict[str, Any] = {"game_key": game, "status": "ok", "n_npz": len(paths)}
        if len(paths) < 2:
            rec["status"] = "skipped"
            rec["reason"] = "need at least 2 *.npz for holdout_last_seed"
            rows.append(rec)
            print(f"[skip] {game}: {rec['reason']}", flush=True)
            continue

        ckpt_path = _template_path(args.action_ckpt_template, game)
        rec["action_checkpoint"] = ckpt_path
        if not os.path.isfile(ckpt_path):
            rec["status"] = "failed"
            rec["reason"] = f"missing action checkpoint: {ckpt_path!r}"
            rows.append(rec)
            print(f"[fail] {game}: {rec['reason']}", flush=True)
            continue

        dual_ckpt: str | None = None
        if args.dual_presence_ckpt_template:
            dual_ckpt = _template_path(args.dual_presence_ckpt_template, game)
            rec["dual_presence_ckpt"] = dual_ckpt
            if not os.path.isfile(dual_ckpt):
                rec["status"] = "failed"
                rec["reason"] = f"missing dual presence checkpoint: {dual_ckpt!r}"
                rows.append(rec)
                print(f"[fail] {game}: {rec['reason']}", flush=True)
                continue

        train_paths = paths[:-1]
        val_paths = [paths[-1]]
        try:
            metrics = eval_holdout_val_action_metrics(
                checkpoint_path=ckpt_path,
                train_npz_paths=train_paths,
                val_npz_paths=val_paths,
                batch=int(args.batch),
                workers=int(args.workers),
                precompute_crops=bool(args.precompute_crops),
                dual_gt_cell_presence_gate_ckpt=dual_ckpt,
                dual_gt_cell_presence_prob_min=float(args.dual_gt_cell_presence_prob_min),
                dual_gt_cell_presence_gate_combine=str(args.dual_gt_cell_presence_gate_combine),
            )
        except Exception:
            rec["status"] = "failed"
            rec["reason"] = traceback.format_exc()
            rows.append(rec)
            print(f"[fail] {game}: exception during eval", flush=True)
            continue

        rec["metrics"] = metrics
        rec["overall_action_acc_pct"] = metrics["overall_action_acc_pct"]
        rec["n_val_samples"] = metrics["n_val_samples"]
        sum_ok += int(metrics["tot_correct"])
        sum_n += int(metrics["tot_n"])
        rows.append(rec)
        print(
            f"[ok] {game}  val_action={metrics['overall_action_acc_pct']:.2f}%  "
            f"n_val={metrics['n_val_samples']}  val_npz={os.path.basename(metrics['val_npz'])}",
            flush=True,
        )

    overall = round(100.0 * sum_ok / max(sum_n, 1), 4) if sum_n else 0.0
    out = {
        "metric": "batch_holdout_val_action_dual_presence",
        "npz_parent": os.path.abspath(args.npz_parent),
        "action_ckpt_template": args.action_ckpt_template,
        "dual_presence_ckpt_template": args.dual_presence_ckpt_template,
        "dual_gt_cell_presence_gate_combine": str(args.dual_gt_cell_presence_gate_combine),
        "dual_gt_cell_presence_prob_min": float(args.dual_gt_cell_presence_prob_min),
        "n_games": len(games),
        "micro_overall_action_acc_pct": overall,
        "micro_tot_correct": int(sum_ok),
        "micro_tot_n": int(sum_n),
        "per_game": rows,
    }
    outp = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps(
            {
                "metric": out["metric"],
                "n_games": out["n_games"],
                "micro_overall_action_acc_pct": out["micro_overall_action_acc_pct"],
                "micro_tot_n": out["micro_tot_n"],
                "saved": outp,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
