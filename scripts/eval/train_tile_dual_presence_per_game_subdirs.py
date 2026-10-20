#!/usr/bin/env python3
"""
Per-game dual-presence tile CNN sweep (one NPZ subfolder per game).

For each immediate child directory of ``--npz_parent`` (the **game key** = folder name),
trains two checkpoints with the same recipe as a single-game dual-gate run:

- **Data:** 20-action RGB NPZs (81 ``pixel_frames`` rows) under ``<npz_parent>/<game>/`` —
  same convention as ``rgb_npz_clean_up_20act_n100`` / ``TILE_RGB_20ACT.md``. Generate with
  ``run_all_mini_games_5act_tile_dual_gpu.py`` (``--skip_train --num_actions 20``) and ``train_cell_player_presence_mlp.py`` per game (see ``example_tile_eval_usage.sh``).
- **Presence:** ``--dual_presence_ckpt_template`` must resolve to a ``CellPresenceMLP`` trained with
  ``train_cell_player_presence_mlp.py`` using the **same** ``cell_px``, ``image_size``, and
  ``pad_mode=constant`` + magenta fill as presence training (see ``example_tile_eval_usage.sh``).
  Match clean_up by training with **``--first_timestep_only``** (rollout starts: logged cells usually
  coincide with a visible avatar); the dual gate still runs this MLP on crops at **each** logged
  endpoint ``t`` and ``t+1`` when filtering rows.
- ``rgb_stack=current_next``, ``precompute_crops=True``
- Optional **center** + required **dual** ``CellPresenceMLP`` gates (paths from templates)
- **Action** head: ``supervise=action`` (joint BCE + one-hot on compact ids)
- **Cell** head (position / 9-way offset): ``supervise=cell``

Train split: all ``*.npz`` in the subfolder **except** the last sorted path → train;
the last file → validation (same as ``--holdout_last_seed`` in the tile CLI). Requires
**≥2** NPZ files per game.

**Presence checkpoints (per game):** The dual gate must use a ``CellPresenceMLP`` trained
with the **same** ``cell_px``, ``image_size``, and padding policy as the tile run. Provide
``--dual_presence_ckpt_template`` with ``{game}`` replaced by the subdirectory name, e.g.::

    outputs/presence_ckpts/{game}_cell_presence_mlp.pt

With ``--dual_gt_cell_presence_gate_combine`` (default ``any``), training keeps rows where
the player is visible at **either** logged endpoint (``t`` or ``t+1``); use ``all`` to
require **both** cells to pass the presence model.

If the resolved path is missing, that game is marked **failed** (the gate is not dropped
silently).

**Target ~95% accuracy:** Use ``--train_until_train_acc_pct`` (default 95). This stops
when **train** accuracy for the **supervised** head(s) reaches the threshold (action-only
or cell-only → that head’s train acc). This is **not** the same as held-out val accuracy.

**Validation:** With one held-out seed, ``n_val_samples`` is often small — val percentages
are noisy. Optional ``--warn_val_action_below`` / ``--warn_val_cell_below`` plus
``--min_val_n`` only add a **warn** status in the manifest when val is below a threshold
and ``n_val_samples`` is large enough.

**Multi-GPU:** run two processes with the same ``--npz_parent`` / ``--out_root`` but
``--num_shards 2 --shard_id 0`` and ``--shard_id 1``, and different ``CUDA_VISIBLE_DEVICES``.
Each process writes ``summary_shard{k}.json`` / ``progress_shard{k}.jsonl`` (unless you
set ``--progress_jsonl`` explicitly). ``tail -f`` those JSONL files to watch val/train
metrics as each game finishes.

**Progress:** append-only JSON lines to ``--progress_jsonl`` after each training head
completes (action then cell), including val accuracies from the trainer return dict.

Example (after placing presence ``.pt`` files per game)::

  python3 scripts/eval/train_tile_dual_presence_per_game_subdirs.py \\
    --npz_parent datasets/rgb_npz_20act_by_game \\
    --out_root outputs/tile_dual_presence_sweep \\
    --dual_presence_ckpt_template outputs/presence/{game}_cell_presence_mlp.pt \\
    --train_until_train_acc_pct 95 --max_epochs 3000 \\
    --lr 3e-3 --batch 256 --hidden 128 --workers 0

Matches the clean_up-style recipe: ``pad_mode=constant``, ``pad_color=255,0,255``.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys
import time
import traceback
from typing import Any, Iterator, TextIO

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import train_per_game_action_tile_model as train_one


class _TeeIO:
    """Write to multiple text streams (e.g. terminal + log file)."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for f in self._streams:
            n = f.write(s)
        return n

    def flush(self) -> None:
        for f in self._streams:
            f.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def tee_stdout(log_path: str) -> Iterator[None]:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    lf = open(log_path, "w", encoding="utf-8", buffering=1)
    try:
        old = sys.stdout
        sys.stdout = _TeeIO(old, lf)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old  # type: ignore[assignment]
        lf.close()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz_parent", required=True, type=str, help="Parent dir: one subfolder per game, each with *.npz")
    p.add_argument("--out_root", required=True, type=str, help="Output root; writes <out_root>/<game>/…")
    p.add_argument(
        "--games",
        nargs="*",
        default=[],
        help="If set, only these subdirectory names (exact).",
    )
    p.add_argument(
        "--dual_presence_ckpt_template",
        required=True,
        type=str,
        help="Path template with {game} for CellPresenceMLP .pt (dual GT gate).",
    )
    p.add_argument(
        "--center_presence_ckpt_template",
        type=str,
        default=None,
        help="Optional path template with {game} for center-cell presence gate.",
    )
    p.add_argument("--dual_gt_cell_presence_prob_min", type=float, default=0.5)
    p.add_argument(
        "--dual_gt_cell_presence_gate_combine",
        type=str,
        choices=("all", "any"),
        default="any",
        help="Dual GT cell presence filter: 'all' = both t and t+1 cells must pass; "
        "'any' = keep rows where either passes (skip only when both absent). Default any.",
    )
    p.add_argument("--center_presence_prob_min", type=float, default=0.5)
    p.add_argument("--train_until_train_acc_pct", type=float, default=95.0)
    p.add_argument("--max_epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cell_px", type=int, default=32)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--pad_mode", choices=("constant", "reflect"), default="constant")
    p.add_argument("--pad_color", type=str, default="255,0,255")
    p.add_argument("--aux_meta", action="store_true", help="Pass aux_meta_channels=True to tile dataset.")
    p.add_argument("--action_filename", type=str, default="action.pt", help="Checkpoint name under each game dir.")
    p.add_argument("--cell_filename", type=str, default="cell.pt", help="Checkpoint name under each game dir.")
    p.add_argument(
        "--resume_action_template",
        type=str,
        default=None,
        help="Optional {game} template for --resume on action training.",
    )
    p.add_argument(
        "--resume_cell_template",
        type=str,
        default=None,
        help="Optional {game} template for --resume on cell training.",
    )
    p.add_argument(
        "--warn_val_action_below",
        type=float,
        default=None,
        help="If set with --min_val_n: warn when val_action_acc_pct is below this.",
    )
    p.add_argument("--warn_val_cell_below", type=float, default=None)
    p.add_argument("--min_val_n", type=int, default=20, help="Min val samples to apply val warn thresholds.")
    p.add_argument("--num_shards", type=int, default=1, help="Split game list across workers (e.g. 2 for 2 GPUs).")
    p.add_argument("--shard_id", type=int, default=0, help="This worker handles games where index %% num_shards == shard_id.")
    p.add_argument(
        "--progress_jsonl",
        type=str,
        default=None,
        help="Append progress JSON lines here (default: out_root/progress.jsonl or progress_shard{k}.jsonl).",
    )
    p.add_argument(
        "--val_log_interval",
        type=int,
        default=5,
        help="Passed to tile trainer: print val_snapshot every N epochs (0=off). Default 5.",
    )
    return p.parse_args()


def _append_progress(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {**obj, "ts": time.time()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


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


def _apply_warn(
    row: dict[str, Any],
    *,
    warn_a: float | None,
    warn_c: float | None,
    min_n: int,
) -> None:
    warns: list[str] = []
    na = row.get("action", {}).get("n_val_samples") or 0
    nc = row.get("cell", {}).get("n_val_samples") or 0
    if warn_a is not None and na >= min_n:
        va = float(row.get("action", {}).get("val_action_acc_pct") or 0.0)
        if va < warn_a:
            warns.append(f"val_action_acc_pct={va:.2f} < {warn_a} (n_val={na})")
    if warn_c is not None and nc >= min_n:
        vc = float(row.get("cell", {}).get("val_cell_acc_pct") or 0.0)
        if vc < warn_c:
            warns.append(f"val_cell_acc_pct={vc:.2f} < {warn_c} (n_val={nc})")
    if warns:
        row["warnings"] = warns
        if row.get("status") == "ok":
            row["status"] = "warn"


def main() -> None:
    args = _parse_args()
    n_sh = int(args.num_shards)
    sid = int(args.shard_id)
    if n_sh < 1 or sid < 0 or sid >= n_sh:
        raise SystemExit("--num_shards must be >= 1 and 0 <= --shard_id < --num_shards")

    only = set(args.games) if args.games else None
    games_all = _discover_games(args.npz_parent, only)
    games = [g for i, g in enumerate(games_all) if i % n_sh == sid]
    if not games_all:
        raise SystemExit(f"No game subdirectories under {args.npz_parent!r} (after --games filter).")
    if not games:
        raise SystemExit(
            f"No games assigned to shard {sid}/{n_sh} (total games={len(games_all)}). "
            "Check --num_shards / --shard_id."
        )

    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    if args.progress_jsonl:
        progress_path = os.path.abspath(args.progress_jsonl)
    else:
        progress_path = os.path.join(
            out_root,
            f"progress_shard{sid}.jsonl" if n_sh > 1 else "progress.jsonl",
        )

    summary: dict[str, Any] = {
        "npz_parent": os.path.abspath(args.npz_parent),
        "out_root": out_root,
        "num_shards": n_sh,
        "shard_id": sid,
        "progress_jsonl": progress_path,
        "train_until_train_acc_pct": float(args.train_until_train_acc_pct),
        "max_epochs": int(args.max_epochs),
        "dual_presence_ckpt_template": str(args.dual_presence_ckpt_template),
        "center_presence_ckpt_template": args.center_presence_ckpt_template,
        "dual_gt_cell_presence_prob_min": float(args.dual_gt_cell_presence_prob_min),
        "dual_gt_cell_presence_gate_combine": str(args.dual_gt_cell_presence_gate_combine),
        "val_log_interval": int(args.val_log_interval),
        "games": [],
    }

    print(
        f"shard {sid}/{n_sh}  n_games_this_shard={len(games)}  total_games={len(games_all)}  "
        f"progress_jsonl={progress_path!r}",
        flush=True,
    )
    for game in games:
        row: dict[str, Any] = {"game_key": game, "status": "ok", "action": {}, "cell": {}}
        paths = _npz_paths_for_game(args.npz_parent, game)
        row["n_npz"] = len(paths)
        if len(paths) < 2:
            row["status"] = "skipped"
            row["reason"] = "need at least 2 *.npz files for holdout_last_seed split"
            summary["games"].append(row)
            print(f"[skip] {game}: {row['reason']}", flush=True)
            continue

        dual_ckpt = _template_path(args.dual_presence_ckpt_template, game)
        if not os.path.isfile(dual_ckpt):
            row["status"] = "failed"
            row["reason"] = f"dual presence checkpoint missing: {dual_ckpt!r}"
            summary["games"].append(row)
            print(f"[fail] {game}: {row['reason']}", flush=True)
            continue

        center_ckpt: str | None = None
        if args.center_presence_ckpt_template:
            center_ckpt = _template_path(args.center_presence_ckpt_template, game)
            if not os.path.isfile(center_ckpt):
                row["status"] = "failed"
                row["reason"] = f"center presence checkpoint missing: {center_ckpt!r}"
                summary["games"].append(row)
                print(f"[fail] {game}: {row['reason']}", flush=True)
                continue

        train_paths = paths[:-1]
        val_paths = [paths[-1]]
        game_out = os.path.join(out_root, game)
        os.makedirs(game_out, exist_ok=True)

        resume_a: str | None = None
        if args.resume_action_template:
            ra = _template_path(args.resume_action_template, game)
            resume_a = ra if os.path.isfile(ra) else None
            if args.resume_action_template and resume_a is None:
                print(f"[note] {game}: resume_action template file not found, training from scratch: {ra!r}", flush=True)

        resume_c: str | None = None
        if args.resume_cell_template:
            rc = _template_path(args.resume_cell_template, game)
            resume_c = rc if os.path.isfile(rc) else None
            if args.resume_cell_template and resume_c is None:
                print(f"[note] {game}: resume_cell template file not found, training from scratch: {rc!r}", flush=True)

        common_kw: dict[str, Any] = {
            "cell_px": int(args.cell_px),
            "image_size": int(args.image_size),
            "num_actions": None,
            "action_label_mode": "compact",
            "aux_meta_channels": bool(args.aux_meta),
            "val_npz_paths": val_paths,
            "input_noise_std": 0.0,
            "input_mode": "rgb",
            "rgb_stack": "current_next",
            "batch": int(args.batch),
            "lr": float(args.lr),
            "workers": int(args.workers),
            "pad_mode": str(args.pad_mode),
            "pad_color": str(args.pad_color),
            "hidden": int(args.hidden),
            "full_batch": False,
            "verify_train_acc": True,
            "optimizer": "adam",
            "train_until_train_acc_pct": float(args.train_until_train_acc_pct),
            "max_epochs": int(args.max_epochs),
            "precompute_crops": True,
            "action_decomposition": "joint",
            "restore_best_val_cell": False,
            "center_presence_gate_ckpt": center_ckpt,
            "center_presence_prob_min": float(args.center_presence_prob_min),
            "dual_gt_cell_presence_gate_ckpt": dual_ckpt,
            "dual_gt_cell_presence_prob_min": float(args.dual_gt_cell_presence_prob_min),
            "dual_gt_cell_presence_gate_combine": str(args.dual_gt_cell_presence_gate_combine),
            "val_log_interval": (int(args.val_log_interval) if int(args.val_log_interval) > 0 else None),
        }

        action_out = os.path.join(game_out, args.action_filename)
        cell_out = os.path.join(game_out, args.cell_filename)
        log_a = os.path.join(game_out, "train_action.log")
        log_c = os.path.join(game_out, "train_cell.log")

        try:
            with tee_stdout(log_a):
                print(f"=== {game} ACTION ===", flush=True)
                info_a = train_one.train_to_checkpoint(
                    train_paths,
                    action_out,
                    supervise="action",
                    epochs=max(1, int(args.max_epochs)),
                    resume_path=resume_a,
                    **common_kw,
                )
            row["action"] = dict(info_a)
            _append_progress(
                progress_path,
                {
                    "event": "action_done",
                    "game_key": game,
                    "shard_id": sid,
                    "num_shards": n_sh,
                    "val_action_acc_pct": info_a.get("val_action_acc_pct"),
                    "train_action_acc_pct": info_a.get("train_action_acc_pct"),
                    "n_val_samples": info_a.get("n_val_samples"),
                    "epochs": info_a.get("epochs"),
                    "checkpoint": os.path.abspath(action_out),
                },
            )
        except Exception:
            row["status"] = "failed"
            row["reason"] = f"action training exception:\n{traceback.format_exc()}"
            summary["games"].append(row)
            print(f"[fail] {game}: action training crashed (see {log_a!r})", flush=True)
            _append_progress(
                progress_path,
                {
                    "event": "action_failed",
                    "game_key": game,
                    "shard_id": sid,
                    "num_shards": n_sh,
                    "reason": row["reason"][:2000],
                },
            )
            continue

        try:
            with tee_stdout(log_c):
                print(f"=== {game} CELL ===", flush=True)
                info_c = train_one.train_to_checkpoint(
                    train_paths,
                    cell_out,
                    supervise="cell",
                    epochs=max(1, int(args.max_epochs)),
                    resume_path=resume_c,
                    **common_kw,
                )
            row["cell"] = dict(info_c)
            _append_progress(
                progress_path,
                {
                    "event": "cell_done",
                    "game_key": game,
                    "shard_id": sid,
                    "num_shards": n_sh,
                    "val_cell_acc_pct": info_c.get("val_cell_acc_pct"),
                    "train_cell_acc_pct": info_c.get("train_cell_acc_pct"),
                    "n_val_samples": info_c.get("n_val_samples"),
                    "epochs": info_c.get("epochs"),
                    "checkpoint": os.path.abspath(cell_out),
                },
            )
        except Exception:
            row["status"] = "failed"
            row["reason"] = f"cell training exception (action ckpt may exist):\n{traceback.format_exc()}"
            summary["games"].append(row)
            print(f"[fail] {game}: cell training crashed (see {log_c!r})", flush=True)
            _append_progress(
                progress_path,
                {
                    "event": "cell_failed",
                    "game_key": game,
                    "shard_id": sid,
                    "num_shards": n_sh,
                    "reason": row["reason"][:2000],
                },
            )
            continue

        _apply_warn(
            row,
            warn_a=args.warn_val_action_below,
            warn_c=args.warn_val_cell_below,
            min_n=int(args.min_val_n),
        )
        summary["games"].append(row)
        st = row.get("status", "ok")
        print(
            f"[{st}] {game}  action_val_acc={float(row['action'].get('val_action_acc_pct') or 0):.2f}%  "
            f"cell_val_acc={float(row['cell'].get('val_cell_acc_pct') or 0):.2f}%",
            flush=True,
        )
        _append_progress(
            progress_path,
            {
                "event": "game_done",
                "game_key": game,
                "shard_id": sid,
                "num_shards": n_sh,
                "status": st,
                "val_action_acc_pct": row["action"].get("val_action_acc_pct"),
                "val_cell_acc_pct": row["cell"].get("val_cell_acc_pct"),
                "train_action_acc_pct": row["action"].get("train_action_acc_pct"),
                "train_cell_acc_pct": row["cell"].get("train_cell_acc_pct"),
                "warnings": row.get("warnings"),
            },
        )

    sum_name = f"summary_shard{sid}.json" if n_sh > 1 else "summary.json"
    sum_path = os.path.join(out_root, sum_name)
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote manifest {sum_path}", flush=True)

    jl_name = f"summary_shard{sid}.jsonl" if n_sh > 1 else "summary.jsonl"
    jl = os.path.join(out_root, jl_name)
    with open(jl, "w", encoding="utf-8") as f:
        for row in summary["games"]:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"Wrote {jl}", flush=True)


if __name__ == "__main__":
    main()
