#!/usr/bin/env python3
"""
Overnight pipeline: **all** ``MINI_GAMES_512`` games, per-game RGB rollouts, then tile CNNs.

For each game (in parallel across **two worker processes / two GPUs**):

1. **Rollouts → NPZ** — ``create_all_games_dataset.py`` with ``--num_actions N`` (default **20**).
   Each logged Melting-Pot action uses 4 env substeps (1 + 3 NOOPs), so **N=20 ⇒ 81 pixel rows**
   per NPZ (same chunking as ``rgb_npz_clean_up_20act_*``). Pass ``--num_actions 5`` only if you
   want the older ``rgb_npz_clean_up_5act_*`` / ``rgb_npz_5act512_*`` rollout length.

   **Sweep vs one-off:** the default **all-games** run is **46 games × 100 seeds** with **fixed 40 Adam epochs**
   unless you pass ``--train_until_train_acc_pct`` / ``--max_epochs``. Single-game jobs like
   ``train_per_game_action_tile_model.py`` often uses a larger epoch cap and train-until.

2. **Train two checkpoints** (same backbone, different supervision; **compact** action ids
   are **re-derived per game** from that game's NPZs, so different Melting-Pot
   ``action_spec`` sizes are handled automatically):

   - ``supervise_action.pt`` — action head only (CE on action).
   - ``supervise_cell.pt`` — 3×3 next-cell offset head only (CE on cell; for inference
     without GT position when combined with your rollout logic).

Both trains use ``--rgb_stack current_next``, ``--precompute_crops``, ``--holdout_last_seed``
(needs ≥2 NPZs), and write logs next to the checkpoints.

**GPUs:** each worker pins ``CUDA_VISIBLE_DEVICES`` to one id from ``--gpu_ids`` (default
``0,1``). RGB NPZ generation is **CPU-only** in the dataset script (no VAE), so two
workers can generate in parallel without competing for CUDA; **tile training** uses
the assigned GPU.

Examples::

  # Full overnight run (2 GPUs)
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py

  # Dry-run commands only
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --dry_run

  # Regenerate NPZs only, skip training
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --skip_train

  # Training only (NPZs already on disk)
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --skip_gen

  # Subset + single GPU
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --gpu_ids 0 --num_workers 1 \\
      --max_games 3 --games clean_up coins

  # Legacy 5-action chunking (optional)
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --games clean_up --num_workers 1 \\
      --num_actions 5 --npz_parent datasets/rgb_npz_5act512_tile_overnight

  # 20-action clean_up-style + train-until (Adam)
  python scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py --games clean_up --num_workers 1 \\
      --num_actions 20 --train_until_train_acc_pct 95 --max_epochs 3000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_mini_games() -> list[str]:
    d = _repo_root() / "scripts" / "dataset"
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from mini_maps import MINI_GAMES_512  # noqa: E402

    return list(MINI_GAMES_512)


def _safe_stem(game: str) -> str:
    return game.replace("/", "_")


def _count_game_npz(npz_dir: Path, game: str) -> int:
    return len(list(npz_dir.glob(f"{game}_s*.npz")))


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    extra_env: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n{'='*80}\n$ {' '.join(cmd)}\n{'='*80}\n")
        logf.flush()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            logf.write(line)
            logf.flush()
        return int(p.wait())


def _one_game_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Runs in a child process. Executes gen (optional) + train action + train cell
    for a single game, pinning CUDA to ``gpu_id`` for training steps.
    """
    repo = Path(payload["repo"])
    game = str(payload["game"])
    gpu_id = int(payload["gpu_id"])
    samples = int(payload["samples_per_game"])
    seed = int(payload["seed"])
    skip_gen = bool(payload["skip_gen"])
    skip_train = bool(payload["skip_train"])
    resume_npz = bool(payload["resume_npz"])
    dry_run = bool(payload["dry_run"])
    epochs = int(payload["epochs"])
    batch = int(payload["batch"])
    lr = float(payload["lr"])
    hidden = int(payload["hidden"])
    workers = int(payload["workers"])
    num_actions = int(payload.get("num_actions", 5))
    train_until = payload.get("train_until_train_acc_pct")
    max_epochs = payload.get("max_epochs")

    py = sys.executable
    gen_py = repo / "scripts" / "dataset" / "create_all_games_dataset.py"
    train_py = repo / "scripts" / "eval" / "train_per_game_action_tile_model.py"

    npz_dir = Path(payload["npz_dir"])
    out_sub = Path(payload["out_sub"])
    ck_action = Path(payload["ck_action"])
    ck_cell = Path(payload["ck_cell"])
    log_dir = Path(payload["log_dir"])

    row: dict[str, Any] = {
        "game": game,
        "gpu_id": gpu_id,
        "npz_dir": str(npz_dir),
        "out_sub": str(out_sub),
        "checkpoint_action": str(ck_action),
        "checkpoint_cell": str(ck_cell),
    }

    cuda_train = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    # ---------- RGB NPZ generation (CPU in dataset script; env left default) ----------
    if not skip_gen:
        gen_cmd = [
            py,
            str(gen_py),
            "--games",
            game,
            "--samples_per_game",
            str(samples),
            "--canvas_size",
            "512",
            "--rgb_npz_dir",
            str(npz_dir),
            "--num_actions",
            str(num_actions),
            "--seed",
            str(seed),
        ]
        if resume_npz:
            gen_cmd.append("--resume")
        row["gen_cmd"] = " ".join(gen_cmd)
        print(f"[gen] {game}  (npz_dir={npz_dir})", flush=True)
        if dry_run:
            row["gen_status"] = "dry_run"
        else:
            npz_dir.mkdir(parents=True, exist_ok=True)
            rc = _run_subprocess(
                gen_cmd,
                cwd=repo,
                log_path=log_dir / "create_rgb_npz.log",
                extra_env=None,
            )
            row["gen_exit"] = rc
            row["gen_status"] = "ok" if rc == 0 else "gen_failed"
            if rc != 0:
                row["train_action_status"] = "skipped_gen_failed"
                row["train_cell_status"] = "skipped_gen_failed"
                row["n_npz"] = _count_game_npz(npz_dir, game)
                return row
    else:
        row["gen_status"] = "skipped_skip_gen"

    n_npz = _count_game_npz(npz_dir, game)
    row["n_npz"] = n_npz
    if n_npz < 2:
        row["train_action_status"] = "skipped_need_2_npz"
        row["train_cell_status"] = "skipped_need_2_npz"
        return row

    if skip_train:
        row["train_action_status"] = "skipped_skip_train"
        row["train_cell_status"] = "skipped_skip_train"
        return row

    # basename_contains: restrict to this game's files (subdir may be game-specific only,
    # but substring is safe for odd layouts).
    common_train = [
        py,
        str(train_py),
        "--val_dir",
        str(npz_dir),
        "--basename_contains",
        game,
        "--holdout_last_seed",
        "--rgb_stack",
        "current_next",
        "--precompute_crops",
        "--epochs",
        str(epochs),
        "--batch",
        str(batch),
        "--lr",
        str(lr),
        "--hidden",
        str(hidden),
        "--workers",
        str(workers),
    ]
    if train_until is not None:
        common_train.extend(["--train_until_train_acc_pct", str(float(train_until))])
        if max_epochs is not None:
            common_train.extend(["--max_epochs", str(int(max_epochs))])

    def _train(supervise: str, out_ckpt: Path, log_name: str) -> tuple[int, str]:
        cmd = [
            *common_train,
            "--supervise",
            supervise,
            "--output",
            str(out_ckpt),
        ]
        key = f"train_cmd_{supervise.replace('+', '_')}"
        row[key] = " ".join(cmd)
        if dry_run:
            return 0, "dry_run"
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        rc = _run_subprocess(
            cmd,
            cwd=repo,
            log_path=log_dir / log_name,
            extra_env=cuda_train,
        )
        return rc, "ok" if rc == 0 else "train_failed"

    out_sub.mkdir(parents=True, exist_ok=True)

    rc_a, st_a = _train("action", ck_action, "train_supervise_action.log")
    row["train_action_exit"] = rc_a
    row["train_action_status"] = st_a

    rc_c, st_c = _train("cell", ck_cell, "train_supervise_cell.log")
    row["train_cell_exit"] = rc_c
    row["train_cell_status"] = st_c

    return row


def main() -> None:
    repo = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples_per_game", type=int, default=100, help="NPZ seeds per game (default 100).")
    ap.add_argument(
        "--npz_parent",
        type=str,
        default=str(repo / "datasets" / "rgb_npz_20act_tile_overnight"),
        help="Parent directory; each game is datasets/.../<game_stem>/",
    )
    ap.add_argument(
        "--out_parent",
        type=str,
        default=str(repo / "outputs" / "per_game_20act_tile_supervised_dualgpu"),
        help="Per-game subdirs with checkpoints + logs.",
    )
    ap.add_argument("--games", nargs="*", default=None, help="Subset of MINI_GAMES_512 keys.")
    ap.add_argument("--max_games", type=int, default=0, help="If >0, only first N games after sort (debug).")
    ap.add_argument("--skip_gen", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--resume_npz", action="store_true", help="Pass --resume to create_all_games_dataset.py")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for rollout generation.")
    ap.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated CUDA device ids for training (round-robin across games).",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Parallel worker processes (typically = number of GPUs).",
    )
    ap.add_argument("--epochs", type=int, default=40, help="Adam epoch cap when not using train_until.")
    ap.add_argument(
        "--num_actions",
        type=int,
        default=20,
        help="Logged Melting-Pot actions per NPZ sample in create_all_games_dataset (20 ⇒ 81 pixel_frames rows; matches clean_up_20act).",
    )
    ap.add_argument(
        "--train_until_train_acc_pct",
        type=float,
        default=None,
        help="If set: pass to tile trainer (Adam) until min supervised train acc%% reaches this.",
    )
    ap.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="With --train_until_train_acc_pct: epoch cap (tile trainer default 5000 if omitted).",
    )
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for tile training.")
    args = ap.parse_args()

    gpu_ids = [int(x.strip()) for x in str(args.gpu_ids).split(",") if x.strip() != ""]
    if not gpu_ids:
        raise SystemExit("--gpu_ids must list at least one integer")

    all_mini = sorted(_import_mini_games())
    games = sorted(args.games) if args.games else list(all_mini)
    if int(args.max_games) > 0:
        games = games[: int(args.max_games)]

    valid = set(all_mini)
    unknown = [g for g in games if g not in valid]
    if unknown:
        raise SystemExit(f"Unknown game keys (not in MINI_GAMES_512): {unknown}")

    npz_parent = Path(args.npz_parent).resolve()
    out_parent = Path(args.out_parent).resolve()
    npz_parent.mkdir(parents=True, exist_ok=True)
    out_parent.mkdir(parents=True, exist_ok=True)

    num_workers = max(1, int(args.num_workers))

    manifest: list[dict[str, Any]] = []
    manifest_path = out_parent / "manifest.json"
    t0 = time.time()

    # Build job payloads (stable sort); assign GPUs round-robin.
    jobs: list[dict[str, Any]] = []
    for i, game in enumerate(games):
        stem = _safe_stem(game)
        npz_dir = npz_parent / stem
        out_sub = out_parent / stem
        log_dir = out_sub / "logs"
        jobs.append(
            {
                "repo": str(repo),
                "game": game,
                "gpu_id": gpu_ids[i % len(gpu_ids)],
                "npz_dir": str(npz_dir),
                "out_sub": str(out_sub),
                "log_dir": str(log_dir),
                "ck_action": str(out_sub / "supervise_action.pt"),
                "ck_cell": str(out_sub / "supervise_cell.pt"),
                "samples_per_game": int(args.samples_per_game),
                "seed": int(args.seed),
                "skip_gen": bool(args.skip_gen),
                "skip_train": bool(args.skip_train),
                "resume_npz": bool(args.resume_npz),
                "dry_run": bool(args.dry_run),
                "epochs": int(args.epochs),
                "batch": int(args.batch),
                "lr": float(args.lr),
                "hidden": int(args.hidden),
                "workers": int(args.workers),
                "num_actions": int(args.num_actions),
                "train_until_train_acc_pct": (
                    float(args.train_until_train_acc_pct)
                    if args.train_until_train_acc_pct is not None
                    else None
                ),
                "max_epochs": int(args.max_epochs) if args.max_epochs is not None else None,
            }
        )

    print(
        f"Games={len(games)}  num_workers={num_workers}  gpu_ids={gpu_ids}  "
        f"npz_parent={npz_parent}\nout_parent={out_parent}",
        flush=True,
    )

    if args.dry_run:
        for j in jobs:
            print(f"  would run: game={j['game']!r} gpu={j['gpu_id']} npz_dir={j['npz_dir']}", flush=True)
        print("Dry run: no subprocesses executed.", flush=True)
        return

    # ProcessPoolExecutor: stagger GPU assignment among concurrent workers.
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        pending: set[Future] = set()
        it = iter(jobs)

        def _submit(nxt: dict[str, Any]) -> Future:
            fut = ex.submit(_one_game_worker, nxt)
            return fut

        for _ in range(min(num_workers, len(jobs))):
            try:
                nxt = next(it)
            except StopIteration:
                break
            pending.add(_submit(nxt))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                manifest.append(row)
                manifest_path.write_text(json.dumps({"games": manifest}, indent=2), encoding="utf-8")
                g = row.get("game", "?")
                print(
                    f"finished {g!r}  gen={row.get('gen_status')}  "
                    f"action={row.get('train_action_status')}  cell={row.get('train_cell_status')}",
                    flush=True,
                )
            for fut in done:
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                pending.add(_submit(nxt))

    dt = time.time() - t0
    summary = {
        "npz_parent": str(npz_parent),
        "out_parent": str(out_parent),
        "n_games": len(manifest),
        "elapsed_min": round(dt / 60.0, 2),
        "gpu_ids": gpu_ids,
        "num_workers": num_workers,
        "games": manifest,
    }
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}  ({len(manifest)} rows) in {dt/60:.1f} min", flush=True)


if __name__ == "__main__":
    # So 'spawn' can import the worker entry on some platforms
    main()
