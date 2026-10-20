#!/usr/bin/env python3
"""
Train a small **MLP** on **single-cell RGB** tiles: binary ``player occupies this grid cell``.

**Positives** (y=1): for each timestep ``t`` and subject ``s`` with valid ``frame_coords[t,s]``,
the grid cell ``(gi,gj) = coords_to_cell_indices(...)`` on the **current** pixel frame
(latent→pixel index ``l2p[t]``), same convention as ``PerGameTileNPZDataset``.

**Caveat:** labels follow **logged grid positions**, not rendered visibility. A cell is
positive if any subject's ``frame_coords`` maps there (e.g. dead-but-still-positioned
agents still count as ``player cell`` for this dataset).

**Negatives** (y=0): random other grid cells on the **same frame** that no subject occupies
(``--neg_per_pos`` negatives per occupied cell, capped by number of free cells).

With ``--first_timestep_only``, only ``frame_coords`` row ``t=0`` is used (pixel frame
``l2p[0]``, usually ``pixel_frames[0]``) per NPZ — fewer samples, biased toward
rollout starts (often visible avatars); still **not** a hard visibility guarantee.

Use case: gate the 9-way cell-offset head when the anchor tile may not show a visible avatar
(e.g. death / invisibility) — if this model says ``P(no player)`` is high, skip or down-weight
cell prediction for that row.

Example (``clean_up`` NPZs, same holdout as tile training). For **dual-gate** tile training,
prefer **``--first_timestep_only``** so positives come from rollout starts where the avatar is
usually visible at the logged cell; save as e.g. ``*_firstframe.pt`` or your own name::

  python scripts/eval/train_cell_player_presence_mlp.py \\
    --val_dir datasets/rgb_npz_clean_up_20act_n100 \\
    --basename_contains clean_up \\
    --holdout_last_seed \\
    --first_timestep_only \\
    --cell_px 32 --image_size 512 \\
    --epochs 40 --batch 256 --lr 1e-3 --hidden 256 \\
    --neg_per_pos 2 \\
    --output outputs/clean_up_20act_cell_presence_mlp_firstframe.pt

Full-trajectory supervision (all ``t``) if you need more diverse negatives::

  python scripts/eval/train_cell_player_presence_mlp.py \\
    --val_dir datasets/rgb_npz_clean_up_20act_n100 \\
    --basename_contains clean_up \\
    --holdout_last_seed \\
    --cell_px 32 --image_size 512 \\
    --epochs 40 --batch 256 --lr 1e-3 --hidden 256 \\
    --neg_per_pos 2 \\
    --output outputs/clean_up_20act_cell_presence_mlp.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL = os.path.dirname(os.path.abspath(__file__))
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

import per_game_tile_crops as ptc
from per_game_tile_model import collect_npz_paths, _num_subjects


class CellPresenceMLP(nn.Module):
    def __init__(self, flat_dim: int, hidden: int = 256):
        super().__init__()
        h = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(flat_dim, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CellPresenceSingleTileDataset(Dataset):
    """
    Preloads all (3, cell_px, cell_px) tiles and binary labels for listed ``npz_paths``.

    ``dataset._meta[i]`` is ``(basename, t, gi, gj, y)`` with ``y`` in ``{0.0, 1.0}``, aligned with ``__getitem__(i)``.
    """

    def __init__(
        self,
        npz_paths: list[str],
        *,
        cell_px: int,
        image_size: int,
        pad_mode: str,
        pad_fill: tuple[float, ...] | None,
        neg_per_pos: int,
        seed: int,
        max_rows: int | None = None,
        first_timestep_only: bool = False,
    ) -> None:
        super().__init__()
        self.paths = [os.path.abspath(p) for p in npz_paths]
        self.cell_px = int(cell_px)
        self.image_size = int(image_size)
        self.pad_mode = str(pad_mode)
        self.pad_fill = pad_fill
        self.first_timestep_only = bool(first_timestep_only)
        c = self.cell_px
        rng = np.random.default_rng(int(seed))

        xs: list[torch.Tensor] = []
        ys: list[float] = []
        meta: list[tuple[str, int, int, int, float]] = []
        cap = max_rows

        def push_tile(
            frame: np.ndarray, gi: int, gj: int, y: float, *, basename: str, t: int
        ) -> bool:
            """Returns False if ``cap`` reached and tile was not stored."""
            if cap is not None and len(xs) >= cap:
                return False
            tile = ptc.extract_single_cell_patch_rgb(
                frame,
                gi,
                gj,
                c,
                pad_mode=self.pad_mode,
                pad_fill=self.pad_fill,
            )
            tch = torch.from_numpy(tile).permute(2, 0, 1).contiguous().float() / 255.0
            xs.append(tch)
            ys.append(y)
            meta.append((basename, int(t), int(gi), int(gj), float(y)))
            return True

        stop = False
        for fi, path in enumerate(self.paths):
            if stop:
                break
            with np.load(path, allow_pickle=True) as d:
                if "pixel_frames" not in d.files or "frame_coords" not in d.files:
                    continue
                coords = d["frame_coords"]
                pf = d["pixel_frames"]
                ns = _num_subjects(d)
                T = int(coords.shape[0])
                l2p = ptc.get_latent_to_pixel(T)
                t_iter = [0] if self.first_timestep_only else list(range(T))
                for t in t_iter:
                    if stop:
                        break
                    pix_i = l2p[min(int(t), len(l2p) - 1)]
                    if pix_i >= len(pf):
                        continue
                    frame = np.ascontiguousarray(pf[pix_i])
                    nh = int(frame.shape[0]) // c
                    nw = int(frame.shape[1]) // c
                    if nh < 1 or nw < 1:
                        continue
                    occupied: set[tuple[int, int]] = set()
                    for s in range(min(ns, coords.shape[1])):
                        cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                        if cx == 0.0 and cy == 0.0:
                            continue
                        gi, gj = ptc.coords_to_cell_indices(cx, cy, c, self.image_size)
                        gi = max(0, min(nw - 1, gi))
                        gj = max(0, min(nh - 1, gj))
                        occupied.add((gi, gj))
                    free = [(i, j) for j in range(nh) for i in range(nw) if (i, j) not in occupied]
                    for gi, gj in sorted(occupied):
                        if not push_tile(frame, gi, gj, 1.0, basename=os.path.basename(path), t=t):
                            stop = True
                            break
                    if stop:
                        break
                    n_want = int(neg_per_pos) * max(1, len(occupied))
                    n_want = min(n_want, len(free))
                    if free and n_want > 0:
                        pick = rng.choice(len(free), size=n_want, replace=False)
                        for pi in pick:
                            ui, uj = free[int(pi)]
                            if not push_tile(frame, ui, uj, 0.0, basename=os.path.basename(path), t=t):
                                stop = True
                                break
                    if stop:
                        break

        if not xs:
            raise RuntimeError("CellPresenceSingleTileDataset: no samples (empty NPZ list or no valid frames).")

        self._x = torch.stack(xs, dim=0)
        self._y = torch.tensor(ys, dtype=torch.float32)
        self._meta = meta
        print(
            f"CellPresenceSingleTileDataset: N={len(self)}  cell_px={c}  "
            f"pos_frac={float(self._y.mean().item()):.4f}  files={len(self.paths)}"
            f"  first_timestep_only={self.first_timestep_only}",
            flush=True,
        )

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]


@torch.no_grad()
def _accuracy_bce(m: nn.Module, dl: DataLoader, dev: torch.device) -> tuple[float, float]:
    m.eval()
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    tot_loss = 0.0
    correct = 0
    tot = 0
    for xb, yb in dl:
        xb = xb.to(dev)
        yb = yb.to(dev)
        logits = m(xb.view(xb.shape[0], -1))
        tot_loss += float(bce(logits, yb).item())
        pred = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((pred == yb).sum().item())
        tot += len(xb)
    return 100.0 * correct / max(tot, 1), tot_loss / max(tot, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_dir", required=True, type=str)
    p.add_argument("--basename_contains", default=None)
    p.add_argument(
        "--holdout_last_seed",
        action="store_true",
        help="Train on all but last sorted NPZ; evaluate on last (needs ≥2 files).",
    )
    p.add_argument("--cell_px", type=int, default=32)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--pad_mode", choices=("reflect", "constant"), default="constant")
    p.add_argument("--pad_color", type=str, default="255,0,255")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--neg_per_pos",
        type=int,
        default=2,
        help="Negatives sampled per occupied cell per frame (same frame as positives).",
    )
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max_rows", type=int, default=None, help="Debug cap on total rows across all files.")
    p.add_argument(
        "--first_timestep_only",
        action="store_true",
        help="Only use frame_coords row t=0 (pixel l2p[0]) per NPZ; fewer rows, rollout-start bias.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32))

    paths = collect_npz_paths(args.val_dir, args.basename_contains)
    if not paths:
        raise SystemExit(f"No npz under {args.val_dir!r}")

    val_paths: list[str] | None = None
    train_paths = paths
    if args.holdout_last_seed:
        if len(paths) >= 2:
            val_paths = [paths[-1]]
            train_paths = paths[:-1]
        else:
            print("WARNING: --holdout_last_seed ignored (need ≥2 files)", flush=True)

    pad_fill = ptc.parse_pad_rgb(args.pad_color) if args.pad_mode == "constant" else None

    ds_tr = CellPresenceSingleTileDataset(
        train_paths,
        cell_px=args.cell_px,
        image_size=args.image_size,
        pad_mode=args.pad_mode,
        pad_fill=pad_fill,
        neg_per_pos=int(args.neg_per_pos),
        seed=int(args.seed),
        max_rows=args.max_rows,
        first_timestep_only=bool(args.first_timestep_only),
    )
    flat = int(ds_tr._x.shape[1] * ds_tr._x.shape[2] * ds_tr._x.shape[3])

    dl_tr = DataLoader(
        ds_tr,
        batch_size=min(int(args.batch), max(1, len(ds_tr))),
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = CellPresenceMLP(flat_dim=flat, hidden=int(args.hidden)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=float(args.lr))
    bce = nn.BCEWithLogitsLoss()

    for ep in range(int(args.epochs)):
        m.train()
        tot = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            logits = m(xb.view(xb.shape[0], -1))
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(xb)
            n += len(xb)
        tr_acc, tr_loss = _accuracy_bce(m, dl_tr, dev)
        print(
            f"epoch {ep+1}/{args.epochs}  train_loss={tot/max(n,1):.4f}  train_acc%={tr_acc:.2f}  "
            f"(mean BCE eval {tr_loss:.4f})",
            flush=True,
        )

    val_acc = val_loss = 0.0
    if val_paths:
        ds_va = CellPresenceSingleTileDataset(
            val_paths,
            cell_px=args.cell_px,
            image_size=args.image_size,
            pad_mode=args.pad_mode,
            pad_fill=pad_fill,
            neg_per_pos=int(args.neg_per_pos),
            seed=int(args.seed) + 1,
            max_rows=args.max_rows,
            first_timestep_only=bool(args.first_timestep_only),
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=min(int(args.batch), max(1, len(ds_va))),
            shuffle=False,
            num_workers=0,
        )
        val_acc, val_loss = _accuracy_bce(m, dl_va, dev)
        print(f"val acc%={val_acc:.2f}  val_mean_bce={val_loss:.6f}  n_val={len(ds_va)}", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ckpt: dict = {
        "backend": "cell_player_presence_mlp",
        "model": m.state_dict(),
        "cell_px": int(args.cell_px),
        "image_size": int(args.image_size),
        "in_ch": 3,
        "flat_dim": int(flat),
        "hidden": int(args.hidden),
        "pad_mode": str(args.pad_mode),
        "pad_fill": list(pad_fill) if pad_fill is not None else None,
        "neg_per_pos": int(args.neg_per_pos),
        "train_npz_paths": [os.path.abspath(p) for p in train_paths],
        "val_npz_paths": [os.path.abspath(p) for p in val_paths] if val_paths else None,
        "decision_threshold": 0.5,
        "first_timestep_only": bool(args.first_timestep_only),
    }
    torch.save(ckpt, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
