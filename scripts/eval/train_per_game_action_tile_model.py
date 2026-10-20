#!/usr/bin/env python3
"""
Train per-game 3×3 cell tile CNN on val_npz (teacher-forced GT crops).

Saves a checkpoint with ``state_dict``, ``cell_px``, ``image_size``, ``num_actions``
(action head width), ``action_orig_ids`` (compact mode), ``hidden``, ``pad_mode`` /
``pad_fill`` (``constant`` + magenta by default; ``reflect`` omits fill), ``action_loss``
(``bce`` for joint compact action; ``ce`` only when ``action_decomposition=factored``),
and ``action_decomposition`` (``joint`` vs ``factored`` when used).

``num_actions`` is only used for ``action_label_mode=fixed``. Default **compact**
labels use only action ids that appear in the filtered NPZs (saved as
``action_orig_ids`` in the checkpoint).

With ``--aux_meta``, the model sees five extra broadcast planes (cell index, time,
subject, seed-file index) so identical-looking crops from different trajectories
are separable; otherwise patch-only fits can plateau below 100%% on tiny sets.

With ``--holdout_last_seed`` (two or more matching NPZs), trains on all but the
last sorted file and reports **val** accuracy on that held-out seed; compact
action ids are defined from **train** only. Checkpoints include ``aux_fi_npz_paths``
when needed so eval matches training ``fi`` normalization.

By default **``--precompute_crops`` is on**: all patches are materialized once after indexing
(one ``np.load`` per NPZ during the cache build), then epochs are fast. Use
``--no-precompute_crops`` only if RAM is tight; streaming mode re-opens ``.npz`` every
sample and is very slow on NFS.

Optional ``--input_noise_std`` adds Gaussian noise (RGB: first 3 channels clamped;
latent: all C channels). ``--input_mode latent`` trains on ``video_latents`` 3×3
cell crops (same grid as RGB); eval needs ``gen_video_latents`` in eval NPZ.

``--rgb_stack current_next`` (RGB only): 6-channel patches (frames ``t`` and ``t+1``
with the **same** crop anchor at ``coords[t]``), same as ``PerGameTileNPZDataset``.
Use this when you will **predict** next cell/orientation at inference to re-center crops.

``--supervise`` (default ``all``): which heads are supervised. **Joint** compact action
uses ``BCEWithLogitsLoss`` + one-hot (independent sigmoids per class, no softmax over
actions). Cell and orientation heads use ``CrossEntropyLoss``. ``action`` / ``cell``
train only that head (shared backbone still updates); use **separate checkpoints** for
action-only vs cell-only so inference can load e.g. a cell specialist for next-cell
offset without orientation/action competing in one loss.

``--action-decomposition factored`` (compact labels, Adam only): four softmax heads
(move / turn / fireZap / fireClean) from Melting Pot ``action_set``, each with its own CE.
Checkpoints include ``game_name`` and ``factored_spec`` for reload. ``train_action_acc_pct``
in the return dict is **joint** accuracy (all four components match); component min/mean
are extra keys.

**GT dual-cell presence gate** (``--dual_gt_cell_presence_gate_ckpt``): after optional
center gate, filters rows using ``CellPresenceMLP`` on **logged** player cells at
``t`` and ``t+1``. With ``--dual_gt_cell_presence_gate_combine all`` (default), **both**
must pass; with ``any``, keep rows where **either** passes (skip only when absent in
**both** frames).

Usage:
  python scripts/eval/train_per_game_action_tile_model.py \\
    --val_dir path/to/val_npz --basename_contains clean_up \\
    --output /tmp/clean_up_tile.pt --epochs 200 --full_batch --hidden 128

  # Near-100%% train fit on small teacher-forced sets (enables ``--aux_meta`` + LBFGS):
  python scripts/eval/train_per_game_action_tile_model.py \\
    --val_dir path/to/val_npz --basename_contains clean_up \\
    --output /tmp/clean_up_memo.pt --memorize --lr 1.0 --hidden 128

  # Legacy fixed 0..N-1 labels:
  python scripts/eval/train_per_game_action_tile_model.py \\
    --val_dir path/to/val_npz --action_label_mode fixed --num_actions 25 ...
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from per_game_tile_model import (
    PerGameTileCNN,
    PerGameTileCNNFactoredAction,
    PerGameTileNPZDataset,
    collect_npz_paths,
)
from mp_action_factorization import build_factored_spec, game_name_from_npz_path
import per_game_tile_crops as ptc


def _min_supervised_train_acc(
    train_aa: float,
    train_ac: float,
    train_ao: float,
    use_a: bool,
    use_c: bool,
    use_o: bool,
) -> float:
    """Minimum train accuracy (%%) among heads that receive supervised loss."""
    parts: list[float] = []
    if use_a:
        parts.append(float(train_aa))
    if use_c:
        parts.append(float(train_ac))
    if use_o:
        parts.append(float(train_ao))
    return min(parts) if parts else 0.0


@torch.no_grad()
def _train_split_accuracy(m: nn.Module, dl: DataLoader, dev: torch.device) -> tuple[float, float, float]:
    m.eval()
    ca = cc = co = n = 0
    for xb, ya, yc, yo in dl:
        xb = xb.to(dev)
        ya = ya.to(dev)
        yc = yc.to(dev)
        yo = yo.to(dev)
        la, lc, lo = m(xb)
        ca += int((la.argmax(-1) == ya).sum().item())
        cc += int((lc.argmax(-1) == yc).sum().item())
        co += int((lo.argmax(-1) == yo).sum().item())
        n += len(xb)
    if n == 0:
        return 0.0, 0.0, 0.0
    return 100.0 * ca / n, 100.0 * cc / n, 100.0 * co / n


def _factored_compact_to_components(
    ya: torch.Tensor,
    orig_t: torch.Tensor,
    tbl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map compact action class ``ya`` to four component class ids (same device as ``ya``)."""
    orig = orig_t[ya]
    t4 = tbl[orig]
    return t4[:, 0], t4[:, 1], t4[:, 2], t4[:, 3]


def _supervised_loss_factored(
    lm: torch.Tensor,
    lt: torch.Tensor,
    lz: torch.Tensor,
    lcl: torch.Tensor,
    lc: torch.Tensor,
    lo: torch.Tensor,
    ym: torch.Tensor,
    yt: torch.Tensor,
    yz: torch.Tensor,
    ycl: torch.Tensor,
    yc: torch.Tensor,
    yo: torch.Tensor,
    ce: nn.CrossEntropyLoss,
    use_a: bool,
    use_c: bool,
    use_o: bool,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if use_a:
        parts.extend([ce(lm, ym), ce(lt, yt), ce(lz, yz), ce(lcl, ycl)])
    if use_c:
        parts.append(ce(lc, yc))
    if use_o:
        parts.append(ce(lo, yo))
    if not parts:
        raise RuntimeError("supervise mask has no active heads")
    return sum(parts)


@torch.no_grad()
def _train_split_accuracy_factored(
    m: nn.Module,
    dl: DataLoader,
    dev: torch.device,
    orig_t: torch.Tensor,
    tbl: torch.Tensor,
    use_a: bool,
    use_c: bool,
    use_o: bool,
) -> tuple[float, float, float, float, float]:
    """Returns (action_min_of_four, cell, orient, action_mean_four, joint_all_four_match)."""
    m.eval()
    sm = st = sz = sc = 0
    sjoint = 0
    cc = co = n = 0
    for xb, ya, yc, yo in dl:
        xb = xb.to(dev)
        ya = ya.to(dev)
        yc = yc.to(dev)
        yo = yo.to(dev)
        lm, lt, lz, lcl, lc, lo = m(xb)
        ym, yt, yz, ycl = _factored_compact_to_components(ya, orig_t.to(dev), tbl.to(dev))
        pm, pt, pz, pcl = lm.argmax(-1), lt.argmax(-1), lz.argmax(-1), lcl.argmax(-1)
        okm = pm == ym
        okt = pt == yt
        okz = pz == yz
        okcl = pcl == ycl
        sm += int(okm.sum().item())
        st += int(okt.sum().item())
        sz += int(okz.sum().item())
        sc += int(okcl.sum().item())
        sjoint += int((okm & okt & okz & okcl).sum().item())
        cc += int((lc.argmax(-1) == yc).sum().item())
        co += int((lo.argmax(-1) == yo).sum().item())
        n += len(xb)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    am = 100.0 * sm / n
    at = 100.0 * st / n
    az = 100.0 * sz / n
    acl = 100.0 * sc / n
    action_min = min(am, at, az, acl)
    action_mean = 0.25 * (am + at + az + acl)
    joint = 100.0 * sjoint / n
    return action_min, 100.0 * cc / n, 100.0 * co / n, action_mean, joint


@torch.no_grad()
def _split_mean_losses_factored(
    m: nn.Module,
    dl: DataLoader,
    dev: torch.device,
    ce: nn.CrossEntropyLoss,
    orig_t: torch.Tensor,
    tbl: torch.Tensor,
    *,
    use_a: bool = True,
    use_c: bool = True,
    use_o: bool = True,
) -> tuple[float, float, float, float, float]:
    """(total_mean, action_mean_sum_of_4, cell, orient, joint_acc_not_used_here)."""
    m.eval()
    tot = ta = tc = to = 0.0
    n = 0
    for xb, ya, yc, yo in dl:
        xb = xb.to(dev)
        ya = ya.to(dev)
        yc = yc.to(dev)
        yo = yo.to(dev)
        lm, lt, lz, lcl, lc, lo = m(xb)
        ym, yt, yz, ycl = _factored_compact_to_components(ya, orig_t.to(dev), tbl.to(dev))
        la_sum = (
            float(ce(lm, ym).item() + ce(lt, yt).item() + ce(lz, yz).item() + ce(lcl, ycl).item())
            if use_a
            else 0.0
        )
        lc_ = float(ce(lc, yc).item()) if use_c else 0.0
        lo_ = float(ce(lo, yo).item()) if use_o else 0.0
        bs = len(xb)
        ta += la_sum * bs
        tc += lc_ * bs
        to += lo_ * bs
        tot += float(_supervised_loss_factored(lm, lt, lz, lcl, lc, lo, ym, yt, yz, ycl, yc, yo, ce, use_a, use_c, use_o).item()) * bs
        n += bs
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    inv = 1.0 / n
    return tot * inv, ta * inv, tc * inv, to * inv, 0.0


def _supervise_mask(supervise: str) -> tuple[bool, bool, bool]:
    s = str(supervise or "all").strip().lower().replace("-", "_")
    if s in ("all", "action_cell_orient", "default"):
        return True, True, True
    if s == "action":
        return True, False, False
    if s == "cell":
        return False, True, False
    if s == "orient":
        return False, False, True
    if s in ("action_cell", "action_and_cell"):
        return True, True, False
    raise ValueError(
        "supervise must be one of: all, action, cell, orient, action_cell "
        f"(got {supervise!r})"
    )


def _supervised_loss(
    la: torch.Tensor,
    lc: torch.Tensor,
    lo: torch.Tensor,
    ya: torch.Tensor,
    yc: torch.Tensor,
    yo: torch.Tensor,
    ce: nn.CrossEntropyLoss,
    use_a: bool,
    use_c: bool,
    use_o: bool,
    *,
    bce_action: nn.BCEWithLogitsLoss | None = None,
    n_action_classes: int = 0,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if use_a:
        if bce_action is None:
            raise RuntimeError("internal: joint action head expects BCEWithLogitsLoss")
        if n_action_classes <= 0:
            raise RuntimeError("n_action_classes required for multilabel BCE action")
        y_oh = F.one_hot(ya.long(), num_classes=n_action_classes).to(dtype=la.dtype)
        parts.append(bce_action(la, y_oh))
    if use_c:
        parts.append(ce(lc, yc))
    if use_o:
        parts.append(ce(lo, yo))
    if not parts:
        raise RuntimeError("supervise mask has no active heads")
    return sum(parts)


@torch.no_grad()
def _split_mean_losses(
    m: nn.Module,
    dl: DataLoader,
    dev: torch.device,
    ce: nn.CrossEntropyLoss,
    *,
    use_a: bool = True,
    use_c: bool = True,
    use_o: bool = True,
    bce_action: nn.BCEWithLogitsLoss | None = None,
    n_action_classes: int = 0,
) -> tuple[float, float, float, float]:
    """Mean per-sample supervised loss sum (``tot``) and each head's mean (CE or BCE for action)."""
    m.eval()
    tot = ta = tc = to = 0.0
    n = 0
    for xb, ya, yc, yo in dl:
        xb = xb.to(dev)
        ya = ya.to(dev)
        yc = yc.to(dev)
        yo = yo.to(dev)
        la, lc, lo = m(xb)
        if use_a:
            if bce_action is None:
                raise RuntimeError("internal: joint action expects BCEWithLogitsLoss")
            y_oh = F.one_hot(ya.long(), num_classes=n_action_classes).to(dtype=la.dtype)
            la_ = bce_action(la, y_oh)
        else:
            la_ = torch.zeros((), device=dev)
        lc_ = ce(lc, yc) if use_c else torch.zeros((), device=dev)
        lo_ = ce(lo, yo) if use_o else torch.zeros((), device=dev)
        bs = len(xb)
        ta += float(la_.item()) * bs
        tc += float(lc_.item()) * bs
        to += float(lo_.item()) * bs
        tot += float(
            _supervised_loss(
                la,
                lc,
                lo,
                ya,
                yc,
                yo,
                ce,
                use_a,
                use_c,
                use_o,
                bce_action=bce_action,
                n_action_classes=n_action_classes,
            ).item()
        ) * bs
        n += bs
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    inv = 1.0 / n
    return tot * inv, ta * inv, tc * inv, to * inv


def train_to_checkpoint(
    npz_paths: list[str],
    output: str,
    *,
    cell_px: int = 32,
    image_size: int = 512,
    num_actions: int | None = None,
    action_label_mode: str = "compact",
    aux_meta_channels: bool = False,
    val_npz_paths: list[str] | None = None,
    input_noise_std: float = 0.0,
    input_mode: str = "rgb",
    rgb_stack: str = "current",
    epochs: int = 12,
    batch: int = 64,
    lr: float = 1e-3,
    workers: int = 0,
    pad_mode: str = "constant",
    pad_color: str = "255,0,255",
    hidden: int = 64,
    full_batch: bool = False,
    verify_train_acc: bool = True,
    optimizer: str = "adam",
    supervise: str = "all",
    train_until_train_acc_pct: float | None = None,
    max_epochs: int | None = None,
    precompute_crops: bool = True,
    action_decomposition: str = "joint",
    restore_best_val_cell: bool = False,
    center_presence_gate_ckpt: str | None = None,
    center_presence_prob_min: float = 0.5,
    dual_gt_cell_presence_gate_ckpt: str | None = None,
    dual_gt_cell_presence_prob_min: float = 0.5,
    dual_gt_cell_presence_gate_combine: str = "all",
    resume_path: str | None = None,
    val_log_interval: int | None = None,
) -> dict:
    """
    Train ``PerGameTileCNN`` on ``npz_paths`` (train) and save ``output``.

    If ``val_npz_paths`` is non-empty, compact action ids are taken from train
    only; val rows whose action is unseen on train are dropped. With
    ``aux_meta_channels``, file-index aux uses the union of train+val paths for
    consistent ``fi`` normalization across splits.

    If ``val_npz_paths`` is empty, ``val_*`` accuracies in the return dict are set
    equal to train (same dataloader): the **full val_npz fit** on all listed
    seeds, not a held-out generalization split.

    If ``train_until_train_acc_pct`` is set (Adam only), training continues until the
    **minimum** train accuracy among **supervised** heads reaches that threshold (or
    ``max_epochs`` is hit). The train DataLoader still uses ``shuffle=True`` (default).

    If ``restore_best_val_cell`` is True (Adam + ``train_until`` + held-out val + cell
    supervised), each epoch evaluates **val cell** accuracy and the saved checkpoint
    restores weights from the epoch with the highest val cell acc (ties → earlier epoch).
    For ``action_decomposition=factored``, the action part of that minimum is the
    **worst** of the four component-head accuracies (move / turn / zap / clean); see
    logged ``action_heads_min`` vs ``action_joint_4match`` (all four correct).

    **Joint** action (default ``action_decomposition=joint``): ``BCEWithLogitsLoss`` with
    one-hot targets over compact action ids — independent sigmoids, no softmax over actions.
    **Factored** action uses four ``CrossEntropyLoss`` heads (move/turn/zap/clean).

    Returns metadata (``num_actions`` head width, ``n_samples``, ``final_loss``, train/val accs, …).
    """
    if not npz_paths:
        raise ValueError("npz_paths is empty")
    if action_label_mode == "fixed" and num_actions is None:
        raise ValueError("action_label_mode=fixed requires num_actions")

    use_a, use_c, use_o = _supervise_mask(supervise)
    ad = str(action_decomposition or "joint").strip().lower()
    factored = ad == "factored"
    action_loss_meta = "bce" if not factored else "ce"
    if factored:
        if str(optimizer).lower() == "lbfgs":
            raise ValueError("action_decomposition=factored is incompatible with optimizer=lbfgs")
        if action_label_mode != "compact":
            raise ValueError("action_decomposition=factored requires action_label_mode=compact")
        if not use_a:
            raise ValueError(
                "action_decomposition=factored requires supervise mode that includes action "
                "(e.g. --supervise action or all)"
            )

    val_npz_paths = [os.path.abspath(p) for p in (val_npz_paths or []) if p]
    train_npz_paths = [os.path.abspath(p) for p in npz_paths]
    all_paths_union = sorted(set(train_npz_paths) | set(val_npz_paths))
    aux_fi_ref = all_paths_union if (aux_meta_channels and val_npz_paths) else None

    pad_fill = None
    if pad_mode == "constant":
        pad_fill = ptc.parse_pad_rgb(pad_color)

    na_fixed = int(num_actions) if num_actions is not None else 25
    ds = PerGameTileNPZDataset(
        train_npz_paths,
        cell_px=cell_px,
        image_size=image_size,
        num_actions=na_fixed,
        pad_mode=pad_mode,
        pad_fill=pad_fill,
        action_label_mode=action_label_mode,
        aux_meta_channels=aux_meta_channels,
        aux_fi_reference_paths=aux_fi_ref,
        input_noise_std=input_noise_std,
        input_mode=input_mode,
        rgb_stack=rgb_stack,
        precompute_crops=precompute_crops,
        center_presence_gate_ckpt=center_presence_gate_ckpt,
        center_presence_prob_min=float(center_presence_prob_min),
        dual_gt_cell_presence_gate_ckpt=dual_gt_cell_presence_gate_ckpt,
        dual_gt_cell_presence_prob_min=float(dual_gt_cell_presence_prob_min),
        dual_gt_cell_presence_gate_combine=str(dual_gt_cell_presence_gate_combine or "all"),
    )
    if len(ds) == 0:
        raise RuntimeError(
            "Dataset empty (need frame_coords, orientations, actions; "
            "rgb: pixel_frames; latent: video_latents)"
        )
    print(
        f"rgb_stack={rgb_stack!r}  input_channels={ds.input_channels}  n_samples={len(ds)}  "
        f"supervise={supervise!r} (loss: action={use_a} cell={use_c} orient={use_o})  "
        f"action_decomposition={ad!r}  "
        f"({'joint action=BCEWithLogits+one_hot' if not factored else 'factored action=4×CE'})",
        flush=True,
    )
    if dual_gt_cell_presence_gate_ckpt and not use_c:
        print(
            "NOTE: --dual_gt_cell_presence_gate_ckpt filters rows by GT cell presence at t and t+1; "
            f"with --supervise={supervise!r} the cell head is not trained, but crops match that subset.",
            flush=True,
        )
    n_cls = ds.num_action_classes
    eff_bs = max(1, len(ds)) if full_batch else max(1, min(int(batch), len(ds)))

    train_until = train_until_train_acc_pct
    if train_until is not None:
        if str(optimizer).lower() != "adam":
            raise ValueError("train_until_train_acc_pct requires optimizer='adam' (LBFGS path is fixed-step)")
        if train_until < 0.0 or train_until > 100.0:
            raise ValueError("train_until_train_acc_pct must be in [0, 100]")
        epoch_budget = int(max_epochs) if max_epochs is not None else 5000
        if epoch_budget <= 0:
            raise ValueError("max_epochs must be positive when train_until_train_acc_pct is set")
        print(
            f"train_until_train_acc_pct={train_until}  max_epochs={epoch_budget}  "
            f"(stop when min supervised train acc%% >= {train_until})",
            flush=True,
        )
    else:
        epoch_budget = int(epochs)

    dl = DataLoader(
        ds,
        batch_size=eff_bs,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    dl_val_track: DataLoader | None = None
    if restore_best_val_cell:
        if train_until is None:
            print(
                "WARNING: --restore_best_val_cell ignored without --train_until_train_acc_pct",
                flush=True,
            )
        elif not val_npz_paths:
            print("WARNING: --restore_best_val_cell ignored without held-out val NPZs", flush=True)
        elif not use_c:
            print("WARNING: --restore_best_val_cell ignored (--supervise must include cell)", flush=True)
        elif str(optimizer).lower() != "adam":
            print("WARNING: --restore_best_val_cell only applies to optimizer=adam", flush=True)
        else:
            ds_val_track = PerGameTileNPZDataset(
                val_npz_paths,
                cell_px=cell_px,
                image_size=image_size,
                num_actions=na_fixed,
                pad_mode=pad_mode,
                pad_fill=pad_fill,
                action_label_mode=action_label_mode,
                aux_meta_channels=aux_meta_channels,
                compact_action_orig_ids=list(ds._action_orig_ids) if action_label_mode == "compact" else None,
                aux_fi_reference_paths=aux_fi_ref,
                input_noise_std=input_noise_std,
                input_mode=input_mode,
                rgb_stack=rgb_stack,
                precompute_crops=precompute_crops,
                center_presence_gate_ckpt=center_presence_gate_ckpt,
                center_presence_prob_min=float(center_presence_prob_min),
                dual_gt_cell_presence_gate_ckpt=dual_gt_cell_presence_gate_ckpt,
                dual_gt_cell_presence_prob_min=float(dual_gt_cell_presence_prob_min),
                dual_gt_cell_presence_gate_combine=str(dual_gt_cell_presence_gate_combine or "all"),
            )
            if len(ds_val_track) > 0:
                dl_val_track = DataLoader(
                    ds_val_track,
                    batch_size=min(256, max(1, len(ds_val_track))),
                    shuffle=False,
                    num_workers=0,
                )
                print(
                    "restore_best_val_cell: tracking val cell acc each epoch; "
                    "final checkpoint uses best-val epoch (ties → earlier).",
                    flush=True,
                )

    dl_val_snap: DataLoader | None = None
    val_log_every = int(val_log_interval) if val_log_interval is not None else 0
    if val_npz_paths and val_log_every > 0:
        ds_val_snap = PerGameTileNPZDataset(
            val_npz_paths,
            cell_px=cell_px,
            image_size=image_size,
            num_actions=na_fixed,
            pad_mode=pad_mode,
            pad_fill=pad_fill,
            action_label_mode=action_label_mode,
            aux_meta_channels=aux_meta_channels,
            compact_action_orig_ids=list(ds._action_orig_ids) if action_label_mode == "compact" else None,
            aux_fi_reference_paths=aux_fi_ref,
            input_noise_std=input_noise_std,
            input_mode=input_mode,
            rgb_stack=rgb_stack,
            precompute_crops=precompute_crops,
            center_presence_gate_ckpt=center_presence_gate_ckpt,
            center_presence_prob_min=float(center_presence_prob_min),
            dual_gt_cell_presence_gate_ckpt=dual_gt_cell_presence_gate_ckpt,
            dual_gt_cell_presence_prob_min=float(dual_gt_cell_presence_prob_min),
            dual_gt_cell_presence_gate_combine=str(dual_gt_cell_presence_gate_combine or "all"),
        )
        if len(ds_val_snap) > 0:
            dl_val_snap = DataLoader(
                ds_val_snap,
                batch_size=min(256, max(1, len(ds_val_snap))),
                shuffle=False,
                num_workers=0,
            )
            print(
                f"val_log_interval={val_log_every}: val_snapshot on epoch 1 then every {val_log_every} epochs",
                flush=True,
            )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  (tile CNN on GPU when cuda available)", flush=True)
    spec = None
    orig_t_dev = tbl_dev = None
    gn: str | None = None
    if factored:
        gn = game_name_from_npz_path(train_npz_paths[0])
        spec = build_factored_spec(gn)
        orig_t_dev = torch.tensor(list(ds._action_orig_ids), dtype=torch.long, device=dev)
        tbl_dev = torch.tensor(spec.joint_to_factored, dtype=torch.long, device=dev)
        print(
            f"action_decomposition=factored  game_name={gn!r}  "
            f"heads: move={spec.n_move} turn={spec.n_turn} zap={spec.n_zap} clean={spec.n_clean}",
            flush=True,
        )
        m = PerGameTileCNNFactoredAction(
            n_move=spec.n_move,
            n_turn=spec.n_turn,
            n_zap=spec.n_zap,
            n_clean=spec.n_clean,
            in_ch=ds.input_channels,
            hidden=hidden,
        ).to(dev)
    else:
        m = PerGameTileCNN(num_actions=n_cls, hidden=hidden, in_ch=ds.input_channels).to(dev)
    ce = nn.CrossEntropyLoss()
    bce_action = nn.BCEWithLogitsLoss() if use_a and not factored else None

    resume_from: str | None = None
    if resume_path and str(resume_path).strip():
        rp = os.path.abspath(str(resume_path).strip())
        try:
            rk = torch.load(rp, map_location="cpu", weights_only=False)
        except TypeError:
            rk = torch.load(rp, map_location="cpu")
        m.load_state_dict(rk["model"], strict=True)
        resume_from = rp
        print(f"resume: loaded model state_dict from {rp!r} (optimizer state not restored)", flush=True)

    def _maybe_val_snapshot(ep1: int) -> None:
        if dl_val_snap is None or val_log_every <= 0:
            return
        if ep1 != 1 and ep1 % val_log_every != 0:
            return
        m.eval()
        if factored:
            assert orig_t_dev is not None and tbl_dev is not None
            _, svc, svo, _, svj = _train_split_accuracy_factored(
                m, dl_val_snap, dev, orig_t_dev, tbl_dev, use_a, use_c, use_o
            )
            print(
                f"  val_snapshot ep={ep1} joint4={svj:.2f}% cell={svc:.2f}% orient={svo:.2f}%",
                flush=True,
            )
        else:
            sva, svc, svo = _train_split_accuracy(m, dl_val_snap, dev)
            print(
                f"  val_snapshot ep={ep1} action={sva:.2f}% cell={svc:.2f}% orient={svo:.2f}%",
                flush=True,
            )
        m.train()

    best_val_cell = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1

    last_loss = 0.0
    epochs_done = 0
    if optimizer == "lbfgs":
        if not full_batch:
            raise ValueError("optimizer=lbfgs is meant for tiny datasets; pass full_batch=True")
        opt = torch.optim.LBFGS(
            m.parameters(),
            lr=float(lr),
            max_iter=30,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        xb, ya, yc, yo = next(iter(dl))
        xb = xb.to(dev)
        ya = ya.to(dev)
        yc = yc.to(dev)
        yo = yo.to(dev)
        if train_until is not None:
            raise ValueError("train_until_train_acc_pct is incompatible with optimizer=lbfgs")
        for ep in range(max(1, epochs)):
            def closure():
                opt.zero_grad()
                la, lc, lo = m(xb)
                loss = _supervised_loss(
                    la,
                    lc,
                    lo,
                    ya,
                    yc,
                    yo,
                    ce,
                    use_a,
                    use_c,
                    use_o,
                    bce_action=bce_action,
                    n_action_classes=n_cls,
                )
                loss.backward()
                return loss

            loss = opt.step(closure)
            last_loss = float(loss.detach().item())
            if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
                print(f"lbfgs step {ep+1}/{epochs}  loss={last_loss:.6f}", flush=True)
        epochs_done = max(1, epochs)
    else:
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        ep_cap = epoch_budget if train_until is not None else int(epochs)
        for ep in range(ep_cap):
            m.train()
            tot = 0.0
            n = 0
            for xb, ya, yc, yo in dl:
                xb = xb.to(dev)
                ya = ya.to(dev)
                yc = yc.to(dev)
                yo = yo.to(dev)
                opt.zero_grad()
                if factored:
                    lm, lt, lz, lcl, lc, lo = m(xb)
                    ym, yt, yz, ycl = _factored_compact_to_components(ya, orig_t_dev, tbl_dev)
                    loss = _supervised_loss_factored(
                        lm, lt, lz, lcl, lc, lo, ym, yt, yz, ycl, yc, yo, ce, use_a, use_c, use_o
                    )
                else:
                    la, lc, lo = m(xb)
                    loss = _supervised_loss(
                        la,
                        lc,
                        lo,
                        ya,
                        yc,
                        yo,
                        ce,
                        use_a,
                        use_c,
                        use_o,
                        bce_action=bce_action,
                        n_action_classes=n_cls,
                    )
                loss.backward()
                opt.step()
                tot += float(loss.item()) * len(xb)
                n += len(xb)
            last_loss = tot / max(n, 1)
            if train_until is not None:
                dl_chk = DataLoader(ds, batch_size=min(256, max(1, len(ds))), shuffle=False, num_workers=0)
                if factored:
                    assert orig_t_dev is not None and tbl_dev is not None
                    chk_amin, chk_ac, chk_ao, chk_amean, chk_joint = _train_split_accuracy_factored(
                        m, dl_chk, dev, orig_t_dev, tbl_dev, use_a, use_c, use_o
                    )
                    crit = _min_supervised_train_acc(chk_amin, chk_ac, chk_ao, use_a, use_c, use_o)
                    print(
                        f"epoch {ep+1}/{ep_cap}  loss={last_loss:.4f}  n_samples={len(ds)}  batch={eff_bs}  "
                        f"train acc%  action_heads_min={chk_amin:.2f} action_heads_mean={chk_amean:.2f} "
                        f"joint_4match={chk_joint:.2f}  cell={chk_ac:.2f} orient={chk_ao:.2f}  "
                        f"supervised_min={crit:.2f}%",
                        flush=True,
                    )
                    _maybe_val_snapshot(ep + 1)
                else:
                    chk_aa, chk_ac, chk_ao = _train_split_accuracy(m, dl_chk, dev)
                    crit = _min_supervised_train_acc(chk_aa, chk_ac, chk_ao, use_a, use_c, use_o)
                    print(
                        f"epoch {ep+1}/{ep_cap}  loss={last_loss:.4f}  n_samples={len(ds)}  batch={eff_bs}  "
                        f"train acc% (all heads) action={chk_aa:.2f} cell={chk_ac:.2f} orient={chk_ao:.2f}  "
                        f"supervised_min={crit:.2f}%",
                        flush=True,
                    )
                    _maybe_val_snapshot(ep + 1)
                if dl_val_track is not None:
                    if factored:
                        assert orig_t_dev is not None and tbl_dev is not None
                        _, v_cell, _, _, _ = _train_split_accuracy_factored(
                            m, dl_val_track, dev, orig_t_dev, tbl_dev, use_a, use_c, use_o
                        )
                    else:
                        _, v_cell, _ = _train_split_accuracy(m, dl_val_track, dev)
                    if v_cell > best_val_cell + 1e-9:
                        best_val_cell = float(v_cell)
                        best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                        best_epoch = ep + 1
                    print(
                        f"  val_cell_track={v_cell:.2f}%  best_val_cell={best_val_cell:.2f}% @epoch {best_epoch}",
                        flush=True,
                    )
                if crit >= float(train_until):
                    print(
                        f"train_until: min supervised train acc reached {crit:.2f}% >= {float(train_until)}%",
                        flush=True,
                    )
                    epochs_done = ep + 1
                    break
            else:
                print(
                    f"epoch {ep+1}/{ep_cap}  loss={last_loss:.4f}  n_samples={len(ds)}  batch={eff_bs}",
                    flush=True,
                )
                _maybe_val_snapshot(ep + 1)
            epochs_done = ep + 1

        if best_state is not None:
            print(
                f"restore_best_val_cell: restoring epoch {best_epoch} "
                f"(val_cell={best_val_cell:.2f}%) for final metrics & checkpoint.",
                flush=True,
            )
            m.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    train_aa = train_ac = train_ao = 0.0
    train_action_component_min = train_action_heads_mean = train_action_joint_match = 0.0
    train_loss_mean = train_la = train_lc = train_lo = 0.0
    if verify_train_acc:
        dl_eval = DataLoader(ds, batch_size=min(256, max(1, len(ds))), shuffle=False, num_workers=0)
        if factored:
            assert orig_t_dev is not None and tbl_dev is not None
            amin, train_ac, train_ao, train_action_heads_mean, train_action_joint_match = (
                _train_split_accuracy_factored(m, dl_eval, dev, orig_t_dev, tbl_dev, use_a, use_c, use_o)
            )
            train_action_component_min = amin
            train_aa = train_action_joint_match
            train_loss_mean, train_la, train_lc, train_lo, _ = _split_mean_losses_factored(
                m, dl_eval, dev, ce, orig_t_dev, tbl_dev, use_a=use_a, use_c=use_c, use_o=use_o
            )
            print(
                f"train acc%: action_joint_4match={train_action_joint_match:.2f} "
                f"action_heads_min={train_action_component_min:.2f} action_heads_mean={train_action_heads_mean:.2f} "
                f"cell={train_ac:.2f} orient={train_ao:.2f}",
                flush=True,
            )
            print(
                "train CE (mean/sample): "
                f"supervised_total={train_loss_mean:.6f}  action_sum4={train_la:.6f} cell={train_lc:.6f} orient={train_lo:.6f}",
                flush=True,
            )
        else:
            train_aa, train_ac, train_ao = _train_split_accuracy(m, dl_eval, dev)
            train_loss_mean, train_la, train_lc, train_lo = _split_mean_losses(
                m,
                dl_eval,
                dev,
                ce,
                use_a=use_a,
                use_c=use_c,
                use_o=use_o,
                bce_action=bce_action,
                n_action_classes=n_cls,
            )
            print(
                f"train acc%: action={train_aa:.2f} cell={train_ac:.2f} orient={train_ao:.2f}",
                flush=True,
            )
            if bce_action is not None:
                print(
                    "train loss (mean/sample): "
                    f"supervised_total={train_loss_mean:.6f}  action_bce={train_la:.6f} "
                    f"cell_ce={train_lc:.6f} orient_ce={train_lo:.6f}",
                    flush=True,
                )
            else:
                print(
                    "train CE (mean/sample): "
                    f"supervised_total={train_loss_mean:.6f}  action={train_la:.6f} cell={train_lc:.6f} orient={train_lo:.6f}",
                    flush=True,
                )

    val_aa = val_ac = val_ao = 0.0
    val_action_component_min = val_action_heads_mean = val_action_joint_match = 0.0
    n_val_samples = 0
    val_loss_mean = val_la = val_lc = val_lo = 0.0
    if val_npz_paths:
        ds_val = PerGameTileNPZDataset(
            val_npz_paths,
            cell_px=cell_px,
            image_size=image_size,
            num_actions=na_fixed,
            pad_mode=pad_mode,
            pad_fill=pad_fill,
            action_label_mode=action_label_mode,
            aux_meta_channels=aux_meta_channels,
            compact_action_orig_ids=list(ds._action_orig_ids) if action_label_mode == "compact" else None,
            aux_fi_reference_paths=aux_fi_ref,
            input_noise_std=input_noise_std,
            input_mode=input_mode,
            rgb_stack=rgb_stack,
            precompute_crops=precompute_crops,
            center_presence_gate_ckpt=center_presence_gate_ckpt,
            center_presence_prob_min=float(center_presence_prob_min),
            dual_gt_cell_presence_gate_ckpt=dual_gt_cell_presence_gate_ckpt,
            dual_gt_cell_presence_prob_min=float(dual_gt_cell_presence_prob_min),
            dual_gt_cell_presence_gate_combine=str(dual_gt_cell_presence_gate_combine or "all"),
        )
        n_val_samples = len(ds_val)
        if n_val_samples > 0:
            dl_val = DataLoader(ds_val, batch_size=min(256, max(1, n_val_samples)), shuffle=False, num_workers=0)
            if factored:
                assert orig_t_dev is not None and tbl_dev is not None
                vmin, val_ac, val_ao, val_action_heads_mean, val_action_joint_match = (
                    _train_split_accuracy_factored(m, dl_val, dev, orig_t_dev, tbl_dev, use_a, use_c, use_o)
                )
                val_action_component_min = vmin
                val_aa = val_action_joint_match
                val_loss_mean, val_la, val_lc, val_lo, _ = _split_mean_losses_factored(
                    m, dl_val, dev, ce, orig_t_dev, tbl_dev, use_a=use_a, use_c=use_c, use_o=use_o
                )
                print(
                    f"val acc%:   action_joint_4match={val_action_joint_match:.2f} "
                    f"action_heads_min={val_action_component_min:.2f} action_heads_mean={val_action_heads_mean:.2f} "
                    f"cell={val_ac:.2f} orient={val_ao:.2f}  (n={n_val_samples})",
                    flush=True,
                )
                print(
                    "val CE (mean/sample):   "
                    f"supervised_total={val_loss_mean:.6f}  action_sum4={val_la:.6f} cell={val_lc:.6f} orient={val_lo:.6f}",
                    flush=True,
                )
            else:
                val_aa, val_ac, val_ao = _train_split_accuracy(m, dl_val, dev)
                val_loss_mean, val_la, val_lc, val_lo = _split_mean_losses(
                    m,
                    dl_val,
                    dev,
                    ce,
                    use_a=use_a,
                    use_c=use_c,
                    use_o=use_o,
                    bce_action=bce_action,
                    n_action_classes=n_cls,
                )
                print(
                    f"val acc%:   action={val_aa:.2f} cell={val_ac:.2f} orient={val_ao:.2f}  (n={n_val_samples})",
                    flush=True,
                )
                if bce_action is not None:
                    print(
                        "val loss (mean/sample):   "
                        f"supervised_total={val_loss_mean:.6f}  action_bce={val_la:.6f} "
                        f"cell_ce={val_lc:.6f} orient_ce={val_lo:.6f}",
                        flush=True,
                    )
                else:
                    print(
                        "val CE (mean/sample):   "
                        f"supervised_total={val_loss_mean:.6f}  action={val_la:.6f} cell={val_lc:.6f} orient={val_lo:.6f}",
                        flush=True,
                    )
        else:
            print("val acc%:   (no labeled val rows after train-vocab filter)", flush=True)

    elif verify_train_acc:
        # No held-out files: report val_* as the same full fit (entire game's val_npz bundle).
        val_aa, val_ac, val_ao = train_aa, train_ac, train_ao
        val_action_component_min = train_action_component_min
        val_action_heads_mean = train_action_heads_mean
        val_action_joint_match = train_action_joint_match
        n_val_samples = len(ds)
        val_loss_mean, val_la, val_lc, val_lo = train_loss_mean, train_la, train_lc, train_lo
        if factored:
            print(
                "full val_npz fit acc% (= train on all listed seeds; memorization target): "
                f"action_joint_4match={val_action_joint_match:.2f} "
                f"action_heads_min={val_action_component_min:.2f} action_heads_mean={val_action_heads_mean:.2f} "
                f"cell={val_ac:.2f} orient={val_ao:.2f}  (n={n_val_samples})",
                flush=True,
            )
            print(
                "full val_npz CE (= train): "
                f"supervised_total={val_loss_mean:.6f}  action_sum4={val_la:.6f} cell={val_lc:.6f} orient={val_lo:.6f}",
                flush=True,
            )
        else:
            print(
                "full val_npz fit acc% (= train on all listed seeds; memorization target): "
                f"action={val_aa:.2f} cell={val_ac:.2f} orient={val_ao:.2f}  (n={n_val_samples})",
                flush=True,
            )
            if bce_action is not None:
                print(
                    "full val_npz loss (= train): "
                    f"supervised_total={val_loss_mean:.6f}  action_bce={val_la:.6f} cell_ce={val_lc:.6f} orient_ce={val_lo:.6f}",
                    flush=True,
                )
            else:
                print(
                    "full val_npz CE (= train): "
                    f"supervised_total={val_loss_mean:.6f}  action={val_la:.6f} cell={val_lc:.6f} orient={val_lo:.6f}",
                    flush=True,
                )

    orig_ids = list(ds._action_orig_ids) if action_label_mode == "compact" else None
    aux_fi_ckpt = all_paths_union if val_npz_paths else list(train_npz_paths)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    ckpt: dict = {
        "model": m.state_dict(),
        "cell_px": cell_px,
        "image_size": image_size,
        "num_actions": int(n_cls),
        "action_label_mode": action_label_mode,
        "action_orig_ids": orig_ids,
        "hidden": int(hidden),
        "npz_paths": train_npz_paths,
        "aux_fi_npz_paths": aux_fi_ckpt,
        "backend": "per_game_tile_cnn",
        "pad_mode": pad_mode,
        "pad_fill": list(pad_fill) if pad_mode == "constant" and pad_fill is not None else None,
        "optimizer": optimizer,
        "aux_meta_channels": bool(aux_meta_channels),
        "in_ch": int(ds.input_channels),
        "input_noise_std": float(input_noise_std),
        "input_mode": str(input_mode),
        "rgb_stack": str(rgb_stack),
        "latent_channels": int(ds._latent_c),
        "center_presence_gate_ckpt": (
            os.path.abspath(center_presence_gate_ckpt) if center_presence_gate_ckpt else None
        ),
        "center_presence_prob_min": float(center_presence_prob_min) if center_presence_gate_ckpt else None,
        "center_presence_gate_stats": getattr(ds, "_center_presence_gate_stats", None),
        "dual_gt_cell_presence_gate_ckpt": (
            os.path.abspath(dual_gt_cell_presence_gate_ckpt) if dual_gt_cell_presence_gate_ckpt else None
        ),
        "dual_gt_cell_presence_prob_min": (
            float(dual_gt_cell_presence_prob_min) if dual_gt_cell_presence_gate_ckpt else None
        ),
        "dual_gt_cell_presence_gate_combine": (
            str(dual_gt_cell_presence_gate_combine or "all")
            if dual_gt_cell_presence_gate_ckpt
            else None
        ),
        "dual_gt_cell_presence_gate_stats": getattr(ds, "_dual_gt_cell_presence_gate_stats", None),
        "resume_from": resume_from,
        "val_log_interval": int(val_log_interval) if val_log_interval and int(val_log_interval) > 0 else None,
        "supervise": str(supervise),
        "supervise_action": bool(use_a),
        "supervise_cell": bool(use_c),
        "supervise_orient": bool(use_o),
        "train_until_train_acc_pct": float(train_until) if train_until is not None else None,
        "max_epochs_budget": int(epoch_budget) if train_until is not None else None,
        "precompute_crops": bool(precompute_crops),
        "action_decomposition": str(ad),
        "action_loss": str(action_loss_meta),
        "best_val_cell_acc_pct": float(best_val_cell) if best_state is not None else None,
        "best_val_cell_epoch": int(best_epoch) if best_state is not None else None,
        "restore_best_val_cell_applied": bool(restore_best_val_cell and best_state is not None),
    }
    if factored and spec is not None and gn is not None:
        ckpt["game_name"] = gn
        ckpt["factored_spec"] = {
            "n_move": int(spec.n_move),
            "n_turn": int(spec.n_turn),
            "n_zap": int(spec.n_zap),
            "n_clean": int(spec.n_clean),
            "move_values": list(spec.move_values),
            "turn_values": list(spec.turn_values),
            "zap_values": list(spec.zap_values),
            "clean_values": list(spec.clean_values),
            "joint_to_factored": [list(t) for t in spec.joint_to_factored],
        }
    torch.save(ckpt, output)
    print(f"Saved {output}", flush=True)
    out = {
        "num_actions": int(n_cls),
        "action_label_mode": action_label_mode,
        "n_action_ids": len(orig_ids) if orig_ids else int(n_cls),
        "n_samples": int(len(ds)),
        "epochs": int(epochs_done),
        "epochs_requested": int(epochs),
        "final_loss": float(last_loss),
        "device": str(dev),
        "train_action_acc_pct": float(train_aa),
        "train_cell_acc_pct": float(train_ac),
        "train_orient_acc_pct": float(train_ao),
        "train_total_ce": float(train_loss_mean),
        "train_action_ce": float(train_la),
        "train_cell_ce": float(train_lc),
        "train_orient_ce": float(train_lo),
        "n_val_samples": int(n_val_samples),
        "val_action_acc_pct": float(val_aa),
        "val_cell_acc_pct": float(val_ac),
        "val_orient_acc_pct": float(val_ao),
        "val_total_ce": float(val_loss_mean),
        "val_action_ce": float(val_la),
        "val_cell_ce": float(val_lc),
        "val_orient_ce": float(val_lo),
        "input_noise_std": float(input_noise_std),
        "input_mode": str(input_mode),
        "rgb_stack": str(rgb_stack),
        "latent_channels": int(ds._latent_c),
        "center_presence_gate_ckpt": (
            os.path.abspath(center_presence_gate_ckpt) if center_presence_gate_ckpt else None
        ),
        "center_presence_prob_min": float(center_presence_prob_min) if center_presence_gate_ckpt else None,
        "center_presence_gate_stats": getattr(ds, "_center_presence_gate_stats", None),
        "dual_gt_cell_presence_gate_ckpt": (
            os.path.abspath(dual_gt_cell_presence_gate_ckpt) if dual_gt_cell_presence_gate_ckpt else None
        ),
        "dual_gt_cell_presence_prob_min": (
            float(dual_gt_cell_presence_prob_min) if dual_gt_cell_presence_gate_ckpt else None
        ),
        "dual_gt_cell_presence_gate_combine": (
            str(dual_gt_cell_presence_gate_combine or "all")
            if dual_gt_cell_presence_gate_ckpt
            else None
        ),
        "dual_gt_cell_presence_gate_stats": getattr(ds, "_dual_gt_cell_presence_gate_stats", None),
        "resume_from": resume_from,
        "val_log_interval": int(val_log_interval) if val_log_interval and int(val_log_interval) > 0 else None,
        "supervise": str(supervise),
        "train_until_train_acc_pct": float(train_until) if train_until is not None else None,
        "max_epochs_budget": int(epoch_budget) if train_until is not None else None,
        "precompute_crops": bool(precompute_crops),
        "action_decomposition": str(ad),
        "action_loss": str(action_loss_meta),
        "best_val_cell_acc_pct": float(best_val_cell) if best_state is not None else None,
        "best_val_cell_epoch": int(best_epoch) if best_state is not None else None,
        "restore_best_val_cell_applied": bool(restore_best_val_cell and best_state is not None),
    }
    if factored:
        out["train_action_component_min_pct"] = float(train_action_component_min)
        out["train_action_heads_mean_pct"] = float(train_action_heads_mean)
        out["train_action_joint_match_pct"] = float(train_action_joint_match)
        out["val_action_component_min_pct"] = float(val_action_component_min)
        out["val_action_heads_mean_pct"] = float(val_action_heads_mean)
        out["val_action_joint_match_pct"] = float(val_action_joint_match)
        if gn is not None:
            out["game_name"] = gn
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_dir", required=True, help="Directory with val_npz *.npz")
    p.add_argument("--basename_contains", default=None, help="Filter: substring of basename (one game / level)")
    p.add_argument(
        "--holdout_last_seed",
        action="store_true",
        help="Use all but the last sorted matching NPZ for training; last file is validation (needs ≥2 files).",
    )
    p.add_argument("--output", required=True, help="Output .pt checkpoint")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load 'model' weights from this .pt before training (same architecture; "
        "Adam state is not restored).",
    )
    p.add_argument("--cell_px", type=int, default=32)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument(
        "--num_actions",
        type=int,
        default=None,
        help="Required for action_label_mode=fixed (raw labels 0..N-1). Ignored for compact.",
    )
    p.add_argument(
        "--action_label_mode",
        choices=("compact", "fixed"),
        default="compact",
        help="compact: only action ids seen in data, K-class head; fixed: raw ids < num_actions.",
    )
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument(
        "--train_until_train_acc_pct",
        type=float,
        default=None,
        help="Adam only: keep training until min train acc%% among **supervised** heads "
        "reaches this (e.g. 100). Train batches stay shuffled; set --max_epochs cap.",
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="With --train_until_train_acc_pct: hard cap on epoch count (default 5000). Ignored otherwise.",
    )
    p.add_argument(
        "--restore_best_val_cell",
        action="store_true",
        help="With --train_until_train_acc_pct + held-out val + cell supervised (Adam): each epoch "
        "eval val cell accuracy; restore the best-val epoch for final metrics and the saved .pt.",
    )
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64, help="Conv width before heads (try 128 to fit small sets).")
    p.add_argument(
        "--full_batch",
        action="store_true",
        help="Use batch_size=len(dataset) each step (helps tiny sets reach 100%% train acc).",
    )
    p.add_argument(
        "--optimizer",
        choices=("adam", "lbfgs"),
        default="adam",
        help="lbfgs: full-batch second-order steps for tiny memorization (use with --full_batch).",
    )
    p.add_argument(
        "--no_verify_train_acc",
        action="store_true",
        help="Skip train-set accuracy print at end.",
    )
    p.add_argument(
        "--aux_meta",
        action="store_true",
        help="Concat 5 broadcast planes (cell, time, subject, file idx) so identical patches are separable.",
    )
    p.add_argument(
        "--memorize",
        action="store_true",
        help="Shorthand: --aux_meta --full_batch --optimizer lbfgs; bumps --epochs to at least 40.",
    )
    p.add_argument(
        "--input_noise_std",
        type=float,
        default=0.0,
        help="Noise: rgb=first 3 ch clamped [0,1]; latent=all C channels (0=off).",
    )
    p.add_argument(
        "--input_mode",
        choices=("rgb", "latent"),
        default="rgb",
        help="latent: crops from video_latents T,C,H,W in each npz (same 3x3 cell grid).",
    )
    p.add_argument(
        "--rgb_stack",
        choices=("current", "current_next"),
        default="current",
        help="RGB only: 'current' = 3ch at t; 'current_next' = 6ch same crop at t and t+1.",
    )
    p.add_argument(
        "--supervise",
        type=str,
        default="all",
        help="Which heads get loss: all | action | cell | orient | action_cell. "
        "Joint action head: BCEWithLogits + one-hot (no softmax over compact ids). "
        "Cell/orient: CrossEntropy.",
    )
    p.add_argument(
        "--precompute_crops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After indexing, load each NPZ once and cache all patches in RAM (default: true; "
        "much faster on NFS than reopening .npz every sample). Use --no-precompute_crops if "
        "RAM is tight. Requires --input_noise_std 0.",
    )
    p.add_argument(
        "--pad_mode",
        choices=("reflect", "constant"),
        default="constant",
        help="constant (default): OOB = --pad_color; reflect: mirror edges.",
    )
    p.add_argument(
        "--action-decomposition",
        dest="action_decomposition",
        choices=("joint", "factored"),
        default="joint",
        help="joint: BCEWithLogits + one-hot on compact action id (K independent sigmoids). "
        "factored: four CrossEntropy heads from MP action_set (compact+Adam only).",
    )
    p.add_argument(
        "--pad_color",
        type=str,
        default="255,0,255",
        help="R,G,B for constant padding (ignored when pad_mode=reflect).",
    )
    p.add_argument(
        "--center_presence_gate_ckpt",
        type=str,
        default=None,
        help="Path to CellPresenceMLP .pt: drop tile rows where center-cell P(present) "
        "is below --center_presence_prob_min (RGB mode only; padding must match presence training).",
    )
    p.add_argument(
        "--center_presence_prob_min",
        type=float,
        default=0.5,
        help="With --center_presence_gate_ckpt: keep row if sigmoid(presence logit) >= this.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_gate_ckpt",
        type=str,
        default=None,
        help="CellPresenceMLP .pt: keep a row only if P(present) passes on **logged** player "
        "cells at t and t+1 (single-cell RGB each; next uses true next cell, not 3×3 center). "
        "RGB only. Typical use: --supervise cell for the offset head.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_prob_min",
        type=float,
        default=0.5,
        help="With --dual_gt_cell_presence_gate_ckpt: per-endpoint P(present) threshold.",
    )
    p.add_argument(
        "--dual_gt_cell_presence_gate_combine",
        type=str,
        choices=("all", "any"),
        default="all",
        help="With --dual_gt_cell_presence_gate_ckpt: 'all' = both t and t+1 cells must pass; "
        "'any' = skip only when **both** fail (either visible is enough).",
    )
    p.add_argument(
        "--val_log_interval",
        type=int,
        default=0,
        help="If >0 and held-out val NPZs are set: every N epochs (and epoch 1) print val_snapshot "
        "action/cell/orient acc%% on the val loader (cheap sanity check during long train_until runs).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    if args.memorize:
        args.aux_meta = True
        args.full_batch = True
        args.optimizer = "lbfgs"
        args.epochs = max(int(args.epochs), 40)
    if args.train_until_train_acc_pct is not None:
        if args.memorize:
            raise SystemExit("--train_until_train_acc_pct cannot be used with --memorize (LBFGS path)")
        t = float(args.train_until_train_acc_pct)
        if t < 0.0 or t > 100.0:
            raise SystemExit("--train_until_train_acc_pct must be in [0, 100]")
    if args.precompute_crops and float(args.input_noise_std) != 0.0:
        raise SystemExit("--precompute_crops requires --input_noise_std 0")
    paths = collect_npz_paths(args.val_dir, args.basename_contains)
    if not paths:
        raise SystemExit(f"No npz under {args.val_dir!r} (filter={args.basename_contains!r})")

    val_paths: list[str] | None = None
    train_paths = paths
    if args.holdout_last_seed:
        if len(paths) >= 2:
            val_paths = [paths[-1]]
            train_paths = paths[:-1]
        else:
            print("WARNING: --holdout_last_seed ignored (need ≥2 matching npz files)", flush=True)

    if str(args.rgb_stack) == "current_next" and str(args.input_mode) != "rgb":
        raise SystemExit("--rgb_stack current_next requires --input_mode rgb")
    if getattr(args, "center_presence_gate_ckpt", None) and str(args.input_mode) != "rgb":
        raise SystemExit("--center_presence_gate_ckpt requires --input_mode rgb")
    if getattr(args, "dual_gt_cell_presence_gate_ckpt", None) and str(args.input_mode) != "rgb":
        raise SystemExit("--dual_gt_cell_presence_gate_ckpt requires --input_mode rgb")

    if getattr(args, "resume", None):
        rpa = os.path.abspath(str(args.resume).strip())
        if not os.path.isfile(rpa):
            raise SystemExit(f"--resume file not found: {rpa!r}")

    if val_paths:
        print(
            "split: --holdout_last_seed → val = last NPZ only (generalization; small n ⇒ noisy %).",
            flush=True,
        )
    else:
        print(
            "split: no holdout → lines labeled 'full val_npz fit' **match train** on all NPZs "
            "under --val_dir (not a held-out generalization set).",
            flush=True,
        )

    train_to_checkpoint(
        train_paths,
        args.output,
        cell_px=args.cell_px,
        image_size=args.image_size,
        num_actions=args.num_actions,
        action_label_mode=args.action_label_mode,
        aux_meta_channels=args.aux_meta,
        val_npz_paths=val_paths,
        input_noise_std=args.input_noise_std,
        input_mode=args.input_mode,
        rgb_stack=str(args.rgb_stack),
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        workers=args.workers,
        pad_mode=args.pad_mode,
        pad_color=args.pad_color,
        hidden=args.hidden,
        full_batch=args.full_batch,
        verify_train_acc=not args.no_verify_train_acc,
        optimizer=args.optimizer,
        supervise=str(args.supervise),
        train_until_train_acc_pct=(
            float(args.train_until_train_acc_pct) if args.train_until_train_acc_pct is not None else None
        ),
        max_epochs=int(args.max_epochs) if args.max_epochs is not None else None,
        precompute_crops=bool(args.precompute_crops),
        action_decomposition=str(args.action_decomposition),
        restore_best_val_cell=bool(args.restore_best_val_cell),
        center_presence_gate_ckpt=args.center_presence_gate_ckpt,
        center_presence_prob_min=float(args.center_presence_prob_min),
        dual_gt_cell_presence_gate_ckpt=args.dual_gt_cell_presence_gate_ckpt,
        dual_gt_cell_presence_prob_min=float(args.dual_gt_cell_presence_prob_min),
        dual_gt_cell_presence_gate_combine=str(args.dual_gt_cell_presence_gate_combine),
        resume_path=os.path.abspath(str(args.resume).strip()) if getattr(args, "resume", None) else None,
        val_log_interval=(int(args.val_log_interval) if int(args.val_log_interval) > 0 else None),
    )


if __name__ == "__main__":
    main()
