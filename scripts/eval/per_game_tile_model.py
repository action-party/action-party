#!/usr/bin/env python3
"""Per-game 3×3 cell tile dataset and small CNN (action + cell + orient heads)."""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in __import__("sys").path:
    __import__("sys").path.insert(0, _EVAL_DIR)

import per_game_tile_crops as ptc

_AUX_META_N = 5  # gi, gj, t, s, file index (all normalized to ~[0,1])


def npz_game_group_key(basename: str) -> str:
    """
    Group val_npz files that share the same level across seeds.

    ``clean_up_s3.npz`` → ``clean_up``; ``chicken_in_the_matrix__arena_s0.npz`` →
    ``chicken_in_the_matrix__arena``. If the pattern does not match, uses the stem.
    """
    bn = os.path.basename(basename)
    m = re.match(r"^(.*)_s\d+\.npz$", bn, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return os.path.splitext(bn)[0]


def group_val_npz_by_game(val_dir: str, skip_substrings: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Map game group key → sorted list of absolute paths to ``*.npz``."""
    skip = [s for s in (skip_substrings or []) if s]
    out: dict[str, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(val_dir, "*.npz"))):
        if any(s in os.path.basename(p) for s in skip):
            continue
        k = npz_game_group_key(os.path.basename(p))
        out.setdefault(k, []).append(os.path.abspath(p))
    return dict(sorted(out.items()))


def infer_num_action_classes(npz_paths: Sequence[str], cap: int = 64, minimum: int = 8) -> int:
    """Largest action index + 1 seen across NPZs (clamped)."""
    m = 0
    for p in npz_paths:
        try:
            with np.load(p, allow_pickle=True) as d:
                if "actions" not in d.files:
                    continue
                a = d["actions"]
                m = max(m, int(np.max(a)))
        except OSError:
            continue
    return min(cap, max(minimum, m + 1))


def _num_subjects(d: np.lib.npyio.NpzFile) -> int:
    if "num_subjects" in d.files:
        return int(d["num_subjects"])
    c = d["frame_coords"]
    return int(c.shape[1])


def aux_meta_planes_torch(
    fi: int,
    n_files: int,
    t: int,
    T_max: int,
    s: int,
    n_sub: int,
    gi: int,
    gj: int,
    nh: int,
    nw: int,
    patch_hw: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Five scalar planes (player cell, time, subject, seed-file index) broadcast
    to patch resolution. Matches training when ``gi,gj`` come from the same
    ``(cx, cy)`` used to extract the patch.
    """
    vals = (
        float(gi) / max(1, nh - 1),
        float(gj) / max(1, nw - 1),
        float(t) / max(1, T_max - 1),
        float(s) / max(1, n_sub - 1),
        float(fi) / max(1, n_files - 1),
    )
    h0, w0 = int(patch_hw[0]), int(patch_hw[1])
    tch = torch.tensor(vals, dtype=dtype, device=device).view(5, 1, 1)
    return tch.expand(5, h0, w0)


def fi_index_for_train_paths(basename: str, train_npz_paths: Sequence[str] | None) -> tuple[int, int]:
    """Return ``(fi, n_files)`` for ``--aux_meta`` eval (basename match to checkpoint paths)."""
    paths = [p for p in (train_npz_paths or []) if p]
    n = max(1, len(paths))
    if not paths:
        return 0, 1
    names = [os.path.basename(p) for p in paths]
    if basename in names:
        return names.index(basename), n
    return 0, n


def filter_tile_index_rows_by_center_presence(
    index_rows: list[tuple[int, int, int, int, int, int, int]],
    npz_paths: list[str],
    *,
    cell_px: int,
    image_size: int,
    pad_mode: str,
    pad_fill: tuple[float, ...] | float | None,
    ckpt_path: str,
    prob_min: float,
    device: torch.device,
) -> tuple[list[tuple[int, int, int, int, int, int, int]], dict[str, Any]]:
    """
    Subset tile dataset rows to those where ``CellPresenceMLP`` on the **center cell**
    of the current RGB frame (same crop as the tile CNN's first RGB block) has
    ``sigmoid(logit) >= prob_min``.

    Requires ``input_mode=rgb`` at the dataset level. Uses the same ``extract_3x3_cell_patch``
    + center slice as the teacher-forced anchor (player cell is the middle of the 3×3).
    """
    from train_cell_player_presence_mlp import CellPresenceMLP  # noqa: PLC0415

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cp_cell = int(ckpt["cell_px"])
    cp_im = int(ckpt["image_size"])
    if cp_cell != int(cell_px) or cp_im != int(image_size):
        raise ValueError(
            f"presence checkpoint cell_px/image_size ({cp_cell}, {cp_im}) != "
            f"tile dataset ({cell_px}, {image_size})"
        )
    pres_pad = str(ckpt.get("pad_mode", "constant"))
    if pres_pad != str(pad_mode):
        raise ValueError(f"presence ckpt pad_mode={pres_pad!r} != dataset pad_mode={pad_mode!r}")
    rf = ckpt.get("pad_fill")
    pres_fill: tuple[float, ...] | None
    if pres_pad == "constant" and rf is not None:
        pres_fill = tuple(float(x) for x in rf)
    else:
        pres_fill = None
    ds_fill = pad_fill if pad_mode == "constant" else None
    if isinstance(ds_fill, tuple) and isinstance(pres_fill, tuple):
        if len(ds_fill) == len(pres_fill) and len(ds_fill) > 0:
            if max(abs(ds_fill[i] - pres_fill[i]) for i in range(len(ds_fill))) > 1e-6:
                raise ValueError(f"presence ckpt pad_fill {pres_fill!r} != dataset pad_fill {ds_fill!r}")
    hidden = int(ckpt["hidden"])
    flat_dim = int(ckpt["flat_dim"])
    m = CellPresenceMLP(flat_dim=flat_dim, hidden=hidden).to(device)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()

    by_fi: dict[int, list[tuple[int, int, int, int, int, int, int]]] = defaultdict(list)
    for row in index_rows:
        by_fi[int(row[0])].append(row)

    kept: list[tuple[int, int, int, int, int, int, int]] = []
    batch_size = 512
    with torch.inference_mode():
        for fi in sorted(by_fi.keys()):
            rows = sorted(by_fi[fi], key=lambda r: (r[1], r[2]))
            buf_x: list[torch.Tensor] = []
            buf_rows: list[tuple[int, int, int, int, int, int, int]] = []

            def _flush() -> None:
                if not buf_x:
                    return
                xt = torch.stack(buf_x, dim=0).to(device)
                logits = m(xt.view(xt.shape[0], -1))
                pr = torch.sigmoid(logits)
                for k, row in enumerate(buf_rows):
                    if float(pr[k].item()) >= float(prob_min):
                        kept.append(row)
                buf_x.clear()
                buf_rows.clear()

            path = npz_paths[fi]
            d = np.load(path, allow_pickle=True)
            try:
                coords = d["frame_coords"]
                pf = d["pixel_frames"]
                l2p = ptc.get_latent_to_pixel(len(coords))
                for row in rows:
                    t, s = int(row[1]), int(row[2])
                    cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                    if cx == 0.0 and cy == 0.0:
                        continue
                    pix_i = l2p[min(int(t), len(l2p) - 1)]
                    frame = pf[pix_i]
                    patch = ptc.extract_3x3_cell_patch(
                        frame,
                        cx,
                        cy,
                        cell_px,
                        image_size,
                        pad_cells=1,
                        pad_mode=pad_mode,
                        pad_fill=pad_fill,
                    )
                    cell_hwc = ptc.center_cell_hwc_from_3x3_patch(patch, cell_px)
                    x_cell = torch.from_numpy(cell_hwc).permute(2, 0, 1).contiguous().float() / 255.0
                    buf_x.append(x_cell)
                    buf_rows.append(row)
                    if len(buf_x) >= batch_size:
                        _flush()
                _flush()
            finally:
                d.close()

    stats: dict[str, Any] = {
        "before": int(len(index_rows)),
        "after": int(len(kept)),
        "presence_ckpt": os.path.abspath(ckpt_path),
        "prob_min": float(prob_min),
    }
    return kept, stats


def filter_tile_index_rows_by_gt_cell_pair_presence_current_next(
    index_rows: list[tuple[int, int, int, int, int, int, int]],
    npz_paths: list[str],
    *,
    cell_px: int,
    image_size: int,
    pad_mode: str,
    pad_fill: tuple[float, ...] | float | None,
    ckpt_path: str,
    prob_min: float,
    device: torch.device,
    combine: str = "all",
) -> tuple[list[tuple[int, int, int, int, int, int, int]], dict[str, Any]]:
    """
    Keep rows where ``CellPresenceMLP`` passes on **both**:
    - GT player grid cell ``(gi, gj)`` on the **current** pixel frame ``l2p[t]``;
    - GT player grid cell ``(gi2, gj2)`` on the **next** pixel frame ``l2p[t+1]``,

    with ``(gi2, gj2)`` inferred from the stored ``y_cell`` offset class (same clamp as
    when building the 9-way label). Uses ``extract_single_cell_patch_rgb`` for each
    cell (full-frame padded crop), **not** the 3×3 patch center for ``t+1`` — so when the
    player moves within the **fixed** ``coords[t]``-anchored 3×3 window, the second check
    is still at their **true** next world cell (matches ``current_next`` semantics).

    RGB / padding rules match ``filter_tile_index_rows_by_center_presence``.

    ``combine``:
    - ``"all"`` (default): keep a row only when presence passes on **both** GT endpoint cells.
    - ``"any"``: keep when **either** endpoint passes (skip rows where the player is absent
      in **both** frames at the logged cells — matches rollout metrics that ignore fully
      invisible transitions).
    """
    from train_cell_player_presence_mlp import CellPresenceMLP  # noqa: PLC0415

    comb = str(combine or "all").strip().lower()
    if comb not in ("all", "any"):
        raise ValueError("combine must be 'all' or 'any'")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    cp_cell = int(ckpt["cell_px"])
    cp_im = int(ckpt["image_size"])
    if cp_cell != int(cell_px) or cp_im != int(image_size):
        raise ValueError(
            f"presence checkpoint cell_px/image_size ({cp_cell}, {cp_im}) != "
            f"tile dataset ({cell_px}, {image_size})"
        )
    pres_pad = str(ckpt.get("pad_mode", "constant"))
    if pres_pad != str(pad_mode):
        raise ValueError(f"presence ckpt pad_mode={pres_pad!r} != dataset pad_mode={pad_mode!r}")
    rf = ckpt.get("pad_fill")
    pres_fill: tuple[float, ...] | None
    if pres_pad == "constant" and rf is not None:
        pres_fill = tuple(float(x) for x in rf)
    else:
        pres_fill = None
    ds_fill = pad_fill if pad_mode == "constant" else None
    if isinstance(ds_fill, tuple) and isinstance(pres_fill, tuple):
        if len(ds_fill) == len(pres_fill) and len(ds_fill) > 0:
            if max(abs(ds_fill[i] - pres_fill[i]) for i in range(len(ds_fill))) > 1e-6:
                raise ValueError(f"presence ckpt pad_fill {pres_fill!r} != dataset pad_fill {ds_fill!r}")
    hidden = int(ckpt["hidden"])
    flat_dim = int(ckpt["flat_dim"])
    m = CellPresenceMLP(flat_dim=flat_dim, hidden=hidden).to(device)
    m.load_state_dict(ckpt["model"], strict=True)
    m.eval()

    by_fi: dict[int, list[tuple[int, int, int, int, int, int, int]]] = defaultdict(list)
    for row in index_rows:
        by_fi[int(row[0])].append(row)

    kept: list[tuple[int, int, int, int, int, int, int]] = []
    pair_batch = 256
    with torch.inference_mode():
        for fi in sorted(by_fi.keys()):
            rows = sorted(by_fi[fi], key=lambda r: (r[1], r[2]))
            buf_cur: list[torch.Tensor] = []
            buf_nxt: list[torch.Tensor] = []
            buf_rows: list[tuple[int, int, int, int, int, int, int]] = []

            def _flush_pairs() -> None:
                if not buf_cur:
                    return
                b = len(buf_cur)
                xc = torch.stack(buf_cur, dim=0).to(device)
                xn = torch.stack(buf_nxt, dim=0).to(device)
                x2 = torch.cat([xc, xn], dim=0)
                logits = m(x2.view(x2.shape[0], -1))
                pr = torch.sigmoid(logits)
                pr_c = pr[:b]
                pr_n = pr[b:]
                thr = float(prob_min)
                for k, row in enumerate(buf_rows):
                    ok_c = float(pr_c[k].item()) >= thr
                    ok_n = float(pr_n[k].item()) >= thr
                    if (ok_c and ok_n) if comb == "all" else (ok_c or ok_n):
                        kept.append(row)
                buf_cur.clear()
                buf_nxt.clear()
                buf_rows.clear()

            path = npz_paths[fi]
            d = np.load(path, allow_pickle=True)
            try:
                coords = d["frame_coords"]
                pf = d["pixel_frames"]
                l2p = ptc.get_latent_to_pixel(len(coords))
                for row in rows:
                    _fi, t, s, y_cell, _ya, gi, gj = row
                    t = int(t)
                    s = int(s)
                    cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                    cx2, cy2 = float(coords[t + 1, s, 0]), float(coords[t + 1, s, 1])
                    if (cx == 0.0 and cy == 0.0) or (cx2 == 0.0 and cy2 == 0.0):
                        continue
                    dgi, dgj = ptc.class_to_cell_delta(int(y_cell))
                    gi2, gj2 = int(gi) + int(dgi), int(gj) + int(dgj)
                    pix_i = l2p[min(t, len(l2p) - 1)]
                    pix_ip = l2p[min(t + 1, len(l2p) - 1)]
                    frame_t = pf[pix_i]
                    frame_tp = pf[pix_ip]
                    hwc_t = ptc.extract_single_cell_patch_rgb(
                        frame_t, int(gi), int(gj), cell_px, pad_mode=pad_mode, pad_fill=pad_fill
                    )
                    hwc_n = ptc.extract_single_cell_patch_rgb(
                        frame_tp, gi2, gj2, cell_px, pad_mode=pad_mode, pad_fill=pad_fill
                    )
                    buf_cur.append(
                        torch.from_numpy(hwc_t).permute(2, 0, 1).contiguous().float() / 255.0
                    )
                    buf_nxt.append(
                        torch.from_numpy(hwc_n).permute(2, 0, 1).contiguous().float() / 255.0
                    )
                    buf_rows.append(row)
                    if len(buf_cur) >= pair_batch:
                        _flush_pairs()
                _flush_pairs()
            finally:
                d.close()

    stats: dict[str, Any] = {
        "before": int(len(index_rows)),
        "after": int(len(kept)),
        "presence_ckpt": os.path.abspath(ckpt_path),
        "prob_min": float(prob_min),
        "mode": "gt_cell_t_and_tplus1",
        "combine": comb,
    }
    return kept, stats


class PerGameTileNPZDataset(Dataset):
    """
    Samples (patch, y_action, y_cell9, y_orient4) from val_npz / eval-style NPZ.

    Patch is float tensor of shape ``[3, H, W]``, ``[6, H, W]`` when ``rgb_stack="current_next"``:
    **same world-fixed 3×3 cell crop** (anchor ``frame_coords[t, s]``) on frame **t** then
    on frame **t+1**, so the player moves **inside** the patch between the two RGB blocks.
    Or ``[8, H, W]`` (with aux meta) in ``[0,1]``.
    y_cell9: 3×3 neighbor offset class from player cell at t to cell at t+1 (clamped).
    y_orient4: orientation at t+1.
    Padding defaults to ``constant`` + ``DEFAULT_PAD_RGB`` (magenta OOB).

    **Action labels** (``action_label_mode``):
    - ``"compact"`` (default): only Melting Pot action ids that appear in this
      dataset are used; targets are ``0 .. K-1`` and ``action_orig_ids[k]`` is
      the original id. Best for small per-game training sets.
    - ``"fixed"``: keep raw integer labels and require ``0 <= ya < num_actions``.

    **Aux meta planes** (``aux_meta_channels=True``): concatenates five channels
    (normalized cell indices, time, subject, seed-file index in ``npz_paths``) so
    identical-looking patches from different trajectories are separable; needed
    to reach ~100%% train accuracy on small sets with repeated local textures.
    When ``aux_fi_reference_paths`` is set (train/val split across seeds), the
    file-index plane uses the index in that list so val NPZs match training
    normalization.

    **Input noise** (``input_noise_std > 0``): RGB mode adds noise to the first
    three channels only; **latent** mode adds i.i.d. noise to all latent channels
    (no clamp). Aux planes stay exact.

    **Temporal convention (RGB, ``rgb_stack``):** Each row uses simulation indices
    ``(t, t+1)`` with ``acts[s, t]`` the logged Melting-Pot action at step ``t`` (the
    one applied in state at ``t`` and producing the transition to ``t+1``). The
    **first** RGB block is ``pixel_frames[l2p[t]]`` cropped around ``frame_coords[t, s]``.
    With ``rgb_stack="current_next"``, the **second** block is ``pixel_frames[l2p[t+1]]``
    using the **same** ``(cx, cy)`` anchor as the first block (fixed window in world space;
    the player’s motion from ``t`` to ``t+1`` is visible inside that window). The map
    ``l2p`` is strictly increasing, so ``t < t+1`` implies **different stored frames**
    whenever ``t+1`` is in range.

    **Precomputed crops** (``precompute_crops=True``): after the index is built, each
    patch tensor is computed once (one ``np.load`` per NPZ file while scanning its rows)
    and stored in RAM. ``__getitem__`` then only indexes tensors — much faster training
    than reopening compressed NPZs every step. Requires ``input_noise_std=0``. Expect
    ~``N * input_channels * (3*cell_px)² * 4`` bytes for ``N=len(dataset)``.

    **Why single-frame accuracy can be well above chance:** The policy that produced
    the logs is a function of **state at t**; the rendered frame at ``t`` is a rich
    (partial) observation of that state, so predicting ``acts[s,t]`` from it is not
    “impossible” — it is **partial observability**, not absence of signal. Adding
    ``t+1`` pixels in the **same** crop shows **motion inside a fixed neighborhood**, which
    can push accuracy higher when displacement/orientation change disambiguates the action.

    **Why 100% train may still fail:** Label / policy noise, identical-looking patches
    with different actions, AMP and finite capacity, and the BCE objective not being
    identical to thresholding ``sigmoid(logit[gt])`` at 0.5 on every sample.

    **Latent tiles** (``input_mode="latent"``): each NPZ must contain
    ``video_latents`` with shape ``(T, C, Hlat, Wlat)`` (e.g. ``T`` matches
    ``frame_coords``, ``C=16``, ``Hlat=Wlat=image_size/8``). Crops use
    ``per_game_tile_crops.extract_3x3_cell_patch_latent_chw`` — same 3×3 **cell**
    neighborhood as RGB, aligned via ``cell_px_latent``.

    **Center-cell presence gate** (``center_presence_gate_ckpt``): after indexing, drops
    rows where ``CellPresenceMLP`` on the **center RGB cell** at time ``t`` (same frame
    as the first RGB block) has ``sigmoid(logit) < center_presence_prob_min``.
    Requires ``input_mode='rgb'``; checkpoint ``cell_px`` / ``image_size`` / padding must
    match the tile dataset. Use to train/eval only when the visibility model says the
    anchor tile shows a player.

    **GT current+next cell presence gate** (``dual_gt_cell_presence_gate_ckpt``): after
    indexing (and after the center gate if both are set), filters rows using
    ``CellPresenceMLP`` on **single-cell** RGB crops at the **logged** grid cells ``(gi,gj)``
    on ``l2p[t]`` and ``(gi2,gj2)`` on ``l2p[t+1]``, with ``(gi2,gj2)`` from the
    stored 9-way offset label (same clamp as ``y_cell``). With ``dual_gt_cell_presence_gate_combine="all"``
    (default), **both** endpoints must pass ``P>=dual_gt_cell_presence_prob_min``. With
    ``"any"``, rows pass if **either** endpoint passes (drop only when the player is absent
    in **both** frames at those cells).
    """

    def __init__(
        self,
        npz_paths: Sequence[str],
        cell_px: int = 32,
        image_size: int = 512,
        num_actions: int = 25,
        pad_mode: str = "constant",
        pad_fill: tuple[float, ...] | float | None = None,
        action_label_mode: str = "compact",
        aux_meta_channels: bool = False,
        compact_action_orig_ids: Sequence[int] | None = None,
        aux_fi_reference_paths: Sequence[str] | None = None,
        input_noise_std: float = 0.0,
        input_mode: str = "rgb",
        rgb_stack: str = "current",
        precompute_crops: bool = False,
        center_presence_gate_ckpt: str | None = None,
        center_presence_prob_min: float = 0.5,
        dual_gt_cell_presence_gate_ckpt: str | None = None,
        dual_gt_cell_presence_prob_min: float = 0.5,
        dual_gt_cell_presence_gate_combine: str = "all",
    ):
        super().__init__()
        self.cell_px = int(cell_px)
        self.image_size = int(image_size)
        self.num_actions = int(num_actions)
        self.pad_mode = str(pad_mode)
        self.pad_fill = pad_fill
        self.action_label_mode = str(action_label_mode)
        self.aux_meta_channels = bool(aux_meta_channels)
        self._input_noise_std = float(input_noise_std)
        self.input_mode = str(input_mode or "rgb").lower()
        self._rgb_stack = str(rgb_stack or "current").lower()
        if self._rgb_stack not in ("current", "current_next"):
            raise ValueError("rgb_stack must be 'current' or 'current_next'")
        if self._rgb_stack == "current_next" and self.input_mode != "rgb":
            raise ValueError("rgb_stack='current_next' requires input_mode='rgb'")
        if self.input_mode not in ("rgb", "latent"):
            raise ValueError("input_mode must be 'rgb' or 'latent'")
        if self.action_label_mode not in ("compact", "fixed"):
            raise ValueError("action_label_mode must be 'compact' or 'fixed'")
        self._paths = [os.path.abspath(p) for p in npz_paths]
        self._latent_c = 3
        if self.input_mode == "latent":
            for path in self._paths:
                with np.load(path, allow_pickle=True) as d0:
                    if "video_latents" not in d0.files:
                        raise ValueError(
                            f"input_mode=latent requires 'video_latents' in {path!r} "
                            "(shape T,C,H,W with T aligned to frame_coords)"
                        )
                    vl = d0["video_latents"]
                    if vl.ndim != 4:
                        raise ValueError(f"video_latents must be 4D T,C,H,W, got {vl.shape!r}")
                    self._latent_c = int(vl.shape[1])
                break
        self._aux_fi_global: list[str] | None = None
        self._aux_fi_index: list[int] | None = None
        if aux_fi_reference_paths is not None and len(aux_fi_reference_paths) > 0:
            self._aux_fi_global = [os.path.abspath(p) for p in aux_fi_reference_paths]
            pmap = {p: i for i, p in enumerate(self._aux_fi_global)}
            self._aux_fi_index = [pmap.get(self._paths[i], i) for i in range(len(self._paths))]
        self._T_per: list[int] = []
        self._ns_per: list[int] = []
        for path in self._paths:
            with np.load(path, allow_pickle=True) as d0:
                c0 = d0["frame_coords"]
                self._T_per.append(int(c0.shape[0]))
                self._ns_per.append(int(c0.shape[1]))
        # (fi, t, s, y_cell, y_act, gi, gj) — y_act compact index if compact else raw
        self._index: list[tuple[int, int, int, int, int, int, int]] = []
        self._action_orig_ids: list[int] = []

        candidates: list[tuple[int, int, int, int, int, int, int]] = []
        for fi, path in enumerate(self._paths):
            d = np.load(path, allow_pickle=True)
            coords = d["frame_coords"]
            acts = d["actions"]
            ori = d["orientations"] if "orientations" in d.files else None
            if ori is None:
                d.close()
                continue
            ns = _num_subjects(d)
            T = int(coords.shape[0])
            n_act = int(acts.shape[1])
            if T < 2:
                d.close()
                continue
            last_t = min(T - 1, n_act)
            for t in range(last_t):
                for s in range(min(ns, coords.shape[1], acts.shape[0])):
                    cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                    if cx == 0 and cy == 0:
                        continue
                    cx2, cy2 = float(coords[t + 1, s, 0]), float(coords[t + 1, s, 1])
                    if cx2 == 0 and cy2 == 0:
                        continue
                    gi, gj = ptc.coords_to_cell_indices(cx, cy, self.cell_px, self.image_size)
                    gi2, gj2 = ptc.coords_to_cell_indices(cx2, cy2, self.cell_px, self.image_size)
                    dgi = max(-1, min(1, gi2 - gi))
                    dgj = max(-1, min(1, gj2 - gj))
                    ya = int(acts[s, t])
                    if ya < 0:
                        continue
                    if self.action_label_mode == "fixed" and ya >= self.num_actions:
                        continue
                    y_cell = ptc.cell_delta_to_class(dgi, dgj)
                    candidates.append((fi, t, s, y_cell, ya, gi, gj))
            d.close()

        if self.action_label_mode == "compact":
            if compact_action_orig_ids is not None:
                orig_ids = sorted({int(x) for x in compact_action_orig_ids})
            else:
                orig_ids = sorted({c[4] for c in candidates})
            self._action_orig_ids = orig_ids
            remap = {o: i for i, o in enumerate(orig_ids)}
            self._index = [
                (fi, t, s, yc, remap[ya], gi, gj)
                for fi, t, s, yc, ya, gi, gj in candidates
                if ya in remap
            ]
        else:
            self._action_orig_ids = list(range(self.num_actions))
            self._index = [tuple(a) for a in candidates]

        self._center_presence_gate_stats: dict[str, Any] | None = None
        ck_gate = center_presence_gate_ckpt
        if ck_gate is not None and str(ck_gate).strip():
            if self.input_mode != "rgb":
                raise ValueError("center_presence_gate_ckpt requires input_mode='rgb'")
            dev_gate = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            new_index, gate_stats = filter_tile_index_rows_by_center_presence(
                self._index,
                self._paths,
                cell_px=self.cell_px,
                image_size=self.image_size,
                pad_mode=self.pad_mode,
                pad_fill=self.pad_fill,
                ckpt_path=str(ck_gate).strip(),
                prob_min=float(center_presence_prob_min),
                device=dev_gate,
            )
            if len(new_index) == 0:
                raise RuntimeError(
                    "center_presence_gate_ckpt removed all rows "
                    "(try lowering center_presence_prob_min or check padding match)"
                )
            self._index = new_index
            self._center_presence_gate_stats = gate_stats
            print(
                f"center_presence_gate: kept {gate_stats['after']}/{gate_stats['before']} rows "
                f"(P(present)>={gate_stats['prob_min']}, ckpt={gate_stats['presence_ckpt']!r})",
                flush=True,
            )

        self._dual_gt_cell_presence_gate_stats: dict[str, Any] | None = None
        dk = dual_gt_cell_presence_gate_ckpt
        dcombine = str(dual_gt_cell_presence_gate_combine or "all").strip().lower()
        if dcombine not in ("all", "any"):
            raise ValueError("dual_gt_cell_presence_gate_combine must be 'all' or 'any'")
        if dk is not None and str(dk).strip():
            if self.input_mode != "rgb":
                raise ValueError("dual_gt_cell_presence_gate_ckpt requires input_mode='rgb'")
            dev_d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            new_index2, dual_stats = filter_tile_index_rows_by_gt_cell_pair_presence_current_next(
                self._index,
                self._paths,
                cell_px=self.cell_px,
                image_size=self.image_size,
                pad_mode=self.pad_mode,
                pad_fill=self.pad_fill,
                ckpt_path=str(dk).strip(),
                prob_min=float(dual_gt_cell_presence_prob_min),
                device=dev_d,
                combine=dcombine,
            )
            if len(new_index2) == 0:
                raise RuntimeError(
                    "dual_gt_cell_presence_gate_ckpt removed all rows "
                    "(try lowering dual_gt_cell_presence_prob_min or check padding match)"
                )
            self._index = new_index2
            self._dual_gt_cell_presence_gate_stats = dual_stats
            print(
                f"dual_gt_cell_presence_gate: kept {dual_stats['after']}/{dual_stats['before']} rows "
                f"(GT cells t and t+1, combine={dual_stats.get('combine', dcombine)!r}, "
                f"P>={dual_stats['prob_min']}, ckpt={dual_stats['presence_ckpt']!r})",
                flush=True,
            )

        self._cached_x: torch.Tensor | None = None
        self._cached_ya: torch.Tensor | None = None
        self._cached_yc: torch.Tensor | None = None
        self._cached_yo: torch.Tensor | None = None
        self._precompute_crops = bool(precompute_crops)
        if self._precompute_crops and float(self._input_noise_std) != 0.0:
            raise ValueError("precompute_crops=True requires input_noise_std=0 (noise would change each epoch)")
        print(
            f"PerGameTileNPZDataset: indexed {len(self._index)} rows from {len(self._paths)} files, "
            f"{self.num_action_classes} action classes; precompute_crops={self._precompute_crops}",
            flush=True,
        )
        if self._precompute_crops and len(self._index) > 0:
            self._build_crop_cache()

    def _load_item_from_npz(self, d: Any, idx: int) -> tuple[torch.Tensor, int, int, int]:
        """Build one training row from an already-open ``np.load`` handle (must match ``idx``'s file)."""
        fi, t, s, y_cell, y_act, gi, gj = self._index[idx]
        coords = d["frame_coords"]
        ori = d["orientations"].astype(np.int64)
        cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
        if self.input_mode == "latent":
            vl = d["video_latents"]
            t_lat = min(int(t), int(vl.shape[0]) - 1)
            lat_chw = np.ascontiguousarray(vl[t_lat], dtype=np.float32)
            lat_fill = 0.0 if self.pad_mode == "constant" else None
            patch = ptc.extract_3x3_cell_patch_latent_chw(
                lat_chw,
                cx,
                cy,
                self.cell_px,
                self.image_size,
                pad_cells=1,
                pad_mode=self.pad_mode,
                pad_fill=lat_fill,
            )
            x_body = torch.from_numpy(patch).permute(2, 0, 1).float()
        else:
            pf = d["pixel_frames"]
            l2p = ptc.get_latent_to_pixel(len(coords))
            pix_i = l2p[min(int(t), len(l2p) - 1)]
            frame = pf[pix_i]
            patch = ptc.extract_3x3_cell_patch(
                frame,
                cx,
                cy,
                self.cell_px,
                self.image_size,
                pad_cells=1,
                pad_mode=self.pad_mode,
                pad_fill=self.pad_fill,
            )
            xt = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            if self._rgb_stack == "current_next":
                pix_ip = l2p[min(int(t + 1), len(l2p) - 1)]
                frame_p = pf[pix_ip]
                patch_p = ptc.extract_3x3_cell_patch(
                    frame_p,
                    cx,
                    cy,
                    self.cell_px,
                    self.image_size,
                    pad_cells=1,
                    pad_mode=self.pad_mode,
                    pad_fill=self.pad_fill,
                )
                xtp = torch.from_numpy(patch_p).permute(2, 0, 1).float() / 255.0
                x_body = torch.cat([xt, xtp], dim=0)
            else:
                x_body = xt
        y_ori = int(ori[t + 1, s]) % 4
        if self.aux_meta_channels:
            nh = self.image_size // self.cell_px
            nw = nh
            if self._aux_fi_global is not None:
                fi_aux = int(self._aux_fi_index[fi]) if self._aux_fi_index is not None else fi
                n_files = max(1, len(self._aux_fi_global))
            else:
                fi_aux = fi
                n_files = max(1, len(self._paths))
            pl = aux_meta_planes_torch(
                fi_aux,
                n_files,
                t,
                self._T_per[fi],
                s,
                self._ns_per[fi],
                gi,
                gj,
                nh,
                nw,
                (x_body.shape[1], x_body.shape[2]),
                dtype=x_body.dtype,
                device=None,
            )
            x = torch.cat([x_body, pl], dim=0)
        else:
            x = x_body
        return x, int(y_act), int(y_cell), int(y_ori)

    def _build_crop_cache(self) -> None:
        """Fill ``_cached_*`` tensors; one ``np.load`` per NPZ file."""
        n = len(self._index)
        by_fi = defaultdict(list)
        for idx in range(n):
            by_fi[self._index[idx][0]].append(idx)
        xs: list[torch.Tensor | None] = [None] * n
        yas = torch.empty(n, dtype=torch.long)
        ycs = torch.empty(n, dtype=torch.long)
        yos = torch.empty(n, dtype=torch.long)
        for fi in sorted(by_fi.keys()):
            path = self._paths[fi]
            d = np.load(path, allow_pickle=True)
            try:
                for idx in by_fi[fi]:
                    x, ya, yc, yo = self._load_item_from_npz(d, idx)
                    xs[idx] = x
                    yas[idx] = ya
                    ycs[idx] = yc
                    yos[idx] = yo
            finally:
                d.close()
        stacked = torch.stack([xs[i] for i in range(n)], dim=0)  # type: ignore[list-item]
        nbytes = int(stacked.numel()) * 4
        print(
            f"precompute_crops: cached N={n} patches x shape={tuple(stacked.shape)} "
            f"(~{nbytes / (1024 ** 3):.3f} GiB float32)",
            flush=True,
        )
        self._cached_x = stacked
        self._cached_ya = yas
        self._cached_yc = ycs
        self._cached_yo = yos

    @property
    def num_action_classes(self) -> int:
        if self.action_label_mode == "compact":
            return max(1, len(self._action_orig_ids))
        return max(1, int(self.num_actions))

    @property
    def rgb_stack(self) -> str:
        return self._rgb_stack

    @property
    def input_channels(self) -> int:
        if self.input_mode == "latent":
            base = self._latent_c
        else:
            base = 6 if self._rgb_stack == "current_next" else 3
        return base + _AUX_META_N if self.aux_meta_channels else base

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx: int):
        if self._cached_x is not None:
            x = self._cached_x[idx]
            if self._input_noise_std > 0.0:
                x = x.clone()
                if self.input_mode == "latent":
                    x = x + torch.randn_like(x) * self._input_noise_std
                else:
                    n_rgb = 6 if self._rgb_stack == "current_next" else 3
                    n_rgb = min(n_rgb, x.shape[0])
                    noise = torch.randn(n_rgb, x.shape[1], x.shape[2], dtype=x.dtype) * self._input_noise_std
                    x[:n_rgb] = (x[:n_rgb] + noise).clamp(0.0, 1.0)
            return (
                x,
                int(self._cached_ya[idx].item()),
                int(self._cached_yc[idx].item()),
                int(self._cached_yo[idx].item()),
            )

        fi, t, s, y_cell, y_act, gi, gj = self._index[idx]
        path = self._paths[fi]
        d = np.load(path, allow_pickle=True)
        try:
            x, y_act, y_cell, y_ori = self._load_item_from_npz(d, idx)
        finally:
            d.close()
        if self._input_noise_std > 0.0:
            x = x.clone()
            if self.input_mode == "latent":
                x = x + torch.randn_like(x) * self._input_noise_std
            else:
                n_rgb = 6 if self._rgb_stack == "current_next" else 3
                n_rgb = min(n_rgb, x.shape[0])
                noise = torch.randn(n_rgb, x.shape[1], x.shape[2], dtype=x.dtype) * self._input_noise_std
                x[:n_rgb] = (x[:n_rgb] + noise).clamp(0.0, 1.0)
        return x, y_act, y_cell, y_ori


class PerGameTileCNN(nn.Module):
    def __init__(self, num_actions: int, in_ch: int = 3, hidden: int = 64):
        super().__init__()
        h = int(hidden)
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
        )
        flat = h * 4
        self.head_action = nn.Linear(flat, num_actions)
        self.head_cell = nn.Linear(flat, 9)
        self.head_orient = nn.Linear(flat, 4)

    def forward(self, x: torch.Tensor):
        h = self.f(x).flatten(1)
        return self.head_action(h), self.head_cell(h), self.head_orient(h)


class PerGameTileCNNFactoredAction(nn.Module):
    """Shared CNN trunk + four MP ``action_set`` component heads (CE = softmax) + cell + orient."""

    def __init__(
        self,
        *,
        n_move: int,
        n_turn: int,
        n_zap: int,
        n_clean: int,
        in_ch: int,
        hidden: int,
    ):
        super().__init__()
        h = int(hidden)
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, h, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
        )
        flat = h * 4
        self.head_move = nn.Linear(flat, int(n_move))
        self.head_turn = nn.Linear(flat, int(n_turn))
        self.head_zap = nn.Linear(flat, int(n_zap))
        self.head_clean = nn.Linear(flat, int(n_clean))
        self.head_cell = nn.Linear(flat, 9)
        self.head_orient = nn.Linear(flat, 4)

    def forward(self, x: torch.Tensor):
        h = self.f(x).flatten(1)
        return (
            self.head_move(h),
            self.head_turn(h),
            self.head_zap(h),
            self.head_clean(h),
            self.head_cell(h),
            self.head_orient(h),
        )


def verify_dual_frame_temporal_consistency(
    ds: PerGameTileNPZDataset,
    *,
    max_index_checks: int = 4096,
    max_patch_checks: int = 256,
) -> dict[str, float]:
    """Cheap checks for ``rgb_stack='current_next'``: distinct ``pixel_frames`` indices; patch MAE.

    Confirms ``l2p[t+1] > l2p[t]`` and that the two 3-channel halves (same crop anchor at ``t``)
    are not bitwise-identical on the sampled rows.
    """
    if ds.input_mode != "rgb" or ds.rgb_stack != "current_next" or len(ds) == 0:
        return {}
    n_ic = min(int(max_index_checks), len(ds))
    step_ic = max(1, len(ds) // n_ic)
    idxs_ic = list(range(0, len(ds), step_ic))[:n_ic]
    d_pix: list[int] = []
    same = 0
    for idx in idxs_ic:
        fi, t, s, *_ = ds._index[idx]
        with np.load(ds._paths[fi], allow_pickle=True) as d:
            coords = d["frame_coords"]
            t_len = int(coords.shape[0])
            l2p = ptc.get_latent_to_pixel(t_len)
            tt = min(int(t), len(l2p) - 1)
            ttp1 = min(int(t + 1), len(l2p) - 1)
            pi, pip = int(l2p[tt]), int(l2p[ttp1])
        d_pix.append(pip - pi)
        if pi == pip:
            same += 1
    arr = np.asarray(d_pix, dtype=np.int64)
    n_pc = min(int(max_patch_checks), len(ds))
    step_pc = max(1, len(ds) // n_pc)
    idxs_pc = list(range(0, len(ds), step_pc))[:n_pc]
    maes: list[float] = []
    for idx in idxs_pc:
        x, _, _, _ = ds[idx]
        maes.append(float((x[:3] - x[3:]).abs().mean()))
    mae_arr = np.asarray(maes, dtype=np.float64)
    n_ic_e = max(len(idxs_ic), 1)
    near_id = float(np.sum(mae_arr < 1e-5)) / float(max(mae_arr.size, 1))
    return {
        "index_rows_checked": float(len(idxs_ic)),
        "pixel_index_delta_min": float(arr.min()) if arr.size else -1.0,
        "pixel_index_delta_mean": float(arr.mean()) if arr.size else -1.0,
        "pixel_index_delta_max": float(arr.max()) if arr.size else -1.0,
        "frac_same_pixel_index": float(same) / float(n_ic_e),
        "patch_mae_curr_vs_next_mean": float(mae_arr.mean()) if mae_arr.size else -1.0,
        "patch_rows_checked": float(len(maes)),
        "frac_patch_near_identical_mae_lt_1e_5": near_id,
    }


def collect_npz_paths(val_dir: str, basename_contains: str | None = None) -> list[str]:
    paths = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    if basename_contains:
        paths = [p for p in paths if basename_contains in os.path.basename(p)]
    return paths
