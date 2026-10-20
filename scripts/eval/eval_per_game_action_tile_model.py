#!/usr/bin/env python3
"""
Evaluate per-game tile CNN on eval_data NPZ with **predicted** grid centering.

Loads ``gen_frames`` (``input_mode=rgb``) or ``gen_video_latents`` T,C,H,W
(``input_mode=latent``, must match checkpoint). Uses ``gt_coords`` when present for labeling; crop center
starts from ``gen_coords`` at t=0 if available (generated positions), else
``gt_coords``. With ``--use_gt_center``, every crop uses GT coords (no drift).

Checkpoints trained with ``rgb_stack=current_next`` need **two** RGB 3×3 blocks (same crop
anchor on consecutive pixel frames); this script matches that layout from ``gen_frames``.

With ``--center_presence_gate_ckpt`` (or the key saved in the tile checkpoint), **action**
accuracy can be gated with ``CellPresenceMLP`` on the **center RGB cell** of each block
(``--action_presence_count_mode``): default counts only when the **current** block's
center passes; ``either_frame_center`` skips a step only when **both** the current and
next block centers are below ``--center_presence_prob_min`` (RGB only). Cell/orient
metrics are always ungated.

Usage:
  python scripts/eval/eval_per_game_action_tile_model.py \\
    --checkpoint /tmp/model.pt \\
    --gen_dir path/to/eval_data \\
    --val_dir path/to/val_npz \\
    --output /tmp/metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Sequence

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

import per_game_tile_crops as ptc
from per_game_tile_model import PerGameTileCNN, aux_meta_planes_torch, fi_index_for_train_paths
from train_cell_player_presence_mlp import CellPresenceMLP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Checkpoint from train_per_game_action_tile_model.py")
    p.add_argument("--gen_dir", required=True, help="eval_data directory with *.npz")
    p.add_argument("--val_dir", default=None, help="val_npz dir for GT orientations (same basename)")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.add_argument("--use_gt_center", action="store_true", help="Ablate: always crop from GT coords (no drift)")
    p.add_argument("--skip", nargs="*", default=["allelopathic_harvest"])
    p.add_argument(
        "--center_presence_gate_ckpt",
        default=None,
        help="CellPresenceMLP .pt: count **action** accuracy only when center-cell P(present) "
        "passes (RGB gen only; omit for legacy behavior). Defaults to tile ckpt key when set.",
    )
    p.add_argument(
        "--center_presence_prob_min",
        type=float,
        default=None,
        help="Threshold for --center_presence_gate_ckpt (default: tile ckpt value or 0.5).",
    )
    p.add_argument(
        "--action_presence_count_mode",
        type=str,
        choices=("current_center", "either_frame_center"),
        default="current_center",
        help="When a presence gate is active: count an action step only if the current "
        "crop center passes (default), or if **either** the current or next RGB block center "
        "passes (skip only when both are absent — use with current_next checkpoints).",
    )
    return p.parse_args()


def _load_ckpt(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@torch.no_grad()
def eval_one_npz_impl(
    path: str,
    m: PerGameTileCNN,
    dev: torch.device,
    cell_px: int,
    image_size: int,
    num_actions: int,
    val_dir: str | None,
    use_gt_center: bool,
    skip: list[str],
    pad_mode: str = "constant",
    pad_fill: tuple[float, ...] | float | None = None,
    action_orig_to_compact: dict[int, int] | None = None,
    aux_meta_channels: bool = False,
    train_npz_paths: Sequence[str] | None = None,
    input_mode: str = "rgb",
    rgb_stack: str = "current",
    pres_m: CellPresenceMLP | None = None,
    pres_prob_min: float = 0.5,
    action_presence_count_mode: str = "current_center",
) -> dict | None:
    bn = os.path.basename(path)
    if any(s in bn for s in skip):
        return None

    ed = np.load(path, allow_pickle=True)
    try:
        if "actions" not in ed.files:
            return None
        input_mode_s = str(input_mode).lower()
        if input_mode_s == "latent":
            if "gen_video_latents" not in ed.files:
                return None
            gvl = ed["gen_video_latents"]
        else:
            if "gen_frames" not in ed.files:
                return None
            gen_frames = ed["gen_frames"]
        if "gt_coords" in ed.files:
            label_c = ed["gt_coords"].copy().astype(np.float64)
        elif "gen_coords" in ed.files:
            label_c = ed["gen_coords"].copy().astype(np.float64)
        else:
            return None
        gen_c = ed["gen_coords"].copy().astype(np.float64) if "gen_coords" in ed.files else label_c.copy()
        actions = ed["actions"]
        num_sub = int(ed.get("num_subjects_real", actions.shape[0]))

        if input_mode_s != "latent":
            H = int(gen_frames.shape[1])
        else:
            H = int(image_size)
        image_size = H if image_size <= 0 else int(image_size)
        for arr in (label_c, gen_c):
            if arr.max() <= 1.0:
                arr *= float(image_size)

        n_latent = int(label_c.shape[0])
        n_act = int(actions.shape[1])
        max_step = min(n_act, n_latent - 1)
        l2p = ptc.get_latent_to_pixel(n_latent)

        gt_orient = np.zeros((n_latent, num_sub), dtype=np.int64)
        if val_dir:
            vp = os.path.join(val_dir, bn)
            if os.path.isfile(vp):
                vd = np.load(vp, allow_pickle=True)
                try:
                    if "orientations" in vd.files:
                        oo = vd["orientations"].astype(np.int64)
                        nt = min(oo.shape[0], n_latent, gt_orient.shape[0])
                        gt_orient[:nt] = oo[:nt, :num_sub]
                finally:
                    vd.close()

        act_ok = act_tot = cell_ok = cell_tot = ori_ok = ori_tot = 0
        pred_gi = np.zeros(num_sub, dtype=np.int64)
        pred_gj = np.zeros(num_sub, dtype=np.int64)

        for s in range(num_sub):
            cx0, cy0 = float(gen_c[0, s, 0]), float(gen_c[0, s, 1])
            pred_gi[s], pred_gj[s] = ptc.coords_to_cell_indices(cx0, cy0, cell_px, image_size)

        rgb_stack_s = str(rgb_stack or "current").lower()
        for t in range(max_step):
            frame_cur = frame_next = None
            if input_mode_s != "latent":
                n_pf = int(len(gen_frames))
                if n_pf <= 0:
                    break
                ti = min(int(t), len(l2p) - 1)
                pix_i = int(l2p[ti])
                if pix_i >= n_pf:
                    break
                frame_cur = gen_frames[pix_i]
                if rgb_stack_s == "current_next":
                    tip = min(int(t + 1), len(l2p) - 1)
                    pix_ip = int(l2p[tip])
                    if pix_ip >= n_pf:
                        pix_ip = n_pf - 1
                    frame_next = gen_frames[pix_ip]
            for s in range(num_sub):
                if int(actions[s, t]) < 0:
                    continue
                cx_gt, cy_gt = float(label_c[t, s, 0]), float(label_c[t, s, 1])
                if use_gt_center:
                    cx, cy = cx_gt, cy_gt
                else:
                    cx = (float(pred_gi[s]) + 0.5) * cell_px
                    cy = (float(pred_gj[s]) + 0.5) * cell_px
                if input_mode_s == "latent":
                    t_lat = min(int(t), int(gvl.shape[0]) - 1)
                    lat_chw = np.ascontiguousarray(gvl[t_lat], dtype=np.float32)
                    lat_fill = 0.0 if pad_mode == "constant" else None
                    patch = ptc.extract_3x3_cell_patch_latent_chw(
                        lat_chw,
                        cx,
                        cy,
                        cell_px,
                        image_size,
                        pad_cells=1,
                        pad_mode=pad_mode,
                        pad_fill=lat_fill,
                    )
                    x = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0).to(dev)
                else:
                    assert frame_cur is not None
                    patch0 = ptc.extract_3x3_cell_patch(
                        frame_cur,
                        cx,
                        cy,
                        cell_px,
                        image_size,
                        pad_cells=1,
                        pad_mode=pad_mode,
                        pad_fill=pad_fill,
                    )
                    t0 = torch.from_numpy(patch0).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
                    if rgb_stack_s == "current_next" and frame_next is not None:
                        patch1 = ptc.extract_3x3_cell_patch(
                            frame_next,
                            cx,
                            cy,
                            cell_px,
                            image_size,
                            pad_cells=1,
                            pad_mode=pad_mode,
                            pad_fill=pad_fill,
                        )
                        t1 = torch.from_numpy(patch1).permute(2, 0, 1).float().unsqueeze(0).to(dev) / 255.0
                        x = torch.cat([t0, t1], dim=1)
                    else:
                        x = t0
                gated_action = True
                if pres_m is not None and input_mode_s != "latent":
                    cpx = cell_px
                    xc = x[:, :3, cpx : 2 * cpx, cpx : 2 * cpx]
                    p_cur = float(torch.sigmoid(pres_m(xc.reshape(1, -1))).item())
                    if (
                        str(action_presence_count_mode) == "either_frame_center"
                        and rgb_stack_s == "current_next"
                        and int(x.shape[1]) >= 6
                    ):
                        xn = x[:, 3:6, cpx : 2 * cpx, cpx : 2 * cpx]
                        p_next = float(torch.sigmoid(pres_m(xn.reshape(1, -1))).item())
                        gated_action = (p_cur >= float(pres_prob_min)) or (p_next >= float(pres_prob_min))
                    else:
                        gated_action = p_cur >= float(pres_prob_min)
                if aux_meta_channels:
                    fi, n_files = fi_index_for_train_paths(bn, train_npz_paths)
                    gi_aux, gj_aux = ptc.coords_to_cell_indices(cx, cy, cell_px, image_size)
                    nh = image_size // cell_px
                    nw = nh
                    ph, pw = int(x.shape[2]), int(x.shape[3])
                    pl = aux_meta_planes_torch(
                        fi,
                        n_files,
                        t,
                        n_latent,
                        s,
                        num_sub,
                        gi_aux,
                        gj_aux,
                        nh,
                        nw,
                        (ph, pw),
                        dtype=x.dtype,
                        device=x.device,
                    ).unsqueeze(0)
                    x = torch.cat([x, pl], dim=1)
                la, lc, lo = m(x)
                pa = int(la.argmax(dim=-1).item())
                pc = int(lc.argmax(dim=-1).item())
                po = int(lo.argmax(dim=-1).item())
                gt_a = int(actions[s, t])
                if action_orig_to_compact is not None:
                    gt_i = action_orig_to_compact.get(gt_a)
                    if gt_i is None:
                        continue
                    if gated_action:
                        act_tot += 1
                        if pa == gt_i:
                            act_ok += 1
                else:
                    if 0 <= gt_a < num_actions:
                        if gated_action:
                            act_tot += 1
                            if pa == gt_a:
                                act_ok += 1

                gi_t, gj_t = ptc.coords_to_cell_indices(cx_gt, cy_gt, cell_px, image_size)
                cx_n, cy_n = float(label_c[t + 1, s, 0]), float(label_c[t + 1, s, 1])
                gi_n, gj_n = ptc.coords_to_cell_indices(cx_n, cy_n, cell_px, image_size)
                dgi = max(-1, min(1, gi_n - gi_t))
                dgj = max(-1, min(1, gj_n - gj_t))
                y_cell = ptc.cell_delta_to_class(dgi, dgj)
                cell_tot += 1
                if pc == y_cell:
                    cell_ok += 1
                dgi_p, dgj_p = ptc.class_to_cell_delta(pc)
                if not use_gt_center:
                    pred_gi[s] = pred_gi[s] + dgi_p
                    pred_gj[s] = pred_gj[s] + dgj_p

                yo = int(gt_orient[t + 1, s]) % 4
                ori_tot += 1
                if po == yo:
                    ori_ok += 1

        def pct(a, b):
            return round(100.0 * a / b, 4) if b else 0.0

        out = {
            "basename": bn,
            "steps": int(max_step),
            "subjects": int(num_sub),
            "use_gt_center": use_gt_center,
            "action_acc_pct": pct(act_ok, act_tot),
            "cell_offset_acc_pct": pct(cell_ok, cell_tot),
            "orient_acc_pct": pct(ori_ok, ori_tot),
            "counts": {"action": [act_ok, act_tot], "cell": [cell_ok, cell_tot], "orient": [ori_ok, ori_tot]},
        }
        if pres_m is not None:
            out["center_presence_action_gated"] = True
            out["center_presence_prob_min"] = float(pres_prob_min)
            out["action_presence_count_mode"] = str(action_presence_count_mode)
        out["rgb_stack"] = str(rgb_stack_s)
        return out
    finally:
        ed.close()


def main():
    args = parse_args()
    ck = _load_ckpt(args.checkpoint)
    cell_px = int(ck.get("cell_px", 32))
    image_size = int(ck.get("image_size", 512))
    num_actions = int(ck.get("num_actions", 25))
    aids = ck.get("action_orig_ids")
    action_orig_to_compact = None
    if aids is not None and len(aids) > 0:
        action_orig_to_compact = {int(o): i for i, o in enumerate(aids)}
    hidden = int(ck.get("hidden", 64))
    aux_meta = bool(ck.get("aux_meta_channels", False))
    input_mode = str(ck.get("input_mode", "rgb")).lower()
    rgb_stack = str(ck.get("rgb_stack", "current"))
    in_ch = int(ck.get("in_ch", 8 if aux_meta else 3))
    pad_mode = ck.get("pad_mode")
    if pad_mode is None:
        pad_mode = "constant"
    else:
        pad_mode = str(pad_mode)
    rf = ck.get("pad_fill")
    if pad_mode == "constant":
        pad_fill = tuple(float(x) for x in rf) if rf is not None else None
    else:
        pad_fill = None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = PerGameTileCNN(num_actions=num_actions, hidden=hidden, in_ch=in_ch).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()

    pres_path = getattr(args, "center_presence_gate_ckpt", None) or ck.get("center_presence_gate_ckpt")
    p_thr = getattr(args, "center_presence_prob_min", None)
    if p_thr is None:
        pc = ck.get("center_presence_prob_min")
        p_thr = float(pc) if pc is not None else 0.5
    pres_m: CellPresenceMLP | None = None
    if pres_path and str(pres_path).strip():
        if input_mode != "rgb":
            raise SystemExit("center presence gate requires RGB tile checkpoint (checkpoint input_mode rgb)")
        pp = os.path.abspath(str(pres_path).strip())
        pck = _load_ckpt(pp)
        if int(pck["cell_px"]) != cell_px or int(pck["image_size"]) != image_size:
            raise SystemExit(
                f"presence ckpt cell_px/image_size mismatch vs tile ckpt "
                f"({int(pck['cell_px'])},{int(pck['image_size'])}) vs ({cell_px},{image_size})"
            )
        pres_pad = str(pck.get("pad_mode", "constant"))
        if pres_pad != pad_mode:
            raise SystemExit(f"presence pad_mode {pres_pad!r} != tile pad_mode {pad_mode!r}")
        rf2 = pck.get("pad_fill")
        pres_fill = tuple(float(x) for x in rf2) if pres_pad == "constant" and rf2 is not None else None
        if pad_mode == "constant" and isinstance(pad_fill, tuple) and isinstance(pres_fill, tuple):
            if len(pad_fill) == len(pres_fill) and len(pad_fill) > 0:
                if max(abs(pad_fill[i] - pres_fill[i]) for i in range(len(pad_fill))) > 1e-6:
                    raise SystemExit(f"presence pad_fill {pres_fill!r} != tile pad_fill {pad_fill!r}")
        pres_m = CellPresenceMLP(flat_dim=int(pck["flat_dim"]), hidden=int(pck["hidden"])).to(dev)
        pres_m.load_state_dict(pck["model"], strict=True)
        pres_m.eval()
        print(f"center_presence_action_gate: ckpt={pp!r}  P(present)>={float(p_thr)}", flush=True)

    if str(args.action_presence_count_mode) == "either_frame_center" and str(rgb_stack) != "current_next":
        print(
            "WARNING: --action_presence_count_mode either_frame_center is intended for "
            "rgb_stack=current_next; falling back to current-center behavior when the second RGB block is absent.",
            flush=True,
        )

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.gen_dir, "*.npz"))):
        r = eval_one_npz_impl(
            fp,
            m,
            dev,
            cell_px,
            image_size,
            num_actions,
            args.val_dir,
            args.use_gt_center,
            list(args.skip or []),
            pad_mode=pad_mode,
            pad_fill=pad_fill,
            action_orig_to_compact=action_orig_to_compact,
            aux_meta_channels=aux_meta,
            train_npz_paths=ck.get("aux_fi_npz_paths") or ck.get("npz_paths"),
            input_mode=input_mode,
            rgb_stack=rgb_stack,
            pres_m=pres_m,
            pres_prob_min=float(p_thr),
            action_presence_count_mode=str(args.action_presence_count_mode),
        )
        if r:
            rows.append(r)

    def agg(key):
        c = sum(x["counts"][key][0] for x in rows)
        t = sum(x["counts"][key][1] for x in rows)
        return c, t

    a0, a1 = agg("action")
    c0, c1 = agg("cell")
    o0, o1 = agg("orient")

    def pct(a, b):
        return round(100.0 * a / b, 4) if b else 0.0

    out = {
        "metric": "per_game_action_tile",
        "checkpoint": os.path.abspath(args.checkpoint),
        "gen_dir": os.path.abspath(args.gen_dir),
        "val_dir": os.path.abspath(args.val_dir) if args.val_dir else None,
        "use_gt_center": args.use_gt_center,
        "pad_mode": pad_mode,
        "pad_fill": (
            list(pad_fill)
            if pad_fill is not None
            else (list(ptc.DEFAULT_PAD_RGB) if pad_mode == "constant" else None)
        ),
        "num_files": len(rows),
        "overall_action_acc_pct": pct(a0, a1),
        "overall_cell_acc_pct": pct(c0, c1),
        "overall_orient_acc_pct": pct(o0, o1),
        "per_file": rows,
        "rgb_stack": str(rgb_stack),
        "action_presence_count_mode": str(args.action_presence_count_mode),
    }
    if pres_m is not None:
        out["center_presence_action_gated"] = True
        out["center_presence_gate_ckpt"] = os.path.abspath(str(pres_path).strip())
        out["center_presence_prob_min"] = float(p_thr)
    outp = args.output or os.path.join(args.gen_dir, "per_game_action_tile_metrics.json")
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("metric", "num_files", "overall_action_acc_pct", "overall_cell_acc_pct", "overall_orient_acc_pct")}, indent=2))
    print(f"Saved {outp}")


if __name__ == "__main__":
    main()
