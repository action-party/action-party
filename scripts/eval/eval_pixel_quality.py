import sys as _sys, os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

"""
Compute PSNR and LPIPS between GT and generated frames across eval runs.

Supports two data formats:
  - npz: eval_data/{game}_s{N}.npz with 'gen_frames' and 'gt_frames' arrays
    (legacy: only gen_frames in eval_data; use --companion_val_npz_dir pointing at
    val_npz/ with matching filenames and 'pixel_frames' for GT)
  - png: multigame_benchmark/{game}/sample_{N}/ with gen_{idx:02d}.png and gt_frame4_next_action.png

Default: one frame per clip (--frame_idx, default 1), matching prior ablation JSONs
(e.g. pixel_metrics_150_151_frame1.json). Use --average_all_frames for mean PSNR/LPIPS
over every aligned frame in each npz (full short-clip rollout).
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dirs",
        nargs="+",
        required=True,
        help="Entries as 'name:format:path[:frame_idx]' where format is 'npz' or 'png'. "
             "Optional per-entry frame_idx overrides --frame_idx.",
    )
    parser.add_argument("--frame_idx", type=int, default=1,
                        help="Frame index to compare for npz/png (ignored if --average_all_frames).")
    parser.add_argument(
        "--average_all_frames",
        action="store_true",
        help="npz only: for each clip, mean PSNR and mean LPIPS over all aligned gen/GT frames.",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--companion_val_npz_dir",
        type=str,
        default=None,
        help="npz: if eval NPZs lack gt_frames, load GT from this dir "
             "(same basename), using key pixel_frames or gt_frames.",
    )
    return parser.parse_args()


def load_lpips_model(device):
    import lpips
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model


def compute_lpips(model, img_a, img_b, device):
    """img_a, img_b: uint8 numpy arrays (H, W, 3). Returns scalar LPIPS."""
    a = torch.from_numpy(img_a).permute(2, 0, 1).float() / 127.5 - 1.0
    b = torch.from_numpy(img_b).permute(2, 0, 1).float() / 127.5 - 1.0
    with torch.no_grad():
        score = model(a.unsqueeze(0).to(device), b.unsqueeze(0).to(device))
    return score.item()


def _crop_match(gen_frame, gt_frame):
    if gen_frame.shape != gt_frame.shape:
        h = min(gen_frame.shape[0], gt_frame.shape[0])
        w = min(gen_frame.shape[1], gt_frame.shape[1])
        return gen_frame[:h, :w], gt_frame[:h, :w]
    return gen_frame, gt_frame


def _load_gt_video_frames(eval_npz_path, data, companion_val_dir):
    """GT frames [T,H,W,3] from eval npz or legacy sibling val_npz."""
    if "gt_frames" in data.files:
        return data["gt_frames"]
    if companion_val_dir:
        fname = os.path.basename(eval_npz_path)
        vpath = os.path.join(companion_val_dir, fname)
        if not os.path.isfile(vpath):
            return None
        vd = np.load(vpath, allow_pickle=True)
        if "pixel_frames" in vd.files:
            return vd["pixel_frames"]
        if "gt_frames" in vd.files:
            return vd["gt_frames"]
    return None


def gather_pairs_npz(eval_dir, frame_idx, companion_val_dir=None):
    """Yield (game, sample_idx, gen_frame, gt_frame) from npz files."""
    pattern = os.path.join(eval_dir, "*.npz")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = re.match(r"(.+)_s(\d+)\.npz$", fname)
        if not m:
            continue
        game, sample = m.group(1), int(m.group(2))
        data = np.load(path, allow_pickle=True)
        gen_frames = data["gen_frames"]
        gt_frames = _load_gt_video_frames(path, data, companion_val_dir)
        if gt_frames is None:
            print(
                f"  [skip] {fname}: no gt_frames in npz and no usable "
                f"companion val (dir={companion_val_dir!r})"
            )
            continue
        if frame_idx >= gt_frames.shape[0] or frame_idx >= gen_frames.shape[0]:
            print(f"  [skip] {fname}: frame_idx {frame_idx} out of range "
                  f"(gen={gen_frames.shape[0]}, gt={gt_frames.shape[0]})")
            continue
        yield game, sample, gen_frames[frame_idx], gt_frames[frame_idx]


def gather_clip_averages_npz(eval_dir, lpips_model, device, companion_val_dir=None):
    """Yield (game, sample_idx, mean_psnr, mean_lpips) averaged over all aligned frames."""
    pattern = os.path.join(eval_dir, "*.npz")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = re.match(r"(.+)_s(\d+)\.npz$", fname)
        if not m:
            continue
        game, sample = m.group(1), int(m.group(2))
        data = np.load(path, allow_pickle=True)
        gen_frames = data["gen_frames"]
        gt_frames = _load_gt_video_frames(path, data, companion_val_dir)
        if gt_frames is None:
            print(
                f"  [skip] {fname}: no gt_frames in npz and no usable "
                f"companion val (dir={companion_val_dir!r})"
            )
            continue
        t_max = min(gen_frames.shape[0], gt_frames.shape[0])
        if t_max <= 0:
            print(f"  [skip] {fname}: no frames")
            continue
        psnrs, lpipses = [], []
        for t in range(t_max):
            gf, gtf = _crop_match(gen_frames[t], gt_frames[t])
            psnrs.append(psnr(gtf, gf, data_range=255))
            lpipses.append(compute_lpips(lpips_model, gf, gtf, device))
        yield game, sample, float(np.mean(psnrs)), float(np.mean(lpipses))


def gather_pairs_png(eval_dir, frame_idx):
    """Yield (game, sample_idx, gen_frame, gt_frame) from PNG directories."""
    for game_dir in sorted(glob.glob(os.path.join(eval_dir, "*"))):
        if not os.path.isdir(game_dir):
            continue
        game = os.path.basename(game_dir)
        for sample_dir in sorted(glob.glob(os.path.join(game_dir, "sample_*"))):
            sample_idx = int(os.path.basename(sample_dir).split("_")[1])
            gen_path = os.path.join(sample_dir, f"gen_{frame_idx:02d}.png")
            gt_path = os.path.join(sample_dir, "gt_frame4_next_action.png")
            if not os.path.exists(gen_path) or not os.path.exists(gt_path):
                continue
            gen_frame = np.array(Image.open(gen_path).convert("RGB"))
            gt_frame = np.array(Image.open(gt_path).convert("RGB"))
            yield game, sample_idx, gen_frame, gt_frame


def evaluate_model(
    name,
    fmt,
    eval_dir,
    frame_idx,
    lpips_model,
    device,
    average_all_frames=False,
    companion_val_dir=None,
):
    """Compute metrics for one model. Returns per-game results dict."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}  (format={fmt}, dir={eval_dir})")
    print(f"{'='*60}")

    game_results = defaultdict(list)
    count = 0

    if average_all_frames:
        if fmt != "npz":
            raise ValueError("--average_all_frames is only supported for format=npz")
        print("  Mode: mean PSNR / mean LPIPS over all aligned frames per clip")
        for game, sample, p, l in gather_clip_averages_npz(
            eval_dir, lpips_model, device, companion_val_dir=companion_val_dir
        ):
            game_results[game].append({"sample": sample, "psnr": p, "lpips": l})
            count += 1
    elif fmt == "npz":
        pairs = gather_pairs_npz(eval_dir, frame_idx, companion_val_dir=companion_val_dir)
        for game, sample, gen_frame, gt_frame in pairs:
            gen_frame, gt_frame = _crop_match(gen_frame, gt_frame)
            p = psnr(gt_frame, gen_frame, data_range=255)
            l = compute_lpips(lpips_model, gen_frame, gt_frame, device)
            game_results[game].append({"sample": sample, "psnr": p, "lpips": l})
            count += 1
    elif fmt == "png":
        pairs = gather_pairs_png(eval_dir, frame_idx)
        for game, sample, gen_frame, gt_frame in pairs:
            gen_frame, gt_frame = _crop_match(gen_frame, gt_frame)
            p = psnr(gt_frame, gen_frame, data_range=255)
            l = compute_lpips(lpips_model, gen_frame, gt_frame, device)
            game_results[game].append({"sample": sample, "psnr": p, "lpips": l})
            count += 1
    else:
        raise ValueError(f"Unknown format: {fmt}")

    print(f"  Processed {count} samples across {len(game_results)} games")
    return dict(game_results)


def print_summary(all_results):
    """Print a formatted summary table."""
    print(f"\n{'='*80}")
    print(f"{'PIXEL QUALITY SUMMARY':^80}")
    print(f"{'='*80}")

    header = f"{'Model':<20} {'Games':>6} {'Samples':>8} {'PSNR (mean)':>12} {'LPIPS (mean)':>13}"
    print(header)
    print("-" * 80)

    for model_name, game_results in all_results.items():
        all_psnr = []
        all_lpips = []
        for game, samples in game_results.items():
            for s in samples:
                all_psnr.append(s["psnr"])
                all_lpips.append(s["lpips"])
        n_games = len(game_results)
        n_samples = len(all_psnr)
        mean_p = np.mean(all_psnr) if all_psnr else 0
        mean_l = np.mean(all_lpips) if all_lpips else 0
        print(f"{model_name:<20} {n_games:>6} {n_samples:>8} {mean_p:>12.2f} {mean_l:>13.4f}")

    print()

    # Per-game breakdown for common games
    all_games = set()
    for game_results in all_results.values():
        all_games.update(game_results.keys())
    common_games = sorted(all_games)

    if common_games:
        print(f"\n{'PER-GAME BREAKDOWN':^80}")
        print(f"{'='*80}")
        model_names = list(all_results.keys())
        col_w = 20
        header = f"{'Game':<35}"
        for mn in model_names:
            header += f" {mn:>{col_w}}"
        print(header)
        print("-" * (35 + col_w * len(model_names) + len(model_names)))

        print("\n  PSNR:")
        for game in common_games:
            row = f"  {game:<33}"
            for mn in model_names:
                samples = all_results[mn].get(game, [])
                if samples:
                    avg = np.mean([s["psnr"] for s in samples])
                    row += f" {avg:>{col_w}.2f}"
                else:
                    row += f" {'---':>{col_w}}"
            print(row)

        print("\n  LPIPS:")
        for game in common_games:
            row = f"  {game:<33}"
            for mn in model_names:
                samples = all_results[mn].get(game, [])
                if samples:
                    avg = np.mean([s["lpips"] for s in samples])
                    row += f" {avg:>{col_w}.4f}"
                else:
                    row += f" {'---':>{col_w}}"
            print(row)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = load_lpips_model(device)

    all_results = {}
    for entry in args.eval_dirs:
        parts = entry.split(":")
        if len(parts) == 4:
            name, fmt, path, fidx = parts
            frame_idx = int(fidx)
        elif len(parts) == 3:
            name, fmt, path = parts
            frame_idx = args.frame_idx
        else:
            raise ValueError(f"Expected 'name:format:path[:frame_idx]', got: {entry}")
        companion = args.companion_val_npz_dir if fmt == "npz" else None
        game_results = evaluate_model(
            name,
            fmt,
            path,
            frame_idx,
            lpips_model,
            device,
            average_all_frames=args.average_all_frames,
            companion_val_dir=companion,
        )
        all_results[name] = game_results

    print_summary(all_results)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
