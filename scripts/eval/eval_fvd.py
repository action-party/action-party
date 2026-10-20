import sys as _sys, os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

"""
Compute FVD (Fréchet Video Distance) using the canonical I3D model.

Uses the same I3D checkpoint (Kinetics-400, Carreira & Zisserman 2017)
as the original FVD paper (Unterthiner et al., 2019) and StyleGAN-V.

Protocol:
  - 16 uniformly sampled frames per video
  - Resized to 224x224
  - Pixel values normalized to [-1, 1]
  - 400-d logit features from I3D
  - Fréchet distance between real and generated feature distributions

Usage:
    python eval_fvd.py \
        --eval_dirs "Ours:outputs_ckpts/eval/gen_139/eval_data" \
                    "Text-Action:outputs_ckpts/eval/gen_144/eval_data"
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm


I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_CACHE = os.path.expanduser("~/.cache/fvd/i3d_torchscript.pt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dirs", nargs="+", required=True,
                   help="Entries as 'name:path'")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--resolution", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


def get_i3d(device):
    """Load the canonical I3D TorchScript model."""
    if not os.path.exists(I3D_CACHE):
        os.makedirs(os.path.dirname(I3D_CACHE), exist_ok=True)
        print(f"  Downloading canonical I3D checkpoint...")
        import urllib.request
        urllib.request.urlretrieve(I3D_URL, I3D_CACHE)
    model = torch.jit.load(I3D_CACHE).to(device).eval()
    return model


def preprocess_video(video_np, num_frames, resolution):
    """Standard FVD preprocessing.

    video_np: (T, H, W, 3) uint8
    Returns: (3, num_frames, resolution, resolution) float32 in [-1, 1]
    """
    T = video_np.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    frames = video_np[indices]

    # (T, H, W, C) -> (C, T, H, W) float [0, 1]
    vid = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

    # Resize spatially to (resolution, resolution)
    # F.interpolate expects (B, C, D, H, W) for 3D
    vid = vid.unsqueeze(0)
    vid = F.interpolate(vid, size=(num_frames, resolution, resolution),
                        mode='trilinear', align_corners=False)
    vid = vid.squeeze(0)

    # Normalize to [-1, 1]
    vid = vid * 2.0 - 1.0
    return vid


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute the Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean_sq, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean_sq):
        covmean_sq = covmean_sq.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean_sq))


def load_videos(eval_dir):
    """Load all (gen, gt) video pairs from NPZ files."""
    gen_videos, gt_videos = [], []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.npz"))):
        data = np.load(path)
        if "gen_frames" not in data or "gt_frames" not in data:
            continue
        gen_videos.append(data["gen_frames"])
        gt_videos.append(data["gt_frames"])
    return gen_videos, gt_videos


@torch.no_grad()
def extract_features(model, videos, num_frames, resolution, device, batch_size):
    """Extract I3D logit features for a list of videos."""
    all_feats = []
    batch = []
    for i, vid in enumerate(videos):
        processed = preprocess_video(vid, num_frames, resolution)
        batch.append(processed)
        if len(batch) == batch_size or i == len(videos) - 1:
            x = torch.stack(batch).to(device)
            feats = model(x)
            all_feats.append(feats.cpu().numpy())
            batch = []
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(videos)}")
    return np.concatenate(all_feats, axis=0)


def compute_fvd(gen_feats, gt_feats):
    """Compute FVD from feature arrays."""
    mu_gen = gen_feats.mean(axis=0)
    mu_gt = gt_feats.mean(axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    sigma_gt = np.cov(gt_feats, rowvar=False)
    return frechet_distance(mu_gt, sigma_gt, mu_gen, sigma_gen)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Protocol: {args.num_frames} frames, {args.resolution}x{args.resolution}, "
          f"canonical I3D (400-d logits)")

    print("\nLoading canonical I3D...")
    model = get_i3d(device)

    gt_feats = None
    results = {}

    for entry in args.eval_dirs:
        name, eval_dir = entry.split(":", 1)

        print(f"\n{'='*60}")
        print(f"  {name}: {eval_dir}")
        print(f"{'='*60}")

        gen_videos, gt_videos = load_videos(eval_dir)
        n = len(gen_videos)
        if n == 0:
            print("  No videos found, skipping")
            continue
        print(f"  {n} video pairs ({gen_videos[0].shape[0]} frames, "
              f"{gen_videos[0].shape[1]}x{gen_videos[0].shape[2]})")

        print("  Extracting gen features...")
        gen_feats = extract_features(
            model, gen_videos, args.num_frames, args.resolution,
            device, args.batch_size)

        if gt_feats is None:
            print("  Extracting GT features (cached for all models)...")
            gt_feats = extract_features(
                model, gt_videos, args.num_frames, args.resolution,
                device, args.batch_size)

        fvd = compute_fvd(gen_feats, gt_feats)
        results[name] = {"fvd": fvd, "n_samples": n, "feat_dim": gen_feats.shape[1]}
        print(f"\n  FVD = {fvd:.2f}")

    print(f"\n{'='*60}")
    print(f"  FVD Summary (canonical I3D, Kinetics-400)")
    print(f"  {args.num_frames} frames @ {args.resolution}x{args.resolution}")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'FVD':>10} {'N':>6}")
    print(f"  {'-'*43}")
    for name, r in results.items():
        print(f"  {name:<25} {r['fvd']:>10.2f} {r['n_samples']:>6}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {args.output_json}")


if __name__ == "__main__":
    main()
