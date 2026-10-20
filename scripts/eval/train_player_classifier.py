#!/usr/bin/env python3
import sys as _sys, os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

"""
Train a player tile classifier from GT validation data.

Uses sklearn MLPClassifier pipelines (see ``--mlp_hidden`` / ``--mlp_alpha``).
For scan-based **pixel_inferred_action** metrics, ``eval_scan_inferred_action.py``
supports ``--hungarian`` (Hungarian detection-to-subject matching in
``visualize_action_video.evaluate_step_correctness``).

Use ``--robust_rollout`` when you care about **long / noisy generated** rollouts:
wider MLP, longer training with early stopping, extra GT negatives, light
**noise/brightness jitter** on positive tiles, optional **hard negatives mined
from generated** frames (false player detections away from GT coords).

For a CNN baseline on the same cached tiles, see ``torch_conv_tile_classifier.py``.

Collects positive tiles (players at known positions with known orientations)
and negative tiles (random non-player regions) from GT pixel frames, then
trains two sklearn MLP classifiers:
  - binary:  player vs not_player
  - 5class:  not_player / player_N / player_E / player_S / player_W

Usage:
    python train_player_classifier.py \
        --val_dir outputs_ckpts/val/val_npz \
        --output  outputs_ckpts/val/player_tile_classifier.pkl

    # For 256 resolution:
    python train_player_classifier.py \
        --val_dir outputs_ckpts/val_256/val_npz \
        --output  outputs_ckpts/val_256/player_tile_classifier.pkl \
        --tile_size 16 \
        --image_size 256

    # Long / noisy rollouts: preset + gen augment + mine false positives on gen
    python train_player_classifier.py \
        --val_dir outputs_ckpts/val/val_npz \
        --output outputs_ckpts/val/player_tile_classifier.pkl \
        --robust_rollout \
        --augment_gen_dirs path/to/long_run/eval_data \
        --mine_gen_eval_dirs path/to/long_run/eval_data
"""
import argparse, os, glob, numpy as np, joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

LATENT_TO_PIXEL_5 = [0, 1, 5, 9, 13]
LATENT_TO_PIXEL_9 = [0, 1, 5, 9, 13, 17, 21, 25, 29]


def get_latent_to_pixel(num_latent):
    if num_latent <= 5:
        return LATENT_TO_PIXEL_5[:num_latent]
    return LATENT_TO_PIXEL_9[:num_latent]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val_dir", required=True,
                   help="Directory with GT validation .npz files")
    p.add_argument("--output", required=True,
                   help="Output path for the trained classifier .pkl")
    p.add_argument("--tile_size", type=int, default=32,
                   help="Size of tile to extract (32 for 512-res, 16 for 256-res)")
    p.add_argument("--image_size", type=int, default=512,
                   help="Pixel resolution of frames (512 or 256)")
    p.add_argument("--augment_gen_dirs", nargs="*", default=[],
                   help="Optional eval_data dirs to augment with generated tiles")
    p.add_argument("--cache", default="/tmp/player_classifier_data.npz",
                   help="Path to cache collected tile data")
    p.add_argument("--skip", nargs="*", default=[],
                   help="Game name substrings to skip during collection")
    p.add_argument(
        "--neg_per_frame_mult",
        type=int,
        default=1,
        help="How many random negative tiles to draw per subject-slot per GT frame (>=1)",
    )
    p.add_argument("--mine_hard_negatives", type=str, default=None,
                   help="Path to existing classifier .pkl to mine hard negatives from GT frames")
    p.add_argument(
        "--mine_gen_eval_dirs",
        nargs="*",
        default=[],
        help="eval_data dirs: mine tiles on **generated** frames where the classifier "
             "fires but the center is far from GT player coords (rollout false positives)",
    )
    p.add_argument("--hard_neg_rounds", type=int, default=2,
                   help="Number of hard negative mining + retrain rounds")
    p.add_argument(
        "--robust_rollout",
        action="store_true",
        help="Preset for noisy long gen eval: wider MLP, more GT negs, tile jitter, "
             "extra mining rounds, early stopping (unless overridden)",
    )
    p.add_argument(
        "--tile_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise in normalized [0,1] tile space for duplicated positives; "
             "e.g. 0.012. With --robust_rollout default 0.012 if left at 0",
    )
    p.add_argument(
        "--tile_brightness_jitter",
        type=float,
        default=0.0,
        help="Per-tile multiplicative brightness ~LogNormal(sigma); try 0.06–0.12. "
             "With --robust_rollout default 0.08 if 0",
    )
    p.add_argument(
        "--mlp_hidden",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes for sklearn MLPClassifier (e.g. 256,128,64)",
    )
    p.add_argument(
        "--mlp_alpha",
        type=float,
        default=1e-4,
        help="L2 penalty on MLP weights (sklearn MLPClassifier alpha)",
    )
    p.add_argument(
        "--mlp_max_iter",
        type=int,
        default=500,
        help="MLP max_iter (800 with --robust_rollout if left at default 500)",
    )
    p.add_argument(
        "--mlp_early_stopping",
        action="store_true",
        help="Enable MLP early stopping on a held-out fraction (good for large tile sets)",
    )
    p.add_argument(
        "--mlp_no_early_stopping",
        action="store_true",
        help="Force-disable early stopping even with --robust_rollout",
    )
    p.add_argument(
        "--eval_val_dir",
        type=str,
        default=None,
        help="After training: evaluate 5-class and binary accuracy on this GT val_npz directory",
    )
    return p.parse_args()


def extract_tile(frame, cx, cy, tile_size):
    h, w = frame.shape[:2]
    half = tile_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    if x0 < 0 or y0 < 0 or x0 + tile_size > w or y0 + tile_size > h:
        return None
    return frame[y0:y0 + tile_size, x0:x0 + tile_size].copy()


def collect_tiles_from_gt(val_dir, tile_size, image_size, skip_substrings, neg_per_frame_mult=1):
    """Collect positive (player) and negative (non-player) tiles from GT frames."""
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    pos_tiles, pos_labels, neg_tiles = [], [], []

    for f in files:
        bn = os.path.basename(f).replace(".npz", "")
        if any(s in bn for s in skip_substrings):
            continue

        d = np.load(f, allow_pickle=True)
        pf = d["pixel_frames"]
        coords = d["frame_coords"]
        orients = d["orientations"]
        num_sub = int(d["num_subjects"])

        l2p = get_latent_to_pixel(len(coords))
        for t in range(len(coords)):
            pix_idx = l2p[t]
            if pix_idx >= len(pf):
                continue
            frame = pf[pix_idx]
            player_positions = set()

            for s in range(num_sub):
                cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                if cx == 0 and cy == 0:
                    continue
                tile = extract_tile(frame, cx, cy, tile_size)
                if tile is not None and tile.shape == (tile_size, tile_size, 3):
                    pos_tiles.append(tile)
                    pos_labels.append(int(orients[t, s]))
                    player_positions.add((int(round(cx)), int(round(cy))))

            # Random negative tiles from same frame
            n_neg_draws = max(1, int(num_sub) * max(1, int(neg_per_frame_mult)))
            for _ in range(n_neg_draws):
                rx = np.random.randint(tile_size // 2, frame.shape[1] - tile_size // 2)
                ry = np.random.randint(tile_size // 2, frame.shape[0] - tile_size // 2)
                too_close = any(abs(rx - px) < tile_size and abs(ry - py) < tile_size
                                for px, py in player_positions)
                if not too_close:
                    tile = extract_tile(frame, rx, ry, tile_size)
                    if tile is not None and tile.shape == (tile_size, tile_size, 3) and tile.max() > 5:
                        neg_tiles.append(tile)

    return np.array(pos_tiles), np.array(pos_labels), np.array(neg_tiles)


def collect_tiles_from_gen(gen_eval_dir, val_dir, tile_size, image_size, skip_substrings):
    """Collect tiles from generated frames at GT positions (augmentation)."""
    files = sorted(glob.glob(os.path.join(gen_eval_dir, "*.npz")))
    pos_tiles, pos_labels = [], []

    for ef in files:
        bn = os.path.basename(ef).replace(".npz", "")
        if any(s in bn for s in skip_substrings):
            continue

        val_path = os.path.join(val_dir, os.path.basename(ef))
        if not os.path.exists(val_path):
            continue

        ed = np.load(ef)
        gen_frames = ed["gen_frames"]
        gt_c = ed["gt_coords"].copy()
        if gt_c.max() <= 1.0:
            gt_c *= image_size
        num_sub = int(ed["num_subjects_real"])

        vd = np.load(val_path, allow_pickle=True)
        gt_orient = vd["orientations"].astype(int)

        l2p = get_latent_to_pixel(len(gt_c))
        for t in range(len(gt_c)):
            pix_idx = l2p[t]
            if pix_idx >= len(gen_frames):
                break
            frame = gen_frames[pix_idx]
            for s in range(num_sub):
                cx, cy = float(gt_c[t, s, 0]), float(gt_c[t, s, 1])
                tile = extract_tile(frame, cx, cy, tile_size)
                if tile is not None and tile.shape == (tile_size, tile_size, 3) and tile.max() > 10:
                    pos_tiles.append(tile)
                    pos_labels.append(int(gt_orient[t, s]))

    if pos_tiles:
        return np.array(pos_tiles), np.array(pos_labels)
    return np.zeros((0, tile_size, tile_size, 3), dtype=np.uint8), np.zeros(0, dtype=np.int64)


def mine_hard_negatives(classifier_path, val_dir, tile_size, image_size,
                        skip_substrings, step=12, min_dist=48):
    """Scan GT frames, find tiles the classifier calls 'player' but are far
    from any real player position. These are environmental sprite false positives."""
    models = joblib.load(classifier_path)
    clf_bin = models["binary"]
    files = sorted(glob.glob(os.path.join(val_dir, "*.npz")))
    hard_negs = []
    half = tile_size // 2

    for f in files:
        bn = os.path.basename(f).replace(".npz", "")
        if any(s in bn for s in skip_substrings):
            continue
        d = np.load(f, allow_pickle=True)
        pf = d["pixel_frames"]
        coords = d["frame_coords"]
        num_sub = int(d["num_subjects"])
        l2p = get_latent_to_pixel(len(coords))

        for t in range(len(coords)):
            pix_idx = l2p[t]
            if pix_idx >= len(pf):
                continue
            frame = pf[pix_idx]
            player_pos = []
            for s in range(num_sub):
                cx, cy = float(coords[t, s, 0]), float(coords[t, s, 1])
                if cx > 0 or cy > 0:
                    player_pos.append((cx, cy))

            h, w = frame.shape[:2]
            for cy in range(half, h - half, step):
                for cx in range(half, w - half, step):
                    too_close = any(
                        abs(cx - px) < min_dist and abs(cy - py) < min_dist
                        for px, py in player_pos)
                    if too_close:
                        continue
                    tile = frame[cy - half:cy + half, cx - half:cx + half]
                    if tile.shape != (tile_size, tile_size, 3) or tile.max() <= 5:
                        continue
                    feat = tile.reshape(1, -1).astype(np.float32) / 255.0
                    if clf_bin.predict(feat)[0] == 1:
                        hard_negs.append(tile.copy())

    return np.array(hard_negs) if hard_negs else np.zeros((0, tile_size, tile_size, 3), dtype=np.uint8)


def mine_hard_negatives_from_generated(
    classifier_path,
    gen_eval_dirs,
    val_dir,
    tile_size,
    image_size,
    skip_substrings,
    step=16,
    min_dist=48,
):
    """Tiles on **generated** frames where the binary classifier fires but the
    center is far from any GT player — typical VAE / long-horizon false positives."""
    models = joblib.load(classifier_path)
    clf_bin = models["binary"]
    half = tile_size // 2
    hard_negs = []

    for gen_eval_dir in gen_eval_dirs:
        if not gen_eval_dir or not os.path.isdir(gen_eval_dir):
            continue
        for ef in sorted(glob.glob(os.path.join(gen_eval_dir, "*.npz"))):
            bn = os.path.basename(ef).replace(".npz", "")
            if any(s in bn for s in skip_substrings):
                continue
            val_path = os.path.join(val_dir, os.path.basename(ef))
            if not os.path.isfile(val_path):
                continue
            ed = np.load(ef, allow_pickle=True)
            if "gen_frames" not in ed.files or "gt_coords" not in ed.files:
                continue
            gen_frames = ed["gen_frames"]
            gt_c = ed["gt_coords"].copy()
            if gt_c.max() <= 1.0:
                gt_c = gt_c * float(image_size)
            num_sub = int(ed.get("num_subjects_real", 1))
            n_latent = len(gt_c)
            l2p = get_latent_to_pixel(n_latent)

            for t in range(n_latent):
                pix_idx = l2p[t]
                if pix_idx >= len(gen_frames):
                    break
                frame = gen_frames[pix_idx]
                player_pos = []
                for s in range(num_sub):
                    cx, cy = float(gt_c[t, s, 0]), float(gt_c[t, s, 1])
                    if cx > 1 or cy > 1:
                        player_pos.append((cx, cy))
                if not player_pos:
                    continue
                h, w = frame.shape[:2]
                for cy in range(half, h - half, step):
                    for cx in range(half, w - half, step):
                        too_close = any(
                            abs(cx - px) < min_dist and abs(cy - py) < min_dist
                            for px, py in player_pos)
                        if too_close:
                            continue
                        tile = frame[cy - half : cy + half, cx - half : cx + half]
                        if tile.shape != (tile_size, tile_size, 3) or tile.max() <= 5:
                            continue
                        feat = tile.reshape(1, -1).astype(np.float32) / 255.0
                        if clf_bin.predict(feat)[0] == 1:
                            hard_negs.append(tile.copy())

    return np.array(hard_negs) if hard_negs else np.zeros((0, tile_size, tile_size, 3), dtype=np.uint8)


def duplicate_jitter_positives(pos_tiles, pos_labels, noise_std, brightness_jitter, rng):
    """Append noisy / brightness-jittered copies of positives (same labels).

    *noise_std*: Gaussian std in **normalized** [0,1] space (same scale as classifier input).
    *brightness_jitter*: LogNormal sigma for multiplicative brightness per tile.
    """
    if noise_std <= 0 and brightness_jitter <= 0:
        return pos_tiles, pos_labels
    if len(pos_tiles) == 0:
        return pos_tiles, pos_labels
    x = pos_tiles.astype(np.float32) / 255.0
    if brightness_jitter > 0:
        sigma = float(brightness_jitter)
        scale = rng.lognormal(mean=0.0, sigma=sigma, size=(len(x), 1, 1, 1)).astype(np.float32)
        x = np.clip(x * scale, 0.0, 1.0)
    if noise_std > 0:
        x = np.clip(x + rng.normal(0.0, float(noise_std), size=x.shape), 0.0, 1.0)
    aug = (x * 255.0).astype(np.uint8)
    return np.concatenate([pos_tiles, aug], axis=0), np.concatenate([pos_labels, pos_labels], axis=0)


def train_classifiers(
    pos_tiles,
    pos_labels,
    neg_tiles,
    hidden_layer_sizes,
    mlp_alpha,
    max_iter=500,
    early_stopping=False,
):
    """Train binary and 5-class classifiers. Returns dict with both pipelines."""
    n_pos = len(pos_tiles)
    n_neg = len(neg_tiles)
    ts = pos_tiles.shape[1]

    # Flatten tiles to feature vectors
    X_pos = pos_tiles.reshape(n_pos, -1).astype(np.float32) / 255.0
    X_neg = neg_tiles.reshape(n_neg, -1).astype(np.float32) / 255.0

    # 5-class: 0=not_player, 1=N, 2=E, 3=S, 4=W
    X_5 = np.vstack([X_neg, X_pos])
    y_5 = np.concatenate([np.zeros(n_neg, dtype=int), pos_labels + 1])

    # Binary: 0=not_player, 1=player
    X_bin = np.vstack([X_neg, X_pos])
    y_bin = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])

    clf_kw = dict(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=int(max_iter),
        random_state=42,
        alpha=mlp_alpha,
    )
    if early_stopping and n_pos + n_neg >= 200:
        clf_kw["early_stopping"] = True
        clf_kw["validation_fraction"] = 0.1
        clf_kw["n_iter_no_change"] = 40

    make_pipeline = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(**clf_kw)),
    ])

    # Cross-validate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Training 5-class classifier...")
    clf_5 = make_pipeline()
    scores_5 = cross_val_score(clf_5, X_5, y_5, cv=cv, scoring="accuracy")
    print(f"  5-class CV accuracy: {scores_5.mean():.3f} +/- {scores_5.std():.3f}")
    clf_5.fit(X_5, y_5)

    print("Training binary classifier...")
    clf_bin = make_pipeline()
    scores_bin = cross_val_score(clf_bin, X_bin, y_bin, cv=cv, scoring="accuracy")
    print(f"  Binary CV accuracy: {scores_bin.mean():.3f} +/- {scores_bin.std():.3f}")
    clf_bin.fit(X_bin, y_bin)

    return {"5class": clf_5, "binary": clf_bin}


def eval_classifiers_on_val_dir(models, val_dir, tile_size, image_size, skip_substrings):
    """Report sklearn accuracy on a held-out GT directory (same tile protocol as training)."""
    from sklearn.metrics import accuracy_score

    pos_tiles, pos_labels, neg_tiles = collect_tiles_from_gt(
        val_dir, tile_size, image_size, skip_substrings
    )
    if len(pos_tiles) == 0:
        print(f"\n[eval_val_dir] No positive tiles in {val_dir!r}, skip.")
        return
    if len(neg_tiles) == 0:
        print(f"\n[eval_val_dir] No negative tiles in {val_dir!r}, skip.")
        return
    n_pos = len(pos_tiles)
    n_neg = len(neg_tiles)
    X_pos = pos_tiles.reshape(n_pos, -1).astype(np.float32) / 255.0
    X_neg = neg_tiles.reshape(n_neg, -1).astype(np.float32) / 255.0
    X_5 = np.vstack([X_neg, X_pos])
    y_5 = np.concatenate([np.zeros(n_neg, dtype=int), pos_labels + 1])
    X_bin = np.vstack([X_neg, X_pos])
    y_bin = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
    p5 = models["5class"].predict(X_5)
    pb = models["binary"].predict(X_bin)
    a5 = accuracy_score(y_5, p5)
    ab = accuracy_score(y_bin, pb)
    print(f"\n[eval_val_dir] {os.path.abspath(val_dir)}")
    print(f"  tiles: n_pos={n_pos} n_neg={n_neg}")
    print(f"  5-class accuracy: {a5:.4f}")
    print(f"  binary accuracy:  {ab:.4f}")


def apply_robust_preset(args):
    """Tune defaults for noisier generated / long-horizon eval (mutates *args*)."""
    if not args.robust_rollout:
        return
    if args.mlp_hidden.strip() == "128,64":
        args.mlp_hidden = "256,128,64"
    if args.hard_neg_rounds == 2:
        args.hard_neg_rounds = 4
    if args.neg_per_frame_mult == 1:
        args.neg_per_frame_mult = 2
    if args.tile_noise_std == 0.0:
        args.tile_noise_std = 0.012
    if args.tile_brightness_jitter == 0.0:
        args.tile_brightness_jitter = 0.08
    if args.mlp_max_iter == 500:
        args.mlp_max_iter = 800
    if not args.mlp_no_early_stopping:
        args.mlp_early_stopping = True
    print("  (--robust_rollout: wider MLP, more negs, tile jitter, extra mining, early stopping)")


def main():
    args = parse_args()
    apply_robust_preset(args)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    use_early = args.mlp_early_stopping and not args.mlp_no_early_stopping

    print(f"Collecting tiles from {args.val_dir} ...")
    print(f"  Tile size: {args.tile_size}x{args.tile_size}, Image size: {args.image_size}")
    if args.robust_rollout:
        print(f"  neg_per_frame_mult={args.neg_per_frame_mult}, tile_noise_std={args.tile_noise_std}, "
              f"tile_brightness_jitter={args.tile_brightness_jitter}")

    pos_tiles, pos_labels, neg_tiles = collect_tiles_from_gt(
        args.val_dir,
        args.tile_size,
        args.image_size,
        args.skip,
        neg_per_frame_mult=args.neg_per_frame_mult,
    )

    print(f"  GT positives: {len(pos_tiles)} (N={sum(pos_labels==0)}, E={sum(pos_labels==1)}, "
          f"S={sum(pos_labels==2)}, W={sum(pos_labels==3)})")
    print(f"  GT negatives: {len(neg_tiles)}")

    # Augment with generated tiles
    for gen_dir in args.augment_gen_dirs:
        print(f"  Augmenting from {gen_dir} ...")
        aug_pos, aug_labels = collect_tiles_from_gen(
            gen_dir, args.val_dir, args.tile_size, args.image_size, args.skip)
        if len(aug_pos) > 0:
            pos_tiles = np.concatenate([pos_tiles, aug_pos])
            pos_labels = np.concatenate([pos_labels, aug_labels])
            print(f"    Added {len(aug_pos)} tiles, total positives: {len(pos_tiles)}")

    if args.tile_noise_std > 0 or args.tile_brightness_jitter > 0:
        n0 = len(pos_tiles)
        pos_tiles, pos_labels = duplicate_jitter_positives(
            pos_tiles, pos_labels, args.tile_noise_std, args.tile_brightness_jitter, rng)
        print(f"  Positive jitter duplicates: {len(pos_tiles) - n0} (total positives {len(pos_tiles)})")

    # Cache
    np.savez(args.cache, pos_tiles=pos_tiles, pos_labels=pos_labels, neg_tiles=neg_tiles)
    print(f"  Cached training data to {args.cache}")

    hidden = tuple(int(x) for x in args.mlp_hidden.split(",") if x.strip())
    if not hidden:
        hidden = (128, 64)

    # Initial train
    models = train_classifiers(
        pos_tiles,
        pos_labels,
        neg_tiles,
        hidden_layer_sizes=hidden,
        mlp_alpha=args.mlp_alpha,
        max_iter=args.mlp_max_iter,
        early_stopping=use_early,
    )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(models, args.output)
    print(f"\nSaved initial classifier to {args.output}")

    # Hard negative mining rounds
    hn_source = args.mine_hard_negatives or args.output
    for rnd in range(args.hard_neg_rounds):
        print(f"\n--- Hard negative mining round {rnd + 1}/{args.hard_neg_rounds} ---")
        hard_negs = mine_hard_negatives(
            hn_source, args.val_dir, args.tile_size, args.image_size, args.skip)
        if len(hard_negs) == 0:
            print("  No hard negatives found, stopping.")
            break
        print(f"  Mined {len(hard_negs)} hard negatives")
        neg_tiles = np.concatenate([neg_tiles, hard_negs])
        print(f"  Total negatives: {len(neg_tiles)}")
        models = train_classifiers(
            pos_tiles,
            pos_labels,
            neg_tiles,
            hidden_layer_sizes=hidden,
            mlp_alpha=args.mlp_alpha,
            max_iter=args.mlp_max_iter,
            early_stopping=use_early,
        )
        joblib.dump(models, args.output)
        hn_source = args.output
        print(f"  Updated classifier saved to {args.output}")

    if args.mine_gen_eval_dirs:
        print("\n--- Mine hard negatives from generated frames ---")
        gen_negs = mine_hard_negatives_from_generated(
            args.output,
            args.mine_gen_eval_dirs,
            args.val_dir,
            args.tile_size,
            args.image_size,
            args.skip,
        )
        if len(gen_negs) == 0:
            print("  No gen false-positive tiles found (classifier may be too conservative on gen).")
        else:
            print(f"  Mined {len(gen_negs)} tiles from generated scans")
            neg_tiles = np.concatenate([neg_tiles, gen_negs])
            print(f"  Total negatives: {len(neg_tiles)}")
            models = train_classifiers(
                pos_tiles,
                pos_labels,
                neg_tiles,
                hidden_layer_sizes=hidden,
                mlp_alpha=args.mlp_alpha,
                max_iter=args.mlp_max_iter,
                early_stopping=use_early,
            )
            joblib.dump(models, args.output)
            print(f"  Updated classifier saved to {args.output}")

    if args.eval_val_dir:
        eval_classifiers_on_val_dir(
            models, args.eval_val_dir, args.tile_size, args.image_size, args.skip
        )

    print(f"\nDone. Final classifier: {args.output}")


if __name__ == "__main__":
    main()
