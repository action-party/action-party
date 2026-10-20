#!/usr/bin/env python3
import sys as _sys, os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

"""
Evaluate action accuracy on custom movement validation set using a trained
player tile classifier.

For every player at every action transition the script checks:
  1. Is the player still present at its GT position? (via binary classifier)
  2. If both current and previous frame have the player, does the orientation
     change match the GT orientation change?
A player disappearing counts as a failed action.

Usage:
    python eval_classifier_metrics.py --gen_dir outputs_ckpts/val/gen_139
    python eval_classifier_metrics.py --gen_dir outputs_ckpts/val/gen_144

    # Evaluate multiple runs at once:
    python eval_classifier_metrics.py \
        --gen_dir outputs_ckpts/val/gen_139 \
                  outputs_ckpts/val/gen_144

    # Custom paths:
    python eval_classifier_metrics.py \
        --gen_dir outputs_ckpts/val/gen_139 \
        --val_dir outputs_ckpts/val/val_npz \
        --classifier outputs_ckpts/val/player_tile_classifier.pkl \
        --skip allelopathic_harvest__open

Output is printed to stdout and saved to <gen_dir>/classifier_action_metrics.txt
"""
import argparse, os, glob, json, numpy as np, joblib
from collections import defaultdict

CELL = 32
LATENT_TO_PIXEL_5 = [0, 1, 5, 9, 13]
LATENT_TO_PIXEL_9 = [0, 1, 5, 9, 13, 17, 21, 25, 29]


def get_latent_to_pixel(num_latent):
    # Pattern: pixel[0]=0, pixel[k]=1+4*(k-1) for k>=1
    # (4 pixel frames per latent step after the initial frame).
    # Handles any num_latent (5 for T=4, 21 for T=20, etc.).
    if num_latent <= 0:
        return []
    return [0] + [1 + 4 * k for k in range(num_latent - 1)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen_dir", nargs="+", required=True,
                   help="One or more generation directories, each containing an eval_data/ subfolder")
    p.add_argument("--val_dir", default="outputs_ckpts/val/val_npz",
                   help="Directory with GT val .npz files")
    p.add_argument("--classifier", default="outputs_ckpts/val/player_tile_classifier.pkl",
                   help="Path to trained classifier .pkl")
    p.add_argument("--tile_size", type=int, default=32,
                   help="Tile size for extraction (32 for 512-res, 16 for 256-res)")
    p.add_argument("--image_size", type=int, default=512,
                   help="Pixel resolution of generated frames (512 or 256)")
    p.add_argument("--skip", nargs="*", default=["allelopathic_harvest__open"],
                   help="Game name substrings to skip")
    p.add_argument("--trackable_only", action="store_true",
                   help="Opt-in: restrict all metrics to subjects the classifier "
                        "detects in the initial GT frame (= tracking-benchmark protocol). "
                        "Default OFF to match the paper's eval (Table / Fig 5 numbers).")
    return p.parse_args()


def extract_tile(frame, cx, cy, tile_size=32):
    h, w = frame.shape[:2]
    half = tile_size // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    if x0 < 0 or y0 < 0 or x0 + tile_size > w or y0 + tile_size > h:
        return None
    return frame[y0:y0 + tile_size, x0:x0 + tile_size].copy()


class TileClassifier:
    def __init__(self, pkl_path, tile_size=32):
        models = joblib.load(pkl_path)
        self.clf_5 = models["5class"]
        self.clf_bin = models["binary"]
        self.tile_size = tile_size

    def classify(self, tile):
        """Returns (is_player: bool, orientation_0to3_or_None)."""
        if tile is None or tile.shape != (self.tile_size, self.tile_size, 3):
            return False, None
        feat = tile.reshape(1, -1).astype(np.float32) / 255.0
        is_player = self.clf_bin.predict(feat)[0] == 1
        if is_player:
            orient = int(self.clf_5.predict(feat)[0]) - 1
            return True, orient
        return False, None


def find_player_by_classifier(classifier, frame, gt_cx, gt_cy, tile_size=32,
                              search_radius=48, step=8):
    """Scan tiles around GT position with classifier. Returns closest (cx, cy) or None."""
    h, w = frame.shape[:2]
    half = tile_size // 2
    best_dist = float("inf")
    best_pos = None
    for dy in range(-search_radius, search_radius + 1, step):
        for dx in range(-search_radius, search_radius + 1, step):
            cx = int(round(gt_cx)) + dx
            cy = int(round(gt_cy)) + dy
            tile = extract_tile(frame, cx, cy, tile_size)
            is_p, _ = classifier.classify(tile)
            if is_p:
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (cx, cy)
    return best_pos


def count_player_clusters(classifier, frame, tile_size=32, step=None, merge_dist=None,
                          conf_thresh=0.999):
    """Count distinct player-classified clusters in a frame (position-independent).
    Uses a high confidence threshold to avoid false positives from VAE artifacts
    and environmental sprites."""
    if step is None:
        step = max(tile_size // 4, 2)
    if merge_dist is None:
        merge_dist = int(tile_size * 0.875)
    h, w = frame.shape[:2]
    half = tile_size // 2
    hits = []
    for cy in range(half, h - half, step):
        for cx in range(half, w - half, step):
            tile = extract_tile(frame, cx, cy, tile_size)
            if tile is None or tile.shape != (tile_size, tile_size, 3):
                continue
            feat = tile.reshape(1, -1).astype(np.float32) / 255.0
            prob = classifier.clf_bin.predict_proba(feat)[0][1]
            if prob >= conf_thresh:
                hits.append((cx, cy))
    if not hits:
        return 0
    parent = list(range(len(hits)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            dx = hits[i][0] - hits[j][0]
            dy = hits[i][1] - hits[j][1]
            if dx * dx + dy * dy < merge_dist * merge_dist:
                union(i, j)
    return len(set(find(i) for i in range(len(hits))))


def evaluate_gen_dir(gen_dir, val_dir, classifier, skip_substrings,
                     tile_size=32, image_size=512, trackable_only=False):
    eval_dir = os.path.join(gen_dir, "eval_data")
    if not os.path.isdir(eval_dir):
        print(f"  ERROR: {eval_dir} not found")
        return None

    efiles = sorted(glob.glob(os.path.join(eval_dir, "*.npz")))
    if not efiles:
        print(f"  ERROR: no .npz files in {eval_dir}")
        return None

    game_results = defaultdict(lambda: {
        "surv_total": 0, "surv_correct": 0,
        "last_surv_total": 0, "last_surv_correct": 0,
        "ori_total": 0, "ori_correct": 0,
        "step_totals": defaultdict(int),
        "step_corrects": defaultdict(int),
        # Paper-protocol MA: denominator = num_sub at every step
        # (disappearance counts as wrong). Kept in parallel with the
        # "strict" MA above (step_totals / step_corrects) which uses
        # denominator = both-present transitions only.
        "step_totals_paper": defaultdict(int),
        "step_corrects_paper": defaultdict(int),
        "step_surv_total": defaultdict(int),
        "step_surv_correct": defaultdict(int),
        "frame_pcd": [],
        "coord_errors": [],
        "visual_errors": [],
        "coord_errors_by_step": defaultdict(list),
        "visual_errors_by_step": defaultdict(list),
        "player_preservation": [],
        "untrackable_count": 0,
        "subject_count": 0,
    })
    has_gen_coords = False

    for ef in efiles:
        bn = os.path.basename(ef).replace(".npz", "")
        if any(s in bn for s in skip_substrings):
            continue

        ed = np.load(ef)
        gen_frames = ed["gen_frames"]
        num_sub = int(ed["num_subjects_real"])

        # Try sibling val_npz; if missing or shorter than gt_coords
        # (e.g. T=20 runs against T=4 val set), fall back to deriving
        # GT orientations from gt_frames via the same classifier used
        # on gen_frames. This is required for the T=20 ablation eval
        # where no matching 21-latent val_npz exists.
        val_path = os.path.join(val_dir, os.path.basename(ef)) if val_dir else None
        vd = None
        if val_path and os.path.exists(val_path):
            vd = np.load(val_path, allow_pickle=True)

        if "gt_coords" in ed:
            gt_c = ed["gt_coords"].copy()
        elif vd is not None and "frame_coords" in vd:
            gt_c = vd["frame_coords"].copy()
        else:
            print(f"  WARNING: no gt_coords in {ef} or sibling val_npz, skipping")
            continue
        if gt_c.max() <= 1.0:
            gt_c *= image_size

        num_latent_gt = len(gt_c)

        if vd is not None and "game_name" in vd.files:
            game = str(vd["game_name"])
        else:
            game = bn.rsplit("_s", 1)[0]

        use_val_orient = (
            vd is not None
            and "orientations" in vd.files
            and len(vd["orientations"]) >= num_latent_gt
        )
        if use_val_orient:
            gt_orient = vd["orientations"].astype(int)
        elif "gt_frames" in ed:
            gt_frames_arr = ed["gt_frames"]
            l2p_gt = get_latent_to_pixel(num_latent_gt)
            gt_orient = np.zeros((num_latent_gt, num_sub), dtype=int)
            for t_lat in range(num_latent_gt):
                pix_idx = l2p_gt[t_lat]
                if pix_idx >= len(gt_frames_arr):
                    break
                frame = gt_frames_arr[pix_idx]
                for s in range(num_sub):
                    cx, cy = float(gt_c[t_lat, s, 0]), float(gt_c[t_lat, s, 1])
                    tile = extract_tile(frame, cx, cy, tile_size)
                    is_p, ori = classifier.classify(tile)
                    gt_orient[t_lat, s] = ori if is_p else 0
        else:
            print(f"  WARNING: no orientations source for {ef} (need val_npz or gt_frames), skipping")
            continue

        num_latent = len(gt_c)
        l2p = get_latent_to_pixel(num_latent)

        # Determine how many latent steps we can actually evaluate
        max_eval_lat = num_latent
        for t_lat in range(num_latent):
            if l2p[t_lat] >= len(gen_frames):
                max_eval_lat = t_lat
                break

        # Classify every tile (all subjects, all steps). We populate
        # pred_present/pred_orient first so we can then restrict all
        # downstream metrics to *trackable* subjects only — i.e. those
        # the classifier can actually detect in the initial GT frame.
        pred_present = np.zeros((num_latent, num_sub), dtype=bool)
        pred_orient = np.full((num_latent, num_sub), -1, dtype=int)

        for t_lat in range(max_eval_lat):
            pix_idx = l2p[t_lat]
            frame = gen_frames[pix_idx]
            for s in range(num_sub):
                cx, cy = float(gt_c[t_lat, s, 0]), float(gt_c[t_lat, s, 1])
                tile = extract_tile(frame, cx, cy, tile_size)
                is_p, ori = classifier.classify(tile)
                pred_present[t_lat, s] = is_p
                if is_p:
                    pred_orient[t_lat, s] = ori

        # Paper-protocol default: count every subject at every step.
        # Opt-in `trackable_only`: restrict to subjects the classifier can
        # see in frame 0 (avoids systematic classifier blind spots on a
        # handful of games like territory__inside_out; matches
        # tracking-benchmark convention). Paper's reported DR uses the
        # default (unfiltered).
        if trackable_only:
            trackable = pred_present[0].copy()
        else:
            trackable = np.ones(num_sub, dtype=bool)
        num_trackable = int(trackable.sum())
        game_results[game]["untrackable_count"] += int(num_sub - num_trackable)
        game_results[game]["subject_count"] += int(num_sub)

        for t_lat in range(max_eval_lat):
            is_last = t_lat == max_eval_lat - 1
            detected = 0
            for s in range(num_sub):
                if not trackable[s]:
                    continue
                is_p = bool(pred_present[t_lat, s])
                if is_p:
                    detected += 1
                game_results[game]["surv_total"] += 1
                game_results[game]["step_surv_total"][t_lat] += 1
                if is_p:
                    game_results[game]["surv_correct"] += 1
                    game_results[game]["step_surv_correct"][t_lat] += 1
                    game_results[game]["ori_total"] += 1
                    if int(pred_orient[t_lat, s]) == int(gt_orient[t_lat, s]):
                        game_results[game]["ori_correct"] += 1
                if is_last:
                    game_results[game]["last_surv_total"] += 1
                    if is_p:
                        game_results[game]["last_surv_correct"] += 1

            game_results[game]["frame_pcd"].append(abs(detected - num_trackable))

        # Position-independent player preservation: compare high-confidence
        # cluster counts between gen frame 0 and gen last frame directly.
        gen_init_clusters = count_player_clusters(classifier, gen_frames[0], tile_size)
        last_pix = l2p[max_eval_lat - 1]
        gen_last_clusters = count_player_clusters(classifier, gen_frames[last_pix], tile_size)
        game_results[game]["player_preservation"].append(
            (num_sub, max(gen_init_clusters, 1), gen_last_clusters))

        # Position error metrics
        gt_c_norm = gt_c / image_size
        if "gen_coords" in ed:
            has_gen_coords = True
            gen_c_raw = ed["gen_coords"].copy()
            gen_c_norm = gen_c_raw / image_size if gen_c_raw.max() > 1.0 else gen_c_raw.copy()

        for t_lat in range(1, max_eval_lat):
            pix_idx = l2p[t_lat]
            frame = gen_frames[pix_idx]
            for s in range(num_sub):
                if not trackable[s]:
                    continue
                gt_n = gt_c_norm[t_lat, s]
                if has_gen_coords:
                    l2 = float(np.sqrt(((gt_n - gen_c_norm[t_lat, s]) ** 2).sum()))
                    game_results[game]["coord_errors"].append(l2)
                    game_results[game]["coord_errors_by_step"][t_lat].append(l2)
                det = find_player_by_classifier(
                    classifier, frame,
                    gt_c[t_lat, s, 0], gt_c[t_lat, s, 1], tile_size)
                if det is not None:
                    det_n = np.array([det[0] / image_size, det[1] / image_size])
                    ve = float(np.sqrt(((gt_n - det_n) ** 2).sum()))
                    game_results[game]["visual_errors"].append(ve)
                    game_results[game]["visual_errors_by_step"][t_lat].append(ve)

        # Action accuracy per transition — emit BOTH variants:
        #   • "strict" MA: denom = both-present trackable transitions.
        #     Isolates action-prediction error from player visibility.
        #   • "paper" MA:  denom = num_sub at every step (disappearance
        #     counts as wrong). Matches the paper's Fig 5 protocol.
        for t in range(1, max_eval_lat):
            for s in range(num_sub):
                if not trackable[s]:
                    continue
                # Paper-protocol denominator: count every subject regardless.
                game_results[game]["step_totals_paper"][t] += 1
                both_present = pred_present[t, s] and pred_present[t - 1, s]
                gt_turn = (int(gt_orient[t, s]) - int(gt_orient[t - 1, s])) % 4
                if both_present:
                    pred_turn = (int(pred_orient[t, s]) - int(pred_orient[t - 1, s])) % 4
                    # Strict MA — denom only counts both-present.
                    game_results[game]["step_totals"][t] += 1
                    if pred_turn == gt_turn:
                        game_results[game]["step_corrects"][t] += 1
                        game_results[game]["step_corrects_paper"][t] += 1

    # Aggregate
    all_steps = set()
    for r in game_results.values():
        all_steps.update(r["step_totals"].keys())
    all_steps = sorted(all_steps)

    totals = {
        "surv_total": 0, "surv_correct": 0,
        "last_surv_total": 0, "last_surv_correct": 0,
        "ori_total": 0, "ori_correct": 0,
        "step_totals": defaultdict(int),
        "step_corrects": defaultdict(int),
        "step_totals_paper": defaultdict(int),
        "step_corrects_paper": defaultdict(int),
        "step_surv_total": defaultdict(int),
        "step_surv_correct": defaultdict(int),
        "all_pcd": [],
        "all_coord_errors": [],
        "all_visual_errors": [],
        "coord_errors_by_step": defaultdict(list),
        "visual_errors_by_step": defaultdict(list),
        "all_preservation": [],
        "untrackable_count": 0,
        "subject_count": 0,
    }
    all_paper_steps = set()
    for r in game_results.values():
        all_paper_steps.update(r["step_totals_paper"].keys())
    all_paper_steps = sorted(all_paper_steps)
    for r in game_results.values():
        for k in ["surv_total", "surv_correct", "last_surv_total",
                   "last_surv_correct", "ori_total", "ori_correct",
                   "untrackable_count", "subject_count"]:
            totals[k] += r[k]
        for step in all_steps:
            totals["step_totals"][step] += r["step_totals"].get(step, 0)
            totals["step_corrects"][step] += r["step_corrects"].get(step, 0)
        for step in all_paper_steps:
            totals["step_totals_paper"][step] += r["step_totals_paper"].get(step, 0)
            totals["step_corrects_paper"][step] += r["step_corrects_paper"].get(step, 0)
        for step, v in r["step_surv_total"].items():
            totals["step_surv_total"][step] += v
        for step, v in r["step_surv_correct"].items():
            totals["step_surv_correct"][step] += v
        totals["all_pcd"].extend(r["frame_pcd"])
        totals["all_coord_errors"].extend(r["coord_errors"])
        totals["all_visual_errors"].extend(r["visual_errors"])
        for step, errs in r["coord_errors_by_step"].items():
            totals["coord_errors_by_step"][step].extend(errs)
        for step, errs in r["visual_errors_by_step"].items():
            totals["visual_errors_by_step"][step].extend(errs)
        totals["all_preservation"].extend(r["player_preservation"])

    action_total = sum(totals["step_totals"].values())
    action_correct = sum(totals["step_corrects"].values())

    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    name = os.path.basename(gen_dir.rstrip("/"))
    p(f"{'=' * 80}")
    p(f"  {name} — Classifier-based metrics")
    p(f"{'=' * 80}")

    pct = lambda n, d: n / max(d, 1) * 100

    trackable_total = totals["subject_count"] - totals["untrackable_count"]
    p(f"  Trackable subjects:   {trackable_total}/{totals['subject_count']}"
      f"  (excluded {totals['untrackable_count']} classifier-invisible at step 0)")
    p(f"  Player survival:      {totals['surv_correct']}/{totals['surv_total']}"
      f"  = {pct(totals['surv_correct'], totals['surv_total']):.1f}%")
    p(f"  Survival (last frame):{totals['last_surv_correct']}/{totals['last_surv_total']}"
      f"  = {pct(totals['last_surv_correct'], totals['last_surv_total']):.1f}%")
    p(f"  Orientation accuracy: {totals['ori_correct']}/{totals['ori_total']}"
      f"  = {pct(totals['ori_correct'], totals['ori_total']):.1f}%")
    p(f"  Action accuracy:      {action_correct}/{action_total}"
      f"  = {pct(action_correct, action_total):.1f}%")
    p(f"  Avg player count diff:{np.mean(totals['all_pcd']):.2f}")

    if totals["all_preservation"]:
        total_gt_sub = sum(ns for ns, _, _ in totals["all_preservation"])
        total_est_init = sum(ei for _, ei, _ in totals["all_preservation"])
        total_est_last = sum(el for _, _, el in totals["all_preservation"])
        pres_pct = total_est_last / max(total_est_init, 1) * 100
        p(f"  Player preservation:  {total_est_last}/{total_est_init}"
          f"  = {pres_pct:.1f}%  (position-independent)")

    if totals["all_coord_errors"]:
        ce = totals["all_coord_errors"]
        ce_px = np.mean(ce) * image_size
        p(f"  Coord prediction err: {ce_px:.1f}px  ({ce_px / tile_size:.2f} tiles)"
          f"  [L2={np.mean(ce):.4f}]")
    if totals["all_visual_errors"]:
        ve = totals["all_visual_errors"]
        ve_px = np.mean(ve) * image_size
        p(f"  Visual position err:  {ve_px:.1f}px  ({ve_px / tile_size:.2f} tiles)"
          f"  [L2={np.mean(ve):.4f}, {len(ve)} detected]")
    p()

    p(f"  Action accuracy by step (strict — denom = both-present transitions):")
    step_accs = {}
    for step in all_steps:
        acc = pct(totals["step_corrects"][step], totals["step_totals"][step])
        p(f"    Step {step}: {totals['step_corrects'][step]}/{totals['step_totals'][step]} = {acc:.1f}%")
        step_accs[step] = acc
    p()

    p(f"  Action accuracy by step (paper protocol — denom = num_sub, disappearance counts as wrong):")
    step_accs_paper = {}
    for step in all_paper_steps:
        acc = pct(totals["step_corrects_paper"][step], totals["step_totals_paper"][step])
        p(f"    Step {step}: {totals['step_corrects_paper'][step]}/{totals['step_totals_paper'][step]} = {acc:.1f}%")
        step_accs_paper[step] = acc
    p()

    # Detection rate (DR) per step — matches the 'detection_rate' column
    # tabulated in the paper, but reported at every latent step instead
    # of aggregated over all latents.
    det_steps = sorted(totals["step_surv_total"].keys())
    step_dr = {}
    if det_steps:
        p(f"  Detection rate (DR) by step:")
        for step in det_steps:
            dr = pct(totals["step_surv_correct"][step], totals["step_surv_total"][step])
            p(f"    Step {step}: {totals['step_surv_correct'][step]}/{totals['step_surv_total'][step]} = {dr:.1f}%")
            step_dr[step] = dr
        p()

    coord_err_by_step_px = {}
    visual_err_by_step_px = {}
    if totals["coord_errors_by_step"] or totals["visual_errors_by_step"]:
        p(f"  Player-position drift vs GT by step (pixels):")
        p(f"    {'step':>4}  {'coord_err':>10}  {'visual_err':>11}  {'n_coord':>7}  {'n_vis':>5}")
        drift_steps = sorted(set(list(totals["coord_errors_by_step"].keys())
                                 + list(totals["visual_errors_by_step"].keys())))
        for step in drift_steps:
            ce_list = totals["coord_errors_by_step"].get(step, [])
            ve_list = totals["visual_errors_by_step"].get(step, [])
            ce_px = float(np.mean(ce_list)) * image_size if ce_list else float("nan")
            ve_px = float(np.mean(ve_list)) * image_size if ve_list else float("nan")
            coord_err_by_step_px[step] = ce_px
            visual_err_by_step_px[step] = ve_px
            p(f"    {step:>4}  {ce_px:>10.1f}  {ve_px:>11.1f}  {len(ce_list):>7d}  {len(ve_list):>5d}")
        p()

    p(f"  Per-game breakdown:")
    p(f"  {'Game':<50} {'Surv':>6} {'Last':>6} {'Orient':>7} {'Action':>7} {'PCD':>5}")
    p(f"  {'-' * 85}")
    for game in sorted(game_results.keys()):
        r = game_results[game]
        st = sum(r["step_totals"].values())
        sc = sum(r["step_corrects"].values())
        p(f"  {game:<50} "
          f"{pct(r['surv_correct'], r['surv_total']):5.1f}% "
          f"{pct(r['last_surv_correct'], r['last_surv_total']):5.1f}% "
          f"{pct(r['ori_correct'], r['ori_total']):6.1f}% "
          f"{pct(sc, st):6.1f}% "
          f"{np.mean(r['frame_pcd']) if r['frame_pcd'] else 0:5.2f}")

    # Save
    out_txt = os.path.join(gen_dir, "classifier_action_metrics.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {out_txt}")

    out_json = os.path.join(gen_dir, "classifier_action_metrics.json")
    act_total_paper = sum(totals["step_totals_paper"].values())
    act_correct_paper = sum(totals["step_corrects_paper"].values())
    json_data = {
        "survival": pct(totals["surv_correct"], totals["surv_total"]),
        "survival_last_frame": pct(totals["last_surv_correct"], totals["last_surv_total"]),
        "orientation_accuracy": pct(totals["ori_correct"], totals["ori_total"]),
        "action_accuracy": pct(action_correct, action_total),
        "action_accuracy_paper": pct(act_correct_paper, act_total_paper),
        "avg_player_count_diff": float(np.mean(totals["all_pcd"])),
        "action_accuracy_by_step": {str(s): step_accs[s] for s in all_steps},
        "action_accuracy_by_step_paper": {str(s): step_accs_paper[s] for s in all_paper_steps},
    }
    if totals["all_preservation"]:
        total_est_init = sum(ei for _, ei, _ in totals["all_preservation"])
        total_est_last = sum(el for _, _, el in totals["all_preservation"])
        json_data["player_preservation_pct"] = total_est_last / max(total_est_init, 1) * 100
    if totals["all_coord_errors"]:
        ce_mean = float(np.mean(totals["all_coord_errors"]))
        json_data["coord_prediction_l2"] = ce_mean
        json_data["coord_prediction_px"] = ce_mean * image_size
        json_data["coord_prediction_tiles"] = ce_mean * image_size / tile_size
    if totals["all_visual_errors"]:
        ve_mean = float(np.mean(totals["all_visual_errors"]))
        json_data["visual_position_l2"] = ve_mean
        json_data["visual_position_px"] = ve_mean * image_size
        json_data["visual_position_tiles"] = ve_mean * image_size / tile_size
        json_data["visual_position_n_detected"] = len(totals["all_visual_errors"])
    if coord_err_by_step_px:
        json_data["coord_err_by_step_px"] = {
            str(s): (None if np.isnan(v) else v) for s, v in coord_err_by_step_px.items()}
    if visual_err_by_step_px:
        json_data["visual_err_by_step_px"] = {
            str(s): (None if np.isnan(v) else v) for s, v in visual_err_by_step_px.items()}
    if step_dr:
        json_data["detection_rate_by_step"] = {str(s): v for s, v in step_dr.items()}
        json_data["detection_rate"] = pct(totals["surv_correct"], totals["surv_total"])
        json_data["detection_rate_last_frame"] = pct(
            totals["last_surv_correct"], totals["last_surv_total"])
    json_data["subject_count_total"] = int(totals["subject_count"])
    json_data["subject_count_trackable"] = int(totals["subject_count"] - totals["untrackable_count"])
    json_data["subject_count_untrackable"] = int(totals["untrackable_count"])
    with open(out_json, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved to {out_json}")

    return json_data


def main():
    args = parse_args()
    global CELL
    CELL = args.tile_size
    clf = TileClassifier(args.classifier, tile_size=args.tile_size)
    skip = args.skip or []

    for gen_dir in args.gen_dir:
        mode = "trackable-only" if args.trackable_only else "paper-protocol (all subjects)"
        print(f"\nEvaluating {gen_dir} (tile_size={args.tile_size}, image_size={args.image_size}, mode={mode})...")
        evaluate_gen_dir(gen_dir, args.val_dir, clf, skip,
                         tile_size=args.tile_size, image_size=args.image_size,
                         trackable_only=args.trackable_only)
        print()


if __name__ == "__main__":
    main()
