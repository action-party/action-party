#!/usr/bin/env python3
import sys as _sys, os as _os
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

"""
Batch ablation evaluation: loads model ONCE, runs inference on multiple samples
of the Melting Pot ``coins`` validation set, then computes classifier-based
action-following metrics.

Output layout (per ablation):

    <output_base>/<name>/
        eval_data/coins_s<idx>.npz       # gen_frames, gt_frames, coords, actions (for pixel/FVD too)
        val_npz/coins_s<idx>.npz         # per-sample GT bundle (frames + coords + actions)
        sample<idx>_combined.gif         # side-by-side gen|GT preview
        classifier_action_metrics.json   # numeric metrics
        classifier_action_metrics.txt    # human-readable metrics

Use ``--output_base`` to redirect the parent dir (default
``outputs_ckpts/ablation_eval``); set it to e.g.
``outputs_ckpts/ablation_eval_rerun_<DATE>`` to avoid overwriting existing runs.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_ablation_eval.py \\
        --config configs/exp-134-coord-token-original-actions.yaml \\
        --checkpoint checkpoints/exp-134-coord-token-original-actions/checkpoint_model_045000/model.pt \\
        --name baseline-134 \\
        --num_samples 25

    512² multi-game view-token (e.g. exp-153): use ``--multi_game`` and the training LMDB
    (``datasets/melting_pot_all_games_512_2k``), not the 256² coins-only set.

        CUDA_VISIBLE_DEVICES=0 python scripts/eval/run_ablation_eval.py \\
            --config configs/exp-153-multi-game-view-token-512.yaml \\
            --checkpoint checkpoints/.../checkpoint_model_005000/model.pt \\
            --name 153-sanity --multi_game \\
            --dataset datasets/melting_pot_all_games_512_2k \\
            --game_name coins --num_samples 1 --skip_classifier

    Quick **debug** smoke (153 / game text): low steps + 1 AR frame + skip bulk T5 cache + no GIF:

        CUDA_VISIBLE_DEVICES=1 python scripts/eval/run_ablation_eval.py \\
          ... \\
          --sampling_steps 2 --max_ar_steps 1 --t5_minimal_cache --no_gif --skip_classifier
"""
import argparse
import glob
import math
import os
import pickle
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import cv2
import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper, WanTextEncoder
from model.action_only_conditioner import ActionOnlySpatialConditioner
from utils.dataset import MultiGameLMDBDataset, PositionTokenLMDBDataset
from utils.view_token_ops import (
    normalized_coords_to_patch_centers,
    gather_view_latent_patches,
    scatter_view_flat_patches_to_latent,
)


# === Embedded helpers (copied from the original inference_coord_diffusion.py
# so this script is self-contained on the open-source branch) ============

def load_checkpoint(generator, checkpoint_path, use_ema=False):
    """Load a generator checkpoint, preferring EMA weights when available."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if use_ema and "generator_ema" in checkpoint:
        print("Using EMA weights")
        state_dict = checkpoint["generator_ema"]
    elif "generator" in checkpoint:
        print("Using regular weights")
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("._fsdp_wrapped_module.", ".").replace("_fsdp_wrapped_module.", "")
        new_state_dict[new_k] = v

    missing, unexpected = generator.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    return generator


def compute_gt_coords(initial_positions, action_sequences, num_frames, speed=16, image_size=256):
    """Compute GT per-frame coords from initial pos + 5-action ABSOLUTE actions.

    Only valid for the 5-action absolute action space (NOOP, NORTH, SOUTH, WEST,
    EAST). For 7-action relative spaces use ``frame_coords`` from the dataset.
    """
    ACTION_DELTAS = {0: (0, 0), 1: (0, -speed), 2: (0, speed), 3: (-speed, 0), 4: (speed, 0)}
    S = initial_positions.shape[0]
    coords = torch.zeros(num_frames, S, 2, device=initial_positions.device)
    coords[0] = initial_positions
    for t in range(1, num_frames):
        for s in range(S):
            action = action_sequences[s, t - 1].item() if t - 1 < action_sequences.shape[1] else 0
            dx, dy = ACTION_DELTAS.get(action, (0, 0))
            coords[t, s, 0] = torch.clamp(coords[t - 1, s, 0] + dx, 0, image_size - 1)
            coords[t, s, 1] = torch.clamp(coords[t - 1, s, 1] + dy, 0, image_size - 1)
    return coords / image_size


# === Argparse ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="Path to experiment .yaml config")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt")
    p.add_argument("--name", required=True, help="Name for output dir (under <output_base>/)")
    p.add_argument("--output_base", default="outputs_ckpts/ablation_eval",
                   help="Parent directory for results (default: outputs_ckpts/ablation_eval)")
    p.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    p.add_argument("--start_idx", type=int, default=0, help="Starting sample index")
    p.add_argument("--sampling_steps", type=int, default=20)
    p.add_argument("--dataset", default="datasets/melting_pot_coins_original_actions")
    p.add_argument("--classifier",
                   default="outputs_ckpts/val_256/player_tile_classifier.pkl")
    p.add_argument("--tile_size", type=int, default=16)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_gif",
        action="store_true",
        help="Skip writing sample*_combined.gif (faster; NPZs unchanged).",
    )
    p.add_argument(
        "--initial_coord_noise_sigma",
        type=float,
        default=0.0,
        help="Std dev of Gaussian noise on **frame-0** coords in normalized [0,1] space "
             "(independent per x,y and subject). Applied to the coord-diffusion initial state "
             "and to conditioner initial_positions (pixels), while **gt_coords for metrics** "
             "stay clean. First video latent stays dataset GT — use this as a coord "
             "mis-registration / user-typo stress test. 0 disables.",
    )
    p.add_argument(
        "--multi_game",
        action="store_true",
        help="Use MultiGameLMDBDataset at --dataset (512² multi-game LMDB). Required for "
             "view-token 153 and other 512×512 multi-game checkpoints.",
    )
    p.add_argument(
        "--game_name",
        type=str,
        default=None,
        help="With --multi_game: use the first sample whose game_name matches (e.g. coins).",
    )
    p.add_argument(
        "--one_sample_per_game",
        action="store_true",
        help="With --multi_game: evaluate exactly one clip per LMDB shard (one per game), "
             "using local index --per_game_local_idx inside each shard. Matches the 46-sample "
             "protocol used for the full multi-game eval. "
             "Incompatible with --game_name.",
    )
    p.add_argument(
        "--per_game_local_idx",
        type=int,
        default=0,
        help="With --one_sample_per_game: which row inside each shard (0 .. num_samples-1). "
             "Use 50 to mirror inference_all_games_coord.py --sample_offset 50.",
    )
    p.add_argument(
        "--eval_num_shards",
        type=int,
        default=1,
        help="Split the resolved sample list across this many workers (contiguous chunks). "
             "Use with --eval_shard_id. When >1, classifier metrics are skipped here; merge "
             "outputs then run scripts/eval/merge_ablation_shards_classify.sh.",
    )
    p.add_argument(
        "--eval_shard_id",
        type=int,
        default=0,
        help="Zero-based worker id in [0, eval_num_shards). Must be < eval_num_shards.",
    )
    p.add_argument(
        "--skip_classifier",
        action="store_true",
        help="Skip eval_classifier_metrics.py subprocess (faster sanity / preview runs).",
    )
    p.add_argument(
        "--max_ar_steps",
        type=int,
        default=None,
        help="Cap how many autoregressive **new** latent frames to generate (debug). "
             "Default: full clip (e.g. 4 for 5-frame data). Use 1 with low --sampling_steps for a smoke test.",
    )
    p.add_argument(
        "--t5_minimal_cache",
        action="store_true",
        help="With use_game_descriptions: skip bulk T5 encoding of all game prompts at startup; "
             "encode only what each sample needs (faster iteration when debugging 153-style runs).",
    )
    p.add_argument(
        "--save_view_mosaic",
        action="store_true",
        help="For view_token_mode only: save an extra GIF with a 3×3 **pixel** tile mosaic per "
             "subject (bins match ActionOnlySpatialConditioner / view-token gather spacing: "
             "2×latent_patch_size px). Top row = generated, bottom = GT.",
    )
    p.add_argument(
        "--save_view_latent_decode",
        action="store_true",
        help="For view_token_mode only: scatter **all** view patches into one zero-padded full latent "
             "(merged), decode once per frame → sample*_view_latent_decoded.gif.",
    )
    p.add_argument(
        "--save_view_latent_decode_separate",
        action="store_true",
        help="For view_token_mode only: decode **each** view micro-patch **separately** (centered in a "
             "small zero latent, not merged into one full frame), build a side×side thumb grid per subject "
             "(P=(2·crop_radius+1)²) → sample*_view_patches_separate.gif. Slower (P× subjects × time decodes).",
    )
    return p.parse_args()


# === Single-sample autoregressive inference ===============================

def run_inference_for_sample(
    sample_idx, dataset, config, generator, vae, conditioner, device, dtype,
    num_subjects, num_action_classes, coord_scale_factor, sampling_steps,
    image_size, output_dir, seed,
    initial_coord_noise_sigma=0.0,
    video_text_embeds=None,
    max_ar_steps=None,
    output_view_latent_decode_path=None,
    output_view_latent_decode_separate_path=None,
):
    """Run AR inference on one sample. Returns dict with frames / coords / actions / l2."""
    torch.manual_seed(seed + sample_idx)
    np.random.seed(seed + sample_idx)

    sample = dataset[sample_idx]
    gt_video_latents = sample["ode_latent"].to(device, dtype=dtype)
    if gt_video_latents.dim() == 5:
        gt_video_latents = gt_video_latents[0]

    initial_positions = sample["initial_positions"].to(device, dtype=dtype)
    if initial_positions.dim() == 1:
        initial_positions = initial_positions.unsqueeze(0)
    if initial_positions.dim() == 2:
        initial_positions = initial_positions.unsqueeze(0)

    T, C, H, W = gt_video_latents.shape
    max_frames = T
    num_actions_per_subj = max_frames - 1
    patch_size = config.get("latent_patch_size", 8)
    effective_patch_size = patch_size * 2
    max_patch_rope = max(0, int(image_size / effective_patch_size) - 1)

    dataset_frame_coords = None
    if "frame_coords" in sample and sample["frame_coords"] is not None:
        dataset_frame_coords = sample["frame_coords"].to(device, dtype=dtype)

    dataset_orientations = None
    if "orientations" in sample and sample["orientations"] is not None:
        o = sample["orientations"]
        dataset_orientations = o.cpu().numpy() if torch.is_tensor(o) else np.asarray(o)
    if dataset_orientations is None:
        try:
            import lmdb as lmdb_lib
            dp = getattr(dataset, "data_path", None) or getattr(dataset, "_data_path", None)
            lmdb_paths = sorted(glob.glob(os.path.join(dp, "*.lmdb"))) if dp else []
            if not lmdb_paths and dp:
                lmdb_paths = sorted(glob.glob(os.path.join(dp, "*", "*.lmdb")))
            for lp in lmdb_paths:
                env = lmdb_lib.open(lp, readonly=True, lock=False)
                with env.begin() as txn:
                    val = txn.get(f"orientations_{sample_idx}_data".encode())
                    if val is not None:
                        dataset_orientations = pickle.loads(val)
                        break
                env.close()
        except Exception:
            pass

    gt_actions = sample["actions"].to(device)
    if gt_actions.dim() == 1:
        gt_actions = gt_actions.unsqueeze(0)
    action_sequences = gt_actions[:num_subjects].contiguous()
    n_action_steps = action_sequences.shape[1]

    use_gt_positions_for_rope = config.get("use_gt_positions_for_rope", False)
    if dataset_frame_coords is not None:
        gt_coords = dataset_frame_coords
    else:
        speed = config.get("position_speed", 16)
        gt_coords = compute_gt_coords(
            initial_positions[0], action_sequences, max_frames,
            speed=speed, image_size=image_size,
        )

    # coord_orientation_in_state ablation: state per subject is 4D (x, y, sin theta, cos theta)
    # instead of 2D. The (sin, cos) channels are NOT scaled by coord_scale_factor.
    coord_orient = bool(config.get("coord_orientation_in_state", False))
    state_dim = 4 if coord_orient else 2

    generated_video = gt_video_latents[:1].unsqueeze(0)
    # Build the initial state: position-only or position+orientation.
    initial_coords_pos = gt_coords[:1].unsqueeze(0).to(device, dtype=dtype)  # [1, 1, S, 2]
    if initial_coord_noise_sigma > 0:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed + sample_idx * 10007) & 0x7FFFFFFF)
        noise = torch.randn(
            initial_coords_pos.shape, device=device, dtype=dtype, generator=g,
        )
        initial_coords_pos = (initial_coords_pos + noise * initial_coord_noise_sigma).clamp(
            0.0, 1.0,
        )
        # Keep conditioner consistent with the perturbed "user" coordinates (pixels).
        ip = initial_positions.clone()
        n0 = initial_coords_pos[0, 0]
        imx = float(image_size)
        ip[0, :, 0] = (n0[:, 0] * imx).clamp(0, image_size - 1)
        ip[0, :, 1] = (n0[:, 1] * imx).clamp(0, image_size - 1)
        initial_positions = ip
    if coord_orient and dataset_orientations is not None:
        # dataset_orientations: numpy [T, S] long with values in {0..3}
        orient0 = torch.tensor(dataset_orientations[0:1], dtype=dtype, device=device).unsqueeze(0)  # [1, 1, S]
        theta0 = orient0 * (math.pi / 2.0)
        sincos0 = torch.stack([torch.sin(theta0), torch.cos(theta0)], dim=-1)  # [1, 1, S, 2]
        initial_coords = torch.cat([initial_coords_pos, sincos0], dim=-1)  # [1, 1, S, 4]
    else:
        initial_coords = initial_coords_pos
    generated_coords = initial_coords.clone()
    position_before_action = config.get("position_before_action", True)
    zero_pos_masked = config.get("zero_position_for_masked", False)
    view_token_mode = bool(config.get("view_token_mode", False))
    view_crop_radius = int(config.get("view_crop_radius", 1))
    view_spatial_h = view_spatial_w = centers_bt = None
    pv = None
    if view_token_mode:
        pv = (2 * view_crop_radius + 1) ** 2
        centers_bt = normalized_coords_to_patch_centers(
            gt_coords[:max_frames].unsqueeze(0),
            image_size=float(image_size),
            pixel_patch_size=patch_size,
            position_before_action=position_before_action,
            num_action_frames=max_frames,
        )
        _, view_spatial_h, view_spatial_w = gather_view_latent_patches(
            gt_video_latents[:max_frames].unsqueeze(0),
            centers_bt,
            crop_radius=view_crop_radius,
        )

    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=config.get("timestep_shift", 5.0))

    n_ar = num_actions_per_subj
    if max_ar_steps is not None:
        n_ar = max(0, min(num_actions_per_subj, max_ar_steps))

    view_patches = None
    with torch.no_grad():
        for frame_idx in range(n_ar):
            sample_scheduler.set_timesteps(sampling_steps, device=device)
            num_context = generated_video.shape[1]
            max_context = max_frames - 1

            if num_context > max_context:
                window_video = generated_video[:, -max_context:]
                window_coords = generated_coords[:, -max_context:]
                window_start = num_context - max_context
            else:
                window_video = generated_video
                window_coords = generated_coords
                window_start = 0

            num_context_in_window = window_video.shape[1]
            target_idx_in_window = num_context_in_window
            num_padding = max_frames - num_context_in_window - 1

            windowed_actions = torch.zeros(1, num_subjects, num_actions_per_subj, dtype=torch.long, device=device)
            action_mask = torch.ones(1, num_subjects, num_actions_per_subj, dtype=torch.float32, device=device)
            action_start = window_start
            action_end = min(action_start + num_actions_per_subj, n_action_steps)
            num_valid_actions = action_end - action_start
            if num_valid_actions > 0:
                for i in range(num_valid_actions):
                    for s in range(num_subjects):
                        windowed_actions[:, s, i] = action_sequences[s, action_start + i]
            for i in range(num_actions_per_subj):
                if i >= num_context_in_window:
                    windowed_actions[:, :, i] = 0
                    action_mask[:, :, i] = 0

            subj_mask = sample.get("subject_mask")
            if subj_mask is not None:
                sm = subj_mask.to(device)
                for s in range(num_subjects):
                    if not sm[s]:
                        windowed_actions[:, s, :] = 0
                        action_mask[:, s, :] = 0

            subject_ids = torch.arange(num_subjects, device=device, dtype=torch.long).unsqueeze(0)

            pred_positions = torch.zeros(1, num_subjects * num_actions_per_subj, 3, device=device, dtype=dtype)
            for s in range(num_subjects):
                for t in range(num_actions_per_subj):
                    idx = s * num_actions_per_subj + t
                    frame_in_window = t + 1
                    if t < num_context_in_window:
                        if position_before_action:
                            coord_idx = min(window_start + t, max_frames - 1)
                        else:
                            coord_idx = min(window_start + t + 1, max_frames - 1)
                        if view_token_mode:
                            coord = gt_coords[coord_idx, s]
                        else:
                            coord = generated_coords[0, coord_idx, s]
                        x_pix = (coord[0] * image_size).clamp(0, image_size - 1)
                        y_pix = (coord[1] * image_size).clamp(0, image_size - 1)
                        pred_positions[0, idx, 0] = frame_in_window
                        pred_positions[0, idx, 1] = (y_pix / effective_patch_size).clamp(0, max_patch_rope).float()
                        pred_positions[0, idx, 2] = (x_pix / effective_patch_size).clamp(0, max_patch_rope).float()

            conditioner_kwargs = dict(
                action_labels=windowed_actions, subject_ids=subject_ids,
                initial_positions=initial_positions, action_mask=action_mask,
            )
            if use_gt_positions_for_rope or num_action_classes > 5:
                window_frame_coords = torch.zeros(
                    1, max_frames, num_subjects, 2, device=device, dtype=dtype,
                )
                for i in range(num_context_in_window):
                    gi = min(window_start + i, max_frames - 1)
                    window_frame_coords[0, i] = gt_coords[gi]
                tgt_g = min(window_start + num_context_in_window, max_frames - 1)
                window_frame_coords[0, target_idx_in_window] = gt_coords[tgt_g]
                conditioner_kwargs["frame_coords"] = window_frame_coords

            embeddings, _ = conditioner(**conditioner_kwargs)
            conditional_dict = {"prompt_embeds": embeddings, "context_positions": pred_positions}
            if video_text_embeds is not None:
                conditional_dict["video_text_embeds"] = video_text_embeds

            sigma_0 = sample_scheduler.sigmas[sample_scheduler.timesteps == 0].item() if 0 in sample_scheduler.timesteps else 0.0148
            sigma_max = sample_scheduler.sigmas[0].item()
            max_timestep = sample_scheduler.timesteps[0].item()

            target_noise_video = torch.randn(1, 1, C, H, W, device=device, dtype=dtype)
            target_video = sigma_max * target_noise_video
            noisy_context_video = (1 - sigma_0) * window_video

            if num_padding > 0:
                padding_video = torch.randn(1, num_padding, C, H, W, device=device, dtype=dtype)
                video_latents = torch.cat([noisy_context_video, target_video, sigma_max * padding_video], dim=1)
            else:
                video_latents = torch.cat([noisy_context_video, target_video], dim=1)

            coord_latents = None
            if not view_token_mode:
                # window_coords carries: (x, y) for state_dim=2 OR (x, y, sin, cos) for state_dim=4.
                # Only scale the (x, y) channels by coord_scale_factor; (sin, cos) stay native [-1, 1].
                window_coords_scaled = window_coords.clone()
                window_coords_scaled[..., :2] = (window_coords[..., :2] - 0.5) * coord_scale_factor
                target_noise_coord = torch.randn(1, 1, num_subjects, state_dim, device=device, dtype=dtype)
                target_coord = sigma_max * target_noise_coord
                noisy_context_coords = (1 - sigma_0) * window_coords_scaled

                if num_padding > 0:
                    padding_coords = torch.randn(1, num_padding, num_subjects, state_dim, device=device, dtype=dtype)
                    coord_latents = torch.cat([noisy_context_coords, target_coord, sigma_max * padding_coords], dim=1)
                else:
                    coord_latents = torch.cat([noisy_context_coords, target_coord], dim=1)

            view_patches = None
            if view_token_mode:
                Dv = C * 2 * 2
                view_patches = torch.randn(1, max_frames, num_subjects, pv, Dv, device=device, dtype=dtype) * sigma_max
                for t in range(num_context_in_window):
                    g = window_start + t
                    lat_1 = window_video[:, t : t + 1]
                    c_1 = centers_bt[:, g : g + 1]
                    patch_g, _, _ = gather_view_latent_patches(lat_1, c_1, crop_radius=view_crop_radius)
                    view_patches[:, g] = (1.0 - sigma_0) * patch_g[:, 0]

            for i, t_val in enumerate(sample_scheduler.timesteps):
                timestep = torch.zeros(1, max_frames, device=device, dtype=torch.float32)
                timestep[:, :num_context_in_window] = 0
                timestep[:, target_idx_in_window] = t_val
                if num_padding > 0:
                    timestep[:, target_idx_in_window + 1:] = max_timestep

                gen_kw = dict(
                    noisy_image_or_video=video_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    coord_num_clean=torch.tensor([num_context_in_window], device=device),
                    zero_position_for_masked=zero_pos_masked,
                )
                if view_token_mode:
                    gen_kw["view_latent_patches"] = view_patches
                    gen_kw["view_spatial_h"] = view_spatial_h
                    gen_kw["view_spatial_w"] = view_spatial_w
                else:
                    gen_kw["subject_coords"] = coord_latents

                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = generator(**gen_kw)

                coord_flow = None
                view_flow = None
                view_x0_pred = None
                if view_token_mode and len(output) == 4:
                    video_flow, video_x0, view_flow, view_x0_pred = output
                elif len(output) == 3:
                    video_flow, video_x0, coord_flow = output
                elif len(output) == 2:
                    video_flow, video_x0 = output
                else:
                    video_flow = output
                    video_x0 = None

                sigma = sample_scheduler.sigmas[i]
                sigma_next = sample_scheduler.sigmas[i + 1] if i + 1 < len(sample_scheduler.sigmas) else 0.0
                dt = sigma_next - sigma
                is_last = (i == len(sample_scheduler.timesteps) - 1)

                tvf = video_flow[:, target_idx_in_window:target_idx_in_window + 1]
                tvl = video_latents[:, target_idx_in_window:target_idx_in_window + 1]
                tvd = video_x0[:, target_idx_in_window:target_idx_in_window + 1] if is_last and video_x0 is not None else tvl + dt * tvf
                video_latents = torch.cat([
                    video_latents[:, :target_idx_in_window], tvd,
                    video_latents[:, target_idx_in_window + 1:]
                    if target_idx_in_window + 1 < video_latents.shape[1]
                    else torch.zeros(1, 0, C, H, W, device=device, dtype=dtype),
                ], dim=1)

                if coord_flow is not None and coord_latents is not None:
                    cf = coord_flow.view(1, max_frames, num_subjects, state_dim)
                    tcf = cf[:, target_idx_in_window:target_idx_in_window + 1]
                    tcl = coord_latents[:, target_idx_in_window:target_idx_in_window + 1]
                    tcd = tcl + dt * tcf
                    coord_latents = torch.cat([
                        coord_latents[:, :target_idx_in_window], tcd,
                        coord_latents[:, target_idx_in_window + 1:]
                        if target_idx_in_window + 1 < coord_latents.shape[1]
                        else torch.zeros(1, 0, num_subjects, state_dim, device=device, dtype=dtype),
                    ], dim=1)

                if view_token_mode and view_flow is not None and view_patches is not None:
                    vf = view_flow.view(1, max_frames, num_subjects, pv, Dv)
                    tvf = vf[:, target_idx_in_window:target_idx_in_window + 1]
                    tvlv = view_patches[:, target_idx_in_window:target_idx_in_window + 1]
                    if is_last and view_x0_pred is not None:
                        tvxv = view_x0_pred.view(1, max_frames, num_subjects, pv, Dv)[
                            :, target_idx_in_window:target_idx_in_window + 1
                        ]
                    else:
                        tvxv = tvlv + dt * tvf
                    tail = (
                        view_patches[:, target_idx_in_window + 1:]
                        if target_idx_in_window + 1 < max_frames
                        else view_patches[:, :0]
                    )
                    view_patches = torch.cat([view_patches[:, :target_idx_in_window], tvxv, tail], dim=1)

            generated_video = torch.cat(
                [generated_video, video_latents[:, target_idx_in_window:target_idx_in_window + 1]], dim=1,
            )
            if not view_token_mode:
                denoised_scaled = coord_latents[:, target_idx_in_window:target_idx_in_window + 1]
                # Unscale only (x, y) channels — (sin, cos) channels are already in their native range.
                denoised_norm = denoised_scaled.clone()
                denoised_norm[..., :2] = (denoised_scaled[..., :2] / coord_scale_factor) + 0.5
                generated_coords = torch.cat([generated_coords, denoised_norm], dim=1)

    if view_token_mode:
        Tgen = generated_video.shape[1]
        generated_coords = gt_coords[:Tgen].unsqueeze(0).to(device=device, dtype=dtype).clone()

    if (
        view_token_mode
        and output_view_latent_decode_path is not None
        and view_patches is not None
    ):
        Tgen = int(generated_video.shape[1])
        sm_t = sample.get("subject_mask")
        sm_dev = sm_t.to(device) if sm_t is not None else None
        z_view = scatter_view_flat_patches_to_latent(
            view_patches[:, :Tgen].float(),
            centers_bt[:, :Tgen],
            latent_hw=(H, W),
            crop_radius=view_crop_radius,
            subject_mask=sm_dev,
        )
        with torch.no_grad():
            view_dec = vae.decode_to_pixel(z_view.float())
        view_dec_np = np.transpose(view_dec[0].cpu().numpy(), (0, 2, 3, 1))
        view_dec_np = np.clip((view_dec_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
        imageio.mimsave(output_view_latent_decode_path, list(view_dec_np), fps=8, loop=0)
        print(f"  Saved view-latent decode GIF: {output_view_latent_decode_path}")

    if (
        view_token_mode
        and output_view_latent_decode_separate_path is not None
        and view_patches is not None
    ):
        Tgen_sep = int(generated_video.shape[1])
        sm_t = sample.get("subject_mask")
        sm_dev = sm_t.to(device) if sm_t is not None else None
        patch_kh, patch_kw = 2, 2
        save_separate_view_patch_decode_gif(
            vae,
            view_patches[:, :Tgen_sep].float(),
            Tgen=Tgen_sep,
            num_subjects=num_subjects,
            patch_count=pv,
            C=C,
            kh=patch_kh,
            kw=patch_kw,
            device=device,
            subject_mask=sm_dev,
            out_path=output_view_latent_decode_separate_path,
        )

    with torch.no_grad():
        video_frames = vae.decode_to_pixel(generated_video.float())
        video_frames = np.transpose(video_frames[0].cpu().numpy(), (0, 2, 3, 1))
        video_frames = np.clip((video_frames + 1) / 2 * 255, 0, 255).astype(np.uint8)

        gt_video_frames = vae.decode_to_pixel(gt_video_latents.float().unsqueeze(0))
        gt_video_frames = np.transpose(gt_video_frames[0].cpu().numpy(), (0, 2, 3, 1))
        gt_video_frames = np.clip((gt_video_frames + 1) / 2 * 255, 0, 255).astype(np.uint8)

    pred_state = generated_coords[0].cpu().float().numpy()  # [T, S, state_dim]
    # Position predictions are always the first 2 channels.
    pred_coords = pred_state[..., :2]
    gt_coords_np = gt_coords.cpu().float().numpy()

    # Recover predicted orientation when state_dim == 4 by snapping (sin, cos) to nearest cardinal.
    pred_orient = None
    if coord_orient and pred_state.shape[-1] == 4:
        pred_sin = pred_state[..., 2]
        pred_cos = pred_state[..., 3]
        # angle = atan2(sin, cos), then nearest of {0, pi/2, pi, 3pi/2} -> {0, 1, 2, 3}.
        # cardinal_idx = round(angle / (pi/2)) mod 4.
        ang = np.arctan2(pred_sin, pred_cos)
        pred_orient = np.mod(np.round(ang / (math.pi / 2.0)).astype(np.int64), 4)

    diff = pred_coords[1:] - gt_coords_np[1:pred_coords.shape[0]]
    l2 = np.sqrt((diff ** 2).sum(axis=-1)).mean()

    return {
        "pred_coords": pred_coords,
        "pred_orient": pred_orient,  # None when state_dim==2
        "gt_coords": gt_coords_np,
        "video_frames": video_frames,
        "gt_video_frames": gt_video_frames,
        "orientations": dataset_orientations,
        "actions": action_sequences.cpu().numpy(),
        "l2_error": l2,
    }


# === GIF preview helpers ==================================================

ACTION_NAMES_7 = {0: "NOOP", 1: "FWD", 2: "BACK", 3: "S_L", 4: "S_R", 5: "T_L", 6: "T_R"}
ORIENT_DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
SUBJECT_COLORS = [
    (0, 0, 255), (255, 100, 0), (0, 200, 0), (200, 0, 200),
    (0, 200, 200), (200, 200, 0), (128, 128, 255), (255, 180, 100),
]


def expand_coords_vae_style(coords, target_len, src_len):
    out = []
    for i in range(target_len):
        latent_idx = 0 if i == 0 else min((i - 1) // 4 + 1, src_len - 1)
        out.append(coords[latent_idx])
    return np.array(out)


def save_combined_gif(video_frames, gt_video_frames, pred_coords, gt_coords,
                      orientations, num_subjects, image_size, actions,
                      num_action_classes, out_path):
    """Save side-by-side GIF: generated (with pred circles + GT crosses) | GT."""
    npf = video_frames.shape[0]
    nlf = pred_coords.shape[0]
    pred_exp = expand_coords_vae_style(pred_coords, npf, nlf)
    gt_exp = expand_coords_vae_style(gt_coords, npf, nlf)

    orient_exp = None
    if orientations is not None and len(orientations) > 0:
        orient_exp = np.zeros(
            (npf, orientations.shape[1] if orientations.ndim > 1 else num_subjects), dtype=np.int64,
        )
        for i in range(npf):
            li = 0 if i == 0 else min((i - 1) // 4 + 1, nlf - 1)
            if li < len(orientations):
                orient_exp[i] = orientations[li]

    action_names = ACTION_NAMES_7 if num_action_classes >= 7 else {i: str(i) for i in range(num_action_classes)}
    combined = []
    for i in range(npf):
        frame = video_frames[i].copy()
        ai = -1 if i == 0 else min((i - 1) // 4, actions.shape[1] - 1)
        if ai < 0:
            label = f"f{i}"
        else:
            a0 = action_names.get(int(actions[0, ai]), "?")
            a1 = action_names.get(int(actions[1, ai]), "?") if actions.shape[0] > 1 else "?"
            label = f"f{i} {a0}|{a1}"
        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        for s in range(num_subjects):
            c = SUBJECT_COLORS[s % len(SUBJECT_COLORS)]
            gx, gy = int(gt_exp[i, s, 0] * image_size), int(gt_exp[i, s, 1] * image_size)
            cv2.line(frame, (gx - 6, gy), (gx + 6, gy), c, 2)
            cv2.line(frame, (gx, gy - 6), (gx, gy + 6), c, 2)
            if orient_exp is not None and s < orient_exp.shape[1]:
                dx, dy = ORIENT_DELTAS.get(int(orient_exp[i, s]), (0, 0))
                cv2.arrowedLine(frame, (gx, gy), (gx + dx * 12, gy + dy * 12), c, 2, tipLength=0.4)
            px, py = int(pred_exp[i, s, 0] * image_size), int(pred_exp[i, s, 1] * image_size)
            cv2.circle(frame, (px, py), 6, c, 2)

        gt_frame = gt_video_frames[min(i, len(gt_video_frames) - 1)].copy()
        for s in range(num_subjects):
            c = SUBJECT_COLORS[s % len(SUBJECT_COLORS)]
            gx, gy = int(gt_exp[i, s, 0] * image_size), int(gt_exp[i, s, 1] * image_size)
            cv2.line(gt_frame, (gx - 6, gy), (gx + 6, gy), c, 2)
            cv2.line(gt_frame, (gx, gy - 6), (gx, gy + 6), c, 2)
            if orient_exp is not None and s < orient_exp.shape[1]:
                dx, dy = ORIENT_DELTAS.get(int(orient_exp[i, s]), (0, 0))
                cv2.arrowedLine(gt_frame, (gx, gy), (gx + dx * 12, gy + dy * 12), c, 2, tipLength=0.4)

        combined.append(np.concatenate([frame, gt_frame], axis=1))

    imageio.mimsave(out_path, combined, fps=8, loop=0)
    print(f"  Saved GIF: {out_path}")


def _view_mosaic_one_frame(
    frame_hwc: np.ndarray,
    gx_norm: float,
    gy_norm: float,
    image_size: int,
    eff_px: int,
    n_bins: int,
) -> np.ndarray:
    """3×3 grid of eff_px×eff_px tiles on the conditioner bin grid around (gx,gy) in norm coords."""
    gx = float(gx_norm) * image_size
    gy = float(gy_norm) * image_size
    ph0 = int(gy / eff_px)
    pw0 = int(gx / eff_px)
    ph0 = max(0, min(n_bins - 1, ph0))
    pw0 = max(0, min(n_bins - 1, pw0))
    H, W, _ = frame_hwc.shape
    mosaic = np.zeros((3 * eff_px, 3 * eff_px, 3), dtype=np.uint8)
    for di in range(3):
        for dj in range(3):
            ph = ph0 + (di - 1)
            pw = pw0 + (dj - 1)
            if ph < 0 or ph >= n_bins or pw < 0 or pw >= n_bins:
                continue
            y0, y1 = ph * eff_px, (ph + 1) * eff_px
            x0, x1 = pw * eff_px, (pw + 1) * eff_px
            y0, y1 = max(0, y0), min(H, y1)
            x0, x1 = max(0, x0), min(W, x1)
            tile = frame_hwc[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            # If clipped at border, paste as much as fits in the mosaic cell
            tH, tW = tile.shape[0], tile.shape[1]
            out_y0, out_x0 = di * eff_px, dj * eff_px
            mosaic[out_y0 : out_y0 + tH, out_x0 : out_x0 + tW] = tile
    return mosaic


def save_view_mosaic_gif(
    video_frames: np.ndarray,
    gt_video_frames: np.ndarray,
    pred_coords: np.ndarray,
    gt_coords: np.ndarray,
    num_subjects: int,
    image_size: int,
    latent_patch_size: int,
    out_path: str,
    subject_mask=None,
):
    """GIF: each frame = [gen mosaics | GT mosaics] stacked vertically (3×3 tiles per subject)."""
    npf = video_frames.shape[0]
    nlf = pred_coords.shape[0]
    pred_exp = expand_coords_vae_style(pred_coords, npf, nlf)
    gt_exp = expand_coords_vae_style(gt_coords, npf, nlf)
    eff_px = latent_patch_size * 2
    n_bins = max(1, image_size // eff_px)

    gap = 4
    frames_out = []
    for i in range(npf):
        gen_fr = video_frames[i]
        gt_fr = gt_video_frames[min(i, len(gt_video_frames) - 1)]
        gen_panels = []
        gt_panels = []
        for s in range(num_subjects):
            if subject_mask is not None and s < len(subject_mask) and not bool(subject_mask[s]):
                continue
            gen_panels.append(
                _view_mosaic_one_frame(
                    gen_fr, pred_exp[i, s, 0], pred_exp[i, s, 1], image_size, eff_px, n_bins,
                )
            )
            gt_panels.append(
                _view_mosaic_one_frame(
                    gt_fr, gt_exp[i, s, 0], gt_exp[i, s, 1], image_size, eff_px, n_bins,
                )
            )
        if not gen_panels:
            continue
        row_g = np.concatenate(gen_panels, axis=1)
        row_gt = np.concatenate(gt_panels, axis=1)
        w = max(row_g.shape[1], row_gt.shape[1])
        if row_g.shape[1] < w:
            pad = np.zeros((row_g.shape[0], w - row_g.shape[1], 3), dtype=np.uint8)
            row_g = np.concatenate([row_g, pad], axis=1)
        if row_gt.shape[1] < w:
            pad = np.zeros((row_gt.shape[0], w - row_gt.shape[1], 3), dtype=np.uint8)
            row_gt = np.concatenate([row_gt, pad], axis=1)
        sep = np.full((gap, w, 3), 255, dtype=np.uint8)
        combo = np.concatenate([row_g, sep, row_gt], axis=0)
        frames_out.append(combo)

    if frames_out:
        imageio.mimsave(out_path, frames_out, fps=8, loop=0)
        print(f"  Saved view-mosaic GIF: {out_path}")


def save_separate_view_patch_decode_gif(
    vae,
    view_btspd: torch.Tensor,
    *,
    Tgen: int,
    num_subjects: int,
    patch_count: int,
    C: int,
    kh: int,
    kw: int,
    device,
    subject_mask,
    out_path: str,
    latent_canvas: int = 32,
    thumb_px: int = 56,
):
    """
    Decode each view micro-patch in isolation: center a [C,kh,kw] tile in a zero latent canvas,
    VAE decode, resize to thumb_px, arrange in a side×side grid per subject (side² = patch_count).
    """
    side = int(round(patch_count ** 0.5))
    assert side * side == patch_count, (patch_count, side)
    top = (latent_canvas - kh) // 2
    left = (latent_canvas - kw) // 2
    if top < 0 or left < 0:
        raise ValueError(f"latent_canvas={latent_canvas} too small for kh,kw={kh},{kw}")

    frames_out = []
    for t in range(Tgen):
        subj_panels = []
        for s in range(num_subjects):
            if subject_mask is not None and s < len(subject_mask) and not bool(subject_mask[s]):
                continue
            grid_np = np.zeros((side * thumb_px, side * thumb_px, 3), dtype=np.uint8)
            for p_idx in range(patch_count):
                last_err = None
                dec = None
                for canvas_try in (latent_canvas, 64):
                    tt = (canvas_try - kh) // 2
                    ll = (canvas_try - kw) // 2
                    z = torch.zeros(1, 1, C, canvas_try, canvas_try, device=device, dtype=torch.float32)
                    z[0, 0, :, tt : tt + kh, ll : ll + kw] = (
                        view_btspd[0, t, s, p_idx].float().view(C, kh, kw)
                    )
                    try:
                        with torch.no_grad():
                            dec = vae.decode_to_pixel(z)
                        last_err = None
                        break
                    except RuntimeError as e:
                        last_err = e
                        dec = None
                if dec is None:
                    raise RuntimeError(
                        f"VAE decode failed for view patch (t={t},s={s},p={p_idx}): {last_err}"
                    )
                fr = dec[0, 0].clamp(-1, 1).cpu().numpy()
                im = np.transpose(fr, (1, 2, 0))
                im = np.clip((im + 1) / 2 * 255, 0, 255).astype(np.uint8)
                im = cv2.resize(im, (thumb_px, thumb_px), interpolation=cv2.INTER_AREA)
                ri, ci = divmod(p_idx, side)
                grid_np[ri * thumb_px : (ri + 1) * thumb_px, ci * thumb_px : (ci + 1) * thumb_px] = im
            subj_panels.append(grid_np)
        if not subj_panels:
            continue
        gap = 6
        h = subj_panels[0].shape[0]
        pieces = []
        for j, p in enumerate(subj_panels):
            if j:
                pieces.append(np.full((h, gap, 3), 255, dtype=np.uint8))
            pieces.append(p)
        frames_out.append(np.concatenate(pieces, axis=1))

    if frames_out:
        imageio.mimsave(out_path, frames_out, fps=8, loop=0)
        print(f"  Saved per-patch view decode GIF: {out_path}")


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_base, args.name)
    eval_data_dir = os.path.join(output_dir, "eval_data")
    val_npz_dir = os.path.join(output_dir, "val_npz")
    os.makedirs(eval_data_dir, exist_ok=True)
    os.makedirs(val_npz_dir, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = OmegaConf.load(args.config)

    print(f"{'=' * 60}")
    print(f"  Batch Ablation Eval: {args.name}")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    if args.multi_game and args.one_sample_per_game:
        print(f"  Samples:    one per game (per_game_local_idx={args.per_game_local_idx})")
    else:
        print(f"  Samples:    {args.num_samples} (starting at {args.start_idx})")
    print(f"  Output:     {output_dir}")
    if int(getattr(args, "eval_num_shards", 1) or 1) > 1:
        print(
            f"  Sharded eval: shard {args.eval_shard_id}/{args.eval_num_shards} "
            f"(same output dir; run classifier once after all shards finish)"
        )
    if args.initial_coord_noise_sigma > 0:
        print(
            f"  Initial coord noise σ (norm [0,1]): {args.initial_coord_noise_sigma}",
        )
    if args.multi_game:
        print(f"  Multi-game LMDB: {args.dataset}")
        if args.game_name:
            print(f"  game_name filter: {args.game_name}")
    if args.max_ar_steps is not None:
        print(f"  DEBUG max_ar_steps: {args.max_ar_steps} (partial rollout; metrics not comparable to full eval)")
    if args.t5_minimal_cache:
        print("  DEBUG t5_minimal_cache: on")
    print(f"{'=' * 60}")

    eval_image_size = int(
        config.get("synthetic_height", config.get("synthetic_width", args.image_size))
    )

    print("\nLoading VAE...")
    vae = WanVAEWrapper().to(device)

    print("Loading model...")
    model_kwargs = dict(config.get("model_kwargs", {}))
    model_kwargs.pop("timestep_shift", None)
    num_subjects = config.get("num_subjects", 2)
    coord_mode = bool(config.get("coord_token_mode", False))
    view_mode = bool(config.get("view_token_mode", False))

    if coord_mode:
        model_kwargs["coord_token_mode"] = True
        model_kwargs["coord_temporal_rope"] = config.get("coord_temporal_rope", False)
        model_kwargs["coord_spatial_rope"] = config.get("coord_spatial_rope", False)
        model_kwargs["coord_layernorm"] = config.get("coord_layernorm", False)
        model_kwargs["num_subjects"] = num_subjects
        model_kwargs["coord_scale_factor"] = config.get("coord_scale_factor", 1.0)
        model_kwargs["coord_fourier_encoding"] = config.get("coord_fourier_encoding", False)
        model_kwargs["coord_fourier_frequencies"] = config.get("coord_fourier_frequencies", 10)
        model_kwargs["coord_direct_embedding"] = config.get("coord_direct_embedding", False)
        model_kwargs["coord_no_encoder_decoder"] = config.get("coord_no_encoder_decoder", False)
        model_kwargs["coord_orientation_in_state"] = config.get("coord_orientation_in_state", False)
    elif view_mode:
        model_kwargs["view_token_mode"] = True
        model_kwargs["view_crop_radius"] = config.get("view_crop_radius", 1)
        model_kwargs["num_subjects"] = num_subjects
    else:
        model_kwargs["num_subjects"] = num_subjects

    if config.get("cross_attn_rope", False):
        model_kwargs["cross_attn_rope"] = True

    # Config `model_kwargs` may repeat keys we pass explicitly below.
    for _dup in (
        "coord_token_mode",
        "view_token_mode",
        "view_crop_radius",
        "per_frame_timesteps",
        "subject_isolation_self_attn",
        "use_flex_attention",
        "per_frame_cross_attn",
        "cross_attn_subject_isolation",
        "image_or_video_shape",
    ):
        model_kwargs.pop(_dup, None)

    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=config.get("timestep_shift", 5.0),
        coord_token_mode=coord_mode,
        view_token_mode=view_mode,
        view_crop_radius=config.get("view_crop_radius", 1),
        per_frame_timesteps=config.get("per_frame_timesteps", False),
        subject_isolation_self_attn=config.get("subject_isolation_self_attn", False),
        use_flex_attention=config.get("use_flex_attention", True),
        per_frame_cross_attn=config.get("per_frame_cross_attn", False),
        cross_attn_subject_isolation=config.get("cross_attn_subject_isolation", True),
        image_or_video_shape=config.get("image_or_video_shape", None),
        **model_kwargs,
    ).to(device)
    generator.eval()
    generator = load_checkpoint(generator, args.checkpoint, use_ema=args.use_ema)
    # Action-only `context` is already model_dim (1536); skip T5 projection there (model checks dim).
    # For `use_game_descriptions`, `video_text_embeds` are raw T5 [B,512,4096] and **must** use the
    # checkpoint's Linear(4096→1536). Setting Identity before load prevented those weights loading and
    # caused runtime shape errors for multi-game 153-style evals.
    if not config.get("use_game_descriptions", False):
        generator.model.text_embedding = torch.nn.Identity()

    game_text_cache = {}
    if config.get("use_game_descriptions", False):
        print("Loading T5 for game-description video cross-attn (null + description cache)...")
        text_encoder = WanTextEncoder().to(device)
        with torch.no_grad():
            null_dict = text_encoder([""])
            null_embed_t5 = null_dict["prompt_embeds"].to(device, dtype)
        generator.model.set_null_embedding_for_cross_attn(null_embed_t5)
        if args.t5_minimal_cache:
            print("   (--t5_minimal_cache: skipping bulk GAME_DESCRIPTIONS T5 cache)")
        else:
            from utils.game_descriptions import GAME_DESCRIPTIONS

            unique_prompts = list(set(GAME_DESCRIPTIONS.values()))
            for i in range(0, len(unique_prompts), 8):
                batch = unique_prompts[i : i + 8]
                with torch.no_grad():
                    embeds = text_encoder(batch)["prompt_embeds"].to(device, dtype)
                for j, prompt in enumerate(batch):
                    game_text_cache[prompt] = embeds[j : j + 1].cpu()
            print(f"   Cached {len(game_text_cache)} game-description T5 embeddings")
        del text_encoder
        torch.cuda.empty_cache()
        if args.t5_minimal_cache:
            print("   Per-sample prompts will be T5-encoded on demand if missing from cache.")

    num_action_classes = config.get("num_action_classes", 5)
    coord_scale_factor = config.get("coord_scale_factor", 4.0)

    conditioner = ActionOnlySpatialConditioner(
        num_action_classes=num_action_classes,
        num_subjects=num_subjects,
        model_dim=generator.model.dim,
        patch_size=config.get("latent_patch_size", 8),
        speed=config.get("position_speed", 16),
        zero_position_for_masked=config.get("zero_position_for_masked", False),
        position_before_action=config.get("position_before_action", True),
        image_size=float(eval_image_size),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    for key in ["multi_subject_conditioner", "spatial_conditioner"]:
        if key in ckpt:
            conditioner.load_state_dict(ckpt[key], strict=False)
            break

    print("Loading dataset...")
    if args.multi_game:
        dataset = MultiGameLMDBDataset(
            args.dataset,
            max_pair=int(1e8),
            max_subjects=config.get("num_subjects", 8),
            max_action_classes=config.get("num_action_classes", 25),
            use_game_descriptions=True,
        )
    else:
        dataset = PositionTokenLMDBDataset(data_path=args.dataset, max_pair=1000)

    if args.multi_game and args.one_sample_per_game:
        if args.game_name:
            raise ValueError("--one_sample_per_game cannot be used with --game_name")
        offset = 0
        sample_indices = []
        for meta in dataset.shard_metadata:
            n = int(meta["num_samples"])
            li = int(args.per_game_local_idx)
            if li < 0 or li >= n:
                raise ValueError(
                    f"--per_game_local_idx={li} invalid for game {meta.get('game_name')!r} "
                    f"(num_samples={n})"
                )
            sample_indices.append(offset + li)
            offset += n
        print(
            f"  one_sample_per_game: {len(sample_indices)} clips "
            f"(per_game_local_idx={args.per_game_local_idx})"
        )
    elif args.multi_game and args.game_name:
        # O(num_shards) via metadata — do not scan all LMDB rows (would load ~92k clips).
        hit_indices = []
        offset = 0
        for meta in dataset.shard_metadata:
            if meta["game_name"] == args.game_name:
                hit_indices.extend(range(offset, offset + meta["num_samples"]))
            offset += meta["num_samples"]
        if not hit_indices:
            raise ValueError(f"No samples found for game_name={args.game_name!r}")
        sample_indices = hit_indices[args.start_idx : args.start_idx + args.num_samples]
        if len(sample_indices) < args.num_samples:
            raise ValueError(
                f"Requested {args.num_samples} samples from start_idx={args.start_idx} but only "
                f"{len(sample_indices)} matches for game_name={args.game_name!r}"
            )
    else:
        sample_indices = list(
            range(args.start_idx, args.start_idx + args.num_samples)
        )

    eval_num_shards = int(getattr(args, "eval_num_shards", 1) or 1)
    eval_shard_id = int(getattr(args, "eval_shard_id", 0) or 0)
    if eval_num_shards < 1:
        raise ValueError("--eval_num_shards must be >= 1")
    if eval_shard_id < 0 or eval_shard_id >= eval_num_shards:
        raise ValueError("--eval_shard_id must satisfy 0 <= id < eval_num_shards")
    if eval_num_shards > 1:
        n_tot = len(sample_indices)
        chunk = (n_tot + eval_num_shards - 1) // eval_num_shards
        s0 = eval_shard_id * chunk
        s1 = min(s0 + chunk, n_tot)
        sample_indices = sample_indices[s0:s1]
        print(
            f"  eval shard {eval_shard_id}/{eval_num_shards}: samples {s0}..{s1 - 1} "
            f"({len(sample_indices)} clips)"
        )
        if not args.skip_classifier:
            print(
                "  (forcing --skip_classifier: run scripts/eval/finish_ablation_eval_classifier.sh "
                "on this output dir after all shards finish)"
            )
            args.skip_classifier = True

    print(f"\nModel loaded. Running inference on {len(sample_indices)} samples...\n")

    l2_errors = []
    for run_i, idx in enumerate(sample_indices):
        print(f"--- Sample {run_i+1}/{len(sample_indices)} (idx={idx}) ---")
        video_text_embeds = None
        if game_text_cache:
            prompt_str = dataset[idx].get("prompts")
            if isinstance(prompt_str, str):
                if prompt_str in game_text_cache:
                    video_text_embeds = game_text_cache[prompt_str].to(device, dtype)
                else:
                    te_one = WanTextEncoder().to(device)
                    with torch.no_grad():
                        video_text_embeds = te_one([prompt_str])["prompt_embeds"].to(device, dtype)
                    del te_one
                    game_text_cache[prompt_str] = video_text_embeds.cpu()
                    torch.cuda.empty_cache()
        try:
            vd_path = None
            vd_sep_path = None
            if view_mode and args.save_view_latent_decode:
                vd_path = os.path.join(output_dir, f"sample{idx}_view_latent_decoded.gif")
            if view_mode and args.save_view_latent_decode_separate:
                vd_sep_path = os.path.join(output_dir, f"sample{idx}_view_patches_separate.gif")
            result = run_inference_for_sample(
                idx, dataset, config, generator, vae, conditioner, device, dtype,
                num_subjects, num_action_classes, coord_scale_factor,
                args.sampling_steps, eval_image_size, output_dir, args.seed,
                initial_coord_noise_sigma=args.initial_coord_noise_sigma,
                video_text_embeds=video_text_embeds,
                max_ar_steps=args.max_ar_steps,
                output_view_latent_decode_path=vd_path,
                output_view_latent_decode_separate_path=vd_sep_path,
            )
        except Exception as e:
            print(f"  Sample {idx} FAILED: {e}")
            continue

        l2_errors.append(result["l2_error"])
        print(f"  L2 error: {result['l2_error']:.4f}")

        sample_name = f"coins_s{idx}.npz"
        gt_coords_pixel = result["gt_coords"].copy()
        if gt_coords_pixel.max() <= 1.0:
            gt_coords_pixel *= eval_image_size
        gen_coords_pixel = result["pred_coords"].copy() * eval_image_size

        eval_npz_kwargs = dict(
            gen_frames=result["video_frames"],
            gt_frames=result["gt_video_frames"],
            gt_coords=gt_coords_pixel,
            gen_coords=gen_coords_pixel,
            num_subjects_real=np.int64(num_subjects),
            actions=result["actions"],
        )
        # When coord_orientation_in_state, also save the model-predicted orientation
        # alongside the standard fields. Downstream eval (eval_classifier_metrics.py)
        # still uses the classifier-derived orient for fair comparison; this is a bonus.
        if result.get("pred_orient") is not None:
            eval_npz_kwargs["pred_orient"] = result["pred_orient"]
        np.savez(os.path.join(eval_data_dir, sample_name), **eval_npz_kwargs)

        gt_orient = (
            result["orientations"]
            if result["orientations"] is not None
            else np.zeros((result["pred_coords"].shape[0], num_subjects), dtype=np.int64)
        )
        _gn = "coins"
        if args.multi_game:
            _gn = str(dataset[idx]["game_name"])
        np.savez(os.path.join(val_npz_dir, sample_name),
                 pixel_frames=result["gt_video_frames"],
                 frame_coords=gt_coords_pixel,
                 orientations=gt_orient,
                 actions=result["actions"],
                 num_subjects=np.int64(num_subjects),
                 game_name=_gn)

        if not args.no_gif:
            try:
                save_combined_gif(
                    result["video_frames"], result["gt_video_frames"],
                    result["pred_coords"], result["gt_coords"],
                    gt_orient, num_subjects, eval_image_size,
                    result["actions"], num_action_classes,
                    os.path.join(output_dir, f"sample{idx}_combined.gif"),
                )
            except Exception as e:
                print(f"  GIF save failed: {e}")

        if args.save_view_mosaic and view_mode:
            try:
                sm = None
                if args.multi_game:
                    sm_t = dataset[idx].get("subject_mask")
                    if sm_t is not None:
                        sm = sm_t.cpu().numpy() if torch.is_tensor(sm_t) else np.asarray(sm_t)
                save_view_mosaic_gif(
                    result["video_frames"], result["gt_video_frames"],
                    result["pred_coords"], result["gt_coords"],
                    num_subjects, eval_image_size,
                    int(config.get("latent_patch_size", 8)),
                    os.path.join(output_dir, f"sample{idx}_view_mosaic.gif"),
                    subject_mask=sm,
                )
            except Exception as e:
                print(f"  View-mosaic GIF failed: {e}")

    print(f"\n{'=' * 60}")
    if l2_errors:
        print(
            f"  Mean L2 error: {np.mean(l2_errors):.4f}  "
            f"({len(l2_errors)}/{len(sample_indices)} samples)"
        )
    else:
        print("  No successful samples")
    print(f"{'=' * 60}")

    if args.skip_classifier:
        print("\nSkipping classifier metrics (--skip_classifier).")
        print(f"\nDone! Outputs: {output_dir}/ (GIF + eval_data/)")
        return

    print(f"\nRunning classifier metrics...")
    metrics_script = os.path.join(REPO_ROOT, "scripts/eval/eval_classifier_metrics.py")
    cmd = [
        sys.executable, metrics_script,
        "--gen_dir", output_dir,
        "--val_dir", val_npz_dir,
        "--classifier", args.classifier,
        "--tile_size", str(args.tile_size),
        "--image_size", str(eval_image_size),
        "--skip", "allelopathic_harvest__open",
    ]
    subprocess.run(cmd)

    print(f"\nDone! Results: {output_dir}/classifier_action_metrics.{{txt,json}}")


if __name__ == "__main__":
    main()
