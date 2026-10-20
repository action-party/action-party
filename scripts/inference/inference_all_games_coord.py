"""
Run coord diffusion inference on all 46 games at a given exp-139 checkpoint.
Produces per-game comparison GIFs (GT left, generated right with coord markers) and summary HTML.

Usage:
  CUDA_VISIBLE_DEVICES=0 python inference_all_games_coord.py \
    --config_path configs/exp-139-multi-game-coord-512.yaml \
    --checkpoint_path checkpoints/exp-139-multi-game-coord-512/checkpoint_model_001500/model.pt
"""

import argparse
import os
import sys

# Add repo root (two dirs up from scripts/inference/) so `utils.*` / `model.*` imports resolve.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import numpy as np
import torch
import imageio
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper, WanTextEncoder
from utils.dataset import MultiGameLMDBDataset
from model.action_only_conditioner import ActionOnlySpatialConditioner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--sample_offset", type=int, default=50,
                        help="Starting offset within each game shard")
    parser.add_argument("--num_offsets", type=int, default=1,
                        help="Number of offsets to iterate (runs sample_offset, sample_offset+1, ...)")
    parser.add_argument("--num_ar_steps", type=int, default=None,
                        help="Number of AR steps (new latent frames to generate). "
                             "Default: max_frames-1 from config (4). "
                             "For longer generation, e.g. 20, actions beyond GT are randomly sampled.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip games that already have output GIF files")
    parser.add_argument("--validation_dir", type=str, default=None,
                        help="Path to validation npz dir (replaces training LMDB)")
    parser.add_argument("--max_samples_per_game", type=int, default=None,
                        help="Limit number of samples per game (e.g. 1 for s0 only)")
    parser.add_argument("--shard_id", type=int, default=None,
                        help="GPU shard index (0-based) for multi-GPU parallel inference")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards (GPUs) for parallel inference")
    parser.add_argument("--window_frames", type=int, default=None,
                        help="Override model window size (latent frames). Use 5 for sliding-window "
                             "AR on models trained with longer sequences (e.g. 21-frame 147 config)")
    parser.add_argument("--games", type=str, default=None,
                        help="Comma-separated list of game names to evaluate (default: all)")
    return parser.parse_args()


def load_validation_samples(validation_dir):
    """Load validation npz files grouped by game."""
    import glob as _glob, re
    game_samples = {}
    for path in sorted(_glob.glob(os.path.join(validation_dir, "*.npz"))):
        fname = os.path.basename(path)
        m = re.match(r"(.+)_s(\d+)\.npz$", fname)
        if not m:
            continue
        game, si = m.group(1), int(m.group(2))
        game_samples.setdefault(game, []).append((si, path))
    return game_samples


def load_checkpoint(generator, conditioner, checkpoint_path, use_ema=False):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if use_ema and 'generator_ema' in checkpoint:
        print("  Using EMA weights")
        state_dict = checkpoint['generator_ema']
    elif 'generator' in checkpoint:
        state_dict = checkpoint['generator']
    else:
        state_dict = checkpoint

    new_sd = {}
    for k, v in state_dict.items():
        new_k = k.replace('._fsdp_wrapped_module.', '.').replace('_fsdp_wrapped_module.', '')
        new_sd[new_k] = v

    missing, unexpected = generator.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  Generator missing keys: {missing}")

    if 'multi_subject_conditioner' in checkpoint:
        cond_sd = checkpoint['multi_subject_conditioner']
        new_cond_sd = {}
        for k, v in cond_sd.items():
            new_k = k.replace('._fsdp_wrapped_module.', '.').replace('_fsdp_wrapped_module.', '')
            new_cond_sd[new_k] = v
        m, u = conditioner.load_state_dict(new_cond_sd, strict=False)
        if m:
            print(f"  Conditioner missing keys: {m}")
    elif 'spatial_conditioner' in checkpoint:
        conditioner.load_state_dict(checkpoint['spatial_conditioner'], strict=False)
    else:
        print("  WARNING: No conditioner weights in checkpoint!")

    return generator, conditioner


def generate_ar_coord(generator, conditioner, initial_latents, initial_coords_norm,
                      action_sequences, frame_coords_gt, prompt_embeds_proj,
                      null_embed_t5, num_subjects_real, config, sample_scheduler,
                      max_frames, channels, height, width, device, dtype,
                      num_ar_steps=None, game_action_count=None):
    """
    AR generation with coordinate diffusion.
    
    Args:
        initial_latents: [1, 1, C, H, W]
        initial_coords_norm: [1, 1, S, 2] in [0,1]
        action_sequences: [1, S, num_actions_gt] GT actions (may be shorter than num_ar_steps)
        frame_coords_gt: [1, T, S, 2] in [0,1] for conditioner RoPE
        prompt_embeds_proj: [1, 512, 4096] raw T5 game text
        null_embed_t5: [1, 512, 4096] raw T5 null text
        num_subjects_real: int, actual number of subjects (before padding)
        num_ar_steps: int, total new latent frames to generate. If None, uses max_frames-1.
    """
    num_subjects = config.get("num_subjects", 8)
    coord_scale = config.get("coord_scale_factor", 4.0)
    total_ar_steps = num_ar_steps if num_ar_steps is not None else (max_frames - 1)
    image_size = config.get("synthetic_height", 512)

    num_window_actions = max_frames - 1  # actions per model window (fixed by architecture)

    # Extend action_sequences to cover all AR steps if needed
    gt_action_len = action_sequences.shape[2]
    if total_ar_steps > gt_action_len:
        n_act = game_action_count if game_action_count is not None else config.get("num_action_classes", 25)
        extra_steps = total_ar_steps - gt_action_len
        extra_actions = torch.randint(
            0, n_act, (1, num_subjects, extra_steps),
            device=device, dtype=torch.long,
        )
        # Zero out padded subjects' extra actions
        for s in range(num_subjects_real, num_subjects):
            extra_actions[:, s, :] = 0
        action_sequences = torch.cat([action_sequences, extra_actions], dim=2)

    generated_video = initial_latents.clone()
    generated_coords = initial_coords_norm.clone()

    for frame_idx in range(total_ar_steps):
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

        num_ctx = window_video.shape[1]
        target_idx = num_ctx
        num_padding = max_frames - num_ctx - 1

        # Build action conditioning with mask
        windowed_actions = torch.zeros(1, num_subjects, num_window_actions, dtype=torch.long, device=device)
        action_mask = torch.zeros(1, num_subjects, num_window_actions, dtype=torch.float32, device=device)
        for i in range(num_window_actions):
            abs_action_idx = window_start + i
            if abs_action_idx < action_sequences.shape[2] and i < num_ctx:
                windowed_actions[:, :, i] = action_sequences[:, :, abs_action_idx]
                action_mask[:, :, i] = 1.0

        # Build frame_coords for conditioner RoPE using generated coords for the current window
        window_frame_coords = torch.zeros(1, max_frames, num_subjects, 2, device=device, dtype=dtype)
        for i in range(num_ctx):
            window_frame_coords[:, i] = generated_coords[:, window_start + i]
        # Target frame: use last generated coord as proxy (before action)
        window_frame_coords[:, target_idx] = generated_coords[:, -1]

        subject_ids = torch.arange(num_subjects, device=device, dtype=torch.long).unsqueeze(0)
        conditioner_kwargs = dict(
            action_labels=windowed_actions,
            subject_ids=subject_ids,
            initial_positions=torch.zeros(1, num_subjects, 2, device=device, dtype=dtype),
            action_mask=action_mask,
            frame_coords=window_frame_coords,
        )
        action_embeds, context_positions = conditioner(**conditioner_kwargs)
        action_embeds = action_embeds.to(dtype=dtype)
        context_positions = context_positions.to(dtype=dtype)

        conditional_dict = {
            "prompt_embeds": action_embeds,
            "context_positions": context_positions,
            "video_text_embeds": prompt_embeds_proj,
        }

        sigma_0 = 0.0148
        sigma_max = 0.999

        # Video latents
        noisy_ctx_video = (1 - sigma_0) * window_video
        target_noise_v = torch.randn(1, 1, channels, height, width, device=device, dtype=dtype)
        if num_padding > 0:
            pad_v = torch.randn(1, num_padding, channels, height, width, device=device, dtype=dtype)
            video_latents = torch.cat([noisy_ctx_video, sigma_max * target_noise_v, sigma_max * pad_v], dim=1)
        else:
            video_latents = torch.cat([noisy_ctx_video, sigma_max * target_noise_v], dim=1)

        # Coord latents — context coords are CLEAN (no sigma_0 scaling),
        # matching training where context frame_coords_scaled are unmodified.
        window_coords_scaled = (window_coords - 0.5) * coord_scale
        noisy_ctx_coords = window_coords_scaled
        target_noise_c = torch.randn(1, 1, num_subjects, 2, device=device, dtype=dtype)
        if num_padding > 0:
            pad_c = torch.randn(1, num_padding, num_subjects, 2, device=device, dtype=dtype)
            coord_latents = torch.cat([noisy_ctx_coords, sigma_max * target_noise_c, sigma_max * pad_c], dim=1)
        else:
            coord_latents = torch.cat([noisy_ctx_coords, sigma_max * target_noise_c], dim=1)

        sample_scheduler.set_timesteps(config.get("sampling_steps", 20))
        sample_scheduler.sigmas = torch.cat([sample_scheduler.sigmas, torch.zeros(1)])
        sample_scheduler.sigmas = sample_scheduler.sigmas.to(device)
        sample_scheduler.timesteps = sample_scheduler.timesteps.to(device)
        max_timestep = sample_scheduler.timesteps[0].item()

        for i, t_val in enumerate(sample_scheduler.timesteps):
            timestep = torch.zeros(1, max_frames, device=device, dtype=torch.float32)
            timestep[:, :num_ctx] = 0
            timestep[:, target_idx] = t_val
            if num_padding > 0:
                timestep[:, target_idx + 1:] = max_timestep

            with torch.autocast(device_type='cuda', dtype=dtype):
                output = generator(
                    noisy_image_or_video=video_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    subject_coords=coord_latents,
                    coord_num_clean=torch.tensor([num_ctx], device=device),
                    zero_position_for_masked=config.get("zero_position_for_masked", False),
                )

            if len(output) == 3:
                video_flow, video_x0, coord_flow = output
            else:
                video_flow = output[0] if isinstance(output, tuple) else output
                coord_flow = None

            sigma = sample_scheduler.sigmas[i]
            sigma_next = sample_scheduler.sigmas[i + 1] if i + 1 < len(sample_scheduler.sigmas) else 0.0
            dt = sigma_next - sigma

            # Update video
            target_vf = video_flow[:, target_idx:target_idx + 1]
            target_vl = video_latents[:, target_idx:target_idx + 1]
            denoised_v = target_vl + dt * target_vf
            ctx_part = video_latents[:, :target_idx]
            if target_idx + 1 < video_latents.shape[1]:
                fut_part = video_latents[:, target_idx + 1:]
                video_latents = torch.cat([ctx_part, denoised_v, fut_part], dim=1)
            else:
                video_latents = torch.cat([ctx_part, denoised_v], dim=1)

            # Update coords
            if coord_flow is not None:
                cf_reshaped = coord_flow.view(1, max_frames, num_subjects, 2)
                target_cf = cf_reshaped[:, target_idx:target_idx + 1]
                target_cl = coord_latents[:, target_idx:target_idx + 1]
                denoised_c = target_cl + dt * target_cf
                ctx_c = coord_latents[:, :target_idx]
                if target_idx + 1 < coord_latents.shape[1]:
                    fut_c = coord_latents[:, target_idx + 1:]
                    coord_latents = torch.cat([ctx_c, denoised_c, fut_c], dim=1)
                else:
                    coord_latents = torch.cat([ctx_c, denoised_c], dim=1)

        generated_video = torch.cat([generated_video, video_latents[:, target_idx:target_idx + 1]], dim=1)
        denoised_norm = (coord_latents[:, target_idx:target_idx + 1] / coord_scale) + 0.5
        denoised_norm = denoised_norm.clamp(0.0, 1.0)
        generated_coords = torch.cat([generated_coords, denoised_norm], dim=1)

    return generated_video, generated_coords, action_sequences


SUBJECT_COLORS = [
    (0, 0, 255), (255, 100, 0), (0, 200, 0), (200, 0, 200),
    (0, 200, 200), (200, 200, 0), (128, 0, 255), (255, 128, 0),
]

SUBJECT_NAMES = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']

ACTION_NAMES_BASE = ['NOOP', 'FWD', 'BACK', 'STR_L', 'STR_R', 'TRN_L', 'TRN_R']

def get_action_name(act_idx):
    if act_idx < len(ACTION_NAMES_BASE):
        return ACTION_NAMES_BASE[act_idx]
    return f"FIRE_{act_idx - 6}"


def pixel_frame_to_latent(i, num_latent):
    """Map pixel frame index to latent frame index (VAE temporal mapping)."""
    if i == 0:
        return 0
    return min((i - 1) // 4 + 1, num_latent - 1)


def pixel_frame_to_action(i, num_actions):
    """Map pixel frame index to action index (action that produced this frame)."""
    if i == 0:
        return -1
    return min((i - 1) // 4, num_actions - 1)


ORIENT_DELTAS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ORIENT_NAMES = ['N', 'E', 'S', 'W']
ARROW_LEN = 12


def compute_expected_orientations(orientations_np, actions_np, num_latent_gen, num_subjects_real):
    """Compute expected orientation for every latent frame, extending beyond GT
    using action-based updates (TURN_LEFT=5 subtracts 1, TURN_RIGHT=6 adds 1)."""
    expected = np.zeros((num_latent_gen, num_subjects_real), dtype=np.int64)
    num_gt = orientations_np.shape[0] if orientations_np is not None else 0
    num_actions = actions_np.shape[1] if actions_np.ndim == 2 else 0

    for t in range(num_latent_gen):
        if t < num_gt and orientations_np is not None:
            expected[t] = orientations_np[t, :num_subjects_real]
        elif t > 0:
            act_idx = t - 1
            for s in range(num_subjects_real):
                prev = int(expected[t - 1, s])
                if act_idx < num_actions:
                    a = int(actions_np[s, act_idx])
                    if a == 5:
                        prev = (prev - 1) % 4
                    elif a == 6:
                        prev = (prev + 1) % 4
                expected[t, s] = prev
    return expected


def compute_expected_positions(gt_coords, orientations_expected, actions_np,
                               num_latent_gen, num_subjects_real, image_size):
    """Compute expected positions for every latent frame by replaying actions
    from the last known GT position using orientation-relative movement.

    Movement per grid step = 32 pixels (one cell at 4x Melting Pot scale).
    Actions: 1=FWD, 2=BACK, 3=STR_L, 4=STR_R move one grid cell relative
    to the player's current facing direction.
    """
    step_norm = 32.0 / image_size
    expected = np.zeros((num_latent_gen, num_subjects_real, 2), dtype=np.float64)
    num_gt = gt_coords.shape[0]
    num_actions = actions_np.shape[1] if actions_np.ndim == 2 else 0

    for t in range(num_latent_gen):
        if t < num_gt:
            expected[t] = gt_coords[t, :num_subjects_real]
        elif t > 0:
            act_idx = t - 1
            for s in range(num_subjects_real):
                px, py = expected[t - 1, s]
                if act_idx < num_actions:
                    a = int(actions_np[s, act_idx])
                    o = int(orientations_expected[t - 1, s])
                    odx, ody = ORIENT_DELTAS.get(o, (0, 0))
                    dx, dy = 0.0, 0.0
                    if a == 1:    # FORWARD
                        dx, dy = odx, ody
                    elif a == 2:  # BACKWARD
                        dx, dy = -odx, -ody
                    elif a == 3:  # STEP_LEFT (90° CCW of facing)
                        dx, dy = ody, -odx
                    elif a == 4:  # STEP_RIGHT (90° CW of facing)
                        dx, dy = -ody, odx
                    px += dx * step_norm
                    py += dy * step_norm
                expected[t, s] = [px % 1.0, py % 1.0]
    return expected.astype(np.float32)


def make_comparison_frames(gen_np, gt_np, pred_coords, gt_coords, actions_np,
                           num_subjects_real, image_size, game_name,
                           orientations_np=None):
    """
    Create side-by-side frames: GT (left) vs Generated (right).
    If generated video is longer than GT, GT side shows a dimmed last frame.

    Annotations:
      - GT crosses (+) on both frames (when GT available), pred circles (o) on generated
      - Per-subject coordinate labels in pixels
      - Per-subject action names at the top (white background)
      - Orientation arrows: GT (solid) + expected (dashed) on generated frames
      - Game name header
    
    Args:
        pred_coords: [T_latent_gen, S, 2] in [0,1]
        gt_coords: [T_latent_gt, S, 2] in [0,1] (may be shorter than pred)
        actions_np: [S, num_actions_total] integer action indices (extended if needed)
        orientations_np: [T_latent_gt, S] integer orientations or None
    """
    num_gen_frames = len(gen_np)
    num_gt_frames = len(gt_np)
    num_frames = max(num_gen_frames, num_gt_frames)
    num_latent_gen = pred_coords.shape[0]
    num_latent_gt = gt_coords.shape[0]
    num_actions = actions_np.shape[1] if actions_np.ndim == 2 else 0
    H, W = gen_np[0].shape[:2]

    expected_orient = compute_expected_orientations(
        orientations_np, actions_np, num_latent_gen, num_subjects_real)
    expected_pos = compute_expected_positions(
        gt_coords, expected_orient, actions_np, num_latent_gen,
        num_subjects_real, image_size)

    combined = []

    last_gt_frame = gt_np[-1].copy() if num_gt_frames > 0 else np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(num_frames):
        if i < num_gen_frames:
            gen_frame = gen_np[i].copy()
        else:
            gen_frame = np.zeros((H, W, 3), dtype=np.uint8)
        
        has_gt = i < num_gt_frames
        gt_frame = gt_np[i].copy() if has_gt else last_gt_frame.copy()

        lat_idx = pixel_frame_to_latent(i, num_latent_gen)
        act_idx = pixel_frame_to_action(i, num_actions)

        for s in range(num_subjects_real):
            color = SUBJECT_COLORS[s % len(SUBJECT_COLORS)]

            # Expected GT position (cross + movement arrow) on BOTH frames
            if lat_idx < num_latent_gen and s < expected_pos.shape[1]:
                gx = int(expected_pos[lat_idx, s, 0] * image_size)
                gy = int(expected_pos[lat_idx, s, 1] * image_size)
                gx_c = max(0, min(gx, W - 1))
                gy_c = max(0, min(gy, H - 1))
                for frame in [gen_frame, gt_frame]:
                    cv2.line(frame, (gx_c - 6, gy_c), (gx_c + 6, gy_c), color, 2)
                    cv2.line(frame, (gx_c, gy_c - 6), (gx_c, gy_c + 6), color, 2)
                # Arrow from game-engine orientation
                o = int(expected_orient[lat_idx, s])
                odx, ody = ORIENT_DELTAS.get(o, (0, 0))
                if odx != 0 or ody != 0:
                    adx = int(odx * ARROW_LEN)
                    ady = int(ody * ARROW_LEN)
                    arrow_end = (gx_c + adx, gy_c + ady)
                    for frame in [gen_frame, gt_frame]:
                        cv2.arrowedLine(frame, (gx_c, gy_c), arrow_end, color, 2, tipLength=0.4)

            # Pred coords (gen frame only): circle marker
            if lat_idx < num_latent_gen:
                px = int(pred_coords[lat_idx, s, 0] * image_size)
                py = int(pred_coords[lat_idx, s, 1] * image_size)
                px_c = max(0, min(px, W - 1))
                py_c = max(0, min(py, H - 1))
                cv2.circle(gen_frame, (px_c, py_c), 7, color, 2)
                label = f"{px},{py}"
                cv2.putText(gen_frame, label, (px_c + 9, py_c + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # Header: white background, dark text, clear action labels
        cols = 2 if num_subjects_real <= 4 else 3
        rows = (num_subjects_real + cols - 1) // cols
        header_h_actual = 22 + rows * 20
        header = np.full((header_h_actual, W * 2, 3), 245, dtype=np.uint8)

        gt_label = "GT" if has_gt else "GT (ended)"
        frame_info = f"pixel {i} | latent {lat_idx}"
        is_rand = act_idx >= 4
        if act_idx >= 0:
            frame_info += f" | step {act_idx}" + (" [RANDOM]" if is_rand else " [GT]")
        cv2.putText(header, f"{gt_label} | {game_name}", (5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        info_color = (0, 0, 180) if is_rand else (80, 80, 80)
        cv2.putText(header, frame_info, (W + 5, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, info_color, 1)

        col_w = (W * 2) // cols
        for s in range(num_subjects_real):
            color = SUBJECT_COLORS[s % len(SUBJECT_COLORS)]
            row_i = s // cols
            col_i = s % cols
            y_pos = 34 + row_i * 20
            x_pos = 5 + col_i * col_w
            if act_idx >= 0 and act_idx < num_actions:
                act_val = int(actions_np[s, act_idx])
                act_name = get_action_name(act_val)
                txt = f"{SUBJECT_NAMES[s]}: {act_name}"
            else:
                txt = f"{SUBJECT_NAMES[s]}: --"
            # Show game-engine orientation in header
            if lat_idx < num_latent_gen and s < expected_orient.shape[1]:
                o = int(expected_orient[lat_idx, s])
                txt += f" [{ORIENT_NAMES[o]}]"
            cv2.putText(header, txt, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        side_by_side = np.concatenate([gt_frame, gen_frame], axis=1)
        full_frame = np.concatenate([header, side_by_side], axis=0)
        combined.append(full_frame)

    return combined


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = OmegaConf.load(args.config_path)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Output dir
    max_frames = config["image_or_video_shape"][1]
    num_ar = args.num_ar_steps if args.num_ar_steps else (max_frames - 1)
    if args.output_dir:
        output_dir = args.output_dir
    else:
        config_name = os.path.splitext(os.path.basename(args.config_path))[0]
        ckpt_dir = os.path.basename(os.path.dirname(args.checkpoint_path))
        step_str = ckpt_dir.replace("checkpoint_model_", "") if "checkpoint_model_" in ckpt_dir else ckpt_dir
        output_dir = f"outputs_ckpts/{config_name}/eval_step{step_str}_n1_ar{num_ar}"
    os.makedirs(output_dir, exist_ok=True)

    num_subjects = config.get("num_subjects", 8)
    image_size = config.get("synthetic_height", 512)
    video_shape = list(config.image_or_video_shape)
    if args.window_frames is not None:
        video_shape[1] = args.window_frames
        print(f"  Overriding model window to {args.window_frames} latent frames (config had {config['image_or_video_shape'][1]})")
    max_frames = video_shape[1]
    channels, height, width = video_shape[2], video_shape[3], video_shape[4]

    # 1. Load model
    print("1. Loading model...")
    model_kwargs = dict(config.get("model_kwargs", {}))
    model_kwargs.pop('timestep_shift', None)
    generator = WanDiffusionWrapper(
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=config.get("timestep_shift", 5.0),
        coord_token_mode=True,
        per_frame_timesteps=config.get("per_frame_timesteps", True),
        subject_isolation_self_attn=config.get("subject_isolation_self_attn", True),
        use_flex_attention=config.get("use_flex_attention", True),
        image_or_video_shape=video_shape,
        coord_temporal_rope=config.get("coord_temporal_rope", True),
        coord_spatial_rope=config.get("coord_spatial_rope", True),
        coord_layernorm=config.get("coord_layernorm", False),
        num_subjects=num_subjects,
        coord_scale_factor=config.get("coord_scale_factor", 4.0),
        coord_fourier_encoding=config.get("coord_fourier_encoding", False),
        coord_direct_embedding=config.get("coord_direct_embedding", False),
        coord_no_encoder_decoder=config.get("coord_no_encoder_decoder", False),
        cross_attn_rope=config.get("cross_attn_rope", True),
        **model_kwargs,
    )
    generator = generator.to(device=device, dtype=dtype).eval()

    # 2. Load conditioner
    print("2. Loading conditioner...")
    conditioner = ActionOnlySpatialConditioner(
        num_action_classes=config.get("num_action_classes", 25),
        num_subjects=num_subjects,
        model_dim=generator.model.dim,
        patch_size=config.get("latent_patch_size", 8),
        speed=config.get("position_speed", 32),
        image_size=float(image_size),
        zero_position_for_masked=config.get("zero_position_for_masked", False),
        position_before_action=config.get("position_before_action", True),
    ).to(device)

    # 3. Load checkpoint
    print(f"3. Loading checkpoint: {args.checkpoint_path}")
    generator, conditioner = load_checkpoint(generator, conditioner, args.checkpoint_path, args.use_ema)

    # 4. Text encoder: encode null + game descriptions
    print("4. Loading text encoder for game descriptions...")
    text_encoder = WanTextEncoder().to(device)

    with torch.no_grad():
        null_dict = text_encoder([""])
        null_embed_t5 = null_dict["prompt_embeds"].to(device, dtype)
        generator.model.set_null_embedding_for_cross_attn(null_embed_t5)

    from utils.game_descriptions import GAME_DESCRIPTIONS, get_game_description
    game_text_cache = {}
    unique_prompts = list(set(GAME_DESCRIPTIONS.values()))
    for i in range(0, len(unique_prompts), 8):
        batch = unique_prompts[i:i+8]
        with torch.no_grad():
            embeds = text_encoder(batch)["prompt_embeds"].to(device, dtype)
        for j, prompt in enumerate(batch):
            game_text_cache[prompt] = embeds[j:j+1].cpu()
    print(f"   Cached {len(game_text_cache)} game text embeddings")

    del text_encoder
    torch.cuda.empty_cache()

    # 5. Load VAE for decoding alongside generator
    print("5. Loading VAE...")
    vae = WanVAEWrapper().to(device, dtype).eval()

    # 6. Load dataset
    use_validation = args.validation_dir is not None
    if use_validation:
        print(f"6. Loading validation set from: {args.validation_dir}")
        val_game_samples = load_validation_samples(args.validation_dir)
        games_to_eval = sorted(val_game_samples.keys())
        game_num_actions = {}
        dataset = None
    else:
        print("6. Loading dataset...")
        dataset = MultiGameLMDBDataset(
            config.data_path, max_pair=int(1e8),
            max_subjects=num_subjects,
            max_action_classes=config.get("num_action_classes", 25),
            use_game_descriptions=True,
        )

        game_offsets = {}
        game_num_actions = {}
        offset = 0
        for meta in dataset.shard_metadata:
            game = meta['game_name']
            n = meta['num_samples']
            game_offsets[game] = (offset, n)
            game_num_actions[game] = meta['num_actions']
            offset += n

    sample_scheduler = FlowMatchScheduler(
        shift=config.get("timestep_shift", 5.0),
        sigma_min=0.0, extra_one_step=True,
    )

    # 7. Generate, decode, and save per game
    if use_validation:
        max_seeds = args.max_samples_per_game
        all_seeds = set()
        for game_name in games_to_eval:
            for val_si, _ in val_game_samples[game_name]:
                all_seeds.add(val_si)
        sorted_seeds = sorted(all_seeds)
        if max_seeds is not None:
            sorted_seeds = sorted_seeds[:max_seeds]

        sample_iter = []
        for seed in sorted_seeds:
            for game_name in games_to_eval:
                for val_si, val_path in val_game_samples[game_name]:
                    if val_si == seed:
                        sample_iter.append((game_name, val_si, val_path))
                        break
    else:
        games_filter = set(args.games.split(",")) if args.games else None
        sample_iter = []
        for si in range(args.num_offsets):
            so = args.sample_offset + si
            for game in sorted(game_offsets.keys()):
                if games_filter and game not in games_filter:
                    continue
                base_offset, n = game_offsets[game]
                if so < n:
                    sample_iter.append((game, base_offset + so, None, so))

    if args.shard_id is not None and args.num_shards is not None:
        total = len(sample_iter)
        sample_iter = [s for i, s in enumerate(sample_iter) if i % args.num_shards == args.shard_id]
        print(f"   Shard {args.shard_id}/{args.num_shards}: {len(sample_iter)}/{total} samples")

    print(f"\n7. Running coord diffusion inference on {len(sample_iter)} samples...")
    results = []

    sample_iter_tagged = not use_validation and args.num_offsets > 1

    for item in tqdm(sample_iter, desc="Samples"):
        if use_validation:
            game_name, out_si, val_path = item
        elif sample_iter_tagged:
            game_name, idx, val_path, out_si = item
        else:
            game_name, idx = item[0], item[1]
            out_si = 0
            val_path = None

        if args.resume:
            existing_mp4 = os.path.join(output_dir, f"{game_name}_s{out_si}.mp4")
            existing_gif = os.path.join(output_dir, f"{game_name}_s{out_si}.gif")
            old_gif = os.path.join(output_dir, f"{game_name}_comparison.gif")
            if os.path.exists(existing_mp4) or os.path.exists(existing_gif) or os.path.exists(old_gif):
                out_name = f"{game_name}_s{out_si}"
                eval_npz = os.path.join(output_dir, "eval_data", f"{out_name}.npz")
                err_str = "n/a"
                ns = 0
                if os.path.exists(eval_npz):
                    ed = np.load(eval_npz)
                    ns = int(ed.get('num_subjects_real', 0))
                    gc, gtc = ed.get('gen_coords'), ed.get('gt_coords')
                    if gc is not None and gtc is not None:
                        min_t = min(gc.shape[0], gtc.shape[0])
                        if min_t > 1:
                            err_str = f"{np.abs(gc[1:min_t,:ns] - gtc[1:min_t,:ns]).mean() * image_size:.1f}"
                results.append({
                    'game': game_name, 'sample': out_si, 'idx': out_si,
                    'num_subjects': ns, 'coord_error_px': err_str,
                    'prompt': get_game_description(game_name)[:80],
                    'gif': f"{out_name}.gif",
                })
                continue

        torch.manual_seed(args.seed + out_si)

        if use_validation:
            val_data = np.load(val_path, allow_pickle=True)
            ode_latent = torch.from_numpy(val_data['video_latents']).float()
            num_subjects_real = int(val_data['num_subjects'])

            if 'game_name' in val_data:
                text_game_name = str(val_data['game_name'])
            else:
                text_game_name = game_name

            raw_actions = torch.from_numpy(val_data['actions']).long()
            if raw_actions.shape[0] < num_subjects:
                pad = torch.zeros(num_subjects - raw_actions.shape[0],
                                  raw_actions.shape[1], dtype=torch.long)
                raw_actions = torch.cat([raw_actions, pad], dim=0)
            actions = raw_actions.unsqueeze(0).to(device)

            raw_coords = torch.from_numpy(val_data['frame_coords']).float()
            raw_coords[:, :, 0] /= image_size
            raw_coords[:, :, 1] /= image_size
            if raw_coords.shape[1] < num_subjects:
                pad = torch.zeros(raw_coords.shape[0],
                                  num_subjects - raw_coords.shape[1], 2)
                raw_coords = torch.cat([raw_coords, pad], dim=1)
            frame_coords = raw_coords.unsqueeze(0).to(device, dtype=dtype)
            gt_pixel_frames = val_data['pixel_frames']
        else:
            sample = dataset[idx]
            ode_latent = sample["ode_latent"]
            if ode_latent.dim() == 5:
                ode_latent = ode_latent.squeeze(0)
            actions = sample["actions"].to(device, dtype=torch.long).unsqueeze(0)
            frame_coords = sample["frame_coords"].to(device, dtype=dtype).unsqueeze(0)
            subject_mask = sample["subject_mask"]
            num_subjects_real = int(subject_mask.sum().item())
            gt_pixel_frames = None

        initial_latents = ode_latent[:1].unsqueeze(0).to(device, dtype)
        gt_latents = ode_latent.unsqueeze(0).to(device, dtype)

        initial_coords = frame_coords[:, :1]

        desc_game = text_game_name if use_validation else game_name
        desc = get_game_description(desc_game)
        if desc in game_text_cache:
            prompt_embeds = game_text_cache[desc].to(device, dtype)
        else:
            prompt_embeds = null_embed_t5

        num_ar = args.num_ar_steps if args.num_ar_steps else (max_frames - 1)

        with torch.no_grad():
            gen_video, gen_coords, extended_actions = generate_ar_coord(
                generator, conditioner,
                initial_latents, initial_coords,
                actions, frame_coords, prompt_embeds, null_embed_t5,
                num_subjects_real, config, sample_scheduler,
                max_frames, channels, height, width, device, dtype,
                num_ar_steps=num_ar,
                game_action_count=game_num_actions.get(game_name),
            )

            gen_decoded = vae.decode_to_pixel(gen_video)
            gt_decoded = vae.decode_to_pixel(gt_latents)

        gen_np = gen_decoded[0].permute(0, 2, 3, 1).cpu().float().numpy()
        gen_np = ((gen_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        if gt_pixel_frames is not None:
            gt_np = gt_pixel_frames
        else:
            gt_np = gt_decoded[0].permute(0, 2, 3, 1).cpu().float().numpy()
            gt_np = ((gt_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        gen_coords_np = gen_coords.cpu().float().numpy()
        gt_coords_np = frame_coords[0].cpu().float().numpy()
        actions_np = extended_actions[0].cpu().numpy()

        orientations = None
        if use_validation:
            if 'orientations' in val_data:
                orientations = val_data['orientations'].astype(np.int64)
        elif 'orientations' in sample:
            orientations = sample.get("orientations", None)
            if orientations is not None:
                orientations = orientations.numpy()

        combined = make_comparison_frames(
            gen_np, gt_np, gen_coords_np[0], gt_coords_np,
            actions_np, num_subjects_real, image_size, game_name,
            orientations_np=orientations,
        )

        out_name = f"{game_name}_s{out_si}"
        gif_path = os.path.join(output_dir, f"{out_name}.gif")
        imageio.mimwrite(gif_path, combined, fps=args.fps)

        mp4_path = os.path.join(output_dir, f"{out_name}.mp4")
        imageio.mimwrite(mp4_path, combined, fps=args.fps, codec='libx264',
                         macro_block_size=1, output_params=['-pix_fmt', 'yuv420p'])

        eval_dir = os.path.join(output_dir, "eval_data")
        os.makedirs(eval_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(eval_dir, f"{out_name}.npz"),
            gen_frames=gen_np,
            gt_frames=gt_np,
            actions=actions_np,
            num_subjects_real=num_subjects_real,
            gen_coords=gen_coords_np[0],
            gt_coords=gt_coords_np,
        )

        gt_t = gt_coords_np.shape[0]
        min_t = min(gen_coords_np.shape[1], gt_t)
        if min_t > 1:
            errors = np.abs(gen_coords_np[0, 1:min_t, :num_subjects_real] -
                          gt_coords_np[1:min_t, :num_subjects_real]) * image_size
            mean_err = errors.mean()
        else:
            mean_err = float('nan')

        results.append({
            'game': game_name,
            'sample': out_si,
            'idx': out_si,
            'num_subjects': num_subjects_real,
            'coord_error_px': f"{mean_err:.1f}",
            'prompt': get_game_description(game_name)[:80],
            'gif': f"{out_name}.gif",
        })
        torch.cuda.empty_cache()

    del generator, conditioner, vae
    torch.cuda.empty_cache()

    # 9. Summary HTML
    print("\n9. Generating summary HTML...")
    ckpt_step = os.path.basename(os.path.dirname(args.checkpoint_path)).replace("checkpoint_model_", "step")
    html_path = os.path.join(output_dir, "summary.html")
    with open(html_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html><head>
<title>Coord Diffusion Inference - {ckpt_step}</title>
<style>
body {{ font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #e94560; text-align: center; }}
.game-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(550px, 1fr)); gap: 20px; }}
.game-card {{ background: #16213e; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
.game-card h3 {{ color: #e94560; margin: 0 0 8px 0; font-size: 14px; }}
.game-card p {{ font-size: 11px; color: #999; margin: 4px 0; line-height: 1.3; }}
.game-card img {{ width: 100%; border-radius: 8px; }}
.label {{ font-size: 12px; color: #666; text-align: center; margin-top: 4px; }}
.error {{ color: #4fc3f7; font-weight: bold; }}
</style>
</head><body>
<h1>Coord Diffusion - {ckpt_step} - 46 Melting Pot Games</h1>
<p style="text-align:center; color:#999;">GT (left) vs Generated (right). Crosses=GT coords, Circles=Predicted coords. {args.sampling_steps} denoising steps.</p>
<div class="game-grid">
""")
        for r in results:
            f.write(f"""<div class="game-card">
<h3>{r['game']} <span class="error">({r['num_subjects']} subj, err: {r['coord_error_px']}px)</span></h3>
<p><b>Prompt:</b> {r['prompt']}...</p>
<img src="{r['gif']}" alt="{r['game']}">
<p class="label">GT (left) | Generated (right) — sample idx {r['idx']}</p>
</div>
""")
        f.write("</div></body></html>")

    print(f"\nDone! Summary at: {html_path}")
    print(f"GIFs at: {output_dir}/")


if __name__ == "__main__":
    main()
