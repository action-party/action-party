"""
Generate Melting Pot dataset for ALL games.

Supports two canvas sizes:
  --canvas_size 1024  (default): 46 games at 4x scale (32px sprites), 1024x1024 canvas
  --canvas_size 512:  46 games at 4x scale with mini maps (max 16x16 grid), 512x512 canvas

Uses mixed trained-bot / random actions, VAE-encodes 17-frame segments into
latent representations at the specified canvas resolution.

Output format: PositionTokenLMDBDataset-compatible LMDB per game.
  - video_latents: [5, 16, H_lat, W_lat]
  - actions: [S, 4] (S=num_players, 4 action steps per sample)
  - frame_coords: [5, S, 2] (pixel coords in canvas)
  - orientations: [5, S]
  - initial_positions: [S, 2]
  - prompts: game name string

Usage:
  # Test on one game at 512x512 with mini maps
  python create_all_games_dataset.py --canvas_size 512 --games coins --samples_per_game 10

  # Full 512 dataset
  python create_all_games_dataset.py --canvas_size 512 --samples_per_game 1000

  # Full 1024 dataset (original)
  python create_all_games_dataset.py --canvas_size 1024 --samples_per_game 1000

  # Resume (skip already-completed games)
  python create_all_games_dataset.py --canvas_size 512 --samples_per_game 1000 --resume

  # RGB-only NPZs (no VAE): one ``*.npz`` per sample with ``pixel_frames`` for
  # ``train_player_classifier.py`` / tile models. Skips LMDB + VAE entirely.
  python create_all_games_dataset.py --canvas_size 512 --games clean_up \\
      --samples_per_game 1000 --rgb_npz_dir datasets/rgb_npz_clean_up_1k
"""

import os
import sys

# GPU selection: must be set BEFORE torch/TF import
# When launched via launch_dataset_generation.py, CUDA_VISIBLE_DEVICES is already set in env.
# This also handles direct command-line usage with --gpu flag via early arg parsing.
if '--gpu' in sys.argv:
    _gpu_idx = sys.argv.index('--gpu')
    if _gpu_idx + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_gpu_idx + 1]

# TF config: suppress warnings and use memory growth (must set env var before TF import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import torch
import numpy as np
import lmdb
import pickle
import cv2
import argparse
import glob
import random as py_random
from tqdm import tqdm
from PIL import Image
import imageio
import warnings
warnings.filterwarnings('ignore', message='.*Received unexpected argument.*')

# Add repo root (two dirs up from scripts/dataset/) to sys.path so `utils.*` imports resolve,
# and scripts/dataset/ itself so sibling `mini_maps` imports resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _HERE)
from utils.wan_wrapper import WanVAEWrapper
from mini_maps import MINI_GAMES_512, apply_mini_config, apply_mini_config_val

# Melting Pot imports
from meltingpot import substrate as substrate_module
from meltingpot import bot as bot_module
from meltingpot.utils.substrates import specs
from dm_env import TimeStep


# =============================================================================
# CONSTANTS
# =============================================================================

# 46 games that fit in 1024x1024 at 4x scale (32px sprites)
GAMES_1024 = [
    'allelopathic_harvest__open',
    'bach_or_stravinsky_in_the_matrix__arena',
    'bach_or_stravinsky_in_the_matrix__repeated',
    'chemistry__three_metabolic_cycles',
    'chemistry__three_metabolic_cycles_with_plentiful_distractors',
    'chemistry__two_metabolic_cycles',
    'chemistry__two_metabolic_cycles_with_distractors',
    'chicken_in_the_matrix__arena',
    'chicken_in_the_matrix__repeated',
    'clean_up',
    'coins',
    'collaborative_cooking__asymmetric',
    'collaborative_cooking__circuit',
    'collaborative_cooking__cramped',
    'collaborative_cooking__crowded',
    'collaborative_cooking__figure_eight',
    'collaborative_cooking__forced',
    'collaborative_cooking__ring',
    'commons_harvest__closed',
    'commons_harvest__open',
    'commons_harvest__partnership',
    'coop_mining',
    'daycare',
    'externality_mushrooms__dense',
    'factory_commons__either_or',
    'fruit_market__concentric_rivers',
    'gift_refinements',
    'paintball__capture_the_flag',
    'paintball__king_of_the_hill',
    'predator_prey__alley_hunt',
    'predator_prey__open',
    'predator_prey__orchard',
    'predator_prey__random_forest',
    'prisoners_dilemma_in_the_matrix__arena',
    'prisoners_dilemma_in_the_matrix__repeated',
    'pure_coordination_in_the_matrix__arena',
    'pure_coordination_in_the_matrix__repeated',
    'rationalizable_coordination_in_the_matrix__arena',
    'rationalizable_coordination_in_the_matrix__repeated',
    'running_with_scissors_in_the_matrix__arena',
    'running_with_scissors_in_the_matrix__one_shot',
    'running_with_scissors_in_the_matrix__repeated',
    'stag_hunt_in_the_matrix__arena',
    'stag_hunt_in_the_matrix__repeated',
    'territory__inside_out',
    'territory__rooms',
]

# Original Melting Pot action space (7 actions)
ACTION_NAMES = ['NOOP', 'FORWARD', 'BACKWARD', 'STEP_LEFT', 'STEP_RIGHT', 'TURN_LEFT', 'TURN_RIGHT']
NUM_ACTIONS = 7

# Orientation mapping
ORIENTATION_MAP = {
    'N': 0, 'E': 1, 'S': 2, 'W': 3,
    0: 0, 1: 1, 2: 2, 3: 3,
}

# Video parameters (defaults for 4 actions; overridden by --num_actions)
STEPS_PER_ACTION = 4            # 1 action + 3 NOOPs (always 4)
ACTIONS_PER_SAMPLE = 4
PIXEL_FRAMES_PER_SAMPLE = 17   # ACTIONS * STEPS_PER_ACTION + 1
LATENT_FRAMES_PER_SAMPLE = 5   # (PIXEL_FRAMES - 1) / 4 + 1
STEPS_PER_SAMPLE = 16           # ACTIONS * STEPS_PER_ACTION
# WAN VAE causal temporal mapping: latent 0 -> pixel 0, latent i (i>0) -> pixel 4i-3
LATENT_TO_PIXEL = [0, 1, 5, 9, 13]


def _set_num_actions(n):
    """Recompute all frame/latent constants from the number of actions."""
    global ACTIONS_PER_SAMPLE, PIXEL_FRAMES_PER_SAMPLE, LATENT_FRAMES_PER_SAMPLE
    global STEPS_PER_SAMPLE, LATENT_TO_PIXEL
    ACTIONS_PER_SAMPLE = n
    STEPS_PER_SAMPLE = n * STEPS_PER_ACTION
    PIXEL_FRAMES_PER_SAMPLE = STEPS_PER_SAMPLE + 1
    LATENT_FRAMES_PER_SAMPLE = (PIXEL_FRAMES_PER_SAMPLE - 1) // 4 + 1
    LATENT_TO_PIXEL = [0] + [4 * i - 3 for i in range(1, LATENT_FRAMES_PER_SAMPLE)]
CANVAS_SIZE = 1024              # Default; overridden by --canvas_size
SCALE = 4                       # 4x upscaling (8px native -> 32px sprites)
SPRITE_SIZE = 8                 # Native Melting Pot sprite size
USE_MINI_MAPS = False           # Set True when --canvas_size 512
USE_VAL_MAPS = False            # Set True for validation with dense spawn points


# =============================================================================
# LOCATION OBSERVER INJECTION
# =============================================================================

def inject_location_observer(config):
    """Modify a substrate config to add LocationObserver to all avatars.
    
    This injects POSITION and ORIENTATION observations for every player,
    working generically across all Melting Pot substrates.
    """
    # 1. Add POSITION and ORIENTATION to individual observation names
    obs_names = set(config.individual_observation_names)
    obs_names.add('POSITION')
    obs_names.add('ORIENTATION')
    config.individual_observation_names = tuple(sorted(obs_names))
    
    # 2. Update timestep_spec to include POSITION and ORIENTATION
    try:
        obs_spec = dict(config.timestep_spec.observation)
        if 'POSITION' not in obs_spec:
            obs_spec['POSITION'] = specs.OBSERVATION['POSITION']
        if 'ORIENTATION' not in obs_spec:
            obs_spec['ORIENTATION'] = specs.OBSERVATION['ORIENTATION']
        config.timestep_spec = config.timestep_spec._replace(observation=obs_spec)
    except Exception as e:
        print(f"  Warning: Could not update timestep_spec: {e}")
    
    # 3. Wrap the lab2d_settings_builder to inject LocationObserver into avatars
    original_builder = config.lab2d_settings_builder
    
    def wrapped_builder(*, roles, config):
        settings = original_builder(roles=roles, config=config)
        game_objects = settings.get('simulation', {}).get('gameObjects', [])
        injected = 0
        for obj in game_objects:
            components = obj.get('components', [])
            has_avatar = any(c.get('component') == 'Avatar' for c in components)
            has_loc_obs = any(c.get('component') == 'LocationObserver' for c in components)
            if has_avatar and not has_loc_obs:
                components.append({
                    'component': 'LocationObserver',
                    'kwargs': {'objectIsAvatar': True, 'alsoReportOrientation': True},
                })
                injected += 1
        return settings
    
    config.lab2d_settings_builder = wrapped_builder
    return config


# =============================================================================
# BOT LOADING
# =============================================================================

def get_bots_for_game(game_name):
    """Get list of available bot names for a substrate."""
    return sorted([b for b in bot_module.BOTS if b.startswith(game_name + '__')])


def load_bot_policies(game_name, num_players, episode_idx=0):
    """Load bot policies for all player slots.
    
    Cycles through available bots with variety, trying different bot types.
    Uses episode_idx to shuffle bot assignment across episodes for diversity.
    """
    available = get_bots_for_game(game_name)
    if not available:
        return None
    
    # Shuffle bot order based on episode index for diversity
    shuffled = list(available)
    rng = py_random.Random(episode_idx * 31337 + hash(game_name))
    rng.shuffle(shuffled)
    
    # Select bots with variety from shuffled order
    selected = []
    used_types = set()
    
    for _ in range(num_players):
        chosen = None
        for bot_name in shuffled:
            bot_type = bot_name.split('__')[-1].rsplit('_', 1)[0]
            if bot_type not in used_types:
                chosen = bot_name
                used_types.add(bot_type)
                break
        if chosen is None:
            # Cycle through available bots (from shuffled order)
            chosen = shuffled[len(selected) % len(shuffled)]
        selected.append(chosen)
    
    # Build policies
    policies = []
    for name in selected:
        try:
            policy = bot_module.build(name)
            policies.append(policy)
        except Exception as e:
            policies.append(None)
    
    return policies


def close_bot_policies(bot_policies):
    """Safely close bot policies to free resources."""
    if bot_policies is None:
        return
    for p in bot_policies:
        if p is not None:
            try:
                p.close()
            except Exception:
                pass


# =============================================================================
# ENVIRONMENT BUILDING
# =============================================================================

def build_game_env(game_name):
    """Build a game environment with LocationObserver injected.
    
    When USE_MINI_MAPS is True (--canvas_size 512), applies mini map
    and reduced player count before building.
    """
    config = substrate_module.get_config(game_name)
    
    if USE_MINI_MAPS and USE_VAL_MAPS:
        config, roles = apply_mini_config_val(game_name, config)
    elif USE_MINI_MAPS:
        config, roles = apply_mini_config(game_name, config)
    else:
        roles = config.default_player_roles
    
    num_players = len(roles)
    
    # Inject LocationObserver
    config = inject_location_observer(config)
    
    # Build environment
    env = substrate_module.build_from_config(config, roles=roles)
    
    return env, num_players, roles


# =============================================================================
# OBSERVATION HELPERS
# =============================================================================

def get_world_frame(obs):
    """Extract the global RGB frame from observations."""
    if 'WORLD.RGB' in obs[0]:
        return obs[0]['WORLD.RGB']
    return obs[0]['RGB']


def get_positions(obs, num_players):
    """Extract grid positions for all players. Returns [S, 2] array of (col, row)."""
    positions = []
    for i in range(num_players):
        if 'POSITION' in obs[i]:
            pos = obs[i]['POSITION']
            positions.append([int(pos[0]), int(pos[1])])  # col, row
        else:
            positions.append([0, 0])
    return np.array(positions, dtype=np.int32)


def get_orientations(obs, num_players):
    """Extract orientations for all players. Returns [S] array."""
    orientations = []
    for i in range(num_players):
        if 'ORIENTATION' in obs[i]:
            orient = obs[i]['ORIENTATION']
            if isinstance(orient, np.ndarray):
                orient = orient.item()
            orientations.append(ORIENTATION_MAP.get(orient, 0))
        else:
            orientations.append(0)
    return np.array(orientations, dtype=np.int64)


def grid_to_canvas_coords(grid_positions, native_h, native_w):
    """Convert grid positions [S, 2] (col, row) to 1024x1024 canvas pixel coords [S, 2] (x, y)."""
    h4x = native_h * SCALE
    w4x = native_w * SCALE
    x_off = (CANVAS_SIZE - w4x) // 2
    y_off = (CANVAS_SIZE - h4x) // 2
    
    canvas_coords = np.zeros_like(grid_positions, dtype=np.float32)
    for s in range(len(grid_positions)):
        col, row = grid_positions[s]
        x_4x = (col * SPRITE_SIZE + SPRITE_SIZE / 2) * SCALE
        y_4x = (row * SPRITE_SIZE + SPRITE_SIZE / 2) * SCALE
        canvas_coords[s, 0] = x_4x + x_off
        canvas_coords[s, 1] = y_4x + y_off
    
    return canvas_coords


def frame_to_canvas(frame, native_h, native_w):
    """Upscale frame to 4x and center-pad to 1024x1024."""
    h4x = native_h * SCALE
    w4x = native_w * SCALE
    frame_4x = cv2.resize(frame, (w4x, h4x), interpolation=cv2.INTER_NEAREST)
    
    padded = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    y_off = (CANVAS_SIZE - h4x) // 2
    x_off = (CANVAS_SIZE - w4x) // 2
    padded[y_off:y_off + h4x, x_off:x_off + w4x] = frame_4x
    
    return padded


# =============================================================================
# ROLLOUT AND SAMPLE GENERATION
# =============================================================================

def run_episode_and_collect_chunks(env, bot_policies, num_players, num_actions_total,
                                    native_h, native_w, bot_prob=0.7, max_chunks=None,
                                    skip_steps=0):
    """Run one episode, collect 17-frame chunks.
    
    Each chunk = 17 pixel frames, 4 actions, 5 latent frame positions.
    At each action step: choose bot (prob=bot_prob) or random action per player.
    Between action steps: NOOP for all players.
    
    Args:
        skip_steps: Random steps to skip at episode start for temporal diversity.
    
    Returns list of chunk dicts.
    """
    timestep = env.reset()
    obs = env.observation()
    
    # Initialize bot states
    bot_states = None
    if bot_policies is not None:
        bot_states = []
        for policy in bot_policies:
            if policy is not None:
                bot_states.append(policy.initial_state())
            else:
                bot_states.append(None)
    
    # Random skip at episode start for temporal diversity
    for _ in range(skip_steps):
        # Use random actions during skip (still update bot states for continuity)
        skip_actions = [py_random.randint(0, num_actions_total - 1) for _ in range(num_players)]
        timestep = env.step(skip_actions)
        if timestep.last():
            return [], True
        obs = env.observation()
        # Update bot states during skip so they stay in sync
        if bot_policies is not None and bot_states is not None:
            for i in range(num_players):
                if bot_policies[i] is not None and bot_states[i] is not None:
                    player_obs = timestep.observation[i]
                    player_reward = timestep.reward[i] if timestep.reward is not None else 0.0
                    player_ts = TimeStep(
                        step_type=timestep.step_type,
                        reward=player_reward,
                        discount=timestep.discount,
                        observation=player_obs,
                    )
                    try:
                        _, new_state = bot_policies[i].step(player_ts, bot_states[i])
                        bot_states[i] = new_state
                    except Exception:
                        pass
    
    chunks = []
    episode_ended = False
    
    while not episode_ended:
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        
        # Start of chunk: current obs is frame 0
        chunk_frames = [get_world_frame(obs)]
        chunk_grid_positions = [get_positions(obs, num_players)]  # latent 0 -> pixel 0
        chunk_orientations = [get_orientations(obs, num_players)]
        chunk_actions = []
        
        chunk_complete = True
        pixel_frame_idx = 0  # track which pixel frame we're at
        
        for action_idx in range(ACTIONS_PER_SAMPLE):
            # Choose actions for this step
            player_actions = []
            for i in range(num_players):
                if bot_policies is not None and bot_policies[i] is not None and bot_states[i] is not None:
                    # Query bot
                    player_obs = timestep.observation[i]
                    player_reward = timestep.reward[i] if timestep.reward is not None else 0.0
                    player_ts = TimeStep(
                        step_type=timestep.step_type,
                        reward=player_reward,
                        discount=timestep.discount,
                        observation=player_obs,
                    )
                    try:
                        bot_action, new_state = bot_policies[i].step(player_ts, bot_states[i])
                        bot_states[i] = new_state
                        
                        if py_random.random() < bot_prob:
                            player_actions.append(int(bot_action))
                        else:
                            player_actions.append(py_random.randint(0, num_actions_total - 1))
                    except Exception:
                        player_actions.append(py_random.randint(0, num_actions_total - 1))
                else:
                    player_actions.append(py_random.randint(0, num_actions_total - 1))
            
            chunk_actions.append(player_actions)
            
            # Take the action step
            timestep = env.step(player_actions)
            if timestep.last():
                episode_ended = True
                chunk_complete = False
                break
            obs = env.observation()
            chunk_frames.append(get_world_frame(obs))
            pixel_frame_idx += 1
            
            # Record position at latent-aligned pixel frames: 1, 5, 9, 13
            # (pixel frame right after each action, before NOOPs)
            next_latent_pixel = LATENT_TO_PIXEL[len(chunk_grid_positions)] if len(chunk_grid_positions) < LATENT_FRAMES_PER_SAMPLE else -1
            if pixel_frame_idx == next_latent_pixel:
                chunk_grid_positions.append(get_positions(obs, num_players))
                chunk_orientations.append(get_orientations(obs, num_players))
            
            # Take 3 NOOP steps
            for _ in range(3):
                timestep = env.step([0] * num_players)
                if timestep.last():
                    episode_ended = True
                    chunk_complete = False
                    break
                obs = env.observation()
                chunk_frames.append(get_world_frame(obs))
                pixel_frame_idx += 1
                
                # Also check NOOP frames for latent alignment (for latent frames 2+ at pixels 5, 9, 13)
                next_latent_pixel = LATENT_TO_PIXEL[len(chunk_grid_positions)] if len(chunk_grid_positions) < LATENT_FRAMES_PER_SAMPLE else -1
                if pixel_frame_idx == next_latent_pixel:
                    chunk_grid_positions.append(get_positions(obs, num_players))
                    chunk_orientations.append(get_orientations(obs, num_players))
            
            if episode_ended:
                break
        
        # Only keep complete chunks (17 frames, 4 actions, 5 position readings)
        if chunk_complete and len(chunk_frames) == PIXEL_FRAMES_PER_SAMPLE:
            grid_pos_array = np.stack(chunk_grid_positions)  # [5, S, 2]
            
            # Skip chunks where any player went OOB (outside visible grid).
            # This can happen in games with torus-like topology (e.g., territory__rooms)
            # where Lab2D allows movement past the map boundary into invisible void.
            world_frame_sample = chunk_frames[0]
            grid_h = world_frame_sample.shape[0] // SPRITE_SIZE
            grid_w = world_frame_sample.shape[1] // SPRITE_SIZE
            cols = grid_pos_array[:, :, 0]  # [5, S]
            rows = grid_pos_array[:, :, 1]  # [5, S]
            has_oob = (cols < 0).any() or (cols >= grid_w).any() or \
                      (rows < 0).any() or (rows >= grid_h).any()
            
            if not has_oob:
                chunks.append({
                    'frames': chunk_frames,                     # list of [H, W, 3] uint8
                    'grid_positions': grid_pos_array,           # [5, S, 2] int32
                    'orientations': np.stack(chunk_orientations),  # [5, S] int64
                    'actions': np.array(chunk_actions, dtype=np.int64),  # [4, S]
                })
    
    return chunks, episode_ended


# =============================================================================
# VAE ENCODING
# =============================================================================

def stack_chunk_pixel_frames_uint8(chunk_frames, native_h, native_w):
    """Stack native frames to canvas RGB ``(T, H, W, 3)`` uint8 (same layout as encode path)."""
    padded = [frame_to_canvas(fr, native_h, native_w) for fr in chunk_frames]
    return np.stack(padded, axis=0).astype(np.uint8)


def count_rgb_npz_exports(rgb_npz_dir: str, game_name: str) -> int:
    pat = os.path.join(rgb_npz_dir, f"{game_name}_s*.npz")
    return len(glob.glob(pat))


def encode_chunk_video(vae, chunk_frames, native_h, native_w, device, dtype):
    """Encode a 17-frame chunk into video latents.
    
    Upscales to 4x, pads to CANVAS_SIZE, normalizes, and VAE-encodes.
    At 1024: -> [5, 16, 128, 128] latents
    At 512:  -> [5, 16, 64, 64] latents
    """
    # Upscale and pad frames
    padded_frames = []
    for frame in chunk_frames:
        padded = frame_to_canvas(frame, native_h, native_w)
        padded_frames.append(padded)
    
    # Stack: [T, H, W, C] -> [C, T, H, W] -> [1, C, T, H, W]
    frames_np = np.stack(padded_frames, axis=0)  # [17, CANVAS, CANVAS, 3]
    frames_t = torch.from_numpy(frames_np).to(dtype).permute(0, 3, 1, 2) / 255.0
    frames_t = frames_t * 2 - 1  # Normalize to [-1, 1]
    frames_t = frames_t.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, 3, 17, CANVAS, CANVAS]
    
    with torch.no_grad():
        latents = vae.encode_to_latent(frames_t)  # [1, 5, 16, H_lat, W_lat]
        latents = latents[0].cpu().float().numpy()  # [5, 16, H_lat, W_lat]
    
    return latents


# =============================================================================
# LMDB WRITING
# =============================================================================

def write_game_to_lmdb(output_path, game_name, samples, native_h, native_w, num_players, game_num_actions=7, shard_id=0):
    """Write all samples for one game to LMDB."""
    shard_dir = os.path.join(output_path, game_name)
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f'shard_{shard_id}.lmdb')
    
    # Estimate size: ~5MB per sample for video latents + overhead
    map_size = max(len(samples) * 8 * 1024 * 1024, 100 * 1024 * 1024)  # At least 100MB
    
    env_lmdb = lmdb.open(shard_path, map_size=map_size)
    
    with env_lmdb.begin(write=True) as txn:
        # Metadata
        txn.put(b'num_samples', str(len(samples)).encode())
        txn.put(b'num_subjects', str(num_players).encode())
        txn.put(b'num_actions', str(game_num_actions).encode())
        txn.put(b'image_size', str(CANVAS_SIZE).encode())
        txn.put(b'speed', str(32).encode())
        txn.put(b'game_name', game_name.encode())
        txn.put(b'native_resolution_h', str(native_h).encode())
        txn.put(b'native_resolution_w', str(native_w).encode())
        txn.put(b'dataset_type', b'melting_pot_all_games')
        txn.put(b'pixel_frames_per_sample', str(PIXEL_FRAMES_PER_SAMPLE).encode())
        txn.put(b'action_names', pickle.dumps(ACTION_NAMES))
        
        if len(samples) > 0:
            txn.put(b'video_latents_shape', str(samples[0]['video_latents'].shape).encode())
            txn.put(b'actions_shape', str(samples[0]['actions'].shape).encode())
            txn.put(b'frame_coords_shape', str(samples[0]['frame_coords'].shape).encode())
            txn.put(b'orientations_shape', str(samples[0]['orientations'].shape).encode())
            txn.put(b'initial_positions_shape', str(samples[0]['initial_positions'].shape).encode())
        
        # Per-sample data
        for i, sample in enumerate(samples):
            txn.put(f'video_latents_{i}_data'.encode(),
                     pickle.dumps(sample['video_latents'].astype(np.float32)))
            txn.put(f'actions_{i}_data'.encode(),
                     pickle.dumps(sample['actions']))
            txn.put(f'frame_coords_{i}_data'.encode(),
                     pickle.dumps(sample['frame_coords'].astype(np.float32)))
            txn.put(f'orientations_{i}_data'.encode(),
                     pickle.dumps(sample['orientations']))
            txn.put(f'initial_positions_{i}_data'.encode(),
                     pickle.dumps(sample['initial_positions'].astype(np.float32)))
            txn.put(f'prompts_{i}_data'.encode(),
                     game_name.encode())
    
    env_lmdb.close()
    return shard_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_sample_visualization(sample, game_name, sample_idx, output_dir, native_h, native_w):
    """Save a GIF visualization of one sample."""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    frames = sample['raw_frames']  # list of [H, W, 3] native resolution
    coords = sample['frame_coords']  # [5, S, 2]
    
    vis_frames = []
    for f_idx, frame in enumerate(frames):
        canvas = frame_to_canvas(frame, native_h, native_w)
        
        # Draw positions at latent frames (every 4th pixel frame)
        if f_idx % 4 == 0:
            latent_idx = f_idx // 4
            if latent_idx < len(coords):
                for s in range(coords.shape[1]):
                    cx = int(coords[latent_idx, s, 0])
                    cy = int(coords[latent_idx, s, 1])
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                             (255, 255, 0), (255, 0, 255), (0, 255, 255)][s % 6]
                    cv2.circle(canvas, (cx, cy), 8, color, 2)
        
        # Downscale for GIF (always show at 512x512)
        vis_size = min(CANVAS_SIZE, 512)
        vis_frame = cv2.resize(canvas, (vis_size, vis_size), interpolation=cv2.INTER_AREA)
        vis_frames.append(vis_frame)
    
    gif_path = os.path.join(vis_dir, f'{game_name}_sample{sample_idx}.gif')
    imageio.mimwrite(gif_path, vis_frames, fps=8)


# =============================================================================
# MAIN
# =============================================================================

def process_game(game_name, vae, device, dtype, args):
    """Generate all samples for one game (LMDB latents, or RGB NPZs under ``--rgb_npz_dir``)."""
    print(f"\n{'='*60}")
    print(f"Processing: {game_name}")
    print(f"{'='*60}")
    rgb_mode = bool(getattr(args, "rgb_npz_dir", None))
    if rgb_mode:
        os.makedirs(args.rgb_npz_dir, exist_ok=True)

    # Build environment
    try:
        env, num_players, roles = build_game_env(game_name)
    except Exception as e:
        print(f"  ERROR building env: {e}")
        return None
    
    # Get native resolution from first observation
    timestep = env.reset()
    obs = env.observation()
    world_frame = get_world_frame(obs)
    native_h, native_w = world_frame.shape[:2]
    h4x, w4x = native_h * SCALE, native_w * SCALE
    
    # Verify it fits in 1024x1024
    if h4x > CANVAS_SIZE or w4x > CANVAS_SIZE:
        print(f"  SKIP: 4x resolution {w4x}x{h4x} exceeds {CANVAS_SIZE}x{CANVAS_SIZE}")
        env.close()
        return None
    
    # Check if POSITION is available
    has_position = 'POSITION' in obs[0]
    if not has_position:
        print(f"  WARNING: POSITION not available in observations!")
    
    # Detect actual action space size from environment spec
    try:
        action_spec = env.action_spec()
        if isinstance(action_spec, (list, tuple)):
            # Per-player specs - get max num_values
            game_num_actions = max(s.num_values for s in action_spec)
        else:
            game_num_actions = action_spec.num_values
    except Exception:
        game_num_actions = NUM_ACTIONS  # fallback to default 7
    
    print(f"  Players: {num_players}")
    print(f"  Native resolution: {native_w}x{native_h}")
    print(f"  4x resolution: {w4x}x{h4x}")
    print(f"  POSITION available: {has_position}")
    print(f"  Action space: {game_num_actions} actions")
    env.close()
    
    # Check bot availability
    available_bots = get_bots_for_game(game_name)
    has_bots = len(available_bots) > 0
    if has_bots:
        print(f"  Available bots: {len(available_bots)} types")
    else:
        print(f"  No bots available, using all random actions")
    
    # Bot probability schedule for diversity: vary per episode
    BOT_PROB_SCHEDULE = [0.5, 0.7, 0.7, 0.9]  # cycle through these
    MAX_SKIP_STEPS = 50  # max random steps to skip at episode start
    
    # Calculate episodes needed
    chunks_per_episode = max(1, 500 // STEPS_PER_SAMPLE)
    episodes_needed = (args.samples_per_game + chunks_per_episode - 1) // chunks_per_episode
    print(f"  Target samples: {args.samples_per_game}")
    print(f"  Chunks per episode: ~{chunks_per_episode}")
    print(f"  Episodes needed: ~{episodes_needed}")
    if rgb_mode:
        print(f"  RGB NPZ export (no VAE): {os.path.abspath(args.rgb_npz_dir)!r}")

    all_samples = []
    exported = count_rgb_npz_exports(args.rgb_npz_dir, game_name) if rgb_mode else 0
    if rgb_mode and exported >= args.samples_per_game:
        print(f"  Already have {exported} npz (>= {args.samples_per_game}), skipping.")
        return exported

    episode_idx = 0
    pbar = tqdm(
        total=args.samples_per_game,
        initial=exported if rgb_mode else 0,
        desc=f"  {game_name}",
    )

    while (exported if rgb_mode else len(all_samples)) < args.samples_per_game:
        # Build fresh env for each episode
        try:
            env, _, _ = build_game_env(game_name)
        except Exception as e:
            print(f"  ERROR rebuilding env at episode {episode_idx}: {e}")
            break

        # Load bots with shuffled assignment for this episode
        bot_policies = load_bot_policies(game_name, num_players, episode_idx=episode_idx) if has_bots else None

        # Vary bot_prob per episode for behavioral diversity
        ep_bot_prob = BOT_PROB_SCHEDULE[episode_idx % len(BOT_PROB_SCHEDULE)]

        # Random skip at episode start for temporal diversity
        ep_skip = py_random.randint(0, MAX_SKIP_STEPS)

        cur = exported if rgb_mode else len(all_samples)
        remaining = args.samples_per_game - cur

        chunks, _ = run_episode_and_collect_chunks(
            env=env,
            bot_policies=bot_policies,
            num_players=num_players,
            num_actions_total=game_num_actions,
            native_h=native_h,
            native_w=native_w,
            bot_prob=ep_bot_prob,
            max_chunks=remaining,
            skip_steps=ep_skip,
        )

        env.close()
        close_bot_policies(bot_policies)

        for chunk in chunks:
            cur = exported if rgb_mode else len(all_samples)
            if cur >= args.samples_per_game:
                break

            try:
                frame_coords = np.zeros((LATENT_FRAMES_PER_SAMPLE, num_players, 2), dtype=np.float32)
                for t in range(LATENT_FRAMES_PER_SAMPLE):
                    frame_coords[t] = grid_to_canvas_coords(
                        chunk['grid_positions'][t], native_h, native_w
                    )
                actions = chunk['actions'].T  # [num_players, ACTIONS_PER_SAMPLE]
                orientations = chunk['orientations']

                if rgb_mode:
                    pixel_frames = stack_chunk_pixel_frames_uint8(
                        chunk['frames'], native_h, native_w
                    )
                    out_path = os.path.join(args.rgb_npz_dir, f"{game_name}_s{exported}.npz")
                    np.savez_compressed(
                        out_path,
                        pixel_frames=pixel_frames,
                        frame_coords=frame_coords,
                        orientations=orientations,
                        actions=actions,
                        num_subjects=np.int64(num_players),
                        game_name=np.str_(game_name),
                    )
                    exported += 1
                else:
                    video_latents = encode_chunk_video(
                        vae, chunk['frames'], native_h, native_w, device, dtype
                    )
                    initial_positions = frame_coords[0]
                    sample = {
                        'video_latents': video_latents,
                        'actions': actions,
                        'frame_coords': frame_coords,
                        'orientations': orientations,
                        'initial_positions': initial_positions,
                    }
                    if len(all_samples) < args.visualize_samples:
                        sample['raw_frames'] = chunk['frames']
                    all_samples.append(sample)

                pbar.update(1)

            except Exception as e:
                print(f"\n  ERROR processing chunk: {e}")
                import traceback

                traceback.print_exc()
                continue

            if not rgb_mode and len(all_samples) % 50 == 0:
                torch.cuda.empty_cache()

        episode_idx += 1

    pbar.close()

    if rgb_mode:
        if exported == 0:
            print(f"  No samples exported for {game_name}!")
            return None
        print(f"  Exported {exported} RGB npz files")
        one = sorted(glob.glob(os.path.join(args.rgb_npz_dir, f"{game_name}_s*.npz")))[0]
        with np.load(one) as z:
            print(f"  Example {os.path.basename(one)}: pixel_frames {z['pixel_frames'].shape}")
        return exported

    if not all_samples:
        print(f"  No samples collected for {game_name}!")
        return None

    print(f"  Collected {len(all_samples)} samples")
    print(f"  Video latents shape: {all_samples[0]['video_latents'].shape}")
    print(f"  Actions shape: {all_samples[0]['actions'].shape}")
    print(f"  Frame coords shape: {all_samples[0]['frame_coords'].shape}")

    # Save visualizations
    for vis_idx in range(min(args.visualize_samples, len(all_samples))):
        if 'raw_frames' in all_samples[vis_idx]:
            save_sample_visualization(
                all_samples[vis_idx], game_name, vis_idx,
                args.output_path, native_h, native_w
            )

    # Remove raw frames before LMDB write (save memory)
    for s in all_samples:
        s.pop('raw_frames', None)

    # Write to LMDB
    shard_path = write_game_to_lmdb(
        args.output_path, game_name, all_samples, native_h, native_w, num_players,
        game_num_actions=game_num_actions, shard_id=args.shard_id
    )
    print(f"  Saved to: {shard_path}")

    # Report storage
    shard_size_mb = os.path.getsize(shard_path + '/data.mdb') / (1024 * 1024)
    print(f"  LMDB size: {shard_size_mb:.1f} MB")

    return len(all_samples)


def main():
    global CANVAS_SIZE, USE_MINI_MAPS
    
    parser = argparse.ArgumentParser(description="Generate multi-game Melting Pot dataset")
    parser.add_argument('--canvas_size', type=int, default=1024, choices=[512, 1024],
                        help='Canvas size: 512 (mini maps, 4x) or 1024 (original, 4x)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory (default: datasets/melting_pot_all_games_{canvas_size})')
    parser.add_argument('--games', type=str, nargs='+', default=None,
                        help='Specific games to process (default: all 46)')
    parser.add_argument('--samples_per_game', type=int, default=1000,
                        help='Number of 17-frame samples per game')
    parser.add_argument('--bot_prob', type=float, default=0.7,
                        help='Probability of using bot action vs random (per player per step)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize_samples', type=int, default=3,
                        help='Number of samples to visualize per game')
    parser.add_argument('--resume', action='store_true',
                        help='Skip games that already have LMDB files')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specific GPU to use (for parallel launches)')
    parser.add_argument('--shard_id', type=int, default=0,
                        help='Shard ID for output LMDB name (shard_{id}.lmdb). '
                             'Use different IDs to add more samples later.')
    parser.add_argument('--num_actions', type=int, default=4,
                        help='Number of actions per sample (default: 4). '
                             'Determines pixel/latent frame counts automatically.')
    parser.add_argument(
        '--rgb_npz_dir',
        type=str,
        default=None,
        help='If set: write one compressed NPZ per sample with pixel_frames (no VAE / no LMDB). '
             'Use with --games for a small set (e.g. one game). Incompatible with resume LMDB checks.',
    )
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Configure number of actions (recomputes all frame constants)
    _set_num_actions(args.num_actions)
    
    # Configure canvas size and mini maps
    CANVAS_SIZE = args.canvas_size
    USE_MINI_MAPS = (args.canvas_size == 512)
    
    if args.output_path is None:
        args.output_path = f'datasets/melting_pot_all_games_{args.canvas_size}'

    rgb_mode = bool(args.rgb_npz_dir)
    if rgb_mode:
        os.makedirs(args.rgb_npz_dir, exist_ok=True)
    
    # Set seeds
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Game list
    valid_games = sorted(MINI_GAMES_512) if USE_MINI_MAPS else GAMES_1024
    games = args.games if args.games else valid_games
    
    # Validate game names
    for g in games:
        if g not in valid_games:
            print(f"Warning: {g} not in valid game list for canvas_size={args.canvas_size}")
    
    print(f"="*60)
    print(f"Multi-Game Melting Pot Dataset Generator")
    print(f"="*60)
    print(f"  Output: {args.output_path}")
    print(f"  Canvas: {CANVAS_SIZE}x{CANVAS_SIZE}")
    print(f"  Mini maps: {USE_MINI_MAPS}")
    print(f"  Games: {len(games)}")
    print(f"  Samples per game: {args.samples_per_game}")
    print(f"  Total samples: {len(games) * args.samples_per_game}")
    print(f"  Bot probability: {args.bot_prob}")
    print(f"  Actions per sample: {ACTIONS_PER_SAMPLE}")
    print(f"  Frames per sample: {PIXEL_FRAMES_PER_SAMPLE} pixel -> {LATENT_FRAMES_PER_SAMPLE} latent")
    if rgb_mode:
        px_bytes = PIXEL_FRAMES_PER_SAMPLE * CANVAS_SIZE * CANVAS_SIZE * 3
        est_gb = len(games) * args.samples_per_game * px_bytes / (1024 ** 3)
        print(f"  Estimated RGB NPZ raw pixel payload (upper bound): ~{est_gb:.1f} GB (compressed smaller)")
    else:
        base_mb = 1.3 if CANVAS_SIZE == 512 else 5.0
        est_mb_per_sample = base_mb * (LATENT_FRAMES_PER_SAMPLE / 5.0)
        print(f"  Estimated storage: ~{len(games) * args.samples_per_game * est_mb_per_sample / 1024:.1f} GB")

    os.makedirs(args.output_path, exist_ok=True)

    if not rgb_mode:
        print("\nLoading VAE...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.bfloat16
        vae = WanVAEWrapper().to(device, dtype).eval()
        print("VAE loaded")
    else:
        vae = None
        device = torch.device('cpu')
        dtype = torch.float32
        print("\nRGB NPZ mode: skipping VAE load.")
    
    # Process each game
    results = {}
    for gi, game_name in enumerate(games):
        # Check if already done (resume mode)
        if args.resume:
            if rgb_mode:
                n = count_rgb_npz_exports(args.rgb_npz_dir, game_name)
                if n >= args.samples_per_game:
                    print(f"\n[{gi+1}/{len(games)}] {game_name}: {n} RGB npz done, skipping")
                    results[game_name] = 'skipped'
                    continue
                if n > 0:
                    print(f"\n[{gi+1}/{len(games)}] {game_name}: resuming from {n}/{args.samples_per_game} npz")
            else:
                shard_path = os.path.join(args.output_path, game_name, f'shard_{args.shard_id}.lmdb')
                if os.path.exists(shard_path):
                    # Verify the shard has the expected number of samples
                    try:
                        env_lmdb = lmdb.open(shard_path, readonly=True, lock=False)
                        with env_lmdb.begin() as txn:
                            n = int(txn.get(b'num_samples').decode())
                        env_lmdb.close()
                        if n >= args.samples_per_game:
                            print(f"\n[{gi+1}/{len(games)}] {game_name}: {n} samples done, skipping")
                            results[game_name] = 'skipped'
                            continue
                        else:
                            print(f"\n[{gi+1}/{len(games)}] {game_name}: only {n}/{args.samples_per_game} samples, regenerating")
                    except Exception:
                        print(f"\n[{gi+1}/{len(games)}] {game_name}: corrupt shard, regenerating")
                        pass
        
        print(f"\n[{gi+1}/{len(games)}]", end="")
        
        try:
            n_samples = process_game(game_name, vae, device, dtype, args)
            results[game_name] = n_samples if n_samples else 'failed'
        except Exception as e:
            print(f"\n  FATAL ERROR for {game_name}: {e}")
            import traceback
            traceback.print_exc()
            results[game_name] = 'error'

        if torch.cuda.is_available() and not rgb_mode:
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    success = sum(1 for v in results.values() if isinstance(v, int))
    skipped = sum(1 for v in results.values() if v == 'skipped')
    failed = sum(1 for v in results.values() if v in ('failed', 'error'))
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print(f"\n  Failed games:")
        for g, v in results.items():
            if v in ('failed', 'error'):
                print(f"    - {g}")
    
    # Total storage
    total_mb = 0
    if rgb_mode:
        for game_name in games:
            for fp in glob.glob(os.path.join(args.rgb_npz_dir, f"{game_name}_s*.npz")):
                total_mb += os.path.getsize(fp) / (1024 * 1024)
    else:
        for game_name in games:
            data_file = os.path.join(args.output_path, game_name, f'shard_{args.shard_id}.lmdb', 'data.mdb')
            if os.path.exists(data_file):
                total_mb += os.path.getsize(data_file) / (1024 * 1024)
    print(f"\n  Total storage (this run output): {total_mb / 1024:.2f} GB")
    print(f"\nDone!")


if __name__ == '__main__':
    main()
