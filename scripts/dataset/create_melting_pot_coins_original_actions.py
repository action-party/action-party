"""
Generate Melting Pot Coins dataset with ORIGINAL action space (relative movement + turns).

Unlike v3 (absolute NSEW), this uses the native Melting Pot action space:
  0: NOOP       - do nothing
  1: FORWARD    - move in facing direction
  2: BACKWARD   - move opposite to facing direction
  3: STEP_LEFT  - strafe left (no turn)
  4: STEP_RIGHT - strafe right (no turn)
  5: TURN_LEFT  - rotate 90° left (no movement)
  6: TURN_RIGHT - rotate 90° right (no movement)

Agents have ORIENTATION (N/S/E/W) which affects how movement actions translate
to grid position changes. Orientation is tracked and stored alongside positions.

Same output format as PositionTokenLMDBDataset:
- video_latents: [T, 16, H, W]
- position_latents: [S, T, 16, H, W] - separate per subject
- actions: [S, num_actions] (now 0-6 instead of 0-4)
- initial_positions: [S, 2]
- frame_coords: [T, S, 2] (normalized)
- orientations: [T, S] (0=N, 1=E, 2=S, 3=W)

Usage:
python create_melting_pot_coins_original_actions.py \
  --output_path datasets/melting_pot_coins_original_actions \
  --samples 2500 \
  --seed 42
"""
import os
import sys
import torch
import numpy as np
import lmdb
import pickle
from tqdm import tqdm
from PIL import Image
import cv2
import argparse

# Add repo root (two dirs up from scripts/dataset/) so `utils.*` imports resolve.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from utils.wan_wrapper import WanVAEWrapper

# Melting Pot imports
from meltingpot import substrate as substrate_module
from meltingpot.configs.substrates import coins as coins_config
from meltingpot.utils.substrates import specs
from meltingpot.utils.substrates import shapes
from meltingpot.utils.substrates import game_object_utils
import random as py_random


# Original action space (7 actions, relative to facing direction)
ACTION_NAMES = ['NOOP', 'FORWARD', 'BACKWARD', 'STEP_LEFT', 'STEP_RIGHT', 'TURN_LEFT', 'TURN_RIGHT']
NUM_ACTIONS = 7
MELTINGPOT_ACTIONS = list(range(NUM_ACTIONS))

# Orientation mapping: MP reports orientation as a string or index
# We map to integers: 0=N, 1=E, 2=S, 3=W
ORIENTATION_MAP = {
    'N': 0, 'E': 1, 'S': 2, 'W': 3,
    0: 0, 1: 1, 2: 2, 3: 3,
}


def get_coin_fixed_spawn(coin_type_a, coin_type_b, initial_coin_type):
    """Create a coin that starts visible and never regenerates."""
    return {
        "name": "coin",
        "components": [
            {
                "component": "StateManager",
                "kwargs": {
                    "initialState": initial_coin_type,
                    "stateConfigs": [
                        {"state": coin_type_a, "layer": "superOverlay", "sprite": coin_type_a},
                        {"state": coin_type_b, "layer": "superOverlay", "sprite": coin_type_b},
                        {"state": "coinWait", "layer": "logic"},
                    ]
                }
            },
            {"component": "Transform"},
            {
                "component": "Appearance",
                "kwargs": {
                    "renderMode": "ascii_shape",
                    "spriteNames": [coin_type_a, coin_type_b],
                    "spriteShapes": [shapes.COIN] * 2,
                    "palettes": [coins_config.COIN_PALETTES[coin_type_a],
                                 coins_config.COIN_PALETTES[coin_type_b]],
                    "noRotates": [False] * 2,
                }
            },
            {
                "component": "Coin",
                "kwargs": {
                    "waitState": "coinWait",
                    "rewardSelfForMatch": 1.0,
                    "rewardSelfForMismatch": 1.0,
                    "rewardOtherForMatch": 0.0,
                    "rewardOtherForMismatch": -2.0,
                }
            },
        ]
    }


def get_ascii_map_with_fixed_coins(map_size, num_coins_per_type=3):
    """Generate ASCII map with specific coin positions."""
    width = map_size
    height = map_size
    
    grid = []
    for row in range(height + 2):
        if row == 0 or row == height + 1:
            grid.append(['W'] * (width + 2))
        else:
            grid.append(['W'] + [' '] * width + ['W'])
    
    # Place spawn points in TOP area
    grid[2][2] = '_'
    grid[2][width - 1] = '_'
    
    # Get available positions
    available = []
    for row in range(1, height + 1):
        for col in range(1, width + 1):
            if grid[row][col] == ' ':
                available.append((row, col))
    
    py_random.shuffle(available)
    
    for i in range(num_coins_per_type):
        if available:
            pos = available.pop()
            grid[pos[0]][pos[1]] = 'A'
    for i in range(num_coins_per_type):
        if available:
            pos = available.pop()
            grid[pos[0]][pos[1]] = 'B'
    
    return '\n'.join([''.join(row) for row in grid])


def create_coins_env(map_size=6, num_coins_per_type=3):
    """Build custom coins environment with RELATIVE coordinates (original action space)."""
    config = coins_config.get_config()
    
    config.min_width = map_size
    config.max_width = map_size
    config.min_height = map_size
    config.max_height = map_size
    
    native_resolution = (map_size + 2) * 8
    
    # Add POSITION and ORIENTATION to individual observations
    config.individual_observation_names = list(config.individual_observation_names) + ['POSITION', 'ORIENTATION']
    
    config.timestep_spec = specs.timestep({
        'RGB': specs.OBSERVATION['RGB'],
        'POSITION': specs.OBSERVATION['POSITION'],
        'ORIENTATION': specs.OBSERVATION['ORIENTATION'],
        'MISMATCHED_COIN_COLLECTED_BY_PARTNER': specs.float64(),
        'WORLD.RGB': specs.rgb(native_resolution, native_resolution),
    })
    
    def custom_build(roles, config):
        assert len(roles) == 2, "Need exactly 2 players"
        
        coin_type_a, coin_type_b = py_random.sample(tuple(coins_config.COIN_PALETTES), k=2)
        
        num_players = len(roles)
        player_color_palettes = coins_config.get_player_color_palettes(
            coin_type_a=coin_type_a, coin_type_b=coin_type_b)
        avatar_objects = game_object_utils.build_avatar_objects(
            num_players, {"avatar": coins_config.get_avatar(coin_type_a)}, player_color_palettes)
        game_object_utils.get_first_named_component(
            avatar_objects[1], "PlayerCoinType")["kwargs"]["coinType"] = coin_type_b
        
        # DO NOT set useAbsoluteCoordinates -- agents use relative movement + orientation
        # Randomize initial orientation for diversity
        for avatar_obj in avatar_objects:
            avatar_comp = game_object_utils.get_first_named_component(avatar_obj, "Avatar")
            if avatar_comp is not None:
                avatar_comp["kwargs"]["randomizeInitialOrientation"] = True
        
        # Add LocationObserver with orientation reporting
        for avatar_obj in avatar_objects:
            avatar_obj["components"].append({
                "component": "LocationObserver",
                "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True},
            })
        
        prefabs = {
            "wall": coins_config.WALL,
            "spawn_point": coins_config.SPAWN_POINT,
            "coin_a": get_coin_fixed_spawn(coin_type_a, coin_type_b, coin_type_a),
            "coin_b": get_coin_fixed_spawn(coin_type_a, coin_type_b, coin_type_b),
        }
        
        char_prefab_map = {"_": "spawn_point", "W": "wall", "A": "coin_a", "B": "coin_b"}
        
        ascii_map = get_ascii_map_with_fixed_coins(map_size, num_coins_per_type)
        
        substrate_definition = dict(
            levelName="coins",
            levelDirectory="meltingpot/lua/levels",
            numPlayers=num_players,
            maxEpisodeLengthFrames=5000,
            spriteSize=8,
            topology="BOUNDED",
            simulation={
                "map": ascii_map,
                "scene": coins_config.SCENE,
                "prefabs": prefabs,
                "charPrefabMap": char_prefab_map,
                "gameObjects": avatar_objects,
            }
        )
        return substrate_definition
    
    config.lab2d_settings_builder = custom_build
    
    roles = config.default_player_roles
    env = substrate_module.build_from_config(config, roles=roles)
    
    return env, native_resolution


def create_position_map_single_subject(position, height=256, width=256, sigma=12.0):
    """Create position map for a single subject as RGB (same Gaussian in all channels)."""
    x, y = position
    
    yy, xx = np.ogrid[:height, :width]
    gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    gaussian = gaussian / (gaussian.max() + 1e-8)
    
    gaussian = gaussian * 2 - 1
    position_map = np.stack([gaussian, gaussian, gaussian], axis=0)  # [3, H, W]
    
    return position_map.astype(np.float32)


def generate_sample(map_size=6, num_coins_per_type=3, num_video_frames=33, output_res=256, sigma=12.0):
    """Generate a single sample with original action space (relative movement + turns)."""
    num_subjects = 2
    
    env, native_res = create_coins_env(map_size=map_size, num_coins_per_type=num_coins_per_type)
    sprite_size = 8
    scale = output_res / native_res
    env.reset()
    
    obs = env.observation()
    
    def grid_to_pixel(obs, player_idx):
        """Convert grid position to pixel (center_x, center_y) in output resolution.
        POSITION returns [col, row] (verified empirically)."""
        col, row = obs[player_idx]['POSITION']
        center_x = (col * sprite_size + sprite_size / 2) * scale
        center_y = (row * sprite_size + sprite_size / 2) * scale
        return (center_x, center_y)
    
    def get_orientation(obs, player_idx):
        """Get agent orientation. Returns integer: 0=N, 1=E, 2=S, 3=W."""
        orient = obs[player_idx]['ORIENTATION']
        if isinstance(orient, (np.ndarray,)):
            orient = orient.item()
        return ORIENTATION_MAP.get(orient, 0)
    
    initial_positions = np.array(
        [grid_to_pixel(obs, 0), grid_to_pixel(obs, 1)], dtype=np.float32
    )
    
    num_latent_frames = (num_video_frames - 1) // 4 + 1
    num_actions = num_latent_frames - 1
    
    # Sample random actions from the full 7-action space
    actions = np.zeros((num_subjects, num_actions), dtype=np.int64)
    for s in range(num_subjects):
        for t in range(num_actions):
            actions[s, t] = np.random.choice(MELTINGPOT_ACTIONS)
    
    video_frames = []
    position_frames_s0 = []
    position_frames_s1 = []
    frame_coords = []
    orientations = []
    
    action_idx = 0
    
    for f in range(num_video_frames):
        if f > 0 and (f - 1) % 4 == 0 and action_idx < num_actions:
            mp_actions = [int(actions[s, action_idx]) for s in range(num_subjects)]
            timestep = env.step(mp_actions)
            action_idx += 1
        elif f > 0:
            timestep = env.step([0, 0])  # NOOP for intermediate frames
        
        obs = env.observation()
        frame = obs[0]['WORLD.RGB']
        
        positions = [grid_to_pixel(obs, 0), grid_to_pixel(obs, 1)]
        orients = [get_orientation(obs, 0), get_orientation(obs, 1)]
        
        frame_256 = cv2.resize(frame, (output_res, output_res), interpolation=cv2.INTER_NEAREST)
        frame_tensor = frame_256.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_tensor = frame_tensor * 2 - 1
        video_frames.append(frame_tensor)
        
        pos_map_s0 = create_position_map_single_subject(positions[0], output_res, output_res, sigma)
        pos_map_s1 = create_position_map_single_subject(positions[1], output_res, output_res, sigma)
        position_frames_s0.append(pos_map_s0)
        position_frames_s1.append(pos_map_s1)
        
        frame_coords.append(np.array(positions, dtype=np.float32))
        orientations.append(np.array(orients, dtype=np.int64))
    
    env.close()
    
    video_frames = np.stack(video_frames, axis=0)
    position_frames_s0 = np.stack(position_frames_s0, axis=0)
    position_frames_s1 = np.stack(position_frames_s1, axis=0)
    frame_coords = np.stack(frame_coords, axis=0)  # [T_pixel, S, 2]
    orientations = np.stack(orientations, axis=0)   # [T_pixel, S]
    
    return video_frames, position_frames_s0, position_frames_s1, actions, initial_positions, frame_coords, orientations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='datasets/melting_pot_coins_original_actions')
    parser.add_argument('--samples', type=int, default=2500)
    parser.add_argument('--map_size', type=int, default=6)
    parser.add_argument('--num_coins_per_type', type=int, default=3)
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--position_sigma', type=float, default=12.0)
    parser.add_argument('--vae_batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize_samples', type=int, default=5)
    args = parser.parse_args()
    
    py_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("Creating Melting Pot Coins dataset with ORIGINAL action space...")
    print(f"  Output: {args.output_path}")
    print(f"  Samples: {args.samples}")
    print(f"  Map size: {args.map_size} (native {(args.map_size + 2) * 8}x{(args.map_size + 2) * 8})")
    print(f"  Target resolution: {args.target_size}x{args.target_size}")
    print(f"  Actions: {ACTION_NAMES} (7 actions, relative to facing direction)")
    print(f"  Position format: [S, T, 16, H, W] (separate per subject)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = WanVAEWrapper().to(device)
    vae.eval()
    print("VAE loaded")
    
    os.makedirs(args.output_path, exist_ok=True)
    vis_dir = os.path.join(args.output_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Storage for batching
    video_batch = []
    pos_s0_batch = []
    pos_s1_batch = []
    batch_actions = []
    batch_positions = []
    batch_frame_coords = []
    batch_orientations = []
    
    # All encoded data
    all_video_latents = []
    all_position_latents = []
    all_actions = []
    all_positions = []
    all_frame_coords = []
    all_orientations = []
    
    # Samples to visualize
    vis_samples = {}
    
    for sample_idx in tqdm(range(args.samples), desc="Generating samples"):
        result = generate_sample(
            map_size=args.map_size,
            num_coins_per_type=args.num_coins_per_type,
            num_video_frames=33,
            output_res=args.target_size,
            sigma=args.position_sigma
        )
        video_frames, pos_frames_s0, pos_frames_s1, actions, initial_positions, frame_coords, orientations = result
        
        # Store for visualization
        if sample_idx < args.visualize_samples:
            vis_samples[sample_idx] = {
                'video': video_frames.copy(),
                'pos_s0': pos_frames_s0.copy(),
                'pos_s1': pos_frames_s1.copy(),
                'actions': actions.copy(),
                'frame_coords': frame_coords.copy(),
                'orientations': orientations.copy(),
            }
        
        video_batch.append(torch.from_numpy(video_frames))
        pos_s0_batch.append(torch.from_numpy(pos_frames_s0))
        pos_s1_batch.append(torch.from_numpy(pos_frames_s1))
        batch_actions.append(actions)
        batch_positions.append(initial_positions)
        batch_frame_coords.append(frame_coords)
        batch_orientations.append(orientations)
        
        if len(video_batch) >= args.vae_batch_size:
            vid_batch = torch.stack(video_batch).permute(0, 2, 1, 3, 4).to(device)
            pos_s0_b = torch.stack(pos_s0_batch).permute(0, 2, 1, 3, 4).to(device)
            pos_s1_b = torch.stack(pos_s1_batch).permute(0, 2, 1, 3, 4).to(device)
            
            with torch.no_grad():
                video_latents = vae.encode_to_latent(vid_batch).cpu().numpy()
                pos_s0_latents = vae.encode_to_latent(pos_s0_b).cpu().numpy()
                pos_s1_latents = vae.encode_to_latent(pos_s1_b).cpu().numpy()
            
            for j in range(len(video_batch)):
                all_video_latents.append(video_latents[j])
                pos_latents = np.stack([pos_s0_latents[j], pos_s1_latents[j]], axis=0)
                all_position_latents.append(pos_latents)
                all_actions.append(batch_actions[j])
                all_positions.append(batch_positions[j])
                # Subsample to latent frames
                fc = batch_frame_coords[j]
                latent_frame_coords = fc[::4][:9]
                all_frame_coords.append(latent_frame_coords)
                ori = batch_orientations[j]
                latent_orientations = ori[::4][:9]
                all_orientations.append(latent_orientations)
            
            video_batch = []
            pos_s0_batch = []
            pos_s1_batch = []
            batch_actions = []
            batch_positions = []
            batch_frame_coords = []
            batch_orientations = []
    
    # Process remaining samples
    if video_batch:
        vid_batch = torch.stack(video_batch).permute(0, 2, 1, 3, 4).to(device)
        pos_s0_b = torch.stack(pos_s0_batch).permute(0, 2, 1, 3, 4).to(device)
        pos_s1_b = torch.stack(pos_s1_batch).permute(0, 2, 1, 3, 4).to(device)
        
        with torch.no_grad():
            video_latents = vae.encode_to_latent(vid_batch).cpu().numpy()
            pos_s0_latents = vae.encode_to_latent(pos_s0_b).cpu().numpy()
            pos_s1_latents = vae.encode_to_latent(pos_s1_b).cpu().numpy()
        
        for j in range(len(video_batch)):
            all_video_latents.append(video_latents[j])
            pos_latents = np.stack([pos_s0_latents[j], pos_s1_latents[j]], axis=0)
            all_position_latents.append(pos_latents)
            all_actions.append(batch_actions[j])
            all_positions.append(batch_positions[j])
            fc = batch_frame_coords[j]
            latent_frame_coords = fc[::4][:9]
            all_frame_coords.append(latent_frame_coords)
            ori = batch_orientations[j]
            latent_orientations = ori[::4][:9]
            all_orientations.append(latent_orientations)
    
    print(f"\nGenerated {len(all_video_latents)} samples")
    print(f"  Video latent shape: {all_video_latents[0].shape}")
    print(f"  Position latent shape: {all_position_latents[0].shape}")
    print(f"  Actions shape: {all_actions[0].shape}")
    print(f"  Frame coords shape: {all_frame_coords[0].shape}")
    print(f"  Orientations shape: {all_orientations[0].shape}")
    
    # Create visualizations with action+orientation annotations
    print("Creating visualizations...")
    orient_names = ['N', 'E', 'S', 'W']
    for sample_idx, data in vis_samples.items():
        video = data['video']
        pos_s0 = data['pos_s0']
        pos_s1 = data['pos_s1']
        actions_data = data['actions']
        coords = data['frame_coords']
        orients = data['orientations']
        
        combined_frames = []
        for f in range(len(video)):
            vid = ((video[f].transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            # Draw position markers and orientation arrows
            vid_annotated = vid.copy()
            for s in range(2):
                cx, cy = int(coords[f, s, 0]), int(coords[f, s, 1])
                color = (255, 0, 0) if s == 0 else (0, 255, 0)
                cv2.circle(vid_annotated, (cx, cy), 8, color, 2)
                
                # Draw orientation arrow
                o = orients[f, s]
                arrow_len = 12
                if o == 0:    # N
                    dx, dy = 0, -arrow_len
                elif o == 1:  # E
                    dx, dy = arrow_len, 0
                elif o == 2:  # S
                    dx, dy = 0, arrow_len
                else:         # W
                    dx, dy = -arrow_len, 0
                cv2.arrowedLine(vid_annotated, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.4)
            
            # Position maps
            p0 = ((pos_s0[f][0] + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            p1 = ((pos_s1[f][0] + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            pos_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            pos_rgb[:, :, 0] = p0
            pos_rgb[:, :, 1] = p1
            
            combined = np.concatenate([vid_annotated, pos_rgb], axis=1)
            combined_frames.append(Image.fromarray(combined))
        
        gif_path = os.path.join(vis_dir, f'sample_{sample_idx}_combined.gif')
        combined_frames[0].save(gif_path, save_all=True, append_images=combined_frames[1:], duration=100, loop=0)
        
        # Print action sequence with orientations
        acts0 = [ACTION_NAMES[a] for a in actions_data[0]]
        acts1 = [ACTION_NAMES[a] for a in actions_data[1]]
        # Show orientation at latent frames
        ori_s0 = [orient_names[orients[f*4, 0]] for f in range(min(9, len(orients)//4 + 1))]
        ori_s1 = [orient_names[orients[f*4, 1]] for f in range(min(9, len(orients)//4 + 1))]
        print(f"  Sample {sample_idx}:")
        print(f"    P0 actions: {acts0}")
        print(f"    P0 orient:  {ori_s0}")
        print(f"    P1 actions: {acts1}")
        print(f"    P1 orient:  {ori_s1}")
    
    # Save to LMDB
    print("\nWriting to LMDB...")
    shard_path = os.path.join(args.output_path, 'shard_0.lmdb')
    env_lmdb = lmdb.open(shard_path, map_size=100 * 1024 * 1024 * 1024)
    
    with env_lmdb.begin(write=True) as txn:
        txn.put(b'num_samples', str(len(all_video_latents)).encode())
        txn.put(b'video_latents_shape', str(all_video_latents[0].shape).encode())
        txn.put(b'position_latents_shape', str(all_position_latents[0].shape).encode())
        txn.put(b'actions_shape', str(all_actions[0].shape).encode())
        txn.put(b'initial_positions_shape', str(all_positions[0].shape).encode())
        txn.put(b'frame_coords_shape', str(all_frame_coords[0].shape).encode())
        txn.put(b'orientations_shape', str(all_orientations[0].shape).encode())
        txn.put(b'num_subjects', str(2).encode())
        txn.put(b'num_actions', str(NUM_ACTIONS).encode())
        txn.put(b'dataset_type', b'melting_pot_coins_original_actions')
        txn.put(b'map_size', str(args.map_size).encode())
        txn.put(b'position_sigma', str(args.position_sigma).encode())
        txn.put(b'action_names', pickle.dumps(ACTION_NAMES))
        
        for i in tqdm(range(len(all_video_latents)), desc="Writing to LMDB"):
            txn.put(f'video_latents_{i}_data'.encode(), pickle.dumps(all_video_latents[i].astype(np.float32)))
            txn.put(f'position_latents_{i}_data'.encode(), pickle.dumps(all_position_latents[i].astype(np.float32)))
            txn.put(f'actions_{i}_data'.encode(), pickle.dumps(all_actions[i]))
            txn.put(f'initial_positions_{i}_data'.encode(), pickle.dumps(all_positions[i]))
            txn.put(f'frame_coords_{i}_data'.encode(), pickle.dumps(all_frame_coords[i]))
            txn.put(f'orientations_{i}_data'.encode(), pickle.dumps(all_orientations[i]))
            txn.put(f'prompts_{i}_data'.encode(), b'')
    
    env_lmdb.close()
    
    print(f"\nSaved to {args.output_path}")
    print(f"  Video latents: {all_video_latents[0].shape}")
    print(f"  Position latents: {all_position_latents[0].shape} (S=2 subjects)")
    print(f"  Actions: {all_actions[0].shape} (7 actions: {ACTION_NAMES})")
    print(f"  Orientations: {all_orientations[0].shape}")
    print(f"  Visualizations: {vis_dir}")
    print("\nDone!")


if __name__ == '__main__':
    main()
