"""
Create 2-subject dataset with RANDOM blob shapes and colors, plus SEPARATE position maps per subject.

Same random shapes as create_random_blobs_dataset.py (metaball, polygon, splat, star, crescent)
with random vibrant colors - but with position_latents for position token training.

Each sample contains:
- video_latents: [T, 16, H, W] - VAE encoded video with 2 subjects
- position_latents: [S, T, 16, H, W] - VAE encoded position maps, SEPARATE per subject
- actions: [S, T-1] - action labels for each subject
- initial_positions: [S, 2] - (x, y) starting positions for each subject

This allows each subject's position tokens to attend to their own actions via masked cross-attention.
"""

import os
import torch
import numpy as np
import lmdb
import pickle
from tqdm import tqdm
import argparse
from scipy import ndimage


# Action definitions
ALL_ACTIONS = [0, 1, 2, 3, 4]  # stay, up, down, left, right
MOVING_ACTIONS = [1, 2, 3, 4]  # up, down, left, right

ACTION_DIRS = {
    0: (0, 0),      # still
    1: (0, -1),     # up (y decreases)
    2: (0, 1),      # down (y increases)
    3: (-1, 0),     # left (x decreases)
    4: (1, 0)       # right (x increases)
}


def generate_random_blob(size, min_fill=0.2, max_fill=0.6):
    """
    Generate a random organic blob shape using multiple methods.
    Same as create_random_blobs_dataset.py
    """
    method = np.random.choice(['metaball', 'polygon', 'splat', 'star', 'crescent'])
    
    binary = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    if method == 'metaball':
        # Multiple overlapping circles of varying sizes
        num_balls = np.random.randint(2, 5)
        for _ in range(num_balls):
            cx = center + np.random.randint(-size//4, size//4)
            cy = center + np.random.randint(-size//4, size//4)
            radius = np.random.randint(size//6, size//3)
            for h in range(size):
                for w in range(size):
                    if (h - cy)**2 + (w - cx)**2 <= radius**2:
                        binary[h, w] = 1
    
    elif method == 'polygon':
        # Random polygon with varying vertices
        num_vertices = np.random.randint(3, 8)
        angles = np.sort(np.random.uniform(0, 2*np.pi, num_vertices))
        radii = np.random.uniform(size//4, size//2.5, num_vertices)
        
        vertices = []
        for angle, radius in zip(angles, radii):
            x = center + int(radius * np.cos(angle))
            y = center + int(radius * np.sin(angle))
            vertices.append((x, y))
        
        # Fill polygon
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(vertices, fill=1)
        binary = np.array(img)
    
    elif method == 'splat':
        # Organic splat shape using bezier-like curves
        num_points = np.random.randint(5, 10)
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        angles += np.random.uniform(-0.3, 0.3, num_points)  # Add jitter
        radii = np.random.uniform(size//5, size//2.2, num_points)
        
        # Create smooth outline
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        points = []
        for angle, radius in zip(angles, radii):
            x = center + int(radius * np.cos(angle))
            y = center + int(radius * np.sin(angle))
            points.append((x, y))
        
        draw.polygon(points, fill=1)
        binary = np.array(img)
    
    elif method == 'star':
        # Star or spiky shape
        num_points = np.random.randint(3, 7)
        outer_radius = np.random.uniform(size//3, size//2.2)
        inner_radius = np.random.uniform(size//6, outer_radius * 0.6)
        
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        points = []
        for i in range(num_points * 2):
            angle = i * np.pi / num_points - np.pi/2
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center + int(radius * np.cos(angle))
            y = center + int(radius * np.sin(angle))
            points.append((x, y))
        
        draw.polygon(points, fill=1)
        binary = np.array(img)
    
    elif method == 'crescent':
        # Crescent or curved shape
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        
        # Main circle
        main_radius = np.random.randint(size//3, size//2)
        draw.ellipse([center - main_radius, center - main_radius,
                      center + main_radius, center + main_radius], fill=1)
        
        # Cut out offset circle
        offset_x = np.random.randint(-size//4, size//4)
        offset_y = np.random.randint(-size//4, size//4)
        cut_radius = np.random.randint(main_radius//2, main_radius)
        draw.ellipse([center + offset_x - cut_radius, center + offset_y - cut_radius,
                      center + offset_x + cut_radius, center + offset_y + cut_radius], fill=0)
        
        binary = np.array(img)
    
    # Add some random bumps/noise for more organic feel
    if np.random.random() < 0.5:
        num_bumps = np.random.randint(1, 4)
        for _ in range(num_bumps):
            bx = np.random.randint(size//4, 3*size//4)
            by = np.random.randint(size//4, 3*size//4)
            br = np.random.randint(2, size//8)
            for h in range(max(0, by-br), min(size, by+br)):
                for w in range(max(0, bx-br), min(size, bx+br)):
                    if (h - by)**2 + (w - bx)**2 <= br**2:
                        binary[h, w] = 1
    
    # Ensure we have something
    if np.sum(binary) < size:
        from PIL import Image, ImageDraw
        img = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(img)
        rx = np.random.randint(size//4, size//2)
        ry = np.random.randint(size//4, size//2)
        draw.ellipse([center - rx, center - ry, center + rx, center + ry], fill=1)
        binary = np.array(img)
    
    # Apply slight smoothing for more organic edges
    binary = ndimage.binary_dilation(binary).astype(np.uint8)
    binary = ndimage.binary_erosion(binary).astype(np.uint8)
    
    # Center the blob
    if np.sum(binary) > 0:
        coords = np.argwhere(binary)
        center_h = (coords[:, 0].min() + coords[:, 0].max()) // 2
        center_w = (coords[:, 1].min() + coords[:, 1].max()) // 2
        shift_h = size // 2 - center_h
        shift_w = size // 2 - center_w
        binary = np.roll(np.roll(binary, shift_h, axis=0), shift_w, axis=1)
    
    return binary.astype(bool)


def generate_random_color():
    """Generate a random vibrant RGB color using HSV."""
    h = np.random.random()  # Hue
    s = np.random.uniform(0.7, 1.0)  # High saturation
    v = np.random.uniform(0.7, 1.0)  # High value
    
    # HSV to RGB conversion
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 2/6:
        r, g, b = x, c, 0
    elif h < 3/6:
        r, g, b = 0, c, x
    elif h < 4/6:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    return (r + m, g + m, b + m)


def create_gaussian_position_map(x, y, height=256, width=256, sigma=12.0):
    """
    Create a grayscale position map with Gaussian at (x, y) for a SINGLE subject.
    Returns [3, H, W] with same value in all RGB channels.
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    gaussian = np.exp(-((x_coords - x)**2 + (y_coords - y)**2) / (2 * sigma**2))
    
    # Replicate to RGB (grayscale)
    position_map = np.stack([gaussian, gaussian, gaussian], axis=0)
    
    # Convert to [-1, 1] range
    position_map = position_map * 2 - 1
    
    return position_map.astype(np.float32)


def draw_blob_on_frame(frame, center_x, center_y, blob_mask, rgb_color):
    """
    Draw a blob shape centered at (center_x, center_y) on the frame.
    
    Args:
        frame: [3, H, W] numpy array in [0, 1] range
        center_x, center_y: center position in pixels
        blob_mask: [size, size] boolean mask
        rgb_color: (r, g, b) tuple in [0, 1] range
    """
    size = blob_mask.shape[0]
    half_size = size // 2
    
    h_start = int(center_y - half_size)
    w_start = int(center_x - half_size)
    
    for dh in range(size):
        for dw in range(size):
            if blob_mask[dh, dw]:
                h = h_start + dh
                w = w_start + dw
                if 0 <= h < frame.shape[1] and 0 <= w < frame.shape[2]:
                    for c in range(3):
                        frame[c, h, w] = rgb_color[c]


def generate_sample_2subj(height=256, width=256, num_video_frames=33, blob_size=48, speed=16, sigma=12.0):
    """
    Generate a single sample with 2 subjects: random blobs, video frames, SEPARATE position maps, and actions.
    
    Returns:
        video_frames: [T_video, 3, H, W] - video frames
        position_frames: [S, T_video, 3, H, W] - SEPARATE position maps per subject
        actions: [S, T_latent-1] - action labels for each subject at latent frame rate
        initial_positions: [S, 2] - (x, y) starting position for each subject
    """
    num_subjects = 2
    
    # Generate random blob shapes and colors for each subject
    blob_masks = [generate_random_blob(blob_size) for _ in range(num_subjects)]
    rgb_colors = [generate_random_color() for _ in range(num_subjects)]
    
    # Random starting positions (with margin, ensuring no overlap)
    margin = blob_size // 2 + 5  # ~29 for blob_size=48
    min_distance = blob_size * 1.5  # Minimum distance between subjects
    
    positions = []
    for s in range(num_subjects):
        for attempt in range(100):
            x = np.random.randint(margin, width - margin)
            y = np.random.randint(margin, height - margin)
            
            # Check distance from existing subjects
            valid = True
            for px, py in positions:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                if dist < min_distance:
                    valid = False
                    break
            
            if valid:
                positions.append([float(x), float(y)])
                break
        else:
            # Fallback: place somewhere
            x = margin + s * (width - 2 * margin) // num_subjects
            y = height // 2
            positions.append([float(x), float(y)])
    
    initial_positions = np.array(positions, dtype=np.float32)  # [S, 2]
    
    # Number of latent frames and actions
    num_latent_frames = (num_video_frames - 1) // 4 + 1  # 33 -> 9
    num_actions = num_latent_frames - 1  # 8
    
    # Generate actions for each subject - 2 unique actions each with random transition
    actions = np.zeros((num_subjects, num_actions), dtype=np.int64)
    
    for s in range(num_subjects):
        action1 = np.random.choice(ALL_ACTIONS)
        action2 = np.random.choice([a for a in ALL_ACTIONS if a != action1])
        transition = np.random.randint(1, num_actions)  # Transition point
        
        for t in range(num_actions):
            actions[s, t] = action1 if t < transition else action2
    
    video_frames = []
    # Separate position frames per subject: [S][T]
    position_frames_per_subject = [[] for _ in range(num_subjects)]
    
    # Current positions for each subject
    current_positions = [[float(positions[s][0]), float(positions[s][1])] for s in range(num_subjects)]
    action_idx = 0
    
    for f in range(num_video_frames):
        # Apply actions at latent frame boundaries BEFORE recording the frame
        if f > 0 and f % 4 == 0 and action_idx < num_actions:
            for s in range(num_subjects):
                dx, dy = ACTION_DIRS[actions[s, action_idx]]
                new_x = current_positions[s][0] + dx * speed
                new_y = current_positions[s][1] + dy * speed
                
                # Clamp to bounds
                current_positions[s][0] = np.clip(new_x, margin, width - margin - 1)
                current_positions[s][1] = np.clip(new_y, margin, height - margin - 1)
            
            action_idx += 1
        
        # Create video frame with both blobs
        frame = np.ones((3, height, width), dtype=np.float32)  # White background
        for s in range(num_subjects):
            draw_blob_on_frame(frame, current_positions[s][0], current_positions[s][1], 
                             blob_masks[s], rgb_colors[s])
        # Convert to [-1, 1]
        frame = frame * 2 - 1
        video_frames.append(frame)
        
        # Create SEPARATE position maps for each subject
        for s in range(num_subjects):
            pos_frame = create_gaussian_position_map(
                current_positions[s][0], current_positions[s][1], 
                height, width, sigma
            )
            position_frames_per_subject[s].append(pos_frame)
    
    video_frames = np.stack(video_frames, axis=0)  # [T, 3, H, W]
    # Stack per-subject position frames: [S, T, 3, H, W]
    position_frames = np.stack([np.stack(pf, axis=0) for pf in position_frames_per_subject], axis=0)
    
    return video_frames, position_frames, actions, initial_positions


def encode_frames_to_latents(frames, vae, device):
    """
    Encode video/position frames through VAE.
    frames: [T, 3, H, W]
    Returns: [T_latent, 16, H//8, W//8]
    """
    frames_tensor = torch.from_numpy(frames).to(device)  # [T, 3, H, W]
    
    # VAE expects [B, 3, T, H, W]
    frames_for_vae = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, T, H, W]
    
    with torch.no_grad():
        latents = vae.encode_to_latent(frames_for_vae)  # [1, T_latent, 16, H//8, W//8]
    
    return latents[0]  # [T_latent, 16, H//8, W//8]


def encode_position_frames_per_subject(position_frames, vae, device):
    """
    Encode position frames for each subject separately.
    position_frames: [S, T, 3, H, W]
    Returns: [S, T_latent, 16, H//8, W//8]
    """
    num_subjects = position_frames.shape[0]
    latents_list = []
    
    for s in range(num_subjects):
        # Encode this subject's position frames
        subj_frames = position_frames[s]  # [T, 3, H, W]
        subj_latents = encode_frames_to_latents(subj_frames, vae, device)  # [T_latent, 16, H//8, W//8]
        latents_list.append(subj_latents.cpu().numpy())
    
    return np.stack(latents_list, axis=0)  # [S, T_latent, 16, H//8, W//8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='datasets/random_blob_position_2subj')
    parser.add_argument('--num_samples', type=int, default=2500)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--num_video_frames', type=int, default=33)
    parser.add_argument('--blob_size', type=int, default=48, help='Size of blob bounding box')
    parser.add_argument('--speed', type=int, default=16)
    parser.add_argument('--sigma', type=float, default=12.0, help='Gaussian sigma for position map')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--visualize', type=int, default=0, help='Number of samples to visualize (saves combined GIF only)')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load VAE
    print("Loading VAE...")
    from utils.wan_wrapper import WanVAEWrapper
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = WanVAEWrapper().to(device)
    vae.eval()
    
    # Create LMDB
    lmdb_path = os.path.join(args.output_path, 'shard_0.lmdb')
    env = lmdb.open(lmdb_path, map_size=100 * 1024 * 1024 * 1024)  # 100GB
    
    num_latent_frames = (args.num_video_frames - 1) // 4 + 1
    num_actions = num_latent_frames - 1
    
    print(f"Generating {args.num_samples} samples with 2 RANDOM blob subjects...")
    print(f"  Shapes: metaball, polygon, splat, star, crescent")
    print(f"  Colors: random vibrant RGB (different per subject)")
    print(f"  Video frames: {args.num_video_frames} -> {num_latent_frames} latent frames")
    print(f"  Actions per sample per subject: {num_actions}")
    print(f"  Position maps: SEPARATE per subject (for masked cross-attention)")
    
    with env.begin(write=True) as txn:
        for i in tqdm(range(args.num_samples)):
            # Generate sample with 2 random blobs
            video_frames, position_frames, actions, initial_positions = generate_sample_2subj(
                height=args.height,
                width=args.width,
                num_video_frames=args.num_video_frames,
                blob_size=args.blob_size,
                speed=args.speed,
                sigma=args.sigma,
            )
            
            # Encode video through VAE
            video_latents = encode_frames_to_latents(video_frames, vae, device)
            video_latents_np = video_latents.cpu().numpy()  # [T, 16, H, W]
            
            # Encode position frames per subject: [S, T, 3, H, W] -> [S, T, 16, H, W]
            position_latents_np = encode_position_frames_per_subject(position_frames, vae, device)
            
            # Store each field
            txn.put(f'video_latents_{i}_data'.encode(), pickle.dumps(video_latents_np))
            txn.put(f'position_latents_{i}_data'.encode(), pickle.dumps(position_latents_np))  # [S, T, 16, H, W]
            txn.put(f'actions_{i}_data'.encode(), pickle.dumps(actions))  # [S, T-1]
            txn.put(f'initial_positions_{i}_data'.encode(), pickle.dumps(initial_positions))  # [S, 2]
        
        # Store metadata
        txn.put(b'num_samples', str(args.num_samples).encode())
        txn.put(b'video_latents_shape', str(video_latents_np.shape).encode())
        txn.put(b'position_latents_shape', str(position_latents_np.shape).encode())  # [S, T, 16, H, W]
        txn.put(b'actions_shape', str(actions.shape).encode())
        txn.put(b'initial_positions_shape', str(initial_positions.shape).encode())
        txn.put(b'num_subjects', str(2).encode())
    
    env.close()
    
    print(f"\nDataset saved to {args.output_path}")
    print(f"  Video latents shape: {video_latents_np.shape}")
    print(f"  Position latents shape: {position_latents_np.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Initial positions shape: {initial_positions.shape}")
    
    # Visualize samples if requested
    if args.visualize > 0:
        print(f"\nVisualizing {args.visualize} samples...")
        from PIL import Image
        
        vis_dir = os.path.join(args.output_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        env = lmdb.open(lmdb_path, readonly=True)
        
        for idx in range(min(args.visualize, args.num_samples)):
            with env.begin() as txn:
                vid_lat = pickle.loads(txn.get(f'video_latents_{idx}_data'.encode()))
                pos_lat = pickle.loads(txn.get(f'position_latents_{idx}_data'.encode()))  # [S, T, 16, H, W]
                acts = pickle.loads(txn.get(f'actions_{idx}_data'.encode()))
                init_pos = pickle.loads(txn.get(f'initial_positions_{idx}_data'.encode()))
            
            num_subjects = pos_lat.shape[0]
            
            # Decode video
            vid_tensor = torch.from_numpy(vid_lat).unsqueeze(0).to(device)
            with torch.no_grad():
                vid_frames = vae.decode_to_pixel(vid_tensor)[0].cpu().numpy()
            vid_frames = np.clip((vid_frames + 1) / 2 * 255, 0, 255).astype(np.uint8)
            
            # Decode position maps for each subject
            pos_frames_per_subj = []
            for s in range(num_subjects):
                pos_tensor = torch.from_numpy(pos_lat[s]).unsqueeze(0).to(device)
                with torch.no_grad():
                    pos_dec = vae.decode_to_pixel(pos_tensor)[0].cpu().numpy()
                pos_dec = np.clip((pos_dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
                pos_frames_per_subj.append(pos_dec)
            
            # Create combined GIF: video | pos_subj0 | pos_subj1
            pil_vid = [Image.fromarray(f.transpose(1, 2, 0)) for f in vid_frames]
            pil_pos0 = [Image.fromarray(f.transpose(1, 2, 0)) for f in pos_frames_per_subj[0]]
            pil_pos1 = [Image.fromarray(f.transpose(1, 2, 0)) for f in pos_frames_per_subj[1]]
            
            combined = []
            for v, p0, p1 in zip(pil_vid, pil_pos0, pil_pos1):
                c = Image.new('RGB', (v.width * 3, v.height))
                c.paste(v, (0, 0))
                c.paste(p0, (v.width, 0))
                c.paste(p1, (v.width * 2, 0))
                combined.append(c)
            
            gif_path = os.path.join(vis_dir, f'sample_{idx}_combined.gif')
            combined[0].save(gif_path, save_all=True, append_images=combined[1:], duration=100, loop=0)
            
            print(f"  Sample {idx}: subj0={acts[0]}, subj1={acts[1]}")
        
        env.close()
        print(f"  Saved to {vis_dir}/")


if __name__ == '__main__':
    main()
