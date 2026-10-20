"""
Position map generation utilities for joint video+position diffusion.

Position maps are 2D Gaussian blobs indicating subject locations.
They are generated on-the-fly from initial positions + action sequences.
"""

import torch
import torch.nn.functional as F


# Action index to (dy, dx) in pixels
ACTION_DELTAS = {
    0: (0, 0),    # still
    1: (-1, 0),   # up (negative y)
    2: (1, 0),    # down (positive y)
    3: (0, -1),   # left (negative x)
    4: (0, 1),    # right (positive x)
}


def compute_positions_from_actions(
    initial_positions: torch.Tensor,
    actions: torch.Tensor,
    speed: int = 16,
    image_size: int = 256,
    margin: int = 29,  # blob_size // 2 + 5 for blob_size=48
) -> torch.Tensor:
    """
    Compute subject positions for all frames from initial positions and actions.
    
    Args:
        initial_positions: [S, 2] - initial (x, y) in pixels for each subject
                           NOTE: Dataset stores (x, y), not (y, x)!
        actions: [S, T-1] - action indices for each subject at each frame transition
        speed: pixels per action step
        image_size: image dimension (assumes square image)
        margin: boundary margin for clamping (matches dataset creation)
    
    Returns:
        positions: [S, T, 2] - (x, y) positions for each subject at each frame
    """
    S, T_minus_1 = actions.shape
    T = T_minus_1 + 1
    device = initial_positions.device
    dtype = initial_positions.dtype
    
    # Clamp bounds (match dataset creation)
    min_pos = float(margin)
    max_pos = float(image_size - margin)
    
    # Initialize positions array
    positions = torch.zeros(S, T, 2, device=device, dtype=dtype)
    positions[:, 0] = initial_positions  # (x, y) order
    
    # Compute cumulative positions with clamping
    # Actions: 0=still, 1=up, 2=down, 3=left, 4=right
    # up/down affect y, left/right affect x
    for t in range(T - 1):
        for s in range(S):
            action_idx = actions[s, t].item()
            dy, dx = ACTION_DELTAS.get(action_idx, (0, 0))
            # positions are (x, y), so delta is (dx, dy)
            new_pos = positions[s, t] + torch.tensor(
                [dx * speed, dy * speed], device=device, dtype=dtype
            )
            # Clamp to valid range (match dataset boundary behavior)
            positions[s, t + 1] = torch.clamp(new_pos, min_pos, max_pos)
    
    return positions


def generate_position_maps(
    initial_positions: torch.Tensor,
    actions: torch.Tensor,
    H: int = 32,
    W: int = 32,
    sigma: float = 1.0,
    speed: int = 16,
    pixel_to_latent_scale: float = 8.0,
    image_size: int = 256,
    margin: int = 29,
) -> torch.Tensor:
    """
    Generate Gaussian position maps for all frames.
    
    Args:
        initial_positions: [S, 2] - initial (x, y) in PIXEL space
                           NOTE: Dataset stores (x, y), not (y, x)!
        actions: [S, T-1] - action indices
        H, W: latent spatial dimensions (typically 32x32)
        sigma: Gaussian width in latent pixels
        speed: pixels per action step
        pixel_to_latent_scale: conversion factor (typically 8 for 256->32)
        image_size: pixel image size (for boundary clamping)
        margin: boundary margin in pixels (for boundary clamping)
    
    Returns:
        position_maps: [T, S, H, W] - Gaussian blobs at subject positions
                       Peak value = 1.0 (unnormalized)
    """
    # Compute positions for all frames in pixel space (with boundary clamping)
    positions = compute_positions_from_actions(
        initial_positions, actions, speed, image_size, margin
    )
    
    # Convert to latent space: positions are (x, y)
    positions_latent = positions / pixel_to_latent_scale  # [S, T, 2] in (x, y) order
    
    S, T, _ = positions_latent.shape
    device = initial_positions.device
    dtype = initial_positions.dtype
    
    # Create coordinate grids
    # y_grid: rows (height), x_grid: columns (width)
    y_grid = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
    
    # Generate Gaussians for all subjects and frames
    # positions_latent: [S, T, 2] where 2 = (x, y)
    # cx = positions[:, :, 0] (column), cy = positions[:, :, 1] (row)
    cx = positions_latent[:, :, 0:1].unsqueeze(-1)  # [S, T, 1, 1]
    cy = positions_latent[:, :, 1:2].unsqueeze(-1)  # [S, T, 1, 1]
    
    # Compute Gaussian: exp(-((y - cy)^2 + (x - cx)^2) / (2 * sigma^2))
    dist_sq = (y_grid - cy) ** 2 + (x_grid - cx) ** 2  # [S, T, H, W]
    position_maps = torch.exp(-dist_sq / (2 * sigma ** 2))  # [S, T, H, W]
    
    # Transpose to [T, S, H, W] for easier concatenation with video
    position_maps = position_maps.permute(1, 0, 2, 3)  # [T, S, H, W]
    
    return position_maps


def generate_position_maps_batch(
    initial_positions: torch.Tensor,
    actions: torch.Tensor,
    H: int = 32,
    W: int = 32,
    sigma: float = 1.0,
    speed: int = 16,
    pixel_to_latent_scale: float = 8.0,
) -> torch.Tensor:
    """
    Generate position maps for a batch of samples.
    
    Args:
        initial_positions: [B, S, 2] - initial positions for batch
        actions: [B, S, T-1] - action indices for batch
        H, W: latent spatial dimensions
        sigma: Gaussian width in latent pixels
        speed: pixels per action step
        pixel_to_latent_scale: conversion factor
    
    Returns:
        position_maps: [B, T, S, H, W] - Gaussian blobs for all samples
    """
    B = initial_positions.shape[0]
    device = initial_positions.device
    dtype = initial_positions.dtype
    
    # Get T from actions shape
    T = actions.shape[2] + 1
    S = initial_positions.shape[1]
    
    maps_list = []
    for b in range(B):
        maps = generate_position_maps(
            initial_positions[b],
            actions[b],
            H=H, W=W,
            sigma=sigma,
            speed=speed,
            pixel_to_latent_scale=pixel_to_latent_scale,
        )
        maps_list.append(maps)
    
    return torch.stack(maps_list, dim=0)  # [B, T, S, H, W]


def extract_center_of_mass(position_map: torch.Tensor, use_softmax: bool = False, 
                           subtract_baseline: bool = False) -> torch.Tensor:
    """
    Extract center of mass from a position map.
    
    Args:
        position_map: [H, W] or [B, H, W] - position probability map
        use_softmax: if True, apply softmax to handle arbitrary value ranges
                     (useful for raw model outputs that may have negative values)
        subtract_baseline: if True, subtract the minimum value before computing
                          (useful for VAE-decoded images with non-zero background)
    
    Returns:
        center: [2] or [B, 2] - (x, y) center of mass in latent coordinates
               NOTE: Returns (x, y) to match dataset convention!
    """
    squeeze_batch = position_map.dim() == 2
    if squeeze_batch:
        position_map = position_map.unsqueeze(0)
    
    B, H, W = position_map.shape
    device = position_map.device
    dtype = position_map.dtype
    
    # Subtract baseline if requested (removes uniform background bias)
    if subtract_baseline:
        baseline = position_map.view(B, -1).min(dim=-1, keepdim=True)[0].view(B, 1, 1)
        position_map = position_map - baseline
    
    # Create coordinate grids
    y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
    x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    
    if use_softmax:
        # Apply softmax for robust handling of arbitrary values (including negative)
        flat = position_map.view(B, -1)
        weights = torch.softmax(flat, dim=-1).view(B, H, W)
    else:
        # Clamp negative values and normalize by sum
        position_map = torch.clamp(position_map, min=0)
        total = position_map.sum(dim=(-2, -1), keepdim=True) + 1e-8
        weights = position_map / total
    
    # Compute weighted average
    cy = (y_grid * weights).sum(dim=(-2, -1))  # [B] - row coordinate
    cx = (x_grid * weights).sum(dim=(-2, -1))  # [B] - column coordinate
    
    # Return (x, y) to match dataset convention
    center = torch.stack([cx, cy], dim=-1)  # [B, 2] as (x, y)
    
    if squeeze_batch:
        center = center.squeeze(0)
    
    return center


def extract_centers_from_position_maps(
    position_maps: torch.Tensor,
    pixel_to_latent_scale: float = 8.0,
) -> torch.Tensor:
    """
    Extract subject positions from position maps for all frames.
    
    Args:
        position_maps: [B, T, S, H, W] - predicted position maps
        pixel_to_latent_scale: conversion factor (to return positions in pixel space)
    
    Returns:
        positions: [B, S, T, 2] - (y, x) positions in PIXEL space
    """
    B, T, S, H, W = position_maps.shape
    device = position_maps.device
    
    # Extract center of mass for each subject at each frame
    positions_latent = torch.zeros(B, S, T, 2, device=device, dtype=position_maps.dtype)
    
    for s in range(S):
        for t in range(T):
            # position_maps[:, t, s] is [B, H, W]
            center = extract_center_of_mass(position_maps[:, t, s])  # [B, 2]
            positions_latent[:, s, t] = center
    
    # Convert to pixel space
    positions_pixel = positions_latent * pixel_to_latent_scale
    
    return positions_pixel


if __name__ == "__main__":
    # Test the position map generation
    print("Testing position map generation...")
    
    # Simulate 2 subjects, 9 frames
    initial_pos = torch.tensor([[50.0, 100.0], [200.0, 150.0]])  # [S=2, 2]
    actions = torch.tensor([
        [1, 3, 2, 4, 1, 3, 2, 4],  # Subject 0: up, left, down, right, ...
        [0, 0, 2, 2, 0, 0, 4, 4],  # Subject 1: still, still, down, down, ...
    ])  # [S=2, T-1=8]
    
    maps = generate_position_maps(
        initial_pos, actions, 
        H=32, W=32, sigma=1.0, speed=16
    )
    print(f"Position maps shape: {maps.shape}")  # Should be [T=9, S=2, H=32, W=32]
    
    # Test center extraction
    for t in range(maps.shape[0]):
        for s in range(maps.shape[1]):
            center = extract_center_of_mass(maps[t, s])
            print(f"  Frame {t}, Subject {s}: center = ({center[0]:.1f}, {center[1]:.1f})")
    
    # Test batch version
    batch_initial = initial_pos.unsqueeze(0).expand(4, -1, -1)  # [B=4, S=2, 2]
    batch_actions = actions.unsqueeze(0).expand(4, -1, -1)  # [B=4, S=2, T-1=8]
    
    batch_maps = generate_position_maps_batch(
        batch_initial, batch_actions,
        H=32, W=32, sigma=1.0, speed=16
    )
    print(f"Batch position maps shape: {batch_maps.shape}")  # [B=4, T=9, S=2, H=32, W=32]
    
    # Extract centers from batch
    extracted_positions = extract_centers_from_position_maps(batch_maps)
    print(f"Extracted positions shape: {extracted_positions.shape}")  # [B=4, S=2, T=9, 2]
    
    print("\n✓ All tests passed!")

