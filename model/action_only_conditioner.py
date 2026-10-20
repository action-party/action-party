"""
Action-Only Spatial Conditioner.

Outputs ONLY action embeddings (no subject embedding) with (frame, h, w) positions
for use with 3D RoPE in cross-attention.

This is a simplified version of SpatialMultiSubjectConditioner that tests whether
spatial RoPE alone (without subject identity) is sufficient for multi-object control.
"""

import torch
import torch.nn as nn


class ActionOnlySpatialConditioner(nn.Module):
    """
    Action-only conditioner with spatial RoPE positions.
    
    Each conditioning token is just an action embedding (no subject part).
    The subject identity comes purely from the spatial RoPE position.
    
    Outputs:
        - embeddings: [B, S*T, model_dim] - action embeddings (projected to model_dim)
        - positions: [B, S*T, 3] - (frame, h_patch, w_patch) per token
    
    Args:
        num_action_classes: Number of action types (5: still, up, down, left, right)
        num_subjects: Number of subjects (for position tracking, not embedding)
        model_dim: Output dimension (1536 for Wan model)
        patch_size: Size of each latent patch in pixels (8 for Wan VAE)
        speed: Pixels moved per frame per action
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        num_subjects: int = 2,
        model_dim: int = 1536,
        patch_size: int = 8,
        speed: int = 6,
        zero_position_for_masked: bool = False,
        position_before_action: bool = False,
        image_size: float = 256.0,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_subjects = num_subjects
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.speed = speed
        self.zero_position_for_masked = zero_position_for_masked
        self.position_before_action = position_before_action
        self.image_size = image_size
        
        # Action embedding - directly to model_dim (no subject part)
        self.action_embedding = nn.Embedding(num_action_classes, model_dim)
        
        # Initialize with small std using FIXED seed for reproducibility
        # This ensures the same embeddings across model instances
        generator = torch.Generator()
        generator.manual_seed(42)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02, generator=generator)
        
        # Freeze embeddings (action semantics are fixed)
        self.action_embedding.weight.requires_grad = False
        
        # Action to direction mapping (for absolute action spaces only)
        # 0: still, 1: up, 2: down, 3: left, 4: right
        # For relative action spaces (7 actions with orientation), positions are
        # provided via frame_coords instead of computed from action deltas.
        if num_action_classes <= 5:
            self.register_buffer('action_dx', torch.tensor([0, 0, 0, -1, 1], dtype=torch.float32))
            self.register_buffer('action_dy', torch.tensor([0, -1, 1, 0, 0], dtype=torch.float32))
        else:
            # Dummy deltas for relative action spaces - won't be used when frame_coords is provided
            self.register_buffer('action_dx', torch.zeros(num_action_classes, dtype=torch.float32))
            self.register_buffer('action_dy', torch.zeros(num_action_classes, dtype=torch.float32))
        
        max_patch = int(image_size / (patch_size * 2)) - 1
        print(f"ActionOnlySpatialConditioner initialized:")
        print(f"  {num_action_classes} actions → {model_dim}d embeddings")
        print(f"  {num_subjects} subjects (position tracking only, no subject embedding)")
        print(f"  Output: {model_dim}d action embeddings + (f, h, w) positions")
        print(f"  Image size: {image_size}, Patch size: {patch_size}px, Grid: {max_patch+1}x{max_patch+1} (positions 0-{max_patch})")
        print(f"  Speed: {speed}px/frame")
        print(f"  Spatial position: {'BEFORE action (where subject IS)' if position_before_action else 'AFTER action (where subject WILL BE)'}")
        print(f"  Masked token position: {'(0,0,0) - no rotation' if zero_position_for_masked else 'initial position at frame 0'}")
        print(f"  Embeddings: FROZEN")
    
    def compute_positions(
        self,
        initial_positions: torch.Tensor,  # [B, S, 2] - (x, y) in pixels
        action_labels: torch.Tensor,       # [B, S, T] action indices
        frame_coords: torch.Tensor = None, # [B, T_frames, S, 2] normalized coords in [0, 1]
    ) -> torch.Tensor:
        """
        Compute (frame, h_patch, w_patch) positions for each conditioning token.
        
        Args:
            initial_positions: [B, S, 2] initial (x, y) in pixels for each subject
            action_labels: [B, S, T] action for each subject for each frame transition
            frame_coords: [B, T_frames, S, 2] optional GT coordinates in [0, 1].
                          When provided, positions are computed directly from these
                          instead of from action deltas. Required for relative action
                          spaces (7 actions with orientation) where action->displacement
                          depends on agent orientation.
        
        Returns:
            positions: [B, S, T, 3] - (frame, h_patch, w_patch) for each token
                       where frame = t+1 (action-as-result semantics)
                       
        Position modes:
            - position_before_action=False (default): h,w is position AFTER action
            - position_before_action=True: h,w is position BEFORE action
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        
        effective_patch_size = self.patch_size * 2  # 8 * 2 = 16
        max_patch = int(self.image_size / effective_patch_size) - 1  # 256->15, 512->31
        
        if frame_coords is not None:
            # USE GT COORDINATES DIRECTLY
            # frame_coords: [B, T_frames, S, 2] in [0, 1] normalized space
            # T_frames = T + 1 (includes frame 0 = initial)
            # action[t] corresponds to transition from frame t to frame t+1
            # With position_before_action: action[t] gets position at frame t
            # Without: action[t] gets position at frame t+1
            
            # Convert normalized coords to pixel coords: [0, 1] -> [0, image_size]
            coords_pixels = frame_coords * self.image_size  # [B, T_frames, S, 2]
            
            if self.position_before_action:
                # action[t] gets position at frame t (where subject IS before taking action)
                # frame_coords[t] for action[t], t=0..T-1
                x_pixels = coords_pixels[:, :T, :, 0]  # [B, T, S]
                y_pixels = coords_pixels[:, :T, :, 1]  # [B, T, S]
            else:
                # action[t] gets position at frame t+1 (where subject WILL BE after action)
                x_pixels = coords_pixels[:, 1:T+1, :, 0]  # [B, T, S]
                y_pixels = coords_pixels[:, 1:T+1, :, 1]  # [B, T, S]
            
            # Transpose to [B, S, T] to match expected layout
            x_pixels = x_pixels.permute(0, 2, 1)  # [B, S, T]
            y_pixels = y_pixels.permute(0, 2, 1)  # [B, S, T]
            
            h_patch = (y_pixels / effective_patch_size).clamp(0, max_patch).long()
            w_patch = (x_pixels / effective_patch_size).clamp(0, max_patch).long()
        else:
            # COMPUTE FROM ACTION DELTAS (original absolute action space logic)
            # Get dx, dy for each action: [B, S, T]
            dx = self.action_dx[action_labels]  # [B, S, T]
            dy = self.action_dy[action_labels]  # [B, S, T]
            
            # Cumulative displacement
            cumsum_dx = torch.cumsum(dx, dim=2)  # [B, S, T]
            cumsum_dy = torch.cumsum(dy, dim=2)  # [B, S, T]
            
            if self.position_before_action:
                # Position BEFORE action: use cumsum shifted by 1 (position at frame t, not t+1)
                prev_cumsum_dx = torch.cat([
                    torch.zeros(B, S, 1, device=device, dtype=dx.dtype),
                    cumsum_dx[:, :, :-1]
                ], dim=2)
                prev_cumsum_dy = torch.cat([
                    torch.zeros(B, S, 1, device=device, dtype=dy.dtype),
                    cumsum_dy[:, :, :-1]
                ], dim=2)
                x_pixels = initial_positions[:, :, 0:1] + prev_cumsum_dx * self.speed
                y_pixels = initial_positions[:, :, 1:2] + prev_cumsum_dy * self.speed
            else:
                # Position AFTER action (default): use full cumsum
                x_pixels = initial_positions[:, :, 0:1] + cumsum_dx * self.speed
                y_pixels = initial_positions[:, :, 1:2] + cumsum_dy * self.speed
            
            h_patch = (y_pixels / effective_patch_size).clamp(0, max_patch).long()  # [B, S, T]
            w_patch = (x_pixels / effective_patch_size).clamp(0, max_patch).long()  # [B, S, T]
        
        # Frame indices: action[t] gets RoPE position t+1 (action-as-result)
        frame_indices = torch.arange(1, T + 1, device=device).view(1, 1, T).expand(B, S, T)
        
        # Stack to [B, S, T, 3]
        positions = torch.stack([frame_indices.float(), h_patch.float(), w_patch.float()], dim=-1)
        
        return positions
    
    def forward(
        self,
        action_labels: torch.Tensor,       # [B, S, T] action indices
        subject_ids: torch.Tensor,         # [B, S] subject indices (unused, kept for API compat)
        initial_positions: torch.Tensor,   # [B, S, 2] initial (x, y) in pixels
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask
        frame_coords: torch.Tensor = None, # [B, T_frames, S, 2] optional GT coords in [0, 1]
    ) -> tuple:
        """
        Generate action embeddings and positions for spatial RoPE.
        
        Args:
            action_labels: [B, S, T] action indices for each subject at each timestep
            subject_ids: [B, S] subject indices (UNUSED - kept for API compatibility)
            initial_positions: [B, S, 2] initial (x, y) position in pixels
            action_mask: [B, S, T] optional mask (1=valid, 0=masked)
            frame_coords: [B, T_frames, S, 2] optional GT coordinates in [0, 1].
                          When provided, spatial RoPE positions are computed from these
                          instead of from action deltas. Required for relative action
                          spaces (e.g., 7 actions with orientation).
        
        Returns:
            embeddings: [B, S*T, model_dim] action embeddings only
            positions: [B, S*T, 3] (frame, h_patch, w_patch) for RoPE
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        dtype = self.action_embedding.weight.dtype
        
        # Action embeddings only: [B, S, T, model_dim]
        embeddings = self.action_embedding(action_labels)
        
        # Compute positions: [B, S, T, 3]
        positions = self.compute_positions(initial_positions, action_labels, frame_coords=frame_coords)
        
        # Apply mask to BOTH embeddings AND positions for future/masked actions
        # This prevents information leakage: the model should not see future action types
        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(-1)  # [B, S, T, 1]
            
            # Zero out embeddings for masked actions (future frames)
            # This ensures no future action type information leaks through
            embeddings = embeddings * mask_expanded  # [B, S, T, model_dim]
            
            if self.zero_position_for_masked:
                # Masked tokens get (0, 0, 0) - no rotation applied
                # This treats all masked tokens uniformly regardless of initial position
                zero_pos = torch.zeros(B, S, T, 3, device=device, dtype=positions.dtype)
                positions = positions * mask_expanded + zero_pos * (1 - mask_expanded)
            else:
                # Masked tokens use initial position at frame 0 (legacy behavior)
                effective_patch_size = self.patch_size * 2
                max_patch = int(self.image_size / effective_patch_size) - 1
                initial_h = (initial_positions[:, :, 1:2] / effective_patch_size).clamp(0, max_patch)
                initial_w = (initial_positions[:, :, 0:1] / effective_patch_size).clamp(0, max_patch)
                initial_pos = torch.stack([
                    torch.zeros_like(initial_h),  # frame 0
                    initial_h,
                    initial_w
                ], dim=-1).expand(-1, -1, T, -1)  # [B, S, T, 3]
                positions = positions * mask_expanded + initial_pos * (1 - mask_expanded)
        
        # Flatten S and T: [B, S*T, ...]
        embeddings = embeddings.view(B, S * T, self.model_dim)
        positions = positions.view(B, S * T, 3)
        
        return embeddings.to(dtype), positions


def create_action_only_conditioner(config, model_dim: int = 1536):
    """Create action-only conditioner from config."""
    return ActionOnlySpatialConditioner(
        num_action_classes=getattr(config, 'num_action_classes', 5),
        num_subjects=getattr(config, 'num_subjects_total', 2),
        model_dim=model_dim,
        patch_size=getattr(config, 'latent_patch_size', 8),
        speed=getattr(config, 'position_speed', 6),
        zero_position_for_masked=getattr(config, 'zero_position_for_masked', False),
        position_before_action=getattr(config, 'position_before_action', False),
        image_size=float(getattr(config, 'synthetic_height', 256)),
    )

