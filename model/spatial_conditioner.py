"""
Spatial Multi-Subject Conditioner.

Outputs action+subject embeddings AND separate (frame, h, w) positions
for use with 3D RoPE in cross-attention.

Unlike PositionAwareMultiSubjectConditioner which embeds positions as content,
this conditioner provides positions as separate tensors for RoPE application.
"""

import torch
import torch.nn as nn


class SpatialMultiSubjectConditioner(nn.Module):
    """
    Multi-subject conditioner that provides positions for spatial RoPE.
    
    Outputs:
        - embeddings: [B, S*T, model_dim] - action+subject concatenated
        - positions: [B, S*T, 3] - (frame, h_patch, w_patch) per token
    
    Args:
        num_action_classes: Number of action types (5: still, up, down, left, right)
        num_subjects: Number of unique subjects
        model_dim: Output dimension (1536 for Wan model)
        patch_size: Size of each latent patch in pixels (8 for Wan VAE)
        speed: Pixels moved per frame per action
        freeze_embeddings: If True, freeze action and subject embeddings
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        num_subjects: int = 2,
        model_dim: int = 1536,
        patch_size: int = 8,
        speed: int = 6,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_subjects = num_subjects
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.speed = speed
        self.freeze_embeddings = freeze_embeddings
        
        # Split model_dim into 2 parts: action and subject
        self.action_dim = model_dim // 2
        self.subject_dim = model_dim - self.action_dim
        
        # Learnable embeddings
        self.action_embedding = nn.Embedding(num_action_classes, self.action_dim)
        self.subject_embedding = nn.Embedding(num_subjects, self.subject_dim)
        
        # Initialize embeddings with small std (matching other embeddings)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        # Orthogonal init for subjects if freezing
        if freeze_embeddings:
            if num_subjects <= self.subject_dim:
                ortho_matrix = torch.zeros(num_subjects, self.subject_dim)
                nn.init.orthogonal_(ortho_matrix)
                ortho_matrix = ortho_matrix * 0.02 * (self.subject_dim ** 0.5)  # Scale to match std=0.02
                self.subject_embedding.weight.data.copy_(ortho_matrix)
            else:
                nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
            
            self.action_embedding.weight.requires_grad = False
            self.subject_embedding.weight.requires_grad = False
            print(f"  Embeddings: FROZEN")
        else:
            nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
        
        # Action to direction mapping (same as dataset)
        # 0: still, 1: up, 2: down, 3: left, 4: right
        self.register_buffer('action_dx', torch.tensor([0, 0, 0, -1, 1], dtype=torch.float32))
        self.register_buffer('action_dy', torch.tensor([0, -1, 1, 0, 0], dtype=torch.float32))
        
        print(f"SpatialMultiSubjectConditioner initialized:")
        print(f"  {num_action_classes} actions × {self.action_dim}d")
        print(f"  {num_subjects} subjects × {self.subject_dim}d")
        print(f"  Output: {model_dim}d embeddings + (f, h, w) positions")
        print(f"  Patch size: {patch_size}px, Speed: {speed}px/frame")
    
    def compute_positions(
        self,
        initial_positions: torch.Tensor,  # [B, S, 2] - (x, y) in pixels
        action_labels: torch.Tensor,       # [B, S, T] action indices
    ) -> torch.Tensor:
        """
        Compute (frame, h_patch, w_patch) positions for each conditioning token.
        
        IMPORTANT: Uses PREVIOUS frame's position (before action) to align with
        coord token spatial RoPE, which also uses the previous frame's position.
        
        This alignment ensures:
        - Coord token at frame t (spatial pos = frame t-1) 
        - Action embedding for step t-1 (spatial pos = frame t-1)
        Both have the same spatial RoPE, enabling proper attention.
        
        Args:
            initial_positions: [B, S, 2] initial (x, y) in pixels for each subject
            action_labels: [B, S, T] action for each subject for each frame transition
        
        Returns:
            positions: [B, S, T, 3] - (frame, h_patch, w_patch) for each token
                       where frame = t+1 (action-as-result semantics)
                       and spatial = position BEFORE action (frame t)
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        
        # Get dx, dy for each action: [B, S, T]
        dx = self.action_dx[action_labels]  # [B, S, T]
        dy = self.action_dy[action_labels]  # [B, S, T]
        
        # Use PREVIOUS frame's position (before action is applied)
        # For action[t] at frame t, use position at frame t (not t+1)
        # This aligns with coord token spatial RoPE which uses position[t-1] for frame t
        
        # Compute cumulative sum SHIFTED by 1 to get position BEFORE each action
        # For t=0: use initial position (frame 0)
        # For t=1: use initial + action[0]*speed (frame 1)
        # For t=T-1: use initial + sum(actions[0:T-1])*speed (frame T-1)
        
        # First, compute position at each frame
        cumsum_dx = torch.cumsum(dx, dim=2)  # [B, S, T] - cumsum after each action
        cumsum_dy = torch.cumsum(dy, dim=2)  # [B, S, T]
        
        # Shift to get position BEFORE each action
        # Position before action[t] = initial + sum(actions[0:t]) * speed
        zero_pad = torch.zeros(B, S, 1, device=device, dtype=cumsum_dx.dtype)
        prev_cumsum_dx = torch.cat([zero_pad, cumsum_dx[:, :, :-1]], dim=2)  # [B, S, T]
        prev_cumsum_dy = torch.cat([zero_pad, cumsum_dy[:, :, :-1]], dim=2)  # [B, S, T]
        
        # Pixel positions at frame t (before action[t] is applied)
        x_pixels = initial_positions[:, :, 0:1] + prev_cumsum_dx * self.speed  # [B, S, T]
        y_pixels = initial_positions[:, :, 1:2] + prev_cumsum_dy * self.speed  # [B, S, T]
        
        # Convert to patch coordinates (clamp to valid range)
        # Model grid is 16x16 (256 pixels / 16 = 16 patches)
        # This accounts for both VAE (8x) and model patch embedding (2x) = 16x total
        effective_patch_size = self.patch_size * 2  # 8 * 2 = 16
        h_patch = (y_pixels / effective_patch_size).clamp(0, 15).long()  # [B, S, T]
        w_patch = (x_pixels / effective_patch_size).clamp(0, 15).long()  # [B, S, T]
        
        # Frame indices: action[t] gets RoPE position t+1 (action-as-result semantics)
        frame_indices = torch.arange(1, T + 1, device=device).view(1, 1, T).expand(B, S, T)
        
        # Stack to [B, S, T, 3]
        positions = torch.stack([frame_indices.float(), h_patch.float(), w_patch.float()], dim=-1)
        
        return positions
    
    def forward(
        self,
        action_labels: torch.Tensor,       # [B, S, T] action indices
        subject_ids: torch.Tensor,         # [B, S] subject indices
        initial_positions: torch.Tensor,   # [B, S, 2] initial (x, y) in pixels
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask
    ) -> tuple:
        """
        Generate action-subject embeddings and positions for spatial RoPE.
        
        Args:
            action_labels: [B, S, T] action indices for each subject at each timestep
            subject_ids: [B, S] subject indices
            initial_positions: [B, S, 2] initial (x, y) position in pixels
            action_mask: [B, S, T] optional mask (1=valid, 0=masked)
        
        Returns:
            embeddings: [B, S*T, model_dim] action+subject embeddings
            positions: [B, S*T, 3] (frame, h_patch, w_patch) for RoPE
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        dtype = self.action_embedding.weight.dtype
        
        # Apply mask if provided
        if action_mask is not None:
            action_labels = action_labels * action_mask.long()
        
        # 1. Action embeddings: [B, S, T, action_dim]
        action_embeds = self.action_embedding(action_labels)
        
        # 2. Subject embeddings: [B, S, subject_dim] -> [B, S, T, subject_dim]
        subject_embeds = self.subject_embedding(subject_ids)
        subject_embeds = subject_embeds.unsqueeze(2).expand(-1, -1, T, -1)
        
        # 3. Concatenate action and subject: [B, S, T, model_dim]
        embeddings = torch.cat([action_embeds, subject_embeds], dim=-1)
        
        # 4. Compute positions: [B, S, T, 3]
        positions = self.compute_positions(initial_positions, action_labels)
        
        # Apply mask to positions: masked positions use initial position
        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(-1)  # [B, S, T, 1]
            # For masked positions, use frame 0 and initial position
            # Use same effective_patch_size as compute_positions (16 = 8 * 2)
            effective_patch_size = self.patch_size * 2
            initial_h = (initial_positions[:, :, 1:2] / effective_patch_size).clamp(0, 15)
            initial_w = (initial_positions[:, :, 0:1] / effective_patch_size).clamp(0, 15)
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


def create_spatial_conditioner(config, model_dim: int = 1536):
    """Create spatial conditioner from config."""
    return SpatialMultiSubjectConditioner(
        num_action_classes=getattr(config, 'num_action_classes', 5),
        num_subjects=getattr(config, 'num_subjects_total', 2),
        model_dim=model_dim,
        patch_size=getattr(config, 'latent_patch_size', 8),
        speed=getattr(config, 'position_speed', 6),
        freeze_embeddings=getattr(config, 'freeze_embeddings', True),
    )

