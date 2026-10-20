"""
Multi-Subject Conditioner with Explicit Position Encoding.

Adds (x, y) position information to conditioning so the model knows
WHERE each subject is, not just WHO it is.

Conditioning vector per subject per frame:
  [action_embed | subject_id_embed | x_pos_embed | y_pos_embed]

Position is computed from initial position + cumulative actions.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal encoding for continuous position values.
    Maps a scalar position to a high-dimensional vector.
    
    IMPORTANT: Scaled to match the norm of random embeddings (std=0.02).
    Without scaling, sinusoidal embeddings have L2 norm ~sqrt(dim/2) ≈ 13.86
    while random embeddings with std=0.02 have L2 norm ~sqrt(dim)*0.02 ≈ 0.39.
    This 35x mismatch causes position to dominate action/subject embeddings!
    """
    
    def __init__(self, dim: int, max_position: float = 256.0, target_std: float = 0.02):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        
        # Precompute frequency bands
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * 
            -(math.log(10000.0) / half_dim)
        )
        self.register_buffer('freqs', freqs)
        
        # Scale factor to match target_std
        # Sinusoidal has L2 norm ~ sqrt(dim/2), we want L2 norm ~ sqrt(dim) * target_std
        # So scale = sqrt(dim) * target_std / sqrt(dim/2) = target_std * sqrt(2)
        self.scale = target_std * math.sqrt(2)
    
    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position: [*] tensor of position values (e.g., x or y coordinates)
        
        Returns:
            encoding: [*, dim] sinusoidal encoding (scaled to match embedding norms)
        """
        # Normalize position to [0, 1] range
        pos_normalized = position / self.max_position
        
        # Expand for frequency multiplication: [*, 1] * [half_dim] -> [*, half_dim]
        pos_expanded = pos_normalized.unsqueeze(-1) * self.freqs.to(position.device)
        
        # Concatenate sin and cos
        encoding = torch.cat([
            torch.sin(pos_expanded),
            torch.cos(pos_expanded)
        ], dim=-1)
        
        # Scale to match embedding norms
        return encoding * self.scale


class PositionAwareMultiSubjectConditioner(nn.Module):
    """
    Multi-subject conditioner with explicit position encoding.
    
    For each subject at each frame, the conditioning is:
      combined = [action_embed | subject_id_embed | x_pos_embed | y_pos_embed]
    
    All concatenated to model_dim total.
    
    Args:
        num_action_classes: Number of action types (5: still, up, down, left, right)
        num_subjects: Number of unique subjects
        model_dim: Output dimension (1536 for Wan model)
        image_size: Size of image for position normalization (256)
        speed: Pixels moved per frame per action (for position computation)
        freeze_embeddings: If True, freeze action and subject embeddings
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        num_subjects: int = 2,
        model_dim: int = 1536,
        image_size: int = 256,
        speed: int = 2,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_subjects = num_subjects
        self.model_dim = model_dim
        self.image_size = image_size
        self.speed = speed
        self.freeze_embeddings = freeze_embeddings
        
        # Split model_dim into 4 parts for concatenation
        # action | subject_id | x_pos | y_pos
        self.action_dim = model_dim // 4
        self.subject_dim = model_dim // 4
        self.x_dim = model_dim // 4
        self.y_dim = model_dim - 3 * (model_dim // 4)  # Handle remainder
        
        # Learnable embeddings
        self.action_embedding = nn.Embedding(num_action_classes, self.action_dim)
        self.subject_embedding = nn.Embedding(num_subjects, self.subject_dim)
        
        # Sinusoidal position encoders
        self.x_encoder = SinusoidalPositionEncoding(self.x_dim, max_position=float(image_size))
        self.y_encoder = SinusoidalPositionEncoding(self.y_dim, max_position=float(image_size))
        
        # Initialize embeddings
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        # Orthogonal init for subjects if freezing
        if freeze_embeddings:
            if num_subjects <= self.subject_dim:
                ortho_matrix = torch.zeros(num_subjects, self.subject_dim)
                nn.init.orthogonal_(ortho_matrix)
                ortho_matrix = ortho_matrix * 0.5
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
        
        print(f"PositionAwareMultiSubjectConditioner initialized:")
        print(f"  {num_action_classes} actions × {self.action_dim}d")
        print(f"  {num_subjects} subjects × {self.subject_dim}d")
        print(f"  X position: {self.x_dim}d (sinusoidal)")
        print(f"  Y position: {self.y_dim}d (sinusoidal)")
        print(f"  Speed: {speed} pixels/frame")
        print(f"  Output: {model_dim}d")
    
    def compute_positions(
        self,
        initial_positions: torch.Tensor,  # [B, S, 2] - (x, y) for each subject
        action_labels: torch.Tensor,       # [B, S, T] action indices
    ) -> tuple:
        """
        Compute positions for each frame based on initial position and actions.
        
        Action semantics: action[t] causes transition from frame t to frame t+1.
        So for frame t, position = initial + sum(actions[0:t]) * speed
        
        Args:
            initial_positions: [B, S, 2] initial (x, y) for each subject
            action_labels: [B, S, T] action for each subject for each transition
        
        Returns:
            x_positions: [B, S, T] x position at start of action t
            y_positions: [B, S, T] y position at start of action t
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        
        # Get dx, dy for each action: [B, S, T]
        dx = self.action_dx[action_labels]  # [B, S, T]
        dy = self.action_dy[action_labels]  # [B, S, T]
        
        # Cumulative displacement: for frame t, sum actions[0:t-1]
        # But action[t] produces frame t+1, so position at action t is after actions[0:t-1]
        # We compute cumulative sum BEFORE current action
        cumsum_dx = torch.zeros(B, S, T, device=device)
        cumsum_dy = torch.zeros(B, S, T, device=device)
        
        if T > 1:
            # cumsum_dx[:, :, t] = sum(dx[:, :, 0:t])
            cumsum_dx[:, :, 1:] = torch.cumsum(dx[:, :, :-1], dim=2)
            cumsum_dy[:, :, 1:] = torch.cumsum(dy[:, :, :-1], dim=2)
        
        # Position = initial + cumulative displacement * speed
        x_positions = initial_positions[:, :, 0:1] + cumsum_dx * self.speed  # [B, S, T]
        y_positions = initial_positions[:, :, 1:2] + cumsum_dy * self.speed  # [B, S, T]
        
        return x_positions, y_positions
    
    def forward(
        self,
        action_labels: torch.Tensor,       # [B, S, T] action indices
        subject_ids: torch.Tensor,          # [B, S] subject indices
        initial_positions: torch.Tensor,   # [B, S, 2] initial (x, y) positions
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask
    ) -> torch.Tensor:
        """
        Generate position-aware action-subject embeddings.
        
        Args:
            action_labels: [B, S, T] action indices for each subject at each timestep
            subject_ids: [B, S] subject indices
            initial_positions: [B, S, 2] initial (x, y) position for each subject
            action_mask: [B, S, T] optional mask (1=valid, 0=masked)
        
        Returns:
            embeddings: [B, S*T, model_dim] combined embeddings
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
        
        # 3. Position embeddings
        x_pos, y_pos = self.compute_positions(initial_positions, action_labels)
        
        # Apply mask to positions: masked positions use initial position
        if action_mask is not None:
            mask_expanded = action_mask.unsqueeze(-1).float()
            initial_x = initial_positions[:, :, 0:1].unsqueeze(2).expand(-1, -1, T, -1)
            initial_y = initial_positions[:, :, 1:2].unsqueeze(2).expand(-1, -1, T, -1)
            x_pos = x_pos * mask_expanded.squeeze(-1) + initial_x.squeeze(-1) * (1 - mask_expanded.squeeze(-1))
            y_pos = y_pos * mask_expanded.squeeze(-1) + initial_y.squeeze(-1) * (1 - mask_expanded.squeeze(-1))
        
        # Encode positions: [B, S, T, x_dim/y_dim]
        x_embeds = self.x_encoder(x_pos)
        y_embeds = self.y_encoder(y_pos)
        
        # 4. Concatenate all parts: [B, S, T, model_dim]
        combined = torch.cat([
            action_embeds,    # [B, S, T, action_dim]
            subject_embeds,   # [B, S, T, subject_dim]
            x_embeds,         # [B, S, T, x_dim]
            y_embeds,         # [B, S, T, y_dim]
        ], dim=-1)
        
        # Flatten S and T: [B, S*T, model_dim]
        combined = combined.view(B, S * T, self.model_dim)
        
        return combined.to(dtype)


def create_position_conditioner(config, model_dim: int = 1536):
    """Create position-aware conditioner from config."""
    return PositionAwareMultiSubjectConditioner(
        num_action_classes=getattr(config, 'num_action_classes', 5),
        num_subjects=getattr(config, 'num_subjects_total', 2),
        model_dim=model_dim,
        image_size=getattr(config, 'position_image_size', 256),
        speed=getattr(config, 'position_speed', 2),
        freeze_embeddings=getattr(config, 'freeze_embeddings', True),
    )

