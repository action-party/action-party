"""
Single Action Per Frame Conditioner.

Outputs ONE action embedding per frame, rotated to the position of the active subject.
This removes the binding problem - there's only one action per frame, and its RoPE
position specifies which spatial location it applies to.

Key simplification:
- Instead of S subjects × T timesteps = S*T tokens
- Just T tokens (one per frame transition)
- Each token is the action + position of the "active" subject for that frame
"""

import torch
import torch.nn as nn


class SingleActionPerFrameConditioner(nn.Module):
    """
    Single action per frame conditioner with spatial RoPE positions.
    
    For each frame transition, outputs exactly ONE action token, positioned at
    the location of the subject performing that action.
    
    Outputs:
        - embeddings: [B, T, model_dim] - one action embedding per frame
        - positions: [B, T, 3] - (frame, h_patch, w_patch) per token
    
    Args:
        num_action_classes: Number of action types (5: still, up, down, left, right)
        model_dim: Output dimension (1536 for Wan model)
        patch_size: Size of each latent patch in pixels (8 for Wan VAE)
        speed: Pixels moved per frame per action
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        model_dim: int = 1536,
        patch_size: int = 8,
        speed: int = 6,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.speed = speed
        
        # Action embedding - directly to model_dim
        self.action_embedding = nn.Embedding(num_action_classes, model_dim)
        
        # Initialize with small std
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        # Freeze embeddings
        self.action_embedding.weight.requires_grad = False
        
        # Action to direction mapping
        # 0: still, 1: up, 2: down, 3: left, 4: right
        self.register_buffer('action_dx', torch.tensor([0, 0, 0, -1, 1], dtype=torch.float32))
        self.register_buffer('action_dy', torch.tensor([0, -1, 1, 0, 0], dtype=torch.float32))
        
        print(f"SingleActionPerFrameConditioner initialized:")
        print(f"  {num_action_classes} actions → {model_dim}d embeddings")
        print(f"  Output: 1 action token per frame (T tokens total)")
        print(f"  Patch size: {patch_size}px, Speed: {speed}px/frame")
        print(f"  Embeddings: FROZEN")
    
    def compute_position_for_subject(
        self,
        initial_position: torch.Tensor,  # [B, 2] - (x, y) in pixels for ONE subject
        action_labels: torch.Tensor,      # [B, T] action indices for that subject
        frame_idx: int,                   # Which frame we're computing position for
    ) -> torch.Tensor:
        """Compute position of a subject at a given frame."""
        B = initial_position.shape[0]
        device = action_labels.device
        
        if frame_idx == 0:
            # Initial position
            x_pixels = initial_position[:, 0]
            y_pixels = initial_position[:, 1]
        else:
            # Cumulative displacement up to this frame
            dx = self.action_dx[action_labels[:, :frame_idx]]  # [B, frame_idx]
            dy = self.action_dy[action_labels[:, :frame_idx]]  # [B, frame_idx]
            
            cumsum_dx = dx.sum(dim=1)  # [B]
            cumsum_dy = dy.sum(dim=1)  # [B]
            
            x_pixels = initial_position[:, 0] + cumsum_dx * self.speed
            y_pixels = initial_position[:, 1] + cumsum_dy * self.speed
        
        h_patch = (y_pixels / self.patch_size).clamp(0, 31).long()
        w_patch = (x_pixels / self.patch_size).clamp(0, 31).long()
        
        return h_patch, w_patch
    
    def forward(
        self,
        action_labels: torch.Tensor,       # [B, S, T] action indices
        subject_ids: torch.Tensor,         # [B, S] subject indices (unused)
        initial_positions: torch.Tensor,   # [B, S, 2] initial (x, y) in pixels
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask (unused here)
        active_subject_per_frame: torch.Tensor = None,  # [B, T] which subject is active per frame
    ) -> tuple:
        """
        Generate single action embedding per frame with spatial RoPE position.
        
        If active_subject_per_frame is not provided, automatically selects the
        subject that is NOT "still" (action != 0) for each frame. If both are
        moving or both are still, uses the first subject.
        
        Args:
            action_labels: [B, S, T] action indices
            subject_ids: [B, S] subject indices (unused)
            initial_positions: [B, S, 2] initial (x, y) positions
            action_mask: [B, S, T] optional mask (applied before selection)
            active_subject_per_frame: [B, T] which subject (0 or 1) is active per frame
        
        Returns:
            embeddings: [B, T, model_dim] one action embedding per frame
            positions: [B, T, 3] (frame, h_patch, w_patch) for RoPE
        """
        B, S, T = action_labels.shape
        device = action_labels.device
        dtype = self.action_embedding.weight.dtype
        
        # Apply mask if provided (set masked actions to "still")
        if action_mask is not None:
            action_labels = action_labels * action_mask.long()
        
        # Determine active subject per frame if not provided
        if active_subject_per_frame is None:
            # Pick ONE consistent subject for the entire video (not per-frame)
            # The subject with MORE movement is the "active" one
            # If equal, prefer subject 0
            is_moving = action_labels != 0  # [B, S, T]
            
            # Count total movement per subject across all frames
            movement_count = is_moving.sum(dim=2)  # [B, S] - total frames with movement
            
            # Subject 1 is active for the video if it has MORE movement than subject 0
            subject1_is_active = movement_count[:, 1] > movement_count[:, 0]  # [B]
            
            # Expand to all frames (same subject for entire video)
            active_subject_per_frame = subject1_is_active.unsqueeze(1).expand(-1, T).long()  # [B, T]
            
            # Debug: show selection (only for batch item 0)
            if B == 1:
                active_s = active_subject_per_frame[0, 0].item()
                s0_moves = movement_count[0, 0].item()
                s1_moves = movement_count[0, 1].item()
                # Only print occasionally to avoid spam during training
                if not hasattr(self, '_debug_printed') or not self._debug_printed:
                    print(f"  Active subject: {active_s} (S0 moves: {s0_moves}, S1 moves: {s1_moves})")
                    self._debug_printed = True
        
        # Gather the active action for each frame
        # action_labels: [B, S, T], active_subject: [B, T]
        active_actions = torch.zeros(B, T, dtype=torch.long, device=device)
        active_positions = torch.zeros(B, T, 3, device=device)
        
        for t in range(T):
            for b in range(B):
                s = active_subject_per_frame[b, t].item()
                active_actions[b, t] = action_labels[b, s, t]
                
                # Compute position of active subject at frame t+1 (action-as-result)
                # The action at t produces frame t+1
                subj_initial_pos = initial_positions[b, s]  # [2]
                subj_actions = action_labels[b, s]  # [T]
                
                # Position after this action
                if t == 0:
                    dx = self.action_dx[subj_actions[0]]
                    dy = self.action_dy[subj_actions[0]]
                    x = subj_initial_pos[0] + dx * self.speed
                    y = subj_initial_pos[1] + dy * self.speed
                else:
                    dx = self.action_dx[subj_actions[:t+1]].sum()
                    dy = self.action_dy[subj_actions[:t+1]].sum()
                    x = subj_initial_pos[0] + dx * self.speed
                    y = subj_initial_pos[1] + dy * self.speed
                
                # Model grid is 16x16 (256 pixels / 16 = 16 patches)
                # This accounts for both VAE (8x) and model patch embedding (2x) = 16x total
                effective_patch_size = self.patch_size * 2  # 8 * 2 = 16
                h_patch = (y / effective_patch_size).clamp(0, 15)
                w_patch = (x / effective_patch_size).clamp(0, 15)
                
                # Frame index is t+1 (action-as-result semantics)
                active_positions[b, t] = torch.tensor([t + 1, h_patch, w_patch], device=device)
        
        # Get action embeddings: [B, T, model_dim]
        embeddings = self.action_embedding(active_actions)
        
        return embeddings.to(dtype), active_positions


def create_single_action_conditioner(config, model_dim: int = 1536):
    """Create single action conditioner from config."""
    return SingleActionPerFrameConditioner(
        num_action_classes=getattr(config, 'num_action_classes', 5),
        model_dim=model_dim,
        patch_size=getattr(config, 'latent_patch_size', 8),
        speed=getattr(config, 'position_speed', 6),
    )

