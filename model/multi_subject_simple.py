"""
Simplified Multi-Subject Conditioner with Learned Subject Embeddings.

Instead of pooling from VAE-encoded prototype images, we use learned embeddings
for each subject (like we do for actions). This is simpler and may learn better
action-subject binding.

The concatenated embedding is: [action_embed | subject_embed] = model_dim
"""

import torch
import torch.nn as nn


class SimpleMultiSubjectConditioner(nn.Module):
    """
    Simple conditioner that uses learned embeddings for both actions and subjects.
    
    Supports two combination modes:
    - 'add': action_embed + subject_embed (both at model_dim, then add)
    - 'concat': [action_embed | subject_embed] (split dimensions)
    
    Args:
        num_action_classes: Number of action types (e.g., 5: still, up, down, left, right)
        num_subjects: Number of unique subjects in the dataset
        model_dim: Output dimension (must match transformer model dim, e.g., 1536)
        combine_mode: 'add' or 'concat'
        freeze_subject_embeddings: If True, use orthogonal init and freeze subject embeddings
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        num_subjects: int = 6,  # e.g., 6 training subjects
        model_dim: int = 1536,
        combine_mode: str = 'add',  # 'add' or 'concat'
        freeze_subject_embeddings: bool = False,  # Whether to freeze subject embeddings
        freeze_action_embeddings: bool = False,   # Whether to freeze action embeddings
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_subjects = num_subjects
        self.model_dim = model_dim
        self.combine_mode = combine_mode
        self.freeze_subject_embeddings = freeze_subject_embeddings
        self.freeze_action_embeddings = freeze_action_embeddings
        
        if combine_mode == 'add':
            # Both embeddings at full model_dim, then add
            self.action_embedding = nn.Embedding(num_action_classes, model_dim)
            self.subject_embedding = nn.Embedding(num_subjects, model_dim)
        elif combine_mode == 'concat':
            # Split dimensions for concatenation
            action_dim = model_dim // 2
            subject_dim = model_dim - action_dim
            self.action_embedding = nn.Embedding(num_action_classes, action_dim)
            self.subject_embedding = nn.Embedding(num_subjects, subject_dim)
        else:
            raise ValueError(f"Unknown combine_mode: {combine_mode}")
        
        # Initialize action embeddings
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        if freeze_action_embeddings:
            self.action_embedding.weight.requires_grad = False
            print(f"  Action embeddings: FROZEN (random init)")
        
        # Initialize subject embeddings
        if freeze_subject_embeddings:
            # Use orthogonal initialization for maximum separation
            # For num_subjects < embed_dim, this gives orthogonal rows
            subject_dim = self.subject_embedding.embedding_dim
            if num_subjects <= subject_dim:
                # Create orthogonal vectors
                ortho_matrix = torch.zeros(num_subjects, subject_dim)
                nn.init.orthogonal_(ortho_matrix)
                # Scale to reasonable norm
                ortho_matrix = ortho_matrix * 0.5
                self.subject_embedding.weight.data.copy_(ortho_matrix)
            else:
                # If more subjects than dimensions, use random init
                nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
            # Freeze the subject embeddings
            self.subject_embedding.weight.requires_grad = False
            print(f"  Subject embeddings: FROZEN (orthogonal init)")
        else:
            nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
        
        print(f"SimpleMultiSubjectConditioner initialized:")
        print(f"  {num_action_classes} actions × {self.action_embedding.embedding_dim}d (frozen={freeze_action_embeddings})")
        print(f"  {num_subjects} subjects × {self.subject_embedding.embedding_dim}d (frozen={freeze_subject_embeddings})")
        print(f"  Combine mode: {combine_mode}")
        print(f"  Output: {model_dim}d")
    
    def forward(
        self,
        action_labels: torch.Tensor,      # [B, S, T] action indices per subject per timestep
        subject_ids: torch.Tensor,         # [B, S] subject indices
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask (1=valid, 0=masked)
    ) -> torch.Tensor:
        """
        Generate action-subject embeddings.
        
        Args:
            action_labels: [B, S, T] action indices for each subject at each timestep
            subject_ids: [B, S] subject indices for each subject in the video
            action_mask: [B, S, T] optional mask for masking future actions
        
        Returns:
            embeddings: [B, S*T, model_dim] combined action-subject embeddings
        """
        B, S, T = action_labels.shape
        
        # Apply mask if provided (set masked actions to 0 = "still")
        if action_mask is not None:
            action_labels = action_labels * action_mask.long()
        
        # Get action embeddings: [B, S, T] -> [B, S, T, embed_dim]
        action_embeds = self.action_embedding(action_labels)
        
        # Get subject embeddings: [B, S] -> [B, S, embed_dim]
        subject_embeds = self.subject_embedding(subject_ids)
        
        # Expand subject embeddings to match temporal dimension: [B, S, embed_dim] -> [B, S, T, embed_dim]
        subject_embeds = subject_embeds.unsqueeze(2).expand(-1, -1, T, -1)
        
        # Combine based on mode
        if self.combine_mode == 'add':
            # Addition: action + subject (both at model_dim)
            combined = action_embeds + subject_embeds
        else:
            # Concatenation: [action | subject]
            combined = torch.cat([action_embeds, subject_embeds], dim=-1)
        
        # Flatten S and T dimensions: [B, S, T, model_dim] -> [B, S*T, model_dim]
        combined = combined.view(B, S * T, self.model_dim)
        
        return combined


class ModulatedMultiSubjectConditioner(nn.Module):
    """
    FiLM-style conditioner: Actions modulate subject embeddings via scale/shift.
    
    Instead of concat/add, actions predict modulation parameters that transform
    the subject embedding. This makes actions work THROUGH subject identity.
    
    Output = scale * subject_embed + shift
    
    Where (scale, shift) are predicted from action embeddings via an MLP.
    
    Args:
        num_action_classes: Number of action types (e.g., 5: still, up, down, left, right)
        num_subjects: Number of unique subjects in the dataset
        model_dim: Output dimension (must match transformer model dim, e.g., 1536)
        action_embed_dim: Hidden dimension for action embeddings (before MLP)
        freeze_subject_embeddings: If True, use orthogonal init and freeze subject embeddings
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        num_subjects: int = 6,
        model_dim: int = 1536,
        action_embed_dim: int = 256,  # Smaller action embedding, expanded via MLP
        freeze_subject_embeddings: bool = True,  # Default to frozen for this approach
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.num_subjects = num_subjects
        self.model_dim = model_dim
        self.action_embed_dim = action_embed_dim
        self.freeze_subject_embeddings = freeze_subject_embeddings
        
        # Subject embeddings - the "base" identity features
        self.subject_embedding = nn.Embedding(num_subjects, model_dim)
        
        # Action embeddings - compact representation
        self.action_embedding = nn.Embedding(num_action_classes, action_embed_dim)
        
        # MLP to predict scale and shift from action
        # Output: 2 * model_dim (scale and shift)
        self.modulation_mlp = nn.Sequential(
            nn.Linear(action_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 2 * model_dim),  # scale and shift
        )
        
        # Initialize subject embeddings
        if freeze_subject_embeddings:
            # Orthogonal init with reasonable norm
            if num_subjects <= model_dim:
                ortho_matrix = torch.zeros(num_subjects, model_dim)
                nn.init.orthogonal_(ortho_matrix)
                # Scale to norm ~1.0 for balanced modulation
                ortho_matrix = ortho_matrix * 1.0
                self.subject_embedding.weight.data.copy_(ortho_matrix)
            else:
                nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
            self.subject_embedding.weight.requires_grad = False
            print(f"  Subject embeddings: FROZEN (orthogonal init, norm=1.0)")
        else:
            nn.init.normal_(self.subject_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize action embeddings
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize modulation MLP for identity-like initialization
        # At init: scale ≈ 1, shift ≈ 0 → output ≈ subject_embed (no modulation)
        nn.init.zeros_(self.modulation_mlp[-1].weight)
        nn.init.zeros_(self.modulation_mlp[-1].bias)
        # Set bias so initial scale=1 (first half of output)
        self.modulation_mlp[-1].bias.data[:model_dim] = 1.0
        # shift=0 already from zeros init
        
        print(f"ModulatedMultiSubjectConditioner initialized:")
        print(f"  {num_action_classes} actions × {action_embed_dim}d → MLP → {2*model_dim}d (scale+shift)")
        print(f"  {num_subjects} subjects × {model_dim}d")
        print(f"  Modulation: scale * subject + shift")
        print(f"  Freeze subjects: {freeze_subject_embeddings}")
        print(f"  Output: {model_dim}d")
    
    def forward(
        self,
        action_labels: torch.Tensor,      # [B, S, T] action indices per subject per timestep
        subject_ids: torch.Tensor,         # [B, S] subject indices
        action_mask: torch.Tensor = None,  # [B, S, T] optional mask (1=valid, 0=masked)
    ) -> torch.Tensor:
        """
        Generate modulated action-subject embeddings.
        
        Args:
            action_labels: [B, S, T] action indices for each subject at each timestep
            subject_ids: [B, S] subject indices for each subject in the video
            action_mask: [B, S, T] optional mask for masking future actions
        
        Returns:
            embeddings: [B, S*T, model_dim] modulated embeddings
        """
        B, S, T = action_labels.shape
        
        # Apply mask if provided (set masked actions to 0 = "still")
        if action_mask is not None:
            action_labels = action_labels * action_mask.long()
        
        # Get subject embeddings: [B, S] -> [B, S, model_dim]
        subject_embeds = self.subject_embedding(subject_ids)
        # Expand to temporal: [B, S, model_dim] -> [B, S, T, model_dim]
        subject_embeds = subject_embeds.unsqueeze(2).expand(-1, -1, T, -1)
        
        # Get action embeddings: [B, S, T] -> [B, S, T, action_embed_dim]
        action_embeds = self.action_embedding(action_labels)
        
        # Predict modulation parameters: [B, S, T, 2*model_dim]
        modulation = self.modulation_mlp(action_embeds)
        scale, shift = modulation.chunk(2, dim=-1)  # each [B, S, T, model_dim]
        
        # Apply FiLM-style modulation: scale * subject + shift
        modulated = scale * subject_embeds + shift
        
        # Flatten S and T dimensions: [B, S, T, model_dim] -> [B, S*T, model_dim]
        modulated = modulated.view(B, S * T, self.model_dim)
        
        return modulated


class SingleSubjectModulatedConditioner(nn.Module):
    """
    FiLM-style conditioner for SINGLE subject (e.g., just blue square).
    
    This is a simpler version for testing the modulation mechanism without
    multi-subject complexity. Has a single "base" embedding that actions
    modulate via scale/shift.
    
    Output = scale * base_embed + shift
    
    Args:
        num_action_classes: Number of action types (e.g., 5: still, up, down, left, right)
        model_dim: Output dimension (must match transformer model dim, e.g., 1536)
        action_embed_dim: Hidden dimension for action embeddings (before MLP)
        freeze_base_embedding: If True, freeze the base embedding (random init, fixed)
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        model_dim: int = 1536,
        action_embed_dim: int = 256,
        freeze_base_embedding: bool = False,
    ):
        super().__init__()
        
        self.num_action_classes = num_action_classes
        self.model_dim = model_dim
        self.action_embed_dim = action_embed_dim
        self.freeze_base_embedding = freeze_base_embedding
        
        # Base embedding - "identity" of the single subject
        # Initialize with norm ~1.0 for balanced modulation
        base_init = torch.randn(model_dim)
        base_init = base_init / base_init.norm() * 1.0  # Normalize to unit norm
        self.base_embedding = nn.Parameter(base_init, requires_grad=not freeze_base_embedding)
        
        # Action embeddings - compact representation
        self.action_embedding = nn.Embedding(num_action_classes, action_embed_dim)
        
        # MLP to predict scale and shift from action
        self.modulation_mlp = nn.Sequential(
            nn.Linear(action_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, 2 * model_dim),  # scale and shift
        )
        
        # Initialize action embeddings
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize modulation MLP for identity: scale=1, shift=0
        nn.init.zeros_(self.modulation_mlp[-1].weight)
        nn.init.zeros_(self.modulation_mlp[-1].bias)
        self.modulation_mlp[-1].bias.data[:model_dim] = 1.0  # scale = 1
        
        base_status = "FROZEN" if freeze_base_embedding else "learnable"
        print(f"SingleSubjectModulatedConditioner initialized:")
        print(f"  {num_action_classes} actions × {action_embed_dim}d → MLP → {2*model_dim}d (scale+shift)")
        print(f"  Base embedding: {model_dim}d ({base_status}, norm=1.0)")
        print(f"  Modulation: scale * base + shift")
        print(f"  Output: {model_dim}d")
    
    def forward(
        self,
        action_labels: torch.Tensor,      # [B, T] action indices per timestep
        action_mask: torch.Tensor = None,  # [B, T] optional mask (1=valid, 0=masked)
    ) -> torch.Tensor:
        """
        Generate modulated action embeddings for single subject.
        
        Args:
            action_labels: [B, T] action indices for each timestep
            action_mask: [B, T] optional mask for masking future actions
        
        Returns:
            embeddings: [B, T, model_dim] modulated embeddings
        """
        B, T = action_labels.shape
        
        # Apply mask if provided (set masked actions to 0 = "still")
        if action_mask is not None:
            action_labels = action_labels * action_mask.long()
        
        # Get base embedding: [model_dim] -> [B, T, model_dim]
        base_embeds = self.base_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        
        # Get action embeddings: [B, T] -> [B, T, action_embed_dim]
        action_embeds = self.action_embedding(action_labels)
        
        # Predict modulation parameters: [B, T, 2*model_dim]
        modulation = self.modulation_mlp(action_embeds)
        scale, shift = modulation.chunk(2, dim=-1)  # each [B, T, model_dim]
        
        # Apply FiLM-style modulation: scale * base + shift
        modulated = scale * base_embeds + shift
        
        return modulated


# For compatibility with existing code
def create_simple_conditioner(config, model_dim: int = 1536) -> SimpleMultiSubjectConditioner:
    """Create conditioner from config."""
    return SimpleMultiSubjectConditioner(
        num_action_classes=getattr(config, 'num_action_classes', 5),
        num_subjects=getattr(config, 'num_subjects_total', 6),
        model_dim=model_dim,
        combine_mode=getattr(config, 'combine_mode', 'add'),
        freeze_subject_embeddings=getattr(config, 'freeze_subject_embeddings', False),
    )

