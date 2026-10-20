"""
Multi-subject action conditioning modules.

These modules enable conditioning the diffusion model on multiple subjects,
where each subject has:
- A prototype image (VAE-encoded latent)
- An action sequence

The subject prototype is patchified (using same weights as video) and pooled
to a single feature vector, which is concatenated with the action embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SubjectFeaturePooler(nn.Module):
    """
    Pools patchified subject features to a single vector per subject.
    
    Uses cross-attention pooling with a learnable query token.
    
    Input: [B, S, num_patches, D]  (S subjects, each with num_patches patch tokens)
    Output: [B, S, D_out]  (single pooled feature per subject)
    """
    
    def __init__(self, dim: int, output_dim: int = None, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Learnable query for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Cross-attention layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, self.output_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, num_patches, D] - patchified subject features
            
        Returns:
            pooled: [B, S, output_dim] - single feature per subject
        """
        B, S, N, D = x.shape
        
        # Normalize patch features
        x = self.norm(x)
        
        # Expand query for all batch and subjects
        query = self.pool_query.expand(B, S, -1)  # [B, S, D]
        
        # Reshape for batch processing: treat (B, S) as batch dimension
        x_flat = x.view(B * S, N, D)  # [B*S, N, D]
        query_flat = query.view(B * S, 1, D)  # [B*S, 1, D]
        
        # Project to Q, K, V
        q = self.q_proj(query_flat)  # [B*S, 1, D]
        k = self.k_proj(x_flat)      # [B*S, N, D]
        v = self.v_proj(x_flat)      # [B*S, N, D]
        
        # Reshape for multi-head attention
        q = q.view(B * S, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B*S, heads, 1, head_dim]
        k = k.view(B * S, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*S, heads, N, head_dim]
        v = v.view(B * S, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B*S, heads, N, head_dim]
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B*S, heads, 1, N]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)  # [B*S, heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B * S, 1, D)  # [B*S, 1, D]
        
        # Project to output dimension
        out = self.out_proj(out)  # [B*S, 1, output_dim]
        
        # Reshape back
        out = out.view(B, S, self.output_dim)  # [B, S, output_dim]
        
        return out


class MultiSubjectActionEmbedding(nn.Module):
    """
    Creates action embeddings for multiple subjects, optionally concatenated
    with pooled subject image features.
    
    For each subject s and frame t:
        action_embed[s, t] = concat(action_embedding[action[s,t]], subject_features[s])
    
    This ties each action to its subject through the concatenated features.
    """
    
    def __init__(
        self,
        num_action_classes: int = 5,
        model_dim: int = 1536,
        action_dim: int = 512,
        subject_dim: int = 1024,
        use_subject_features: bool = True
    ):
        super().__init__()
        self.num_action_classes = num_action_classes
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.subject_dim = subject_dim
        self.use_subject_features = use_subject_features
        
        # Action embedding (smaller dimension since it's concatenated with subject)
        self.action_embedding = nn.Embedding(num_action_classes, action_dim)
        nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        
        if use_subject_features:
            # Subject feature projection (if subject features aren't already subject_dim)
            self.subject_proj = nn.Linear(subject_dim, subject_dim)
            
            # Verify dimensions add up
            assert action_dim + subject_dim == model_dim, \
                f"action_dim ({action_dim}) + subject_dim ({subject_dim}) must equal model_dim ({model_dim})"
        else:
            # If not using subject features, action embedding needs full model_dim
            self.action_embedding = nn.Embedding(num_action_classes, model_dim)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        action_labels: torch.Tensor,
        subject_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            action_labels: [B, S, T] - action indices for each subject at each frame
            subject_features: [B, S, subject_dim] - pooled subject features (optional)
            
        Returns:
            action_embeds: [B, S*T, model_dim] - flattened action embeddings for cross-attention
                           Ordering: [subj0_t0, subj0_t1, ..., subj0_tT, subj1_t0, ...]
        """
        B, S, T = action_labels.shape
        
        # Get action embeddings: [B, S, T, action_dim]
        action_embeds = self.action_embedding(action_labels)
        
        if self.use_subject_features and subject_features is not None:
            # Project subject features: [B, S, subject_dim]
            subj_feats = self.subject_proj(subject_features)
            
            # Expand subject features across time: [B, S, 1, subject_dim] -> [B, S, T, subject_dim]
            subj_feats = subj_feats.unsqueeze(2).expand(-1, -1, T, -1)
            
            # Concatenate: [B, S, T, action_dim + subject_dim] = [B, S, T, model_dim]
            combined = torch.cat([action_embeds, subj_feats], dim=-1)
        else:
            combined = action_embeds  # [B, S, T, model_dim]
        
        # Flatten S and T for cross-attention sequence: [B, S*T, model_dim]
        combined = combined.view(B, S * T, self.model_dim)
        
        return combined


class MultiSubjectConditioner(nn.Module):
    """
    Complete conditioning module for multi-subject action control.
    
    Takes:
    - Subject prototype latents: [B, S, C, H, W]
    - Action labels: [B, S, T]
    
    Returns:
    - Action embeddings with subject features: [B, S*T, model_dim]
    """
    
    def __init__(
        self,
        latent_channels: int = 16,
        model_dim: int = 1536,
        patch_size: tuple = (1, 2, 2),
        num_action_classes: int = 5,
        action_dim: int = 512,
        subject_dim: int = 1024,
        num_pool_heads: int = 8,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        
        # Patch embedding for subject prototypes (same as video patch embedding)
        # Will be initialized from the model's patch_embedding weights
        self.subject_patch_embedding = nn.Conv3d(
            latent_channels, model_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Pooler to compress patches to single feature
        self.pooler = SubjectFeaturePooler(
            dim=model_dim,
            output_dim=subject_dim,
            num_heads=num_pool_heads
        )
        
        # Action + subject embedding
        self.action_subject_embed = MultiSubjectActionEmbedding(
            num_action_classes=num_action_classes,
            model_dim=model_dim,
            action_dim=action_dim,
            subject_dim=subject_dim,
            use_subject_features=True
        )
        
    def initialize_from_patch_embedding(self, patch_embedding: nn.Conv3d):
        """Copy weights from the model's patch_embedding."""
        with torch.no_grad():
            self.subject_patch_embedding.weight.copy_(patch_embedding.weight)
            if patch_embedding.bias is not None and self.subject_patch_embedding.bias is not None:
                self.subject_patch_embedding.bias.copy_(patch_embedding.bias)
    
    def forward(
        self,
        subject_prototypes: torch.Tensor,
        action_labels: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            subject_prototypes: [B, S, C, H, W] - VAE-encoded prototype images
            action_labels: [B, S, T] - action indices per subject per frame
            action_mask: [B, S, T] - optional mask for valid actions (1=valid, 0=masked)
            
        Returns:
            action_embeds: [B, S*T, model_dim] - action embeddings for cross-attention
        """
        B, S, C, H, W = subject_prototypes.shape
        
        # Add temporal dimension for Conv3d: [B, S, C, H, W] -> [B*S, C, 1, H, W]
        protos = subject_prototypes.view(B * S, C, 1, H, W)
        
        # Patchify: [B*S, C, 1, H, W] -> [B*S, model_dim, 1, H', W']
        patch_tokens = self.subject_patch_embedding(protos)
        
        # Flatten patches: [B*S, model_dim, 1, H', W'] -> [B*S, num_patches, model_dim]
        patch_tokens = patch_tokens.squeeze(2).flatten(2).transpose(1, 2)
        num_patches = patch_tokens.shape[1]
        
        # Reshape for pooler: [B*S, num_patches, model_dim] -> [B, S, num_patches, model_dim]
        patch_tokens = patch_tokens.view(B, S, num_patches, self.model_dim)
        
        # Pool to single feature per subject: [B, S, subject_dim]
        subject_features = self.pooler(patch_tokens)
        
        # Create action embeddings with subject features: [B, S*T, model_dim]
        action_embeds = self.action_subject_embed(action_labels, subject_features)
        
        # Apply mask if provided
        if action_mask is not None:
            # Flatten mask: [B, S, T] -> [B, S*T]
            mask = action_mask.view(B, -1)
            # Zero out masked positions
            action_embeds = action_embeds * mask.unsqueeze(-1)
        
        return action_embeds


# Convenience function to create multi-subject conditioner from config
def create_multi_subject_conditioner(args, model_dim: int = 1536):
    """Create MultiSubjectConditioner from training config."""
    return MultiSubjectConditioner(
        latent_channels=getattr(args, "latent_channels", 16),
        model_dim=model_dim,
        patch_size=getattr(args, "patch_size", (1, 2, 2)),
        num_action_classes=getattr(args, "num_action_classes", 5),
        action_dim=getattr(args, "action_dim", 512),
        subject_dim=getattr(args, "subject_dim", 1024),
        num_pool_heads=getattr(args, "num_pool_heads", 8),
    )

