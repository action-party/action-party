# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#
# ----------------------------------------------------------------------------
# Self-Forcing-Games note
# ----------------------------------------------------------------------------
# This file is the upstream Wan DiT with additions marked by
# ``=== COORD-TOKEN EXTENSION === [start]/[end]`` banners. Standalone helpers
# (FlexAttention block-mask builders and FourierCoordEncoder) live in
# :mod:`wan.modules.coord_token`; they are re-exported below for
# backwards-compatibility with callers that imported them from this module.
# New cross-attention classes and a coord-aware ``rope_apply`` remain inline
# because they are tightly coupled to the DiT internals.
# ----------------------------------------------------------------------------
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat

from .attention import flash_attention

# === COORD-TOKEN EXTENSION === [start]
# Re-export FlexAttention availability + our subject-isolation utilities so that
# existing ``from .model import get_self_attn_block_mask`` style imports keep
# working. See wan/modules/coord_token.py for the definitions.
from .coord_token import (
    FLEX_ATTENTION_AVAILABLE,
    flex_attention,
    create_block_mask,
    get_compiled_flex_attention,
    get_self_attn_block_mask,
    get_coord_self_attn_block_mask,
    FourierCoordEncoder,
)
# === COORD-TOKEN EXTENSION === [end]

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# Note: FourierCoordEncoder was moved to wan/modules/coord_token.py and is
# re-exported at the top of this file.


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, coord_token_info=None, position_token_info=None):
    """
    Apply 3D RoPE to video tokens and temporal RoPE to coordinate tokens.
    
    Args:
        x: [B, total_seq_len, num_heads, head_dim] - input tensor
        grid_sizes: [B, 3] - (F, H, W) grid dimensions per sample
        freqs: precomputed frequency parameters
        coord_token_info: Optional dict with {'num_tokens': T*S, 'num_frames': T, 'num_subjects': S}
                         If provided, applies temporal RoPE to the last num_tokens tokens
        position_token_info: Optional dict with {'num_tokens': F*H*W, 'grid_sizes': [B, 3]}
                            If provided, applies same 3D RoPE to position tokens (after video tokens)
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs for 3D RoPE (temporal, height, width)
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Handle video tokens (if any)
        if seq_len > 0:
            # precompute multipliers for video tokens (3D RoPE)
            x_video = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                    freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding to video tokens
            x_video = torch.view_as_real(x_video * freqs_i).flatten(2)
        else:
            # No video tokens (coord_only_no_video mode)
            # After flatten(2), video tokens are [seq_len, n, head_dim]
            # So empty should be [0, n, head_dim] to match extra_tokens shape
            x_video = x.new_empty(0, n, x.size(-1))  # [0, n, head_dim]
        
        # Handle extra tokens
        extra_tokens = x[i, seq_len:]
        
        # Handle position tokens with 3D RoPE (with temporal offset to distinguish from video)
        if position_token_info is not None and extra_tokens.size(0) > 0:
            num_position_tokens = position_token_info['num_tokens']
            num_subjects_in_pos = position_token_info.get('num_subjects', 1)
            tokens_per_subject = position_token_info.get('tokens_per_subject', num_position_tokens)
            pos_grid = position_token_info['grid_sizes'][i] if position_token_info['grid_sizes'].dim() > 1 else position_token_info['grid_sizes']
            pf, ph, pw = pos_grid.tolist()
            pos_seq_len = pf * ph * pw
            
            # Temporal offset for position tokens (0.1 frames) to distinguish from video tokens
            temporal_offset = position_token_info.get('temporal_offset', 0.1)
            
            if extra_tokens.size(0) >= num_position_tokens and pos_seq_len > 0:
                # Extract position tokens (first num_position_tokens of extra tokens)
                position_tokens = extra_tokens[:num_position_tokens]
                remaining_tokens = extra_tokens[num_position_tokens:] if extra_tokens.size(0) > num_position_tokens else None
                
                # Apply 3D RoPE with temporal offset for position tokens
                position_tokens_complex = torch.view_as_complex(
                    position_tokens.to(torch.float64).reshape(num_position_tokens, n, -1, 2)
                )
                
                # Build 3D RoPE frequencies with temporal offset
                # For temporal component: use t + offset instead of t
                # RoPE formula: exp(i * t * freq) -> exp(i * (t + offset) * freq) = exp(i * t * freq) * exp(i * offset * freq)
                # freqs_split[0] has shape [max_frames, dim/3] with values exp(i * t * freq_k)
                # We need to compute the offset rotation: exp(i * offset * freq_k)
                
                # Get base frequencies for computing offset rotation
                # freqs_split[0][1] / freqs_split[0][0] gives exp(i * freq_k) for each k
                # So offset rotation = (freqs_split[0][1] / freqs_split[0][0]) ** offset
                if pf > 1:
                    # Compute the per-step rotation and raise to power of offset
                    step_rotation = freqs_split[0][1:2] / freqs_split[0][0:1]  # exp(i * freq_k)
                    offset_rotation = step_rotation ** temporal_offset  # exp(i * offset * freq_k)
                    # Apply offset to all temporal positions
                    temporal_freqs_offset = freqs_split[0][:pf] * offset_rotation
                else:
                    temporal_freqs_offset = freqs_split[0][:pf]
                
                # Build 3D RoPE for ONE subject's worth of position tokens
                pos_freqs_single = torch.cat([
                    temporal_freqs_offset.view(pf, 1, 1, -1).expand(pf, ph, pw, -1),
                    freqs_split[1][:ph].view(1, ph, 1, -1).expand(pf, ph, pw, -1),
                    freqs_split[2][:pw].view(1, 1, pw, -1).expand(pf, ph, pw, -1)
                ], dim=-1).reshape(pos_seq_len, 1, -1)
                
                # For multi-subject: tile the RoPE pattern for each subject
                # Each subject gets the SAME 3D RoPE based on (frame, height, width)
                # Subject differentiation comes from cross-attention masking, not RoPE
                if num_subjects_in_pos > 1:
                    # Repeat the single-subject RoPE pattern for all subjects
                    pos_freqs = pos_freqs_single.repeat(num_subjects_in_pos, 1, 1)
                else:
                    pos_freqs = pos_freqs_single
                
                # Handle padding (if position tokens are padded beyond expected)
                if num_position_tokens > pos_freqs.size(0):
                    pos_freqs = torch.cat([
                        pos_freqs,
                        torch.ones(num_position_tokens - pos_freqs.size(0), 1, pos_freqs.size(-1), device=pos_freqs.device, dtype=pos_freqs.dtype)
                    ], dim=0)
                
                # Apply rotation
                position_tokens_rotated = torch.view_as_real(position_tokens_complex * pos_freqs).flatten(2)
                
                # Reassemble extra tokens
                if remaining_tokens is not None:
                    extra_tokens = torch.cat([position_tokens_rotated, remaining_tokens], dim=0)
                else:
                    extra_tokens = position_tokens_rotated
        
        # Handle coordinate tokens (if any)
        if coord_token_info is not None and extra_tokens.size(0) > 0:
            num_coord_tokens = coord_token_info['num_tokens']
            T = coord_token_info['num_frames']
            S = coord_token_info['num_subjects']
            
            if extra_tokens.size(0) >= num_coord_tokens:
                # Extract coord tokens (last num_coord_tokens of extra tokens)
                coord_tokens = extra_tokens[-num_coord_tokens:]
                other_tokens = extra_tokens[:-num_coord_tokens] if extra_tokens.size(0) > num_coord_tokens else None
                
                # Apply RoPE to coord tokens
                # Token order is: frame0_subj0, frame0_subj1, frame1_subj0, frame1_subj1, ...
                coord_tokens_complex = torch.view_as_complex(
                    coord_tokens.to(torch.float64).reshape(num_coord_tokens, n, -1, 2)
                )
                
                # Build frame indices for each coord token
                frame_indices = torch.arange(T, device=x.device).repeat_interleave(S)
                
                # Check if spatial positions are provided
                has_spatial = 'spatial_h' in coord_token_info and 'spatial_w' in coord_token_info
                
                if has_spatial:
                    # FULL 3D RoPE: use actual spatial positions for h, w
                    # spatial_h, spatial_w: [B, T*S] - we use sample i
                    h_indices = coord_token_info['spatial_h'][i]  # [T*S]
                    w_indices = coord_token_info['spatial_w'][i]  # [T*S]
                    
                    freq_frame = freqs_split[0][frame_indices]  # [T*S, frame_dim]
                    freq_h = freqs_split[1][h_indices]  # [T*S, h_dim]
                    freq_w = freqs_split[2][w_indices]  # [T*S, w_dim]
                else:
                    # TEMPORAL-ONLY RoPE: identity for h, w (position 0)
                    freq_frame = freqs_split[0][frame_indices]  # [T*S, frame_dim]
                    freq_h = freqs_split[1][0:1].expand(num_coord_tokens, -1)  # [T*S, h_dim]
                    freq_w = freqs_split[2][0:1].expand(num_coord_tokens, -1)  # [T*S, w_dim]
                
                coord_freqs = torch.cat([freq_frame, freq_h, freq_w], dim=-1).unsqueeze(1)  # [T*S, 1, c]
                
                # Apply rotation
                coord_tokens_rotated = torch.view_as_real(coord_tokens_complex * coord_freqs).flatten(2)
                
                # Reassemble
                if other_tokens is not None:
                    extra_tokens = torch.cat([other_tokens, coord_tokens_rotated], dim=0)
                else:
                    extra_tokens = coord_tokens_rotated
        
        x_i = torch.cat([x_video, extra_tokens], dim=0)

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


# === COORD-TOKEN EXTENSION === [start]
# temporal_rope_apply and the two specialized cross-attentions below were added
# on top of the upstream Wan DiT to support coord-token / per-subject attention.
# @amp.autocast(enabled=False)
def temporal_rope_apply(x, grid_sizes, freqs, coord_token_info=None):
    """
    Apply RoPE only along the temporal (frame) dimension.
    All channels use frame index as position (no spatial encoding).
    
    Args:
        x: [B, seq_len, num_heads, head_dim] - input tensor
        grid_sizes: [B, 3] - (F, H, W) grid dimensions per sample
        freqs: precomputed frequency parameters, shape [max_seq_len, head_dim//2]
        coord_token_info: Optional dict with {'num_tokens': T*S, 'num_frames': T, 'num_subjects': S}
                         If provided, applies temporal RoPE to the last num_tokens tokens
    
    Returns:
        x with temporal-only RoPE applied
    """
    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Handle video tokens (if any)
        if seq_len > 0:
            x_video = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))

            # Use frame index as position for all patches in that frame
            # freqs[:f] gives frequencies for frames 0..f-1
            # Expand to all spatial positions (h*w) within each frame
            freqs_i = freqs[:f, :c].view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)

            # apply rotary embedding to video tokens
            x_video = torch.view_as_real(x_video * freqs_i).flatten(2)
        else:
            # No video tokens (coord_only_no_video mode)
            # After flatten(2), video tokens are [seq_len, n, head_dim]
            # So empty should be [0, n, head_dim] to match extra_tokens shape
            x_video = x.new_empty(0, n, x.size(-1))  # [0, n, head_dim]

        # Handle extra tokens (including coordinate tokens)
        extra_tokens = x[i, seq_len:]
        if coord_token_info is not None and extra_tokens.size(0) > 0:
            num_coord_tokens = coord_token_info['num_tokens']
            T = coord_token_info['num_frames']
            S = coord_token_info['num_subjects']
            
            if extra_tokens.size(0) >= num_coord_tokens:
                # Extract coord tokens (last num_coord_tokens of extra tokens)
                coord_tokens = extra_tokens[-num_coord_tokens:]
                other_tokens = extra_tokens[:-num_coord_tokens] if extra_tokens.size(0) > num_coord_tokens else None
                
                # Apply TEMPORAL RoPE to coord tokens
                coord_tokens_complex = torch.view_as_complex(
                    coord_tokens.to(torch.float64).reshape(num_coord_tokens, n, -1, 2)
                )
                
                # Build frame indices for each coord token
                # Token order is: frame0_subj0, frame0_subj1, frame1_subj0, frame1_subj1, ...
                frame_indices = torch.arange(T, device=x.device).repeat_interleave(S)
                
                # Get temporal frequencies for each token's frame
                coord_freqs = freqs[:T, :c][frame_indices].unsqueeze(1)  # [T*S, 1, c]
                
                # Apply rotation
                coord_tokens_rotated = torch.view_as_real(coord_tokens_complex * coord_freqs).flatten(2)
                
                # Reassemble
                if other_tokens is not None:
                    extra_tokens = torch.cat([other_tokens, coord_tokens_rotated], dim=0)
                else:
                    extra_tokens = coord_tokens_rotated
        
        x_i = torch.cat([x_video, extra_tokens], dim=0)

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, coord_token_info=None, position_token_info=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            coord_token_info: Optional dict for coordinate token RoPE
            position_token_info: Optional dict for position token RoPE (same 3D RoPE as video)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs, coord_token_info=coord_token_info, position_token_info=position_token_info)
        k = rope_apply(k, grid_sizes, freqs, coord_token_info=coord_token_info, position_token_info=position_token_info)
        
        # Check if we need subject-isolation mask for multi-subject position/coord tokens
        # Only apply if explicitly enabled via subject_isolation flag in position_token_info or coord_token_info
        use_subject_isolation_pos = (
            position_token_info is not None and 
            position_token_info.get('subject_isolation', False) and
            position_token_info.get('num_subjects', 1) > 1 and
            position_token_info.get('num_tokens', 0) > 0
        )
        
        # Coord token subject isolation: similar to position tokens but simpler layout
        use_subject_isolation_coord = (
            coord_token_info is not None and 
            coord_token_info.get('subject_isolation', False) and
            coord_token_info.get('num_subjects', 1) > 1 and
            coord_token_info.get('num_tokens', 0) > 0
        )
        
        if use_subject_isolation_pos:
            # Subject-isolated self-attention: each subject's pos tokens only see their own
            # Layout: [video tokens, S0 pos tokens, S1 pos tokens, ..., padding]
            video_len = position_token_info['video_seq_lens'][0].item()
            num_pos_tokens = position_token_info['num_tokens']
            num_subjects = position_token_info['num_subjects']
            pos_per_subject = num_pos_tokens // num_subjects
            total_len = video_len + num_pos_tokens
            padded_len = s  # Full padded sequence length
            
            # q, k, v are [B, L, n, d] - need to transpose to [B, n, L, d] for attention
            q = q.transpose(1, 2)  # [B, n, L, d]
            k = k.transpose(1, 2)  # [B, n, L, d]
            v = v.transpose(1, 2)  # [B, n, L, d]
            
            # Check if position tokens should NOT attend to video
            position_no_video_attn = position_token_info.get('position_no_video_attn', False)
            
            # Try FlexAttention first (5x faster than SDPA) if enabled
            use_flex = position_token_info.get('use_flex_attention', True)
            compiled_flex = get_compiled_flex_attention() if use_flex else None
            if compiled_flex is not None and FLEX_ATTENTION_AVAILABLE:
                # Slice to actual tokens (exclude padding) for efficient FlexAttention
                q_actual = q[:, :, :total_len, :]  # [B, n, total_len, d]
                k_actual = k[:, :, :total_len, :]
                # Cast v to match q dtype (RoPE returns float32, v is bfloat16)
                v_actual = v[:, :, :total_len, :].to(q.dtype)
                
                # Get cached block mask (or create it)
                block_mask = get_self_attn_block_mask(
                    video_len, pos_per_subject, num_subjects, total_len, x.device,
                    position_no_video_attn=position_no_video_attn
                )
                
                # Apply FlexAttention with block mask
                out_actual = compiled_flex(q_actual, k_actual, v_actual, block_mask=block_mask)
                
                # Pad output back to original length
                if padded_len > total_len:
                    padding = torch.zeros(b, n, padded_len - total_len, d, device=out_actual.device, dtype=out_actual.dtype)
                    x = torch.cat([out_actual, padding], dim=2)
                else:
                    x = out_actual
            else:
                # Fallback to SDPA with explicit mask (also slice to actual tokens for efficiency)
                q_actual = q[:, :, :total_len, :]
                k_actual = k[:, :, :total_len, :]
                v_actual = v[:, :, :total_len, :]
                
                attn_mask = torch.zeros(total_len, total_len, device=x.device, dtype=q.dtype)
                
                # Build attention mask based on mode
                for subj in range(num_subjects):
                    pos_start = video_len + subj * pos_per_subject
                    pos_end = video_len + (subj + 1) * pos_per_subject if subj < num_subjects - 1 else video_len + num_pos_tokens
                    
                    # Block this subject from attending to OTHER subjects' position tokens
                    for other_subj in range(num_subjects):
                        if other_subj != subj:
                            other_start = video_len + other_subj * pos_per_subject
                            other_end = video_len + (other_subj + 1) * pos_per_subject if other_subj < num_subjects - 1 else video_len + num_pos_tokens
                            attn_mask[pos_start:pos_end, other_start:other_end] = float('-inf')
                    
                    # If position_no_video_attn, also block position tokens from attending to video
                    if position_no_video_attn:
                        attn_mask[pos_start:pos_end, :video_len] = float('-inf')
                
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                out_actual = torch.nn.functional.scaled_dot_product_attention(
                    q_actual, k_actual, v_actual, attn_mask=attn_mask, dropout_p=0.0
                )
                
                # Pad output back to original length
                if padded_len > total_len:
                    padding = torch.zeros(b, n, padded_len - total_len, d, device=out_actual.device, dtype=out_actual.dtype)
                    x = torch.cat([out_actual, padding], dim=2)
                else:
                    x = out_actual
            
            # Transpose back: [B, n, L, d] -> [B, L, n, d]
            x = x.transpose(1, 2)
        elif use_subject_isolation_coord:
            # Subject-isolated self-attention for coord tokens
            # Layout is FRAME-MAJOR: [video tokens, f0_s0, f0_s1, f1_s0, f1_s1, ..., padding]
            # Token at index (video_len + i) belongs to subject (i % S) and frame (i // S)
            num_coord_tokens = coord_token_info['num_tokens']
            num_subjects = coord_token_info['num_subjects']
            T = coord_token_info['num_frames']
            video_len = s - num_coord_tokens
            total_len = video_len + num_coord_tokens
            padded_len = s
            
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            q_actual = q[:, :, :total_len, :]
            k_actual = k[:, :, :total_len, :]
            v_actual = v[:, :, :total_len, :].to(q.dtype)
            
            use_flex = coord_token_info.get('use_flex_attention', True)
            compiled_flex = get_compiled_flex_attention() if use_flex else None
            if compiled_flex is not None and FLEX_ATTENTION_AVAILABLE:
                block_mask = get_coord_self_attn_block_mask(
                    video_len, num_coord_tokens, num_subjects, total_len, x.device
                )
                out_actual = compiled_flex(q_actual, k_actual, v_actual, block_mask=block_mask)
            else:
                # Vectorized SDPA fallback
                coord_ids = torch.arange(num_coord_tokens, device=x.device)
                subj_q = coord_ids % num_subjects
                subj_k = coord_ids % num_subjects
                cross_subj = subj_q.unsqueeze(1) != subj_k.unsqueeze(0)  # [C, C]
                attn_mask = torch.zeros(total_len, total_len, device=x.device, dtype=q.dtype)
                attn_mask[video_len:, video_len:][cross_subj] = float('-inf')
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                out_actual = torch.nn.functional.scaled_dot_product_attention(
                    q_actual, k_actual, v_actual, attn_mask=attn_mask, dropout_p=0.0
                )
            
            if padded_len > total_len:
                padding = torch.zeros(b, n, padded_len - total_len, d, device=out_actual.device, dtype=out_actual.dtype)
                x = torch.cat([out_actual, padding], dim=2)
            else:
                x = out_actual
            
            x = x.transpose(1, 2)
        else:
            # Use fast flash attention (no subject isolation needed)
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanGanCrossAttention(WanSelfAttention):

    def forward(self, x, context, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        qq = self.norm_q(self.q(context)).view(b, 1, -1, d)

        kk = self.norm_k(self.k(x)).view(b, -1, n, d)
        vv = self.v(x).view(b, -1, n, d)

        # compute attention
        x = flash_attention(qq, kk, vv)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(
            dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanTemporalActionCrossAttention(WanSelfAttention):
    """
    Cross-attention with temporal RoPE for action conditioning.
    
    Uses "action-as-result" semantics:
    - action[i] = action that produces frame i+1 (transition from frame i to i+1)
    - action[i] gets RoPE position i+1 to align with the frame it produces
    
    Applies temporal RoPE to both:
    - Queries (video frame tokens): rotated by their frame index (0, 1, ..., N-1)
    - Keys (action embeddings): N-1 actions rotated by positions 1, 2, ..., N-1
    
    This allows frame i+1 to attend strongly to action[i] via RoPE alignment.
    Frame 0 (initial frame) has no corresponding action.
    """

    def forward(self, x, context, context_lens, grid_sizes=None, freqs=None, coord_token_info=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - video tokens (F*H*W patches)
            context(Tensor): Shape [B, N-1, C] - N-1 action embeddings for N frames
            context_lens(Tensor): Shape [B] - unused but kept for interface compatibility
            grid_sizes(Tensor): Shape [B, 3] - (F, H, W) grid dimensions
            freqs(Tensor): RoPE frequency parameters
            coord_token_info: Optional dict for coordinate token RoPE
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # compute query from video tokens
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        # compute key, value from action embeddings
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Apply temporal RoPE if grid_sizes and freqs are provided
        if grid_sizes is not None and freqs is not None:
            # Apply temporal RoPE to queries (video tokens + coord tokens)
            # Frames 0, 1, ..., N-1 get RoPE positions 0, 1, ..., N-1
            q = temporal_rope_apply(q, grid_sizes, freqs, coord_token_info=coord_token_info)
            
            # Apply temporal RoPE to keys (action embeddings) with position offset
            # action[i] produces frame i+1, so it should get RoPE position i+1
            # This aligns action[i] with frame i+1's queries
            num_actions = k.size(1)
            k = self._temporal_rope_apply_with_offset(k, num_actions, freqs, offset=1)
        
        # compute attention
        x = flash_attention(q, k, v, k_lens=None)  # No masking needed
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    
    def _temporal_rope_apply_with_offset(self, x, num_positions, freqs, offset=0):
        """
        Apply temporal RoPE with a position offset.
        
        Args:
            x: [B, seq_len, num_heads, head_dim] - input tensor
            num_positions: Number of positions (should equal seq_len)
            freqs: RoPE frequency parameters [max_seq_len, head_dim//2]
            offset: Starting position offset (default 0)
        
        For action-as-result: offset=1 means positions 1, 2, ..., N-1 instead of 0, 1, ..., N-2
        """
        b = x.size(0)
        n, c = x.size(2), x.size(3) // 2
        seq_len = x.size(1)
        
        # Use freqs[offset:offset+num_positions] instead of freqs[:num_positions]
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(b, seq_len, n, -1, 2))
        freqs_slice = freqs[offset:offset + num_positions, :c].view(1, num_positions, 1, -1)
        
        # Apply rotary embedding
        x_rotated = torch.view_as_real(x_complex * freqs_slice).flatten(3)
        
        return x_rotated.type_as(x)


class WanMultiSubjectCrossAttention(WanSelfAttention):
    """
    Cross-attention with temporal RoPE for multi-subject action conditioning.
    
    Handles S subjects, each with T=N-1 actions for N frames.
    Context shape: [B, S*T, C] where actions are laid out as:
        [subj0_action0, ..., subj0_actionT-1, subj1_action0, ..., subjS-1_actionT-1]
    
    Each subject's action[i] gets RoPE position i+1 (action-as-result semantics).
    The model learns to attend to the correct subject's action through the concatenated
    subject features in the action embedding.
    """
    
    def __init__(self, *args, num_subjects: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_subjects = num_subjects

    def forward(self, x, context, context_lens, grid_sizes=None, freqs=None, num_subjects=None, coord_token_info=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C] - video tokens (F*H*W patches)
            context(Tensor): Shape [B, S*T, C] - S*T action embeddings
            context_lens(Tensor): Shape [B] - unused
            grid_sizes(Tensor): Shape [B, 3] - (F, H, W) grid dimensions
            freqs(Tensor): RoPE frequency parameters
            num_subjects(int): Number of subjects (overrides self.num_subjects if provided)
            coord_token_info: Optional dict for coordinate token RoPE
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        S = num_subjects if num_subjects is not None else self.num_subjects
        
        # compute query from video tokens
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        # compute key, value from action embeddings
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Apply temporal RoPE if grid_sizes and freqs are provided
        if grid_sizes is not None and freqs is not None:
            # Apply temporal RoPE to queries (video tokens + coord tokens)
            q = temporal_rope_apply(q, grid_sizes, freqs, coord_token_info=coord_token_info)
            
            # Apply temporal RoPE to keys (action embeddings)
            # context is [B, S*T, C] -> reshape to [B, S, T, n, d]
            total_actions = k.size(1)
            T = total_actions // S
            
            # Reshape to [B, S, T, n, d]
            k_reshaped = k.view(b, S, T, n, d)
            
            # Apply temporal RoPE per-subject with offset=1
            # Each subject's action[i] gets position i+1
            k_rotated_list = []
            for s in range(S):
                k_s = k_reshaped[:, s]  # [B, T, n, d]
                k_s_rotated = self._temporal_rope_apply_with_offset(k_s, T, freqs, offset=1)
                k_rotated_list.append(k_s_rotated)
            
            # Stack and flatten back to [B, S*T, n, d]
            k = torch.stack(k_rotated_list, dim=1).view(b, S * T, n, d)
        
        # compute attention
        x = flash_attention(q, k, v, k_lens=None)
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x
    
    def _temporal_rope_apply_with_offset(self, x, num_positions, freqs, offset=0):
        """Apply temporal RoPE with a position offset."""
        b = x.size(0)
        n, c = x.size(2), x.size(3) // 2
        seq_len = x.size(1)
        
        x_complex = torch.view_as_complex(x.to(torch.float64).reshape(b, seq_len, n, -1, 2))
        freqs_slice = freqs[offset:offset + num_positions, :c].view(1, num_positions, 1, -1)
        x_rotated = torch.view_as_real(x_complex * freqs_slice).flatten(3)
        
        return x_rotated.type_as(x)


from .spatial_cross_attn import WanSpatialCrossAttention, WanMaskedPositionCrossAttention, WanCoordActionCrossAttention
# === COORD-TOKEN EXTENSION === [end]

WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
    # Coord/subject-aware cross-attentions added for this work:
    'temporal_action_cross_attn': WanTemporalActionCrossAttention,
    'multi_subject_cross_attn': WanMultiSubjectCrossAttention,
    'spatial_cross_attn': WanSpatialCrossAttention,
    'masked_position_cross_attn': WanMaskedPositionCrossAttention,
    'dual_stream_cross_attn': WanMaskedPositionCrossAttention,  # video->null uses same class
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 separate_position_modulation=False,
                 cross_attn_rope=False,
                 per_frame_cross_attn=False,
                 cross_attn_subject_isolation=True):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.cross_attn_type = cross_attn_type
        self.eps = eps
        self.separate_position_modulation = separate_position_modulation
        self.cross_attn_rope = cross_attn_rope

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # Cross-attention with optional RoPE for spatial correspondence
        cross_attn_kwargs = {}
        if cross_attn_type in ('masked_position_cross_attn', 'dual_stream_cross_attn'):
            cross_attn_kwargs['cross_attn_rope'] = cross_attn_rope
            cross_attn_kwargs['per_frame_cross_attn'] = per_frame_cross_attn
            cross_attn_kwargs['subject_isolation'] = cross_attn_subject_isolation
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps,
                                                                      **cross_attn_kwargs)
        
        # Dual-stream: second cross-attention for coord tokens -> actions (independent weights)
        self.cross_attn_action = None
        if cross_attn_type == 'dual_stream_cross_attn':
            self.cross_attn_action = WanCoordActionCrossAttention(
                dim, num_heads, (-1, -1), qk_norm, eps,
                cross_attn_rope=cross_attn_rope
            )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # Separate modulation for position tokens (when enabled)
        # Initialize with same distribution - will be copied from modulation after checkpoint loading
        if separate_position_modulation:
            self.modulation_pos = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        else:
            self.modulation_pos = None

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        context_positions=None,
        coord_token_info=None,
        position_token_info=None,
        video_text_embeds=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] (uniform) OR [B, T, 6, C] (per-frame)
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context_positions(Tensor): Shape [B, S*T, 3] - (frame, h, w) for spatial cross-attn
            coord_token_info: Optional dict for coordinate token RoPE
            position_token_info: Optional dict for position token RoPE (same 3D RoPE as video)
            video_text_embeds: Optional[Tensor] Shape [B, 512, 1536] projected game text for video cross-attn
        """
        # Check if per-frame timesteps mode: e has shape [B, T, 6, C]
        per_frame = e.dim() == 4
        
        if per_frame:
            # Per-frame modulation: e is [B, T, 6, C]
            num_frames = e.size(1)
            
            # IMPORTANT: Use video_seq_lens from position_token_info to get ORIGINAL video token count
            # After the fix, seq_lens includes both video AND position tokens
            # So we need the original video count for proper per-frame modulation splitting
            if position_token_info is not None and 'video_seq_lens' in position_token_info:
                num_video_tokens = position_token_info['video_seq_lens'][0].item()
            elif coord_token_info is not None and 'video_seq_lens' in coord_token_info:
                num_video_tokens = coord_token_info['video_seq_lens'][0].item()
            else:
                # Fallback for non-position-token mode
                num_video_tokens = seq_lens[0].item()
            video_frame_seqlen = num_video_tokens // num_frames
            
            # Check if we have extra tokens (position tokens OR coord tokens)
            # Get actual position token count from position_token_info, NOT from x.size()
            has_position_tokens = (position_token_info is not None and position_token_info.get('num_tokens', 0) > 0)
            has_coord_tokens = (coord_token_info is not None and coord_token_info.get('num_tokens', 0) > 0)
            has_extra_tokens = has_position_tokens or has_coord_tokens
            
            # For coord tokens, we need to set up similar variables
            # IMPORTANT: coord tokens are FRAME-MAJOR: [f0_s0, f0_s1, f1_s0, f1_s1, ...]
            # Position tokens are SUBJECT-MAJOR: [s0_f0, s0_f1, ..., s1_f0, s1_f1, ...]
            is_frame_major_layout = False
            if has_coord_tokens and not has_position_tokens:
                # Coord token mode: coord tokens at the end of sequence
                num_coord_tokens = coord_token_info['num_tokens']  # e.g., 18
                num_coord_subjects = coord_token_info.get('num_subjects', 1)  # e.g., 2
                coord_T = coord_token_info.get('num_frames', num_frames)  # e.g., 9
                # Set these for compatibility with position token code paths
                num_actual_pos_tokens = num_coord_tokens
                num_pos_subjects = num_coord_subjects
                tokens_per_subject = num_coord_tokens // num_coord_subjects  # 9 per subject
                tokens_per_frame_per_subject = tokens_per_subject // num_frames  # 1 per frame per subject
                is_frame_major_layout = True  # Coord tokens use frame-major layout
            
            if has_position_tokens:
                # Position tokens: need to know layout for correct per-frame modulation
                # Layout is [subj0_all (tokens_per_subject), subj1_all (tokens_per_subject), ...]
                # Each subject has [f0, f1, ..., f8] contiguous tokens
                num_actual_pos_tokens = position_token_info['num_tokens']  # e.g., 4608
                num_pos_subjects = position_token_info.get('num_subjects', 1)  # e.g., 2
                tokens_per_subject = position_token_info.get('tokens_per_subject', num_actual_pos_tokens)  # e.g., 2304
                tokens_per_frame_per_subject = tokens_per_subject // num_frames  # e.g., 256
                
            # Add modulation per-frame: expand modulation to match
            # self.modulation is [1, 6, dim], need to expand to [1, 1, 6, dim] for broadcasting
            e_raw = e  # Keep raw timestep embedding for position modulation
            e = (self.modulation.unsqueeze(1) + e_raw).chunk(6, dim=2)
            # Each e[i] is now [B, T, 1, C]
            
            # Separate modulation for position tokens (when enabled)
            if self.modulation_pos is not None and has_extra_tokens:
                e_pos = (self.modulation_pos.unsqueeze(1) + e_raw).chunk(6, dim=2)
            else:
                e_pos = None  # Use same modulation as video
            
            # Helper function to apply per-frame modulation to both video and position/coord tokens
            def apply_perframe_modulation(x, e_scale, e_shift, video_frame_seqlen, has_pos, e_pos_scale=None, e_pos_shift=None):
                """Apply per-frame modulation: x * (1 + e_scale) + e_shift
                x: [B, seq_len, C] - includes video, position, and padding
                e_scale, e_shift: [B, T, 1, C] - for video tokens
                e_pos_scale, e_pos_shift: [B, T, 1, C] - for position tokens (if separate modulation)
                """
                if has_pos:
                    B, total_len, C = x.shape
                    # Split into video, position, and padding tokens
                    # IMPORTANT: num_actual_pos_tokens is from position_token_info, not from x.size()
                    x_video = x[:, :num_video_tokens]  # [B, video_tokens, C]
                    x_pos = x[:, num_video_tokens:num_video_tokens + num_actual_pos_tokens]  # [B, pos_tokens, C]
                    x_padding = x[:, num_video_tokens + num_actual_pos_tokens:]  # [B, padding, C]
                    
                    # Apply modulation to video tokens: [B, T, tokens_per_frame, C]
                    x_video_unflat = x_video.unflatten(dim=1, sizes=(num_frames, video_frame_seqlen))
                    x_video_mod = (x_video_unflat * (1 + e_scale) + e_shift).flatten(1, 2)
                    
                    # Use separate position modulation if provided, else use video modulation
                    pos_scale = e_pos_scale if e_pos_scale is not None else e_scale
                    pos_shift = e_pos_shift if e_pos_shift is not None else e_shift
                    
                    if is_frame_major_layout:
                        # COORD TOKENS: frame-major layout [f0_s0, f0_s1, f1_s0, f1_s1, ...]
                        # Reshape to [B, T, S * tokens_per_frame_per_subject, C]
                        x_pos_reshaped = x_pos.view(B, num_frames, num_pos_subjects * tokens_per_frame_per_subject, C)
                        # e_scale/e_shift are [B, T, 1, C] - broadcasts directly with [B, T, S*tpf, C]
                        x_pos_mod = x_pos_reshaped * (1 + pos_scale) + pos_shift
                        # Flatten back: [B, T * S * tpf, C] - preserves frame-major layout
                        x_pos_mod = x_pos_mod.view(B, num_frames * num_pos_subjects * tokens_per_frame_per_subject, C)
                    else:
                        # POSITION TOKENS: subject-major layout [subj0_all, subj1_all, ...]
                        # Each subject has [f0, f1, ..., f8] contiguous tokens
                        # Reshape to [B, S, T, tokens_per_frame_per_subject, C]
                        x_pos_reshaped = x_pos.view(B, num_pos_subjects, num_frames, tokens_per_frame_per_subject, C)
                        # e_scale/e_shift are [B, T, 1, C], need [B, 1, T, 1, C] for broadcasting
                        pos_scale_exp = pos_scale.unsqueeze(1)  # [B, 1, T, 1, C]
                        pos_shift_exp = pos_shift.unsqueeze(1)  # [B, 1, T, 1, C]
                        x_pos_mod = x_pos_reshaped * (1 + pos_scale_exp) + pos_shift_exp
                        # Reshape back to [B, S * T * tokens_per_frame, C] maintaining [subj0_all, subj1_all] layout
                        x_pos_mod = x_pos_mod.view(B, num_pos_subjects * num_frames * tokens_per_frame_per_subject, C)
                    
                    # Concatenate back with padding (padding is not modulated)
                    return torch.cat([x_video_mod, x_pos_mod, x_padding], dim=1)
                else:
                    x_unflat = x.unflatten(dim=1, sizes=(num_frames, video_frame_seqlen))
                    return (x_unflat * (1 + e_scale) + e_shift).flatten(1, 2)
            
            def apply_perframe_output_modulation(y, e_scale, video_frame_seqlen, has_pos, e_pos_scale=None):
                """Apply per-frame output modulation: y * e_scale
                y: [B, seq_len, C] - includes video, position, and padding
                e_scale: [B, T, 1, C] - for video tokens
                e_pos_scale: [B, T, 1, C] - for position tokens (if separate modulation)
                """
                if has_pos:
                    B, total_len, C = y.shape
                    y_video = y[:, :num_video_tokens]
                    y_pos = y[:, num_video_tokens:num_video_tokens + num_actual_pos_tokens]
                    y_padding = y[:, num_video_tokens + num_actual_pos_tokens:]
                    
                    y_video_unflat = y_video.unflatten(dim=1, sizes=(num_frames, video_frame_seqlen))
                    y_video_mod = (y_video_unflat * e_scale).flatten(1, 2)
                    
                    # Use separate position modulation if provided
                    pos_scale = e_pos_scale if e_pos_scale is not None else e_scale
                    
                    if is_frame_major_layout:
                        # COORD TOKENS: frame-major layout
                        y_pos_reshaped = y_pos.view(B, num_frames, num_pos_subjects * tokens_per_frame_per_subject, C)
                        y_pos_mod = (y_pos_reshaped * pos_scale).view(B, num_frames * num_pos_subjects * tokens_per_frame_per_subject, C)
                    else:
                        # POSITION TOKENS: subject-major layout
                        y_pos_reshaped = y_pos.view(B, num_pos_subjects, num_frames, tokens_per_frame_per_subject, C)
                        pos_scale_exp = pos_scale.unsqueeze(1)  # [B, 1, T, 1, C]
                        y_pos_mod = y_pos_reshaped * pos_scale_exp
                        y_pos_mod = y_pos_mod.view(B, num_pos_subjects * num_frames * tokens_per_frame_per_subject, C)
                    
                    return torch.cat([y_video_mod, y_pos_mod, y_padding], dim=1)
                else:
                    y_unflat = y.unflatten(dim=1, sizes=(num_frames, video_frame_seqlen))
                    return (y_unflat * e_scale).flatten(1, 2)
            
            # self-attention (needs flattened input)
            # Pass separate position modulation if available
            e_pos_scale_1, e_pos_shift_0 = (e_pos[1], e_pos[0]) if e_pos is not None else (None, None)
            attn_input = apply_perframe_modulation(self.norm1(x), e[1], e[0], video_frame_seqlen, has_extra_tokens,
                                                   e_pos_scale=e_pos_scale_1, e_pos_shift=e_pos_shift_0)
            
            # Check if position tokens should skip self-attention entirely (video can't see position)
            # Note: These options only apply to position_token_mode, not coord_token_mode
            position_skip_self_attn = (position_token_info is not None and 
                                       position_token_info.get('position_skip_self_attn', False) and
                                       has_position_tokens)
            
            # Check if position tokens should not be updated by self-attention (but video CAN see them)
            position_no_self_attn_update = (position_token_info is not None and 
                                            position_token_info.get('position_no_self_attn_update', False) and
                                            has_position_tokens)
            
            if position_skip_self_attn:
                # Position tokens skip self-attention entirely (pure residual path)
                # Only video tokens go through self-attention - video CAN'T see position
                B, total_len, C = attn_input.shape
                
                # Split into video and position+padding
                attn_input_video = attn_input[:, :num_video_tokens]
                attn_input_pos = attn_input[:, num_video_tokens:num_video_tokens + num_actual_pos_tokens]
                attn_input_padding = attn_input[:, num_video_tokens + num_actual_pos_tokens:]
                
                # Only video tokens go through self-attention (with adjusted seq_lens)
                video_only_seq_lens = torch.clamp(seq_lens - num_actual_pos_tokens, min=0)
                y_video = self.self_attn(attn_input_video, video_only_seq_lens, grid_sizes, freqs, 
                                         coord_token_info=coord_token_info, position_token_info=None)
                
                # Position tokens get zeros (they skip self-attention, will only get residual)
                y_pos = torch.zeros_like(attn_input_pos)
                y_padding = torch.zeros_like(attn_input_padding)
                
                # Reconstruct full y
                y = torch.cat([y_video, y_pos, y_padding], dim=1)
            elif position_no_self_attn_update:
                # All tokens go through self-attention (video CAN see position)
                # But position token outputs are zeroed (position doesn't update from self-attn)
                y = self.self_attn(attn_input, seq_lens, grid_sizes, freqs, coord_token_info=coord_token_info, position_token_info=position_token_info)
                
                # Zero out position token outputs (they only get residual)
                B, total_len, C = y.shape
                y_video = y[:, :num_video_tokens]
                y_pos_zeroed = torch.zeros_like(y[:, num_video_tokens:num_video_tokens + num_actual_pos_tokens])
                y_padding = y[:, num_video_tokens + num_actual_pos_tokens:]
                y = torch.cat([y_video, y_pos_zeroed, y_padding], dim=1)
            else:
                y = self.self_attn(attn_input, seq_lens, grid_sizes, freqs, coord_token_info=coord_token_info, position_token_info=position_token_info)
            
            # Apply per-frame output modulation
            e_pos_scale_2 = e_pos[2] if e_pos is not None else None
            x = x + apply_perframe_output_modulation(y, e[2], video_frame_seqlen, has_extra_tokens, e_pos_scale=e_pos_scale_2)
            
            # cross-attention & ffn with per-frame modulation
            def cross_attn_ffn_perframe(x, context, context_lens, e, e_pos, grid_sizes, freqs, context_positions, coord_token_info, position_token_info):
                if self.cross_attn_type in ('temporal_action_cross_attn', 'multi_subject_cross_attn'):
                    x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            coord_token_info=coord_token_info)
                elif self.cross_attn_type == 'spatial_cross_attn':
                    x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            context_positions=context_positions,
                                            coord_token_info=coord_token_info)
                elif self.cross_attn_type == 'dual_stream_cross_attn':
                    # DUAL STREAM: separate cross-attention for video (->text) and coord (->actions)
                    normed = self.norm3(x)
                    num_subjects = coord_token_info.get('num_subjects', 1) if coord_token_info is not None else 1
                    num_coord = coord_token_info['num_tokens'] if coord_token_info is not None else 0
                    
                    if num_coord > 0 and self.cross_attn_action is not None:
                        # Split: video tokens | coord tokens | padding
                        normed_video = normed[:, :-num_coord]  # video + any padding before coord
                        normed_coord = normed[:, -num_coord:]  # coord tokens only
                        
                        # Video -> game text (WanMaskedPositionCrossAttention, no coord_token_info)
                        video_out = self.cross_attn(normed_video, context, context_lens,
                                                    grid_sizes=grid_sizes, freqs=freqs,
                                                    context_positions=context_positions,
                                                    num_subjects=num_subjects,
                                                    empty_prompt_embed=video_text_embeds)
                        
                        # Coord -> actions (WanCoordActionCrossAttention, with subject isolation + RoPE)
                        coord_out = self.cross_attn_action(normed_coord, context,
                                                           context_lens=context_lens,
                                                           grid_sizes=grid_sizes, freqs=freqs,
                                                           context_positions=context_positions,
                                                           coord_token_info=coord_token_info,
                                                           num_subjects=num_subjects)
                        
                        # Merge back: [video_out, coord_out]
                        x = x + torch.cat([video_out, coord_out], dim=1)
                    else:
                        # Fallback: no coord tokens, just video -> text
                        x = x + self.cross_attn(normed, context, context_lens,
                                                grid_sizes=grid_sizes, freqs=freqs,
                                                context_positions=context_positions,
                                                num_subjects=num_subjects,
                                                empty_prompt_embed=video_text_embeds)
                elif self.cross_attn_type == 'masked_position_cross_attn':
                    # Masked cross-attn: video→game text, position/coord→actions
                    # Get num_subjects from position_token_info or coord_token_info
                    if position_token_info is not None:
                        num_subjects = position_token_info.get('num_subjects', 1)
                    elif coord_token_info is not None:
                        num_subjects = coord_token_info.get('num_subjects', 1)
                    else:
                        num_subjects = 1
                    x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            context_positions=context_positions,
                                            coord_token_info=coord_token_info,
                                            position_token_info=position_token_info,
                                            num_subjects=num_subjects,
                                            empty_prompt_embed=video_text_embeds)
                else:
                    x = x + self.cross_attn(self.norm3(x), context, context_lens)
                
                # FFN with per-frame modulation - pass separate position modulation if available
                e_pos_scale_4, e_pos_shift_3 = (e_pos[4], e_pos[3]) if e_pos is not None else (None, None)
                ffn_input = apply_perframe_modulation(self.norm2(x), e[4], e[3], video_frame_seqlen, has_extra_tokens,
                                                      e_pos_scale=e_pos_scale_4, e_pos_shift=e_pos_shift_3)
                y = self.ffn(ffn_input)
                e_pos_scale_5 = e_pos[5] if e_pos is not None else None
                x = x + apply_perframe_output_modulation(y, e[5], video_frame_seqlen, has_extra_tokens, e_pos_scale=e_pos_scale_5)
                return x
            
            x = cross_attn_ffn_perframe(x, context, context_lens, e, e_pos, grid_sizes, freqs, context_positions, coord_token_info, position_token_info)
        else:
            # Standard uniform modulation: e is [B, 6, C]
            # Check if we have position tokens for separate modulation
            has_position_tokens_uniform = (position_token_info is not None and position_token_info.get('num_tokens', 0) > 0)
            use_separate_pos_mod = (self.modulation_pos is not None and has_position_tokens_uniform)
            
            e_raw = e  # Keep raw timestep embedding for position modulation
            e = (self.modulation + e_raw).chunk(6, dim=1)
            
            # Separate modulation for position tokens (when enabled)
            if use_separate_pos_mod:
                e_pos = (self.modulation_pos + e_raw).chunk(6, dim=1)
                num_video_tokens_uniform = position_token_info['video_seq_lens'][0].item()
                num_actual_pos_tokens_uniform = position_token_info['num_tokens']
            else:
                e_pos = None
            
            # Helper functions for uniform modulation with separate position modulation
            def apply_uniform_modulation(x_normed, e_scale, e_shift, e_pos_scale=None, e_pos_shift=None):
                """Apply uniform modulation: x * (1 + e_scale) + e_shift"""
                if use_separate_pos_mod:
                    x_video = x_normed[:, :num_video_tokens_uniform]
                    x_pos = x_normed[:, num_video_tokens_uniform:num_video_tokens_uniform + num_actual_pos_tokens_uniform]
                    x_padding = x_normed[:, num_video_tokens_uniform + num_actual_pos_tokens_uniform:]
                    
                    x_video_mod = x_video * (1 + e_scale) + e_shift
                    x_pos_mod = x_pos * (1 + e_pos_scale) + e_pos_shift
                    return torch.cat([x_video_mod, x_pos_mod, x_padding], dim=1)
                else:
                    return x_normed * (1 + e_scale) + e_shift
            
            def apply_uniform_output_modulation(y, e_scale, e_pos_scale=None):
                """Apply uniform output modulation: y * e_scale"""
                if use_separate_pos_mod:
                    y_video = y[:, :num_video_tokens_uniform]
                    y_pos = y[:, num_video_tokens_uniform:num_video_tokens_uniform + num_actual_pos_tokens_uniform]
                    y_padding = y[:, num_video_tokens_uniform + num_actual_pos_tokens_uniform:]
                    
                    y_video_mod = y_video * e_scale
                    y_pos_mod = y_pos * e_pos_scale
                    return torch.cat([y_video_mod, y_pos_mod, y_padding], dim=1)
                else:
                    return y * e_scale

        # self-attention
            attn_input = apply_uniform_modulation(self.norm1(x), e[1], e[0], 
                                                  e_pos[1] if e_pos else None, e_pos[0] if e_pos else None)
            y = self.self_attn(attn_input, seq_lens, grid_sizes,
                freqs, coord_token_info=coord_token_info, position_token_info=position_token_info)
            x = x + apply_uniform_output_modulation(y, e[2], e_pos[2] if e_pos else None)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, e_pos, grid_sizes, freqs, context_positions, coord_token_info, position_token_info):
            # Pass grid_sizes and freqs for temporal action/multi-subject cross-attention
            if self.cross_attn_type in ('temporal_action_cross_attn', 'multi_subject_cross_attn'):
                x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            coord_token_info=coord_token_info)
            elif self.cross_attn_type == 'spatial_cross_attn':
                # Spatial cross-attention with 3D RoPE based on context positions
                x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                        grid_sizes=grid_sizes, freqs=freqs,
                                        context_positions=context_positions,
                                        coord_token_info=coord_token_info)
            elif self.cross_attn_type == 'dual_stream_cross_attn':
                # DUAL STREAM: separate cross-attention for video (->text) and coord (->actions)
                normed = self.norm3(x)
                num_subjects = coord_token_info.get('num_subjects', 1) if coord_token_info is not None else 1
                num_coord = coord_token_info['num_tokens'] if coord_token_info is not None else 0
                
                if num_coord > 0 and self.cross_attn_action is not None:
                    normed_video = normed[:, :-num_coord]
                    normed_coord = normed[:, -num_coord:]
                    
                    video_out = self.cross_attn(normed_video, context, context_lens,
                                                grid_sizes=grid_sizes, freqs=freqs,
                                                context_positions=context_positions,
                                                num_subjects=num_subjects,
                                                empty_prompt_embed=video_text_embeds)
                    
                    coord_out = self.cross_attn_action(normed_coord, context,
                                                       context_lens=context_lens,
                                                       grid_sizes=grid_sizes, freqs=freqs,
                                                       context_positions=context_positions,
                                                       coord_token_info=coord_token_info,
                                                       num_subjects=num_subjects)
                    
                    x = x + torch.cat([video_out, coord_out], dim=1)
                else:
                    x = x + self.cross_attn(normed, context, context_lens,
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            context_positions=context_positions,
                                            num_subjects=num_subjects,
                                            empty_prompt_embed=video_text_embeds)
            elif self.cross_attn_type == 'masked_position_cross_attn':
                    # Masked cross-attn: video→game text, position/coord→actions
                    # Get num_subjects from position_token_info or coord_token_info
                    if position_token_info is not None:
                        num_subjects = position_token_info.get('num_subjects', 1)
                    elif coord_token_info is not None:
                        num_subjects = coord_token_info.get('num_subjects', 1)
                    else:
                        num_subjects = 1
                    x = x + self.cross_attn(self.norm3(x), context, context_lens, 
                                            grid_sizes=grid_sizes, freqs=freqs,
                                            context_positions=context_positions,
                                            coord_token_info=coord_token_info,
                                            position_token_info=position_token_info,
                                            num_subjects=num_subjects,
                                            empty_prompt_embed=video_text_embeds)
            else:
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                
                # FFN with uniform modulation
                ffn_input = apply_uniform_modulation(self.norm2(x), e[4], e[3],
                                                     e_pos[4] if e_pos else None, e_pos[3] if e_pos else None)
                y = self.ffn(ffn_input)
                x = x + apply_uniform_output_modulation(y, e[5], e_pos[5] if e_pos else None)
            return x

            x = cross_attn_ffn(x, context, context_lens, e, e_pos, grid_sizes, freqs, context_positions, coord_token_info, position_token_info)
        return x


class GanAttentionBlock(nn.Module):

    def __init__(self,
                 dim=1536,
                 ffn_dim=8192,
                 num_heads=12,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        # self.norm1 = WanLayerNorm(dim, eps)
        # self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
        #   eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.cross_attn = WanGanCrossAttention(dim, num_heads,
                                               (-1, -1),
                                               qk_norm,
                                               eps)

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        context,
        # seq_lens,
        # grid_sizes,
        # freqs,
        # context,
        # context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        # e = (self.modulation + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # # self-attention
        # y = self.self_attn(
        #     self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes,
        #     freqs)
        # # with amp.autocast(dtype=torch.float32):
        # x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context):
            token = context + self.cross_attn(self.norm3(x), context)
            y = self.ffn(self.norm2(token)) + token  # * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            # x = x + y * e[5]
            return y

        x = cross_attn_ffn(x, context)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class RegisterTokens(nn.Module):
    def __init__(self, num_registers: int, dim: int):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(num_registers, dim) * 0.02)
        self.rms_norm = WanRMSNorm(dim, eps=1e-6)

    def forward(self):
        return self.rms_norm(self.register_tokens)

    def reset_parameters(self):
        nn.init.normal_(self.register_tokens, std=0.02)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 cross_attn_type=None,
                 eps=1e-6,
                 position_channels=0,
                 videojam_mode=False,
                 videojam_position_residual=True,
                 videojam_identity_init=False,
                 coord_token_mode=False,
                 coord_temporal_rope=False,
                 coord_spatial_rope=False,
                 coord_layernorm=False,
                 coord_fourier_encoding=False,
                 coord_fourier_frequencies=10,
                 coord_direct_embedding=False,
                 coord_no_encoder_decoder=False,
                 coord_scale_factor=1.0,
                 num_subjects=2,
                 per_frame_timesteps=False,
                 position_token_mode=False,
                 position_temporal_offset=0.0,
                 subject_isolation_self_attn=False,
                 position_no_video_attn=False,
                 position_skip_self_attn=False,
                 position_no_self_attn_update=False,
                 use_flex_attention=True,
                 separate_position_modulation=False,
                 pixel_position_mode=False,
                 pixel_position_num_subjects=2,
                 cross_attn_rope=False,
                 per_frame_cross_attn=False,
                 cross_attn_subject_isolation=True):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            cross_attn_type (`str`, *optional*, defaults to None):
                Cross-attention type: None (use model_type default), 't2v_cross_attn', 
                'i2v_cross_attn', or 'temporal_action_cross_attn'
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
            position_channels (`int`, *optional*, defaults to 0):
                Number of position map channels for joint position diffusion (VideoJAM-style).
                If > 0, adds separate position_embedding and position_head for position maps.
            videojam_mode (`bool`, *optional*, defaults to False):
                If True, uses true VideoJAM-style architecture:
                - Position latents are VAE-encoded (16 channels, same as video)
                - Both use the same patch_embedding
                - Concatenate along channel dim, project with Win+ (2*dim → dim)
                - After transformer, project with Wout+ (dim → 2*dim)
                - Split back into video and position predictions
            coord_token_mode (`bool`, *optional*, defaults to False):
                If True, uses coordinate token architecture:
                - Subject coordinates (x, y) are encoded as separate tokens
                - Tokens are concatenated with video patches in sequence dimension
                - After transformer, coordinate tokens are decoded to (x, y) predictions
            coord_temporal_rope (`bool`, *optional*, defaults to False):
                If True, applies temporal RoPE to coordinate tokens so they have
                frame-index position encoding. This helps the model know which
                frame each coordinate token corresponds to.
            coord_spatial_rope (`bool`, *optional*, defaults to False):
                If True, also applies SPATIAL RoPE to coordinate tokens based on
                their actual (x, y) position. This makes coord tokens "appear" at
                the spatial location of the subject, enabling natural attention
                alignment with video patches at that location.
            coord_layernorm (`bool`, *optional*, defaults to False):
                If True, applies LayerNorm after coord_encoder to normalize coord
                token representations to match video patch statistics. This can
                help stabilize training by preventing scale mismatches.
                - More explicit coordinate supervision than diffusing position maps
            num_subjects (`int`, *optional*, defaults to 2):
                Number of subjects to track when coord_token_mode=True.
            per_frame_timesteps (`bool`, *optional*, defaults to False):
                If True, enables per-frame timestep conditioning for AR training.
                Each frame can have a different diffusion timestep, allowing:
                - Context frames: t=0 (clean)
                - Target frame(s): varying noise levels
                This is similar to CausalWanModel's per-frame modulation.
        """

        super().__init__()
        self.per_frame_timesteps = per_frame_timesteps

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.local_attn_size = 21
        self.position_channels = position_channels
        self.videojam_mode = videojam_mode
        self.videojam_position_residual = videojam_position_residual
        self.videojam_identity_init = videojam_identity_init
        self.coord_token_mode = coord_token_mode
        self.coord_temporal_rope = coord_temporal_rope
        self.coord_spatial_rope = coord_spatial_rope
        self.coord_layernorm = coord_layernorm
        self.num_subjects = num_subjects
        self.coord_only_no_video = False  # Set by CausalDiffusion when coord_only_no_video is True
        self.position_token_mode = position_token_mode  # Position maps as sequential tokens with 3D RoPE
        self.position_temporal_offset = position_temporal_offset  # Temporal RoPE offset for position tokens to distinguish from video
        self.subject_isolation_self_attn = subject_isolation_self_attn  # If True, isolate position tokens per subject in self-attention
        self.position_no_video_attn = position_no_video_attn  # If True, position tokens cannot attend to video tokens (only own subject's position)
        self.position_skip_self_attn = position_skip_self_attn  # If True, position tokens completely skip self-attention (pure residual)
        self.position_no_self_attn_update = position_no_self_attn_update  # If True, position tokens don't update from self-attn but video can see them
        self.use_flex_attention = use_flex_attention  # If True, use FlexAttention for masked self-attention (faster)
        self.separate_position_modulation = separate_position_modulation  # If True, use separate adaLN modulation for position tokens
        self.pixel_position_mode = pixel_position_mode  # If True, diffuse pixel-space position maps (not VAE encoded)
        self.pixel_position_num_subjects = pixel_position_num_subjects  # Number of subjects/channels for pixel position maps
        self.cross_attn_rope = cross_attn_rope  # If True, apply 3D RoPE in cross-attention for spatial correspondence

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # Pixel position mode: embed raw pixel position maps (1 channel per subject) to model dim
        # Position maps are at latent spatial resolution (e.g., 32x32), one channel per subject
        # Each subject is embedded SEPARATELY to enable subject isolation masking
        if pixel_position_mode:
            # Conv3d to embed SINGLE subject position pixels: [B, 1, T, H, W] -> [B, dim, T, H/2, W/2]
            # Uses same kernel/stride as patch_embedding for consistent spatial structure
            self.position_pixel_embedding_per_subj = nn.Conv3d(
                1, dim, kernel_size=patch_size, stride=patch_size)
            # Output head for position prediction per subject: 1 channel output
            # We'll apply this to each subject's tokens separately
            self.position_pixel_head_per_subj = Head(dim, 1, patch_size, eps)
            print(f"  Using PIXEL POSITION MODE: diffusing raw position maps ({pixel_position_num_subjects} subjects, each 1 channel)")
            print(f"    position_pixel_embedding_per_subj: Conv3d(1 -> {dim})")
            print(f"    position_pixel_head_per_subj: {dim} -> 1 channel (applied per subject)")
        
        # VideoJAM mode: position latents are VAE-encoded (16 channels)
        # Use same patch_embedding for both, then project with Win+/Wout+
        if videojam_mode:
            # Win+: project concatenated [video_embed, position_embed] to joint representation
            self.win_proj = nn.Linear(dim * 2, dim)
            # Wout+: project joint representation back to [video, position]
            self.wout_proj = nn.Linear(dim, dim * 2)
            # No separate position embedding in VideoJAM mode
            self.position_embedding = None
        elif position_channels > 0:
            # Legacy mode: separate position embedding for raw position maps
            # Position embedding is added to video embedding (not concatenated in sequence)
            # This preserves pretrained weights completely
            self.position_embedding = nn.Conv3d(
                position_channels, dim, kernel_size=patch_size, stride=patch_size)
            self.win_proj = None
            self.wout_proj = None
        else:
            self.position_embedding = None
            self.win_proj = None
            self.wout_proj = None
        
        # I2V cross-attention: separate patch embedding for first frame
        # Weights will be copied from patch_embedding after checkpoint loading
        # self.latent_patch_embedding = nn.Conv3d(
        #     in_dim, dim, kernel_size=patch_size, stride=patch_size)
        
        # Action conditioning: learnable action embeddings (optional)
        # Will be dynamically created by diffusion.py if use_action_conditioning=True
        # self.action_embedding = nn.Embedding(num_action_classes, text_dim)
        
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        # Use explicit cross_attn_type if provided, else default based on model_type
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.cross_attn_type = cross_attn_type
        self.per_frame_cross_attn = per_frame_cross_attn
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              separate_position_modulation=separate_position_modulation,
                              cross_attn_rope=cross_attn_rope,
                              per_frame_cross_attn=per_frame_cross_attn,
                              cross_attn_subject_isolation=cross_attn_subject_isolation)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # Joint position diffusion: separate head for position maps
        if videojam_mode:
            # VideoJAM mode: position head outputs same channels as video (16 from VAE)
            self.position_head = Head(dim, out_dim, patch_size, eps)
        elif position_channels > 0:
            # Legacy mode: position head outputs position_channels
            self.position_head = Head(dim, position_channels, patch_size, eps)
        else:
            self.position_head = None
        
        # Position token mode: separate head for position tokens (when separate_position_modulation is enabled)
        # This allows position tokens to have their own modulation in the head layer
        if position_token_mode and separate_position_modulation:
            self.position_token_head = Head(dim, out_dim, patch_size, eps)
        else:
            self.position_token_head = None
        
        # Coordinate token mode: encode (x, y) per subject as tokens, concat in sequence
        self.coord_fourier_encoding = coord_fourier_encoding
        self.coord_direct_embedding = coord_direct_embedding
        self.coord_no_encoder_decoder = coord_no_encoder_decoder
        self.coord_scale_factor = coord_scale_factor
        if coord_token_mode:
            if coord_no_encoder_decoder:
                # PURE SLICING: no learned encoder/decoder
                # Input: [x, y, 0, 0, ..., 0] (pad with zeros)
                # Output: slice first 2 dims (no learnable params)
                self.coord_encoder = None
                self.coord_encoder_norm = None
                self.coord_decoder = None
                print(f"  Using PURE SLICING: no encoder/decoder (baseline)")
            elif getattr(self, 'coord_encoder_only', False):
                # ENCODER ONLY: learned encoder, no decoder (slice output)
                self.coord_encoder = nn.Linear(2, dim, bias=False)
                self.coord_encoder_norm = None
                self.coord_decoder = None  # Slice first 2 dims
                print(f"  Using ENCODER ONLY: learned 2→{dim}, slice output")
            elif coord_direct_embedding:
                # DIRECT EMBEDDING: pad (x,y) to dim
                # Input: [x, y, 0, 0, ..., 0] (dim-dimensional)
                # Output: USE LEARNED DECODER for proper gradient flow!
                # Without decoder, only 2/1536 dims get gradients (0.13% coverage)
                # With decoder, all 1536 dims contribute to the 2 output dims
                self.coord_encoder = None
                self.coord_encoder_norm = None
                self.coord_decoder = nn.Linear(dim, 2, bias=True)  # LEARNED decoder!
                print(f"  Using DIRECT coord embedding WITH learned decoder (full gradient flow)")
            elif coord_fourier_encoding:
                # FIXED Fourier encoding - creates distinguishable features from step 0
                self.coord_encoder = FourierCoordEncoder(dim, num_frequencies=coord_fourier_frequencies)
                self.coord_encoder_norm = None  # Not needed with Fourier encoding
                self.coord_decoder = nn.Linear(dim, 2, bias=False)
                print(f"  Using Fourier coord encoder with {coord_fourier_frequencies} frequencies")
            else:
                # Learned linear: (x, y) -> dim (no bias - forces learning from input)
                self.coord_encoder = nn.Linear(2, dim, bias=False)
                # Optional LayerNorm to match video token statistics
                if coord_layernorm:
                    self.coord_encoder_norm = nn.LayerNorm(dim, eps=eps)
                else:
                    self.coord_encoder_norm = None
                # Decoder: dim -> (x, y) (no bias - prevents constant offset)
                self.coord_decoder = nn.Linear(dim, 2, bias=False)
        else:
            self.coord_encoder = None
            self.coord_encoder_norm = None
            self.coord_decoder = None
        
        # Position token mode: position maps as sequential tokens with 3D RoPE
        # Uses same patch_embedding AND same head as video (fully shared pipeline)
        if position_token_mode:
            print(f"  Using POSITION TOKEN MODE: position maps as sequential tokens with same 3D RoPE")
            print(f"    Shares: patch_embedding, transformer blocks, and output head with video")

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        *args,
        **kwargs
    ):
        # if kwargs.get('classify_mode', False) is True:
        # kwargs.pop('classify_mode')
        # return self._forward_classify(*args, **kwargs)
        # else:
        return self._forward(*args, **kwargs)

    def set_null_embedding_for_cross_attn(self, null_embedding: torch.Tensor):
        """
        Set the pre-computed null embedding (T5-encoded "") for masked cross-attention.
        This is used by WanMaskedPositionCrossAttention to give video tokens the same
        null conditioning as base Wan CFG.
        
        Args:
            null_embedding: [1, 512, 4096] - Raw T5 encoder output for ""
        
        Should be called once after model creation.
        The raw T5 output is projected through self.text_embedding (4096->1536)
        just like regular text context in base Wan.
        """
        # Project through text_embedding just like regular context
        # This matches what _forward() does: context = self.text_embedding(...)
        with torch.no_grad():
            # Pad to text_len if needed (same as _forward)
            if null_embedding.size(1) < self.text_len:
                pad = null_embedding.new_zeros(
                    null_embedding.size(0), 
                    self.text_len - null_embedding.size(1), 
                    null_embedding.size(2)
                )
                null_embedding = torch.cat([null_embedding, pad], dim=1)
            elif null_embedding.size(1) > self.text_len:
                null_embedding = null_embedding[:, :self.text_len, :]
            
            # Move to same device and dtype as model parameters
            param = next(self.parameters())
            null_embedding = null_embedding.to(device=param.device, dtype=param.dtype)
            
            # Project through text_embedding: 4096 -> 1536
            projected_null = self.text_embedding(null_embedding)
        
        for block in self.blocks:
            if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'set_null_embedding'):
                block.cross_attn.set_null_embedding(projected_null)
        print(f"  Set null embedding for masked cross-attention: raw T5 {null_embedding.shape} -> projected {projected_null.shape}")

    def _forward(
        self,
        x,
        t,
        context,
        seq_len,
        classify_mode=False,
        concat_time_embeddings=False,
        register_tokens=None,
        cls_pred_branch=None,
        gan_ca_blocks=None,
        clip_fea=None,
        y=None,
        first_frame_latent=None,
        context_positions=None,
        subject_coords=None,
        coord_num_clean=None,  # [B] - number of clean frames per batch item for spatial RoPE
        zero_position_for_masked=False,  # If True, use (0,0) for noisy frame spatial RoPE
        position_latents=None,  # [B, C, T, H, W] per-subject position latents (position_token_mode)
        position_pixels=None,  # [B, S, T, H, W] raw pixel position maps (pixel_position_mode)
        video_text_embeds=None,  # [B, 512, 4096] pre-encoded game text for video cross-attn
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Joint position diffusion: split input into video and position channels
        position_input = None
        if self.videojam_mode:
            # VideoJAM mode: input has shape [C_video + C_video, F, H, W] (32 channels total)
            # Both video and position are VAE-encoded with 16 channels each
            video_input = [u[:self.in_dim] for u in x]  # First 16 channels = video
            position_input = [u[self.in_dim:] for u in x]  # Next 16 channels = position
            x = video_input
        elif self.position_channels > 0 and self.position_embedding is not None:
            # Legacy mode: input has shape [C_video + C_pos, F, H, W]
            # Split into video [C_video, F, H, W] and position [C_pos, F, H, W]
            video_input = [u[:self.in_dim] for u in x]  # First in_dim channels = video
            position_input = [u[self.in_dim:] for u in x]  # Remaining = position
            x = video_input

        # COORD ONLY NO VIDEO: Skip video embedding entirely, just use coord tokens
        if getattr(self, 'coord_only_no_video', False) and self.coord_token_mode and subject_coords is not None:
            # No video patches - only coordinate tokens
            B_coords, T_coords, S_coords, _ = subject_coords.shape
            
            # Create dummy grid_sizes and seq_lens for compatibility
            # grid_sizes: [B, 3] = (F, H, W) - set to (0, 0, 0) so seq_len=0 (no video tokens)
            # Spatial RoPE for coords will use coord_token_info instead
            grid_sizes = torch.tensor([[0, 0, 0]] * B_coords, dtype=torch.long, device=device)
            seq_lens = torch.zeros(B_coords, dtype=torch.long, device=device)  # No video tokens
            
            # Store actual latent dims for coord spatial RoPE (used in coord_token_info)
            self._coord_only_latent_h = 32
            self._coord_only_latent_w = 32
            
            # x will be set to coord tokens below, start with empty
            x = None
        else:
            # embeddings (video)
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        
        # Joint position diffusion: handle position embeddings
        position_embeddings = None  # Store for residual at output (only used if not identity_init)
        if self.videojam_mode and position_input is not None:
            # VideoJAM mode: use same patch_embedding for position, then project with Win+
            p = [self.patch_embedding(u.unsqueeze(0)) for u in position_input]
            p = [u.flatten(2).transpose(1, 2) for u in p]
            p = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in p
            ])
            
            # Concatenate video and position embeddings
            concat_vp = torch.cat([x, p], dim=-1)  # [B, seq_len, 2*dim]
            
            if self.videojam_identity_init:
                # Identity init: Win+ = [I, 0], so Win+(concat(v, p)) = v at init
                # No residual needed - Win+ directly outputs the representation
                x = self.win_proj(concat_vp)
            else:
                # Zero init + residual: when Win+ = 0, x = video_embed (preserves pretrained)
                position_embeddings = p  # Store for residual at output
                x = x + self.win_proj(concat_vp)
        elif position_input is not None and self.position_embedding is not None:
            # Legacy mode: separate position embedding, then add
            p = [self.position_embedding(u.unsqueeze(0)) for u in position_input]
            p = [u.flatten(2).transpose(1, 2) for u in p]
            p = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                          dim=1) for u in p
            ])
            # Add position embedding to video embedding
            x = x + p
        
        # Coordinate token mode: encode subject coordinates as extra tokens
        num_coord_tokens = 0
        coord_T, coord_S = 0, 0
        video_seq_lens_for_coords = seq_lens.clone() if seq_lens is not None else None  # Will be set properly if coord tokens are added
        if self.coord_token_mode and subject_coords is not None:
            # subject_coords: [B, T, S, 2] - (x, y) per subject per frame
            B_coords, T_coords, S_coords, _ = subject_coords.shape
            coord_T, coord_S = T_coords, S_coords
            # Flatten to [B, T*S, 2]
            coords_flat = subject_coords.view(B_coords, T_coords * S_coords, 2)
            
            # Encode to [B, T*S, dim]
            if self.coord_no_encoder_decoder or self.coord_direct_embedding:
                # DIRECT/PURE: pad (x,y) with zeros to dim
                # [B, T*S, 2] -> [B, T*S, dim] with first 2 dims = coords
                coord_tokens = coords_flat.new_zeros(B_coords, T_coords * S_coords, self.dim)
                coord_tokens[..., :2] = coords_flat
            elif self.coord_encoder is not None:
                coord_tokens = self.coord_encoder(coords_flat)
                # Optional LayerNorm for stability
                if self.coord_encoder_norm is not None:
                    coord_tokens = self.coord_encoder_norm(coord_tokens)
            else:
                raise ValueError("coord_token_mode requires either encoder or direct embedding")
            num_coord_tokens = coord_tokens.shape[1]
            
            # Store ORIGINAL video token count BEFORE adding coord tokens
            video_seq_lens_for_coords = seq_lens.clone()
            
            if x is None:
                # COORD ONLY MODE: x is just coord tokens (no video)
                x = coord_tokens
                # In coord-only mode, video_seq_lens is 0 (no video tokens)
                video_seq_lens_for_coords = torch.zeros(B_coords, dtype=torch.long, device=x.device)
            else:
                # Normal mode: concatenate with video tokens
                x = torch.cat([x, coord_tokens], dim=1)
            
            # Update seq_lens to include coord tokens
            seq_lens = seq_lens + num_coord_tokens
        
        # Position token mode: embed position latents as sequential tokens with 3D RoPE
        num_position_tokens = 0
        position_grid_sizes = None
        num_subjects_in_position = 1
        tokens_per_subject = 0
        if self.position_token_mode and position_latents is not None:
            # position_latents can be:
            # - [B, C, T, H, W] for single subject (S=1)
            # - [B, S, C, T, H, W] for multi-subject (S>1)
            # Embed each subject's position latent using same patch_embedding as video
            # This ensures position tokens live in the same embedding space
            
            # Detect multi-subject format
            if position_latents.dim() == 6:
                # Multi-subject: [B, S, C, T, H, W]
                B_pos, S_pos = position_latents.shape[:2]
                num_subjects_in_position = S_pos
                
                # Embed each subject separately, then concatenate
                all_p_tokens = []
                all_p_grids = []
                for s in range(S_pos):
                    # Extract subject s: [B, C, T, H, W]
                    subj_latents = position_latents[:, s]
                    subj_tokens = self.patch_embedding(subj_latents)  # [B, dim, F, H_out, W_out]
                    # Get grid sizes from EMBEDDED tokens (not input)
                    if s == 0:
                        all_p_grids = subj_tokens.shape[2:]  # (F, H_out, W_out)
                    subj_tokens = subj_tokens.flatten(2).transpose(1, 2)  # [B, F*H*W, dim]
                    all_p_tokens.append(subj_tokens)
                
                # Get grid sizes from embedded output (correct dimensions after patch conv)
                position_grid_sizes = torch.tensor(
                    [list(all_p_grids)] * B_pos,
                    dtype=torch.long, device=position_latents.device
                )
                
                # tokens_per_subject: how many tokens per subject
                tokens_per_subject = all_p_tokens[0].shape[1]
                
                # Concatenate all subjects: [B, S * F*H*W, dim]
                p_tokens = torch.cat(all_p_tokens, dim=1)
                num_position_tokens = p_tokens.shape[1]
                
            elif isinstance(position_latents, list):
                # List of [C, T, H, W] per sample
                p_tokens = [self.patch_embedding(u.unsqueeze(0)) for u in position_latents]
                position_grid_sizes = torch.stack(
                    [torch.tensor(u.shape[2:], dtype=torch.long) for u in p_tokens])
                p_tokens = [u.flatten(2).transpose(1, 2) for u in p_tokens]
                # Concatenate without padding - will be padded after combining with video
                p_tokens = torch.cat(p_tokens)
                num_position_tokens = (position_grid_sizes[:, 0] * position_grid_sizes[:, 1] * position_grid_sizes[:, 2]).max().item()
                tokens_per_subject = num_position_tokens
            else:
                # Single subject: Tensor [B, C, T, H, W]
                p_tokens = self.patch_embedding(position_latents)
                position_grid_sizes = torch.tensor(
                    [list(p_tokens.shape[2:])] * p_tokens.shape[0],
                    dtype=torch.long, device=p_tokens.device
                )  # [B, 3] = (F, H, W)
                # Flatten: [B, dim, F, H, W] -> [B, F*H*W, dim]
                p_tokens = p_tokens.flatten(2).transpose(1, 2)
                num_position_tokens = p_tokens.shape[1]
                tokens_per_subject = num_position_tokens
            
            # Store ORIGINAL video token count BEFORE updating seq_lens
            video_seq_lens = seq_lens.clone()
            actual_video_len = video_seq_lens[0].item()  # Assumes uniform batch
            
            # FIXED: Insert position tokens RIGHT AFTER video tokens, BEFORE padding
            # Current x layout: [video_actual (2304), video_padding (seq_len - 2304)]
            # Target layout: [video_actual (2304), position_actual (S*2304), padding]
            
            # Extract video tokens (before padding)
            x_video_actual = x[:, :actual_video_len]  # [B, 2304, dim]
            
            # Concatenate: video_actual + position_actual (all subjects)
            x_combined = torch.cat([x_video_actual, p_tokens], dim=1)
            
            # Pad to original seq_len if needed
            if x_combined.size(1) < seq_len:
                padding = x_combined.new_zeros(
                    x_combined.size(0), 
                    seq_len - x_combined.size(1), 
                    x_combined.size(2)
                )
                x = torch.cat([x_combined, padding], dim=1)
            else:
                x = x_combined
            
            # CRITICAL: Update seq_lens to include ALL position tokens (all subjects)!
            seq_lens = seq_lens + num_position_tokens
            
            if not hasattr(self, '_printed_position_token_debug'):
                print(f"[POSITION TOKEN MODE] Video tokens: {video_seq_lens.max().item()}, "
                      f"Position tokens: {num_position_tokens} ({num_subjects_in_position} subjects x {tokens_per_subject} each), "
                      f"Total seq_lens: {seq_lens.max().item()}")
                self._printed_position_token_debug = True
        
        # Pixel position mode: embed raw pixel position maps as tokens
        # Each subject is embedded SEPARATELY to enable subject isolation in self-attention
        # and per-subject cross-attention masking (same as position_token_mode)
        if self.pixel_position_mode and position_pixels is not None:
            # position_pixels: [B, S, T, H, W] where S=num_subjects, H=W=32 (latent spatial)
            B_pos, S_pos, T_pos, H_pos, W_pos = position_pixels.shape
            num_subjects_in_position = S_pos
            
            # Embed each subject separately using position_pixel_embedding
            # This requires a separate embedding per subject (or shared with 1 input channel)
            all_p_tokens = []
            all_p_grids = None
            for s in range(S_pos):
                # Extract subject s: [B, 1, T, H, W] (add channel dim for Conv3d)
                subj_pixels = position_pixels[:, s:s+1]  # [B, 1, T, H, W]
                # Embed this subject's position map
                # Note: position_pixel_embedding expects S channels, but we're doing per-subject
                # So we need a single-channel embedding for each subject
                subj_tokens = self.position_pixel_embedding_per_subj(subj_pixels)  # [B, dim, T, H', W']
                if all_p_grids is None:
                    all_p_grids = subj_tokens.shape[2:]  # (T, H', W')
                subj_tokens = subj_tokens.flatten(2).transpose(1, 2)  # [B, T*H'*W', dim]
                all_p_tokens.append(subj_tokens)
            
            # Get grid sizes from embedded output
            position_grid_sizes = torch.tensor(
                [list(all_p_grids)] * B_pos,
                dtype=torch.long, device=position_pixels.device
            )  # [B, 3] = (T, H', W')
            
            # tokens_per_subject: how many tokens per subject
            tokens_per_subject = all_p_tokens[0].shape[1]
            
            # Concatenate all subjects: [B, S * T*H'*W', dim]
            p_tokens = torch.cat(all_p_tokens, dim=1)
            num_position_tokens = p_tokens.shape[1]
            
            # Store original video token count
            video_seq_lens = seq_lens.clone()
            actual_video_len = video_seq_lens[0].item()
            
            # Insert position tokens after video tokens, before padding
            x_video_actual = x[:, :actual_video_len]
            x_combined = torch.cat([x_video_actual, p_tokens], dim=1)
            
            # Pad to original seq_len if needed
            if x_combined.size(1) < seq_len:
                padding = x_combined.new_zeros(
                    x_combined.size(0),
                    seq_len - x_combined.size(1),
                    x_combined.size(2)
                )
                x = torch.cat([x_combined, padding], dim=1)
            else:
                x = x_combined
            
            # Update seq_lens to include position tokens
            seq_lens = seq_lens + num_position_tokens
            
            if not hasattr(self, '_printed_pixel_position_debug'):
                print(f"[PIXEL POSITION MODE] Video tokens: {video_seq_lens.max().item()}, "
                      f"Position tokens: {num_position_tokens} ({num_subjects_in_position} subjects x {tokens_per_subject} each), "
                      f"Input shape: {position_pixels.shape}, "
                      f"Total seq_lens: {seq_lens.max().item()}")
                self._printed_pixel_position_debug = True
        
        # Build position_token_info for RoPE and block modulation
        position_token_info = None
        if (self.position_token_mode or self.pixel_position_mode) and position_grid_sizes is not None:
            position_token_info = {
                'num_tokens': num_position_tokens,
                'grid_sizes': position_grid_sizes.to(x.device),
                'video_seq_lens': video_seq_lens,  # Original video token count for per-frame modulation
                'temporal_offset': getattr(self, 'position_temporal_offset', 0.1),  # Temporal RoPE offset to distinguish from video
                'num_subjects': num_subjects_in_position,  # Actual number of subjects in position latents
                'tokens_per_subject': tokens_per_subject,  # For masked cross-attention subject-level masking
                'subject_isolation': getattr(self, 'subject_isolation_self_attn', False),  # Subject isolation in self-attention
                'position_no_video_attn': getattr(self, 'position_no_video_attn', False),  # Position tokens don't attend to video
                'position_skip_self_attn': getattr(self, 'position_skip_self_attn', False),  # Position tokens skip self-attention entirely
                'position_no_self_attn_update': getattr(self, 'position_no_self_attn_update', False),  # Position tokens don't update but video can see them
                'use_flex_attention': getattr(self, 'use_flex_attention', True),  # Use FlexAttention for masked attention
                # num_action_tokens will be added later after context length is captured
            }

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        if self.per_frame_timesteps and t.dim() == 2:
            # Per-frame timesteps: t is [B, T]
            # Like CausalWanModel: flatten, embed, then unflatten
            B_t, T_t = t.shape
            e_flat = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e_flat).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            # e0 shape: [B, T, 6, dim] - per-frame modulation vectors
            # For head layer: use mean of per-frame embeddings [B*T, dim] -> [B, T, dim] -> [B, dim]
            e = e_flat.unflatten(0, (B_t, T_t)).mean(dim=1)
            # Debug: print shape once
            if not hasattr(self, '_printed_perframe_e0_debug'):
                print(f"[MODEL] Per-frame timestep mode: t shape {t.shape} -> e0 shape {e0.shape}, e_head shape {e.shape}")
                self._printed_perframe_e0_debug = True
        else:
            # Standard: single timestep per batch item, t is [B]
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            # e0 shape: [B, 6, dim] - uniform modulation
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        # Capture original context lengths BEFORE padding (critical for masked cross-attention)
        # Without this, the mask is computed on padded length (512) instead of actual action count (16)
        original_context_lens = [u.size(0) for u in context]
        num_action_tokens = original_context_lens[0] if original_context_lens else 0
        
        # Add num_action_tokens to position_token_info for correct mask calculation
        # Without this, the padded context length (512) is used, breaking subject isolation masking
        # NOTE: coord_token_info is built later (after context padding), so we add to it there
        if position_token_info is not None:
            position_token_info['num_action_tokens'] = num_action_tokens
        
        # Stack and pad context
        context = torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
        ])
        
        # Only apply text_embedding if context is in T5 dimension (4096), not model dim
        # Action embeddings are already in model dimension (1536) and should skip projection
        if context.size(-1) != self.dim:
            context = self.text_embedding(context)

        # Project pre-encoded game text through text_embedding for video cross-attention
        # video_text_embeds: [B, 512, 4096] → projected_video_text: [B, 512, 1536]
        # This is used by WanMaskedPositionCrossAttention as empty_prompt_embed,
        # replacing the generic null embedding with game-specific text conditioning.
        projected_video_text = None
        if video_text_embeds is not None:
            projected_video_text = self.text_embedding(video_text_embeds)

        # Process first_frame_latent for i2v cross-attention (if provided)
        if first_frame_latent is not None:
            # Apply latent_patch_embedding: [B, C, 1, H, W] -> [B, model_dim, 1, H', W']
            latent_tokens = self.latent_patch_embedding(first_frame_latent)
            # Flatten to tokens: [B, model_dim, 1, H', W'] -> [B, num_patches, model_dim]
            latent_tokens = latent_tokens.squeeze(2).flatten(2).transpose(1, 2)
            # Prepend to context
            context = torch.cat([latent_tokens, context], dim=1)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Pad context_positions to match context length (text_len)
        # context_positions: [B, num_tokens, 3] -> [B, text_len, 3]
        if context_positions is not None:
            batch_size = context_positions.size(0)
            num_tokens = context_positions.size(1)
            if num_tokens < self.text_len:
                # Pad with zeros (frame=0, h=0, w=0) for padding tokens
                padding = context_positions.new_zeros(batch_size, self.text_len - num_tokens, 3)
                context_positions = torch.cat([context_positions, padding], dim=1)

        # Build coord_token_info for RoPE if coord tokens are present AND temporal rope is enabled
        coord_token_info = None
        if self.coord_token_mode and self.coord_temporal_rope and num_coord_tokens > 0:
            # video_seq_lens_for_coords was set when we added coord tokens
            # It contains the original video token count before adding coord tokens
            coord_token_info = {
                'num_tokens': num_coord_tokens,
                'num_frames': coord_T,
                'num_subjects': coord_S,
                'subject_isolation': getattr(self, 'subject_isolation_self_attn', False),
                'use_flex_attention': getattr(self, 'use_flex_attention', True),
                'video_seq_lens': video_seq_lens_for_coords,
            }
            
            # Add spatial positions if spatial rope is enabled
            if self.coord_spatial_rope and subject_coords is not None:
                # subject_coords: [B, T, S, 2] - normalized (x, y) in range [0, 1]
                # Convert to patch positions based on grid_sizes
                # grid_sizes: [B, 3] where each row is (F, H, W)
                # x maps to W (width), y maps to H (height)
                B_coords = subject_coords.shape[0]
                
                # Get H, W - in coord_only mode, use stored dims since grid_sizes is (0,0,0)
                if getattr(self, 'coord_only_no_video', False):
                    H = getattr(self, '_coord_only_latent_h', 32)
                    W = getattr(self, '_coord_only_latent_w', 32)
                else:
                    _, H, W = grid_sizes[0].tolist()
                
                # Spatial RoPE for coord tokens:
                # NEW BEHAVIOR (fixed coord-RoPE, default):
                # - Context frames (t < num_clean): use position[t-1] (or position[0] for t=0)
                # - Denoising frame (t == num_clean): use position[num_clean-1] (last clean)
                # - Masked frames (t > num_clean): use (0, 0) -> no spatial rotation
                #
                # OLD BEHAVIOR (legacy coord-RoPE, use_legacy_coord_rope=True):
                # - Context frames: same as above
                # - ALL noisy frames: use last clean position on H//2 grid
                
                # IMPORTANT: subject_coords may be SCALED (e.g., (x-0.5)*4 to match noise distribution)
                # We need to UNSCALE to get back to [0, 1] normalized coords for RoPE position
                if self.coord_scale_factor != 1.0:
                    # Unscale: (scaled / scale_factor) + 0.5 = original
                    subject_coords_unscaled = (subject_coords / self.coord_scale_factor) + 0.5
                else:
                    subject_coords_unscaled = subject_coords
                
                # Check if we should use legacy behavior (for checkpoints trained with old code)
                use_legacy = getattr(self, '_use_legacy_coord_rope', False)
                
                rope_coords = torch.zeros_like(subject_coords_unscaled)  # [B, T, S, 2] - default (0,0)
                
                for b in range(B_coords):
                    # Determine num_clean for this batch item
                    if coord_num_clean is not None:
                        nc = coord_num_clean[b].item() if hasattr(coord_num_clean[b], 'item') else int(coord_num_clean[b])
                    else:
                        nc = coord_T  # If not provided, assume all are clean (inference fallback)
                    
                    # Frame 0: uses its own position (always clean context)
                    rope_coords[b, 0] = subject_coords_unscaled[b, 0]
                    
                    # Frames 1 to nc-1 (context): use previous frame's position
                    for t in range(1, min(nc, coord_T)):
                        rope_coords[b, t] = subject_coords_unscaled[b, t - 1]
                    
                    if use_legacy:
                        # OLD: all noisy frames use last clean position
                        if nc > 0 and nc < coord_T:
                            last_clean_pos = subject_coords_unscaled[b, nc - 1]
                            rope_coords[b, nc:] = last_clean_pos.unsqueeze(0).expand(coord_T - nc, -1, -1)
                    else:
                        # NEW: denoising frame uses last clean, masked frames use (0,0)
                        if nc > 0 and nc < coord_T:
                            rope_coords[b, nc] = subject_coords_unscaled[b, nc - 1]
                        # Frames nc+1 onwards stay at (0, 0)
                
                # Convert normalized coords to patch indices
                if use_legacy:
                    # OLD: H//2 grid (legacy coord-RoPE bug)
                    coord_h = (rope_coords[..., 1] * (H // 2)).long().clamp(0, H // 2 - 1)  # [B, T, S]
                    coord_w = (rope_coords[..., 0] * (W // 2)).long().clamp(0, W // 2 - 1)  # [B, T, S]
                else:
                    # NEW: correct H grid matching video patch grid [0, H-1]
                    coord_h = (rope_coords[..., 1] * H).long().clamp(0, H - 1)  # [B, T, S]
                    coord_w = (rope_coords[..., 0] * W).long().clamp(0, W - 1)  # [B, T, S]
                
                # Flatten to [B, T*S]
                coord_h_flat = coord_h.view(B_coords, coord_T * coord_S)
                coord_w_flat = coord_w.view(B_coords, coord_T * coord_S)
                
                coord_token_info['spatial_h'] = coord_h_flat  # [B, T*S]
                coord_token_info['spatial_w'] = coord_w_flat  # [B, T*S]

        # Add num_action_tokens to coord_token_info for correct subject isolation masking
        # MUST be done here (after coord_token_info is built), not earlier when it's still None
        if coord_token_info is not None:
            coord_token_info['num_action_tokens'] = num_action_tokens

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            context_positions=context_positions,
            coord_token_info=coord_token_info,
            position_token_info=position_token_info,
            video_text_embeds=projected_video_text)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        final_x = None
        if classify_mode:
            assert register_tokens is not None
            assert gan_ca_blocks is not None
            assert cls_pred_branch is not None

            final_x = []
            registers = repeat(register_tokens(), "n d -> b n d", b=x.shape[0])
            # x = torch.cat([registers, x], dim=1)

        gan_idx = 0
        for ii, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

            if classify_mode and ii in [13, 21, 29]:
                gan_token = registers[:, gan_idx: gan_idx + 1]
                final_x.append(gan_ca_blocks[gan_idx](x, gan_token))
                gan_idx += 1

        if classify_mode:
            final_x = torch.cat(final_x, dim=1)
            if concat_time_embeddings:
                final_x = cls_pred_branch(torch.cat([final_x, 10 * e[:, None, :]], dim=1).view(final_x.shape[0], -1))
            else:
                final_x = cls_pred_branch(final_x.view(final_x.shape[0], -1))

        # Coordinate token mode: split coordinate tokens and decode
        coord_pred = None
        if self.coord_token_mode and num_coord_tokens > 0:
            if getattr(self, 'coord_only_no_video', False):
                # COORD ONLY MODE: x is all coord tokens, no video
                x_coord_tokens = x
                # Use decoder for all modes (direct embedding now has learned projection)
                if self.coord_decoder is not None:
                    coord_pred = self.coord_decoder(x_coord_tokens)
                else:
                    # Fallback: slice first 2 dims (shouldn't happen with new code)
                    coord_pred = x_coord_tokens[..., :2]
                # Return early - no video to decode
                return coord_pred  # Just return coord predictions
            else:
                # Normal mode: split video and coord tokens
                x_video_tokens = x[:, :-num_coord_tokens]
                x_coord_tokens = x[:, -num_coord_tokens:]
                # Decode coordinates: [B, T*S, dim] -> [B, T*S, 2]
                # Use decoder for all modes (direct embedding now has learned projection)
                if self.coord_decoder is not None:
                    coord_pred = self.coord_decoder(x_coord_tokens)
                else:
                    # Fallback: slice first 2 dims (shouldn't happen with new code)
                    coord_pred = x_coord_tokens[..., :2]
                # Continue with video tokens only
                x = x_video_tokens
        
        # Position token mode: split position tokens and decode
        position_pred = None
        if self.position_token_mode and num_position_tokens > 0:
            # Get actual video token count from position_token_info
            actual_video_tokens = position_token_info['video_seq_lens'][0].item()
            num_subjects_in_pos = position_token_info.get('num_subjects', 1)
            tokens_per_subject = position_token_info.get('tokens_per_subject', num_position_tokens)
            
            # Split: x = [video_tokens (actual_video_tokens), position_tokens (num_position_tokens), padding]
            x_video_tokens = x[:, :actual_video_tokens]
            x_position_tokens = x[:, actual_video_tokens:actual_video_tokens + num_position_tokens]
            
            # Use separate position_token_head when available, otherwise use shared head
            pos_head = self.position_token_head if self.position_token_head is not None else self.head
            
            if num_subjects_in_pos > 1:
                # Multi-subject: split position tokens by subject and decode each
                B_pos = x_position_tokens.size(0)
                position_preds_per_subj = []
                for s in range(num_subjects_in_pos):
                    start_idx = s * tokens_per_subject
                    end_idx = (s + 1) * tokens_per_subject
                    subj_tokens = x_position_tokens[:, start_idx:end_idx]
                    
                    # Apply position head (separate or shared) to decode position tokens
                    subj_output = pos_head(subj_tokens, e)
                    # Unpatchify using position grid sizes (same as video)
                    subj_pred = self.unpatchify(subj_output, position_grid_sizes, c=self.out_dim)
                    position_preds_per_subj.append(torch.stack(subj_pred))  # [B, C, T, H, W]
                
                # Stack subjects: [S, B, C, T, H, W] -> [B, S, C, T, H, W]
                position_pred_stacked = torch.stack(position_preds_per_subj, dim=0)  # [S, B, C, T, H, W]
                position_pred = position_pred_stacked.permute(1, 0, 2, 3, 4, 5)  # [B, S, C, T, H, W]
            else:
                # Single subject: apply position head (separate or shared) to decode position tokens
                position_output = pos_head(x_position_tokens, e)
                # Unpatchify using position grid sizes (same as video)
                position_pred = self.unpatchify(position_output, position_grid_sizes, c=self.out_dim)
            
            # Continue with video tokens only
            x = x_video_tokens
        
        # Pixel position mode: split position tokens and decode with position_pixel_head_per_subj
        if self.pixel_position_mode and num_position_tokens > 0:
            # Get actual video token count from position_token_info
            actual_video_tokens = position_token_info['video_seq_lens'][0].item()
            num_subjects_in_pos = position_token_info.get('num_subjects', self.pixel_position_num_subjects)
            tokens_per_subject = position_token_info.get('tokens_per_subject', num_position_tokens // num_subjects_in_pos)
            
            # Split: x = [video_tokens, position_tokens, padding]
            x_video_tokens = x[:, :actual_video_tokens]
            x_position_tokens = x[:, actual_video_tokens:actual_video_tokens + num_position_tokens]
            
            # Decode each subject's position tokens separately
            B_pos = x_position_tokens.size(0)
            position_preds_per_subj = []
            for s in range(num_subjects_in_pos):
                start_idx = s * tokens_per_subject
                end_idx = (s + 1) * tokens_per_subject
                subj_tokens = x_position_tokens[:, start_idx:end_idx]
                
                # Apply per-subject head: decode to 1 channel
                subj_output = self.position_pixel_head_per_subj(subj_tokens, e)
                # Unpatchify using position grid sizes, output has 1 channel
                subj_pred = self.unpatchify(subj_output, position_grid_sizes, c=1)
                # subj_pred is list of [1, T, H, W] per batch item
                position_preds_per_subj.append(torch.stack(subj_pred))  # [B, 1, T, H, W]
            
            # Stack subjects: [S, B, 1, T, H, W] -> [B, S, T, H, W]
            position_pred_stacked = torch.stack(position_preds_per_subj, dim=0)  # [S, B, 1, T, H, W]
            position_pred = position_pred_stacked.permute(1, 0, 2, 3, 4, 5).squeeze(2)  # [B, S, T, H, W]
            
            # Continue with video tokens only
            x = x_video_tokens

        # VideoJAM mode: apply Wout+ to get separate video and position representations
        if self.videojam_mode:
            # Project to 2*dim: [B, seq_len, dim] -> [B, seq_len, 2*dim]
            x_proj = self.wout_proj(x)
            # Split into video and position parts: each [B, seq_len, dim]
            x_video_part = x_proj[..., :self.dim]
            x_position_part = x_proj[..., self.dim:]
            
            if self.videojam_identity_init:
                # Identity init: Wout+ = [[I], [0]], so Wout+(x) = [x, 0] at init
                # No residuals needed - Wout+ directly outputs video and position representations
                x_video_repr = x_video_part
                x_position_repr = x_position_part
            else:
                # Zero init + residuals
                # Video: x + video_delta (when Wout+ = 0, this is just x)
                x_video_repr = x + x_video_part
                # Position: optionally add position_embeddings as residual
                if self.videojam_position_residual and position_embeddings is not None:
                    # SYMMETRIC residual: when Wout+ = 0, x_position = original position embeddings
                    x_position_repr = position_embeddings + x_position_part
                else:
                    # Legacy behavior: position_repr is just the delta (zero when Wout+=0)
                    x_position_repr = x_position_part
            
            # Apply video head
            x_video = self.head(x_video_repr, e)
            x_video = self.unpatchify(x_video, grid_sizes, c=self.out_dim)
            
            # Apply position head (same head structure, outputs same as video: 16 channels)
            x_position = self.position_head(x_position_repr, e)
            x_position = self.unpatchify(x_position, grid_sizes, c=self.out_dim)
            
            # Concatenate video and position along channel dimension: [32, F, H, W]
            x = [torch.cat([v, p], dim=0) for v, p in zip(x_video, x_position)]
        else:
            # head (video)
            x_video = self.head(x, e)

            # unpatchify video
            x_video = self.unpatchify(x_video, grid_sizes, c=self.out_dim)

            # Joint position diffusion: also apply position head and concatenate
            if self.position_head is not None and self.position_channels > 0:
                x_position = self.position_head(x, e)
                x_position = self.unpatchify(x_position, grid_sizes, c=self.position_channels)
                # Concatenate video and position along channel dimension
                x = [torch.cat([v, p], dim=0) for v, p in zip(x_video, x_position)]
            else:
                x = x_video

        if classify_mode:
            return torch.stack(x), final_x
        
        # Position token mode: return both video and position predictions
        if self.position_token_mode and position_pred is not None:
            # position_pred can be:
            # - [B, S, C, T, H, W] for multi-subject (already stacked tensor)
            # - list of [C, T, H, W] for single subject (needs stacking)
            if isinstance(position_pred, torch.Tensor):
                # Multi-subject: already [B, S, C, T, H, W]
                return torch.stack(x), position_pred
            else:
                # Single subject: list of [C, T, H, W] -> [B, C, T, H, W]
                return torch.stack(x), torch.stack(position_pred)
        
        # Pixel position mode: return both video and pixel-space position predictions
        if self.pixel_position_mode and position_pred is not None:
            # position_pred: [B, S, T, H, W] pixel-space position maps
            return torch.stack(x), position_pred
        
        # Coordinate token mode: return both video and coordinate predictions
        if self.coord_token_mode and coord_pred is not None:
            return torch.stack(x), coord_pred

        return torch.stack(x)

    def _forward_classify(
        self,
        x,
        t,
        context,
        seq_len,
        register_tokens,
        cls_pred_branch,
        clip_fea=None,
        y=None,
        first_frame_latent=None,
    ):
        r"""
        Feature extraction through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of video features with original input shapes [C_block, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Process first_frame_latent for i2v cross-attention (if provided)
        if first_frame_latent is not None:
            # Apply latent_patch_embedding: [B, C, 1, H, W] -> [B, model_dim, 1, H', W']
            latent_tokens = self.latent_patch_embedding(first_frame_latent)
            # Flatten to tokens: [B, model_dim, 1, H', W'] -> [B, num_patches, model_dim]
            latent_tokens = latent_tokens.squeeze(2).flatten(2).transpose(1, 2)
            # Prepend to context
            context = torch.cat([latent_tokens, context], dim=1)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # TODO: Tune the number of blocks for feature extraction
        for block in self.blocks[:16]:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # unpatchify
        x = self.unpatchify(x, grid_sizes, c=self.dim // 4)
        return torch.stack(x)

    def unpatchify(self, x, grid_sizes, c=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim if c is None else c
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        
        # Initialize position-related components
        # This ensures pretrained model behavior is preserved initially
        if self.videojam_mode:
            if self.videojam_identity_init:
                # Identity initialization: Win+ = [I, 0], Wout+ = [[I], [0]]
                # This preserves pretrained behavior WITHOUT residual connections
                if self.win_proj is not None:
                    # Win+: [2*dim, dim] -> we want Win+(concat(v, p)) = v
                    # Weight shape: [dim, 2*dim], first dim columns are identity, rest are zero
                    nn.init.zeros_(self.win_proj.weight)
                    nn.init.zeros_(self.win_proj.bias)
                    # Set first dim columns to identity
                    with torch.no_grad():
                        self.win_proj.weight[:, :self.dim].copy_(torch.eye(self.dim))
                    print("✓ Win+ projection identity-initialized [I, 0]")
                if self.wout_proj is not None:
                    # Wout+: [dim, 2*dim] -> we want Wout+(x) = [x, 0]
                    # Weight shape: [2*dim, dim], first dim rows are identity, rest are zero
                    nn.init.zeros_(self.wout_proj.weight)
                    nn.init.zeros_(self.wout_proj.bias)
                    # Set first dim rows to identity
                    with torch.no_grad():
                        self.wout_proj.weight[:self.dim, :].copy_(torch.eye(self.dim))
                    print("✓ Wout+ projection identity-initialized [[I], [0]]")
            else:
                # Zero initialization: requires residual connections
                if self.win_proj is not None:
                    nn.init.zeros_(self.win_proj.weight)
                    nn.init.zeros_(self.win_proj.bias)
                    print("✓ Win+ projection zero-initialized")
                if self.wout_proj is not None:
                    nn.init.zeros_(self.wout_proj.weight)
                    nn.init.zeros_(self.wout_proj.bias)
                    print("✓ Wout+ projection zero-initialized")
            
            # Position head: always zero-init (it's a new head, not pretrained)
            if self.position_head is not None:
                nn.init.zeros_(self.position_head.head.weight)
                if self.position_head.head.bias is not None:
                    nn.init.zeros_(self.position_head.head.bias)
                print("✓ Position head zero-initialized")
        else:
            # Legacy mode: zero-init position embedding and head
            if self.position_embedding is not None:
                nn.init.zeros_(self.position_embedding.weight)
                if self.position_embedding.bias is not None:
                    nn.init.zeros_(self.position_embedding.bias)
                print("✓ Position embedding zero-initialized")
            if self.position_head is not None:
                nn.init.zeros_(self.position_head.head.weight)
                if self.position_head.head.bias is not None:
                    nn.init.zeros_(self.position_head.head.bias)
                print("✓ Position head zero-initialized")
        
        # Coordinate token mode: 
        # - Encoder: Fourier (fixed) or Linear (learned with large init)
        # - Decoder: always learned with small init for stable flow predictions
        if self.coord_token_mode:
            if self.coord_encoder is not None and not self.coord_fourier_encoding:
                # Only init if using learned linear encoder
                nn.init.normal_(self.coord_encoder.weight, std=0.1)  # Larger for distinguishable encodings
                print("✓ Coord encoder normal-initialized (std=0.1, no bias)")
            if self.coord_decoder is not None:
                if self.coord_direct_embedding:
                    # Direct embedding with decoder - initialize as identity (not used anymore)
                    nn.init.zeros_(self.coord_decoder.weight)
                    nn.init.zeros_(self.coord_decoder.bias)
                    self.coord_decoder.weight.data[0, 0] = 1.0
                    self.coord_decoder.weight.data[1, 1] = 1.0
                    print("✓ Coord decoder identity-initialized for direct embedding")
                else:
                    nn.init.normal_(self.coord_decoder.weight, std=0.02)
                    print("✓ Coord decoder normal-initialized (std=0.02, no bias)")
            elif self.coord_direct_embedding:
                print("✓ Direct embedding: using first 2 dims slice (no decoder)")
            elif self.coord_no_encoder_decoder:
                print("✓ Pure slicing mode: no encoder/decoder (baseline)")

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def init_position_modulation_from_video(self):
        r"""
        Initialize position modulation parameters from video modulation parameters.
        
        This method should be called after loading a pretrained checkpoint when
        separate_position_modulation=True. It copies the video modulation weights
        (self.modulation in each block) to position modulation (self.modulation_pos),
        and copies the video head weights to position_token_head.
        
        This ensures that position tokens start with the same modulation behavior
        as video tokens, allowing fine-tuning to differentiate them.
        """
        if not self.separate_position_modulation:
            print("Warning: init_position_modulation_from_video called but separate_position_modulation=False")
            return
        
        copied_blocks = 0
        for block in self.blocks:
            if hasattr(block, 'modulation_pos') and block.modulation_pos is not None:
                with torch.no_grad():
                    # Copy block modulation weights
                    block.modulation_pos.data = block.modulation.data.clone()
                copied_blocks += 1
        
        if copied_blocks > 0:
            print(f"✓ Copied video modulation to position modulation in {copied_blocks} blocks")
        
        # Copy position_token_head weights from video head
        # Use proper cloning to avoid gradient graph issues
        if self.position_token_head is not None and self.head is not None:
            with torch.no_grad():
                # Clone and copy each parameter separately
                # Head linear layer
                self.position_token_head.head.weight.copy_(self.head.head.weight.detach().clone())
                if self.head.head.bias is not None and self.position_token_head.head.bias is not None:
                    self.position_token_head.head.bias.copy_(self.head.head.bias.detach().clone())
                
                # Head modulation
                self.position_token_head.modulation.copy_(self.head.modulation.detach().clone())
                
                # LayerNorm (WanLayerNorm may not have learnable weight/bias if elementwise_affine=False)
                if hasattr(self.head.norm, 'weight') and self.head.norm.weight is not None:
                    if hasattr(self.position_token_head.norm, 'weight') and self.position_token_head.norm.weight is not None:
                        self.position_token_head.norm.weight.copy_(self.head.norm.weight.detach().clone())
                if hasattr(self.head.norm, 'bias') and self.head.norm.bias is not None:
                    if hasattr(self.position_token_head.norm, 'bias') and self.position_token_head.norm.bias is not None:
                        self.position_token_head.norm.bias.copy_(self.head.norm.bias.detach().clone())
            
            print("✓ Copied video head to position_token_head (linear, modulation)")
