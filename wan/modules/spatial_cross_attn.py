"""
Spatial Cross-Attention with 3D RoPE.

Applies the SAME 3D RoPE (frame, height, width) to both:
- Video patch queries (based on patch position)
- Conditioning keys (based on subject position)

This gives EXPLICIT spatial correspondence - patches automatically attend
more strongly to conditioning at matching spatial positions.
"""

import torch
import torch.nn as nn
from .model import WanSelfAttention, rope_apply
from .attention import flash_attention

# Global flag to print cross-attn RoPE verification only once
_CROSS_ATTN_ROPE_VERIFIED = False

# FlexAttention for fast masked cross-attention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None

# Module-level cache for compiled flex_attention and cross-attention block masks
_compiled_flex_attention = None
_cross_attn_block_mask_cache = {}


def get_compiled_flex_attention():
    """Get or create the compiled flex_attention function (cached globally)."""
    global _compiled_flex_attention
    if _compiled_flex_attention is None and FLEX_ATTENTION_AVAILABLE:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def get_cross_attn_block_mask(video_len: int, pos_per_subject: int, num_subjects: int,
                               null_len: int, actions_per_subject: int, 
                               total_query_len: int, num_keys: int, device: torch.device):
    """
    Get or create a block mask for masked position cross-attention.
    
    Pattern:
    - Video tokens (0:video_len) → attend ONLY to null tokens (0:null_len)
    - S_i position tokens → attend ONLY to S_i actions (null_len + i*aps : null_len + (i+1)*aps)
    """
    global _cross_attn_block_mask_cache
    cache_key = (video_len, pos_per_subject, num_subjects, null_len, 
                 actions_per_subject, total_query_len, num_keys, str(device))
    
    if cache_key not in _cross_attn_block_mask_cache:
        pos_start = video_len
        
        def mask_mod(b, h, q_idx, kv_idx):
            # Is this a video token query?
            is_video_query = q_idx < video_len
            
            # Is this a null key (for video tokens)?
            is_null_key = kv_idx < null_len
            
            # Video tokens attend ONLY to null keys
            video_attends = is_video_query & is_null_key
            
            # For position tokens, compute subject ownership
            q_subject = (q_idx - pos_start) // pos_per_subject
            
            # Compute which subject this key belongs to (in the action part)
            # Actions start at null_len, and are grouped by subject
            action_key_idx = kv_idx - null_len  # offset into action tokens
            kv_subject = action_key_idx // actions_per_subject
            
            # Is this an action key (not null)?
            is_action_key = kv_idx >= null_len
            
            # Position token is valid (within the position token range)
            is_pos_query = (q_idx >= pos_start) & (q_idx < pos_start + num_subjects * pos_per_subject)
            
            # Position tokens attend ONLY to their own subject's actions
            pos_attends = is_pos_query & is_action_key & (q_subject == kv_subject)
            
            return video_attends | pos_attends
        
        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=1,
            Q_LEN=total_query_len,
            KV_LEN=num_keys,
            device=device
        )
        _cross_attn_block_mask_cache[cache_key] = block_mask
    
    return _cross_attn_block_mask_cache[cache_key]


_coord_cross_attn_block_mask_cache = {}


def get_coord_cross_attn_block_mask(video_len: int, coord_len: int, num_subjects: int,
                                     null_len: int, actions_per_subject: int,
                                     total_query_len: int, num_keys: int,
                                     per_frame: bool, device: torch.device):
    """
    Block mask for coord-token cross-attention (frame-major layout).
    
    Coord token at query index (video_len + i) belongs to subject (i % S), frame (i // S).
    
    Patterns:
    - Video tokens (0:video_len) → null keys (0:null_len) only
    - per_frame=False: coord(t, s) → all actions for subject s
    - per_frame=True:  coord(t=0, s) → null keys; coord(t>0, s) → action[t-1] for subject s
    """
    global _coord_cross_attn_block_mask_cache
    cache_key = (video_len, coord_len, num_subjects, null_len,
                 actions_per_subject, total_query_len, num_keys, per_frame, str(device))
    
    if cache_key not in _coord_cross_attn_block_mask_cache:
        coord_start = video_len
        
        if per_frame:
            def mask_mod(b, h, q_idx, kv_idx):
                is_video_q = q_idx < video_len
                is_null_k = kv_idx < null_len
                video_ok = is_video_q & is_null_k
                
                is_coord_q = q_idx >= coord_start
                coord_i = q_idx - coord_start
                q_subj = coord_i % num_subjects
                q_frame = coord_i // num_subjects
                
                is_frame_0 = q_frame == 0
                frame0_ok = is_coord_q & is_frame_0 & is_null_k
                
                action_offset = kv_idx - null_len
                kv_subj = action_offset // actions_per_subject
                action_within = action_offset % actions_per_subject
                is_action_k = kv_idx >= null_len
                
                frameN_ok = (is_coord_q & (~is_frame_0) & is_action_k
                             & (q_subj == kv_subj)
                             & (action_within == (q_frame - 1)))
                
                return video_ok | frame0_ok | frameN_ok
        else:
            def mask_mod(b, h, q_idx, kv_idx):
                is_video_q = q_idx < video_len
                is_null_k = kv_idx < null_len
                video_ok = is_video_q & is_null_k
                
                is_coord_q = q_idx >= coord_start
                coord_i = q_idx - coord_start
                q_subj = coord_i % num_subjects
                
                action_offset = kv_idx - null_len
                kv_subj = action_offset // actions_per_subject
                is_action_k = kv_idx >= null_len
                
                coord_ok = is_coord_q & is_action_k & (q_subj == kv_subj)
                
                return video_ok | coord_ok
        
        block_mask = create_block_mask(
            mask_mod, B=1, H=1,
            Q_LEN=total_query_len, KV_LEN=num_keys,
            device=device
        )
        _coord_cross_attn_block_mask_cache[cache_key] = block_mask
    
    return _coord_cross_attn_block_mask_cache[cache_key]


def rope_apply_with_positions(x, positions, grid_sizes, freqs):
    """
    Apply 3D RoPE to tokens based on explicit (f, h, w) positions.
    
    This matches the frequency structure of rope_apply:
    - freqs is [max_seq, c] where c = head_dim // 2 (complex numbers)
    - freqs is split into [frame_dim, h_dim, w_dim] parts
    - Each position (f, h, w) gets freqs_frame[f] ⊕ freqs_h[h] ⊕ freqs_w[w]
    
    Args:
        x: [B, seq_len, num_heads, head_dim] - input tokens
        positions: [B, seq_len, 3] - (frame, h_patch, w_patch) for each token
        grid_sizes: [B, 3] - (F, H, W) grid dimensions (unused, kept for API compat)
        freqs: RoPE frequencies [max_seq, c] where c = head_dim // 2 (COMPLEX)
    
    Returns:
        x with 3D RoPE applied based on positions
    """
    b, seq_len, n, d = x.shape
    c = d // 2  # freqs dimension (complex)
    device = x.device
    
    # Ensure freqs is on the right device
    if freqs.device != device:
        freqs = freqs.to(device)
    
    # Ensure positions is on the right device
    if positions.device != device:
        positions = positions.to(device)
    
    # Split frequencies into frame, height, width parts (same structure as rope_apply)
    # frame gets c - 2*(c//3), h gets c//3, w gets c//3
    freq_dims = [c - 2 * (c // 3), c // 3, c // 3]
    freqs_split = freqs.split(freq_dims, dim=1)
    freqs_f = freqs_split[0]  # [max_seq, freq_dim_f]
    freqs_h = freqs_split[1]  # [max_seq, freq_dim_h]
    freqs_w = freqs_split[2]  # [max_seq, freq_dim_w]
    
    output = []
    for i in range(b):
        x_i = x[i]  # [seq_len, n, d]
        pos_i = positions[i]  # [seq_len, 3] - (f, h, w) per token
        
        # Get frame, height, width indices (clamp to valid range)
        f_idx = pos_i[:, 0].long().clamp(0, freqs_f.size(0) - 1)  # [seq_len]
        h_idx = pos_i[:, 1].long().clamp(0, freqs_h.size(0) - 1)  # [seq_len]
        w_idx = pos_i[:, 2].long().clamp(0, freqs_w.size(0) - 1)  # [seq_len]
        
        # Gather frequencies for each position
        freq_f_i = freqs_f[f_idx]  # [seq_len, freq_dim_f]
        freq_h_i = freqs_h[h_idx]  # [seq_len, freq_dim_h]
        freq_w_i = freqs_w[w_idx]  # [seq_len, freq_dim_w]
        
        # Concatenate to full frequency: [seq_len, c]
        freqs_i = torch.cat([freq_f_i, freq_h_i, freq_w_i], dim=-1)
        
        # Reshape freqs for broadcasting with heads: [seq_len, 1, c]
        freqs_i = freqs_i.unsqueeze(1)
        
        # Convert x to complex: [seq_len, n, d] -> [seq_len, n, c] (complex)
        x_complex = torch.view_as_complex(
            x_i.to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        
        # Apply RoPE rotation: multiply by complex frequencies
        x_rotated = x_complex * freqs_i
        
        # Convert back to real: [seq_len, n, c] (complex) -> [seq_len, n, d]
        x_rotated = torch.view_as_real(x_rotated).flatten(2)
        
        output.append(x_rotated)
    
    return torch.stack(output).type_as(x)


class WanSpatialCrossAttention(WanSelfAttention):
    """
    Cross-attention with FULL 3D RoPE for both queries and keys.
    
    Handles S subjects, each with T=N-1 actions for N frames.
    Context shape: [B, S*T, C]
    Positions shape: [B, S*T, 3] - (frame, h_patch, w_patch) per conditioning token
    
    Key difference from WanMultiSubjectCrossAttention:
    - Queries get 3D RoPE based on video patch position (f, h, w)
    - Keys get 3D RoPE based on subject position at that frame
    - This creates explicit spatial correspondence!
    """
    
    def __init__(self, *args, num_subjects: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_subjects = num_subjects

    def forward(self, x, context, context_lens, grid_sizes=None, freqs=None, 
                context_positions=None, num_subjects=None, coord_token_info=None):
        """
        Args:
            x(Tensor): Shape [B, L1, C] - video tokens (F*H*W patches)
            context(Tensor): Shape [B, S*T, C] - S*T action-subject embeddings
            context_lens(Tensor): Shape [B] - unused
            grid_sizes(Tensor): Shape [B, 3] - (F, H, W) grid dimensions
            freqs(Tensor): RoPE frequency parameters
            context_positions(Tensor): Shape [B, S*T, 3] - (frame, h, w) for each conditioning token
            num_subjects(int): Number of subjects
            coord_token_info: Optional dict for coordinate token RoPE
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        S = num_subjects if num_subjects is not None else self.num_subjects
        
        # Compute query from video tokens
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        # Compute key, value from conditioning
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        
        # Apply 3D RoPE if available
        if grid_sizes is not None and freqs is not None:
            # Apply 3D RoPE to queries (video patches + coord tokens)
            q = rope_apply(q, grid_sizes, freqs, coord_token_info=coord_token_info)
            
            # Apply 3D RoPE to keys based on context_positions
            if context_positions is not None:
                k = rope_apply_with_positions(k, context_positions, grid_sizes, freqs)
            else:
                # Fallback: just use temporal RoPE if no positions provided
                # This shouldn't happen in normal use
                pass
        
        # Compute attention
        x = flash_attention(q, k, v, k_lens=None)
        
        # Output projection
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCoordActionCrossAttention(WanSelfAttention):
    """
    Cross-attention for coord tokens ONLY -> action embeddings.
    
    Used in dual_stream_cross_attn mode where video and coord tokens
    have separate cross-attention modules with independent weights.
    
    Features:
    - Input: only coord tokens as queries (NOT video tokens)
    - Keys/values: action embeddings only (no null embedding)
    - Subject-isolation mask: coord tokens for subject s attend only to subject s's actions
    - 3D RoPE on coord queries (frame, prev_h, prev_w) and action keys (frame, h, w)
    
    Coord token layout is FRAME-MAJOR: [f0_s0, f0_s1, f1_s0, f1_s1, ...]
    Action layout is SUBJECT-MAJOR: [s0_a0, s0_a1, ..., s1_a0, s1_a1, ...]
    """
    
    def __init__(self, *args, cross_attn_rope: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cross_attn_rope = cross_attn_rope
    
    def forward(self, x_coord, context, context_lens=None, grid_sizes=None, freqs=None,
                context_positions=None, coord_token_info=None, num_subjects=None):
        """
        Args:
            x_coord(Tensor): Shape [B, coord_len, C] - coord tokens ONLY
            context(Tensor): Shape [B, S*T, C] - action embeddings (S subjects, T actions each)
                             May be padded to text_len; actual count from coord_token_info.
            context_lens(Tensor): unused
            grid_sizes(Tensor): Shape [B, 3] - (F, H, W) grid dimensions
            freqs(Tensor): RoPE frequency parameters
            context_positions(Tensor): Shape [B, S*T, 3] - (frame, h, w) for each action
            coord_token_info: Dict with num_tokens, num_frames, num_subjects, spatial_h, spatial_w
            num_subjects(int): Number of subjects
        
        Returns:
            Tensor: Shape [B, coord_len, C] - cross-attended coord tokens
        """
        b, n, d = x_coord.size(0), self.num_heads, self.head_dim
        coord_len = x_coord.size(1)
        
        if coord_token_info is None:
            # No coord info - just return zeros (shouldn't happen in practice)
            return torch.zeros_like(x_coord)
        
        S = num_subjects if num_subjects is not None else coord_token_info.get('num_subjects', 2)
        T = coord_token_info.get('num_frames', coord_len // S)
        
        # Determine actual action count (context may be padded to text_len=512)
        # CRITICAL: context_positions is also padded to text_len! Must use coord_token_info
        if 'num_action_tokens' in coord_token_info:
            num_action_tokens = coord_token_info['num_action_tokens']
        else:
            # Fallback: S subjects * (T-1) actions for T frames
            num_action_tokens = min(context.size(1), S * max(T - 1, 1))
        actions_per_subject = num_action_tokens // S if S > 0 else num_action_tokens
        
        # Slice context to actual action tokens (drop padding)
        action_context = context[:, :num_action_tokens]  # [B, S*T_actions, C]
        
        # Q, K, V projections
        q = self.norm_q(self.q(x_coord)).view(b, coord_len, n, d)
        k = self.norm_k(self.k(action_context)).view(b, num_action_tokens, n, d)
        v = self.v(action_context).view(b, num_action_tokens, n, d)
        
        # Apply 3D RoPE if enabled
        if self.cross_attn_rope and freqs is not None and context_positions is not None:
            if 'spatial_h' in coord_token_info and 'spatial_w' in coord_token_info:
                spatial_h = coord_token_info['spatial_h']  # [B, T*S]
                spatial_w = coord_token_info['spatial_w']  # [B, T*S]
                coord_S = coord_token_info['num_subjects']
                
                # Build frame indices for coord queries: FRAME-MAJOR [t0_s0, t0_s1, t1_s0, ...]
                frame_indices = torch.arange(T, device=x_coord.device).view(T, 1).expand(T, coord_S).reshape(-1)
                frame_indices = frame_indices.unsqueeze(0).expand(b, -1).to(dtype=q.dtype)  # [B, T*S]
                
                # Build coord query positions: [B, coord_len, 3]
                coord_query_positions = torch.stack([
                    frame_indices, 
                    spatial_h.to(dtype=q.dtype), 
                    spatial_w.to(dtype=q.dtype)
                ], dim=-1)
                
                # Apply RoPE to coord queries
                q = rope_apply_with_positions(q, coord_query_positions, grid_sizes, freqs)
                
                # Apply RoPE to action keys
                actual_actions = min(num_action_tokens, context_positions.size(1))
                k = rope_apply_with_positions(k, context_positions[:, :actual_actions], grid_sizes, freqs)
        
        # Transpose for SDPA: [B, coord_len, n, d] -> [B, n, coord_len, d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Build subject-isolation mask
        # Coord token layout: FRAME-MAJOR [f0_s0, f0_s1, f1_s0, f1_s1, ...]
        # Token at index i belongs to subject (i % S)
        # Action layout: SUBJECT-MAJOR [s0_a0, s0_a1, ..., s1_a0, s1_a1, ...]
        mask_val = -1e4 if q.dtype == torch.float16 else -1e9
        mask = torch.full((coord_len, num_action_tokens), mask_val, device=x_coord.device, dtype=q.dtype)
        
        T_coord = coord_len // S if S > 0 else coord_len
        for s in range(S):
            # Actions for subject s (subject-major layout)
            action_start = s * actions_per_subject
            action_end = (s + 1) * actions_per_subject
            
            # Coord tokens for subject s are at indices: t*S + s for t in range(T_coord)
            for t in range(T_coord):
                token_idx = t * S + s
                mask[token_idx, action_start:action_end] = 0.0
        
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, coord_len, num_actions]
        
        # Apply SDPA with mask
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0
        )
        
        # Output: [B, n, coord_len, d] -> [B, coord_len, C]
        out = out.transpose(1, 2).flatten(2)
        return self.o(out)


class WanMaskedPositionCrossAttention(WanSelfAttention):
    """
    Cross-attention with SUBJECT-LEVEL masking for position token mode.
    
    Masking (per subject, not per frame):
    - Video tokens → attend ONLY to empty prompt (T5-encoded "" like CFG)
    - Position tokens (subject S) → attend to ALL actions for subject S
    
    This separates subjects but allows full temporal attention within a subject.
    Currently supports 1 subject (all position tokens attend to all actions).
    
    The empty prompt embedding is pre-computed from T5 encoder at init (like CFG null).
    
    If cross_attn_rope=True, applies 3D RoPE to position tokens and action embeddings
    for explicit spatial correspondence.
    """
    
    def __init__(self, *args, num_subjects: int = 1, cross_attn_rope: bool = False, 
                 per_frame_cross_attn: bool = False, subject_isolation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_subjects = num_subjects
        self.cross_attn_rope = cross_attn_rope
        self.per_frame_cross_attn = per_frame_cross_attn
        self.subject_isolation = subject_isolation
        
        # Pre-compute T5-encoded empty string "" for null embedding (like CFG)
        # Will be set by set_null_embedding() after model creation
        # Use persistent=False so it's NOT saved in state_dict (re-computed on load)
        self.register_buffer('null_embedding', None, persistent=False)
    
    def set_null_embedding(self, null_embed: torch.Tensor):
        """
        Set the pre-computed null embedding (T5-encoded "").
        Call this once after model creation with the text encoder's output for "".
        
        Args:
            null_embed: [1, seq_len, dim] - T5 encoder output for "" (projected)
        """
        self.null_embedding = null_embed.detach()
        self._cross_attn_mask_cache = {}
    
    def _build_cross_attn_mask(self, query_len, key_len, video_len, null_len,
                                pos_len, coord_len, S, pos_per_subject,
                                actions_per_subject, position_token_info,
                                coord_token_info, device, dtype):
        """Build and cache the cross-attention mask (vectorized, no Python loops)."""
        cache_key = (query_len, key_len, video_len, null_len, pos_len, coord_len,
                     S, pos_per_subject, actions_per_subject,
                     self.per_frame_cross_attn, self.subject_isolation, str(device))
        
        if not hasattr(self, '_cross_attn_mask_cache'):
            self._cross_attn_mask_cache = {}
        
        if cache_key in self._cross_attn_mask_cache:
            cached = self._cross_attn_mask_cache[cache_key]
            if cached.dtype == dtype:
                return cached
            return cached.to(dtype)
        
        mask_val = -1e4 if dtype == torch.float16 else -1e9
        mask = torch.full((query_len, key_len), mask_val, device=device, dtype=dtype)
        
        # Video tokens → null tokens
        mask[:video_len, :null_len] = 0.0
        
        # Position tokens (subject-major layout)
        if pos_len > 0:
            if self.per_frame_cross_attn and position_token_info is not None:
                pos_grid = position_token_info.get('grid_sizes', None)
                if pos_grid is not None:
                    pf = pos_grid[0, 0].item() if pos_grid.dim() > 1 else pos_grid[0].item()
                    patches_per_frame = pos_per_subject // pf
                else:
                    pf = actions_per_subject
                    patches_per_frame = pos_per_subject // pf
                
                for s in range(S):
                    pos_subj_start = video_len + s * pos_per_subject
                    action_subj_start = null_len + s * actions_per_subject
                    frame_0_start = pos_subj_start
                    frame_0_end = pos_subj_start + patches_per_frame
                    mask[frame_0_start:frame_0_end, :null_len] = 0.0
                    for t in range(1, pf):
                        if (t - 1) < actions_per_subject:
                            f_start = pos_subj_start + t * patches_per_frame
                            f_end = pos_subj_start + (t + 1) * patches_per_frame
                            mask[f_start:f_end, action_subj_start + (t - 1)] = 0.0
            else:
                for s in range(S):
                    ps = video_len + s * pos_per_subject
                    pe = video_len + pos_len if s == S - 1 else video_len + (s + 1) * pos_per_subject
                    a_s = null_len + s * actions_per_subject
                    a_e = null_len + (s + 1) * actions_per_subject
                    mask[ps:pe, a_s:a_e] = 0.0
        
        # Coord tokens (frame-major layout) - VECTORIZED
        if coord_len > 0 and S > 0:
            T_coord = coord_len // S
            token_indices = torch.arange(coord_len, device=device)
            subj_ids = token_indices % S
            frame_ids = token_indices // S
            
            if self.subject_isolation and self.per_frame_cross_attn:
                is_frame_0 = frame_ids == 0
                frame_0_tokens = token_indices[is_frame_0]
                mask[video_len + frame_0_tokens.unsqueeze(1),
                     torch.arange(null_len, device=device).unsqueeze(0)] = 0.0
                
                for s in range(S):
                    action_start = null_len + s * actions_per_subject
                    s_mask = subj_ids == s
                    for t in range(1, T_coord):
                        if (t - 1) < actions_per_subject:
                            t_mask = frame_ids == t
                            tok = token_indices[s_mask & t_mask]
                            if tok.numel() > 0:
                                mask[video_len + tok, action_start + (t - 1)] = 0.0
            elif self.subject_isolation:
                for s in range(S):
                    a_s = null_len + s * actions_per_subject
                    a_e = null_len + (s + 1) * actions_per_subject
                    s_tokens = video_len + token_indices[subj_ids == s]
                    mask[s_tokens.unsqueeze(1),
                         torch.arange(a_s, a_e, device=device).unsqueeze(0)] = 0.0
            else:
                all_a_s = null_len
                all_a_e = null_len + S * actions_per_subject
                mask[video_len:video_len + coord_len, all_a_s:all_a_e] = 0.0
        
        self._cross_attn_mask_cache[cache_key] = mask
        return mask
    
    def forward(self, x, context, context_lens, grid_sizes=None, freqs=None, 
                context_positions=None, num_subjects=None, coord_token_info=None,
                position_token_info=None, empty_prompt_embed=None):
        """
        Args:
            x(Tensor): Shape [B, L, C] - video tokens + position/coord tokens
            context(Tensor): Shape [B, S*T, C] - action embeddings (S subjects, T actions each)
            context_positions(Tensor): Shape [B, S*T, 3] - (frame, h, w) for each action
            position_token_info: Dict with video_seq_lens, grid_sizes (for position token mode)
            coord_token_info: Dict with num_tokens, spatial_h, spatial_w (for coord token mode)
        """
        global _CROSS_ATTN_ROPE_VERIFIED  # Declare global at start of function
        
        b, n, d = x.size(0), self.num_heads, self.head_dim
        total_query_len = x.size(1)
        C = x.size(-1)
        S = num_subjects if num_subjects is not None else self.num_subjects
        
        # Determine video/position/coord split
        # Priority: position_token_info > coord_token_info > all video
        use_coord_mode = (coord_token_info is not None and position_token_info is None)
        
        if position_token_info is not None and 'video_seq_lens' in position_token_info:
            video_len = position_token_info['video_seq_lens'][0].item()
            pos_len = position_token_info.get('num_tokens', 0)
            coord_len = 0
        elif use_coord_mode:
            # Coord token mode: last num_tokens are coord tokens
            coord_len = coord_token_info['num_tokens']
            video_len = total_query_len - coord_len
            pos_len = 0  # No position tokens in coord mode
        else:
            video_len = total_query_len
            pos_len = 0
            coord_len = 0
        
        # CRITICAL: Use ACTUAL action token count, not padded context.size(1)
        # context may be padded to text_len (512), but actual actions are much fewer (S*T)
        # Check position_token_info, then coord_token_info, then fall back to context.size(1)
        if position_token_info is not None and 'num_action_tokens' in position_token_info:
            num_action_tokens = position_token_info['num_action_tokens']
        elif coord_token_info is not None and 'num_action_tokens' in coord_token_info:
            num_action_tokens = coord_token_info['num_action_tokens']
        else:
            num_action_tokens = context.size(1)  # Fallback (may be padded - legacy behavior)
        actions_per_subject = num_action_tokens // S if S > 0 else num_action_tokens
        
        # Empty prompt = T5-encoded "" (like CFG null)
        if empty_prompt_embed is None:
            if self.null_embedding is not None:
                # Use pre-computed T5-encoded ""
                empty_prompt_embed = self.null_embedding.to(device=x.device, dtype=x.dtype)
                # Expand to batch size if needed
                if empty_prompt_embed.size(0) == 1 and b > 1:
                    empty_prompt_embed = empty_prompt_embed.expand(b, -1, -1)
                
                # NOTE: No scaling here - the model was trained with the T5-projected null
                # embedding as-is. The null std (~0.001) is much smaller than action std
                # (~0.02), but the model learned to work with this. Adding scaling here
                # would introduce a training-inference mismatch.
            else:
                # Fallback to zeros if null_embedding not set - THIS SHOULD NOT HAPPEN
                # Print warning only once
                if not hasattr(self, '_warned_null_fallback'):
                    print("WARNING: null_embedding not set in WanMaskedPositionCrossAttention! "
                          "Using zeros fallback. This may degrade performance.")
                    self._warned_null_fallback = True
                # Use 512 tokens to match expected T5 output length
                empty_prompt_embed = torch.zeros(b, 512, C, device=x.device, dtype=x.dtype)
        
        # Keys: [null_tokens (512), action_0_s0, action_1_s0, ..., action_0_s1, ...]
        null_len = empty_prompt_embed.size(1)  # 512 tokens from T5
        context_with_null = torch.cat([empty_prompt_embed, context], dim=1)  # [B, 512+S*T, C]
        num_keys = context_with_null.size(1)
        
        # Q, K, V projections
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context_with_null)).view(b, -1, n, d)
        v = self.v(context_with_null).view(b, -1, n, d)
        
        # Apply 3D RoPE if enabled, context_positions provided, AND we have position tokens
        # This gives spatial correspondence: position patches attend to actions at matching (f, h, w)
        if self.cross_attn_rope and context_positions is not None and freqs is not None and pos_len > 0:
            # One-time verification that cross-attention RoPE is actually being applied
            if not _CROSS_ATTN_ROPE_VERIFIED:
                _CROSS_ATTN_ROPE_VERIFIED = True
                print(f"[CROSS-ATTN ROPE ACTIVE] Applying 3D RoPE in cross-attention: "
                      f"pos_len={pos_len}, num_actions={context_positions.shape[1]}, "
                      f"context_positions sample: {context_positions[0, :3].tolist()}")
            # Position token queries: apply 3D RoPE based on their (f, h, w) grid position
            # Position tokens are laid out as [S0_tokens, S1_tokens, ...] where each subject
            # has F*H*W tokens in (f, h, w) order
            pos_grid = position_token_info['grid_sizes'][0] if position_token_info else grid_sizes[0]
            pf, ph, pw = pos_grid.tolist() if hasattr(pos_grid, 'tolist') else pos_grid
            pos_per_subject = pf * ph * pw
            
            # Build position indices for position token queries (VECTORIZED)
            # Each subject's tokens have the same (f, h, w) pattern laid out as [f*h*w] per subject
            # Create base indices for one subject: [pos_per_subject, 3]
            f_idx = torch.arange(pf, device=x.device).view(pf, 1, 1).expand(pf, ph, pw).reshape(-1)
            h_idx = torch.arange(ph, device=x.device).view(1, ph, 1).expand(pf, ph, pw).reshape(-1)
            w_idx = torch.arange(pw, device=x.device).view(1, 1, pw).expand(pf, ph, pw).reshape(-1)
            base_positions = torch.stack([f_idx, h_idx, w_idx], dim=-1).to(dtype=q.dtype)  # [pos_per_subject, 3]
            
            # Tile for all subjects: [S * pos_per_subject, 3]
            pos_query_positions = base_positions.repeat(S, 1)  # [pos_len, 3]
            pos_query_positions = pos_query_positions.unsqueeze(0).expand(b, -1, -1)  # [B, pos_len, 3]
            
            # Apply RoPE to position token queries only (video tokens don't need RoPE for null attention)
            q_pos = q[:, video_len:video_len + pos_len]  # [B, pos_len, n, d]
            q_pos_rope = rope_apply_with_positions(q_pos, pos_query_positions, grid_sizes, freqs)
            q = torch.cat([q[:, :video_len], q_pos_rope, q[:, video_len + pos_len:]], dim=1)
            
            # Action embedding keys: apply 3D RoPE based on context_positions
            # Build full key positions: null tokens get (0,0,0), actions get context_positions
            key_positions = torch.zeros(b, num_keys, 3, device=x.device, dtype=k.dtype)
            # Null tokens (0:null_len) keep (0,0,0) - no meaningful spatial position
            # Action tokens (null_len:null_len+num_action_tokens) get context_positions
            actual_actions = min(num_action_tokens, context_positions.size(1))
            key_positions[:, null_len:null_len + actual_actions, :] = context_positions[:, :actual_actions, :]
            
            # Apply RoPE to action keys (skip null tokens which have no spatial meaning)
            k_actions = k[:, null_len:null_len + actual_actions]  # [B, num_actions, n, d]
            k_actions_rope = rope_apply_with_positions(k_actions, context_positions[:, :actual_actions], grid_sizes, freqs)
            k = torch.cat([k[:, :null_len], k_actions_rope, k[:, null_len + actual_actions:]], dim=1)
        
        # COORD TOKEN MODE: Apply 3D RoPE to coord tokens (similar to position tokens but simpler)
        # Coord tokens: 1 per subject per frame, with RoPE based on (frame, prev_h, prev_w)
        if self.cross_attn_rope and use_coord_mode and coord_token_info is not None and freqs is not None and coord_len > 0:
            # One-time verification for coord token mode (reuse global flag from position token block)
            if not _CROSS_ATTN_ROPE_VERIFIED:
                _CROSS_ATTN_ROPE_VERIFIED = True
                print(f"[CROSS-ATTN ROPE ACTIVE - COORD MODE] Applying 3D RoPE in cross-attention: "
                      f"coord_len={coord_len}, num_actions={context_positions.shape[1] if context_positions is not None else 0}, "
                      f"spatial_h sample: {coord_token_info.get('spatial_h', 'N/A')[:3] if 'spatial_h' in coord_token_info else 'N/A'}")
            
            T = coord_token_info['num_frames']
            coord_S = coord_token_info['num_subjects']
            
            # Build coord token query positions: (frame, h, w) for each coord token
            # Layout is FRAME-MAJOR: [t0_s0, t0_s1, t1_s0, t1_s1, ...] - all subjects for frame 0, then frame 1, etc.
            
            if 'spatial_h' in coord_token_info and 'spatial_w' in coord_token_info:
                # Use provided spatial positions (previous clean positions)
                spatial_h = coord_token_info['spatial_h']  # [B, T*S]
                spatial_w = coord_token_info['spatial_w']  # [B, T*S]
                
                # Build frame indices: T frames per subject, S subjects
                # coord_token layout in model.py: [B, T*S] flattened from [B, T, S]
                # So token 0 = (t=0, s=0), token 1 = (t=0, s=1), token 2 = (t=1, s=0), etc.
                frame_indices = torch.arange(T, device=x.device).view(T, 1).expand(T, coord_S).reshape(-1)  # [T*S]
                frame_indices = frame_indices.unsqueeze(0).expand(b, -1).to(dtype=q.dtype)  # [B, T*S]
                
                # Stack to get [B, coord_len, 3] positions
                coord_query_positions = torch.stack([frame_indices, spatial_h.to(dtype=q.dtype), spatial_w.to(dtype=q.dtype)], dim=-1)  # [B, coord_len, 3]
                
                # Apply RoPE to coord token queries
                q_coord = q[:, video_len:video_len + coord_len]  # [B, coord_len, n, d]
                q_coord_rope = rope_apply_with_positions(q_coord, coord_query_positions, grid_sizes, freqs)
                q = torch.cat([q[:, :video_len], q_coord_rope, q[:, video_len + coord_len:]], dim=1)
                
                # Apply RoPE to action keys (same as position token mode)
                if context_positions is not None:
                    key_positions = torch.zeros(b, num_keys, 3, device=x.device, dtype=k.dtype)
                    actual_actions = min(num_action_tokens, context_positions.size(1))
                    key_positions[:, null_len:null_len + actual_actions, :] = context_positions[:, :actual_actions, :]
                    
                    k_actions = k[:, null_len:null_len + actual_actions]
                    k_actions_rope = rope_apply_with_positions(k_actions, context_positions[:, :actual_actions], grid_sizes, freqs)
                    k = torch.cat([k[:, :null_len], k_actions_rope, k[:, null_len + actual_actions:]], dim=1)
        
        # Transpose for attention: [B, L, n, d] -> [B, n, L, d]
        q = q.transpose(1, 2)  # [B, n, L, d]
        k = k.transpose(1, 2)  # [B, n, K, d]
        v = v.transpose(1, 2)  # [B, n, K, d]
        
        pos_tokens_per_subject = pos_len // S if S > 0 and pos_len > 0 else pos_len
        coord_tokens_per_subject = coord_len // S if S > 0 and coord_len > 0 else coord_len
        # Actual tokens: video + position + coord (whichever is active)
        actual_query_len = video_len + pos_len + coord_len  # Actual tokens (excluding padding)
        padded_query_len = total_query_len  # Full padded length
        
        q_actual = q[:, :, :actual_query_len, :]
        
        # SDPA with cached mask (faster than FlexAttention for the small asymmetric
        # cross-attention matrix: ~5160 queries x ~552 keys)
        mask = self._build_cross_attn_mask(
            actual_query_len, num_keys, video_len, null_len,
            pos_len, coord_len, S, pos_tokens_per_subject,
            actions_per_subject, position_token_info, coord_token_info,
            x.device, q.dtype
        )
        mask = mask.unsqueeze(0).unsqueeze(0)
        out_actual = torch.nn.functional.scaled_dot_product_attention(
            q_actual, k, v, attn_mask=mask, dropout_p=0.0
        )
        
        if padded_query_len > actual_query_len:
            padding = torch.zeros(b, n, padded_query_len - actual_query_len, d,
                                  device=out_actual.device, dtype=out_actual.dtype)
            out = torch.cat([out_actual, padding], dim=2)
        else:
            out = out_actual
        
        # Output
        out = out.transpose(1, 2).flatten(2)  # [B, L, C]
        return self.o(out)

