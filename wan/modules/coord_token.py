"""Coord-token / subject-isolation helpers added for the Melting Pot work.

This module collects the purely-standalone additions we made on top of the
upstream Wan DiT (``wan/modules/model.py``) so readers can see our delta from
the fork in one place:

* :data:`FLEX_ATTENTION_AVAILABLE`, :func:`get_compiled_flex_attention`:
  FlexAttention availability + compiled callable shared by the DiT and by
  :mod:`wan.modules.spatial_cross_attn`.
* :func:`get_self_attn_block_mask`: subject-isolation block mask used when
  position-token style supplementary tokens are appended to the video.
* :func:`get_coord_self_attn_block_mask`: subject-isolation block mask used in
  the coord-token layout (``[video_tokens, f0_s0, f0_s1, ..., f1_s0, ...]``).
* :class:`FourierCoordEncoder`: fixed Fourier positional encoding used to map
  2D coordinates into the coord-token embedding space.

Everything that touches the model internals (``rope_apply``, the cross-attention
classes such as ``WanTemporalActionCrossAttention`` and
``WanMultiSubjectCrossAttention``) still lives in ``model.py`` so the DiT stays
in one place; look for ``=== COORD-TOKEN EXTENSION ===`` banners there.
"""
import math

import torch
import torch.nn as nn

# FlexAttention for fast masked attention (PyTorch 2.5+)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True

    # Raise dynamo cache limits to avoid thrashing when many distinct block-mask
    # shapes are compiled (FlexAttention block-size limits).
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096

    flex_attention = torch.compile(flex_attention)
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    flex_attention = None
    create_block_mask = None


__all__ = [
    "FLEX_ATTENTION_AVAILABLE",
    "flex_attention",
    "create_block_mask",
    "get_compiled_flex_attention",
    "get_self_attn_block_mask",
    "get_coord_self_attn_block_mask",
    "FourierCoordEncoder",
]


def get_compiled_flex_attention():
    """Return the (module-level compiled) ``flex_attention`` callable if available."""
    return flex_attention if FLEX_ATTENTION_AVAILABLE else None


# Module-level caches keyed by (layout, device) so we only pay the
# `create_block_mask` cost once per unique attention shape.
_self_attn_block_mask_cache = {}
_coord_self_attn_block_mask_cache = {}


def get_self_attn_block_mask(
    video_len: int,
    tokens_per_subject: int,
    num_subjects: int,
    total_len: int,
    device: torch.device,
    position_no_video_attn: bool = False,
):
    """Block mask for subject-isolated self-attention with position tokens.

    Assumed layout: ``[video_tokens (video_len), subj0_pos, subj1_pos, ...]``.

    When ``position_no_video_attn=False`` (default):
      * video queries see everything,
      * subject-``i`` position queries see video + their own subject's position
        tokens; cross-subject attention is blocked.

    When ``position_no_video_attn=True``:
      * video queries still see everything,
      * position tokens only see position tokens of the same subject (fully
        isolated from video). This is the stricter variant used in ablations.
    """
    global _self_attn_block_mask_cache
    cache_key = (video_len, tokens_per_subject, num_subjects, total_len, str(device), position_no_video_attn)

    if cache_key not in _self_attn_block_mask_cache:
        pos_start = video_len

        if position_no_video_attn:
            def mask_mod(b, h, q_idx, kv_idx):
                is_video_query = q_idx < video_len
                is_pos_query = q_idx >= video_len
                is_kv_video = kv_idx < video_len
                q_subject = (q_idx - pos_start) // tokens_per_subject
                kv_subject = (kv_idx - pos_start) // tokens_per_subject
                is_same_subject = q_subject == kv_subject
                pos_can_see = is_pos_query & (~is_kv_video) & is_same_subject
                return is_video_query | pos_can_see
        else:
            def mask_mod(b, h, q_idx, kv_idx):
                is_video_query = q_idx < video_len
                q_subject = (q_idx - pos_start) // tokens_per_subject
                kv_subject = (kv_idx - pos_start) // tokens_per_subject
                is_kv_video = kv_idx < video_len
                is_same_subject = q_subject == kv_subject
                return is_video_query | is_kv_video | is_same_subject

        block_mask = create_block_mask(
            mask_mod,
            B=1,
            H=1,
            Q_LEN=total_len,
            KV_LEN=total_len,
            device=device,
        )
        _self_attn_block_mask_cache[cache_key] = block_mask

    return _self_attn_block_mask_cache[cache_key]


def get_coord_self_attn_block_mask(
    video_len: int,
    num_coord_tokens: int,
    num_subjects: int,
    total_len: int,
    device: torch.device,
):
    """Block mask for subject-isolated self-attention with coord tokens.

    Assumed FRAME-MAJOR layout:
    ``[video_tokens, f0_s0, f0_s1, ..., f1_s0, f1_s1, ...]``.

    The coord token at index ``video_len + i`` belongs to subject
    ``i % num_subjects``. Video tokens see everything; each subject's coord
    tokens see video + same-subject coord tokens.
    """
    global _coord_self_attn_block_mask_cache
    cache_key = (video_len, num_coord_tokens, num_subjects, total_len, str(device))

    if cache_key not in _coord_self_attn_block_mask_cache:
        def mask_mod(b, h, q_idx, kv_idx):
            is_video_query = q_idx < video_len
            is_kv_video = kv_idx < video_len
            q_subject = (q_idx - video_len) % num_subjects
            kv_subject = (kv_idx - video_len) % num_subjects
            is_same_subject = q_subject == kv_subject
            return is_video_query | is_kv_video | is_same_subject

        block_mask = create_block_mask(
            mask_mod, B=1, H=1, Q_LEN=total_len, KV_LEN=total_len, device=device,
        )
        _coord_self_attn_block_mask_cache[cache_key] = block_mask

    return _coord_self_attn_block_mask_cache[cache_key]


class FourierCoordEncoder(nn.Module):
    """Fixed Fourier positional encoding for 2D coordinates.

    Maps ``(x, y)`` in ``[0, 1]`` to a high-dimensional feature vector using
    sin/cos at multiple frequency bands. Compared to a learned linear layer,
    this:

    * works from step 0 (no learning needed),
    * creates distinguishable features for nearby coordinates, and
    * uses full dimensionality rather than the rank-2 features a linear
      projection of ``(x, y)`` would produce.
    """

    def __init__(self, dim, num_frequencies=10):
        super().__init__()
        self.dim = dim
        self.num_frequencies = num_frequencies

        # Each (x, y) gets sin/cos per frequency => 4 features per freq.
        self.raw_dim = 4 * num_frequencies

        freqs = math.pi * (2.0 ** torch.arange(num_frequencies))
        self.register_buffer("freqs", freqs)

        if self.raw_dim != dim:
            self.proj = nn.Linear(self.raw_dim, dim, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
        else:
            self.proj = None

    def forward(self, coords):
        """
        Args:
            coords: ``[..., 2]`` tensor of ``(x, y)`` coordinates in ``[0, 1]``.

        Returns:
            ``[..., dim]`` Fourier features in the dtype of ``coords``.
        """
        x = coords[..., 0:1]
        y = coords[..., 1:2]

        x_scaled = x.float() * self.freqs
        y_scaled = y.float() * self.freqs

        features = torch.cat(
            [torch.sin(x_scaled), torch.cos(x_scaled), torch.sin(y_scaled), torch.cos(y_scaled)],
            dim=-1,
        )

        if self.proj is not None:
            features = self.proj(features.type_as(self.proj.weight))

        return features.type_as(coords)
