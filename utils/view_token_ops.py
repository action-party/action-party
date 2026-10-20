"""
Latent-space view tokens: 3×3 neighborhoods on the DiT patch grid (player-centered).

Patch grid matches Wan Conv3d patch_embedding with kernel/stride (1, 2, 2) on latent
[B, C, T, H_lat, W_lat]: each output cell (ph, pw) covers latent indices [2*ph:2*ph+2], [2*pw:2*pw+2].
Player center (ph_c, pw_c) is derived from normalized frame_coords the same way as
ActionOnlySpatialConditioner (effective_patch_size = 2 * pixel_patch_size in latent index space).
"""
from __future__ import annotations

from typing import Tuple

import torch


def latent_patch_grid_hw(latent_h: int, latent_w: int, kh: int = 2, kw: int = 2) -> Tuple[int, int]:
    """Output patch grid size for patch_embedding (1,2,2) on latent [T,H,W]."""
    assert latent_h % kh == 0 and latent_w % kw == 0, (latent_h, latent_w, kh, kw)
    return latent_h // kh, latent_w // kw


def normalized_coords_to_patch_centers(
    frame_coords: torch.Tensor,
    *,
    image_size: float,
    pixel_patch_size: int,
    position_before_action: bool,
    num_action_frames: int,
) -> torch.Tensor:
    """
    Map normalized coords [B, T_all, S, 2] to patch centers (ph, pw) on the DiT patch grid.

    Same 16px binning as ActionOnlySpatialConditioner (effective_patch_size = 2 * pixel_patch_size).
    For 256×256 video and 32×32 latent, Wan patch_embedding (1,2,2) yields a 16×16 token grid;
    these bin indices match (ph, pw) for RoPE and latent gather.

    Returns:
        centers: [B, T_use, S, 2] long — (ph, pw) row/col on the patch grid
    """
    effective_patch_px = float(pixel_patch_size * 2)
    max_patch = int(image_size / effective_patch_px) - 1

    coords_px = frame_coords * image_size

    if position_before_action:
        x_px = coords_px[:, :num_action_frames, :, 0]
        y_px = coords_px[:, :num_action_frames, :, 1]
    else:
        x_px = coords_px[:, 1 : num_action_frames + 1, :, 0]
        y_px = coords_px[:, 1 : num_action_frames + 1, :, 1]

    ph = (y_px / effective_patch_px).clamp(0, max_patch).long()
    pw = (x_px / effective_patch_px).clamp(0, max_patch).long()
    return torch.stack([ph, pw], dim=-1)


def gather_view_latent_patches(
    latent_btchw: torch.Tensor,
    centers: torch.Tensor,
    crop_radius: int = 1,
    kh: int = 2,
    kw: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract P=(2*r+1)^2 latent micro-patches per (t,s), each micro-patch is [C, kh, kw].

    Args:
        latent_btchw: [B, T, C, H_lat, W_lat]
        centers: [B, T, S, 2] long — (ph, pw) on *output* patch grid (H_p=W_p=16 for 32 latent)
        crop_radius: 1 -> 3×3 neighborhoods on the patch grid

    Returns:
        flat: [B, T, S, P, C * kh * kw]
        spatial_h: [B, T, S, P] — global patch row per token (for RoPE)
        spatial_w: [B, T, S, P] — global patch col per token
    """
    B, T, C, H_lat, W_lat = latent_btchw.shape
    Bc, Tc, S, _ = centers.shape
    assert Bc == B and Tc == T, (centers.shape, latent_btchw.shape)
    device = latent_btchw.device
    H_p, W_p = latent_patch_grid_hw(H_lat, W_lat, kh, kw)
    side = 2 * crop_radius + 1
    P = side * side
    flat_dim = C * kh * kw

    spatial_h = latent_btchw.new_zeros(B, T, S, P, dtype=torch.long)
    spatial_w = latent_btchw.new_zeros(B, T, S, P, dtype=torch.long)
    micro = []

    b_ix = torch.arange(B, device=device).view(B, 1, 1).expand(B, T, S)
    t_ix = torch.arange(T, device=device).view(1, T, 1).expand(B, T, S)

    pi = 0
    for dh in range(-crop_radius, crop_radius + 1):
        for dw in range(-crop_radius, crop_radius + 1):
            ph = (centers[..., 0] + dh).clamp(0, H_p - 1)
            pw = (centers[..., 1] + dw).clamp(0, W_p - 1)
            spatial_h[:, :, :, pi] = ph
            spatial_w[:, :, :, pi] = pw
            h_base = ph * kh
            w_base = pw * kw
            cells = []
            for ii in range(kh):
                for jj in range(kw):
                    h = (h_base + ii).long()
                    w = (w_base + jj).long()
                    cells.append(latent_btchw[b_ix, t_ix, :, h, w])
            micro.append(torch.cat(cells, dim=-1))
            pi += 1

    out = torch.stack(micro, dim=3)
    assert out.shape == (B, T, S, P, flat_dim)
    return out, spatial_h, spatial_w


def scatter_view_flat_patches_to_latent(
    view_btspd: torch.Tensor,
    centers_bts2: torch.Tensor,
    *,
    latent_hw: tuple[int, int],
    crop_radius: int = 1,
    kh: int = 2,
    kw: int = 2,
    subject_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pad denoised view micro-patches into a **full-size** latent tensor (zeros elsewhere), for VAE decode.

    Inverse layout of ``gather_view_latent_patches``: for each (b,t,s,p), reshape flat [D] to
    [C, kh, kw] and write to the latent cell at patch-grid offset matching ``gather``.

    Overlapping (b,t,*,*,*) writes from different subjects use **last** writer (rare for separated players).

    Args:
        view_btspd: [B, T, S, P, C*kh*kw]
        centers_bts2: [B, T, S, 2] long — (ph, pw) on DiT patch grid
        latent_hw: (H_lat, W_lat) full latent spatial size (e.g. 64, 64 for 512² video)
        subject_mask: optional [S] bool tensor — scatter only subjects marked True

    Returns:
        z: [B, T, C, H_lat, W_lat]
    """
    B, T, S, P, D = view_btspd.shape
    H_lat, W_lat = latent_hw
    C = D // (kh * kw)
    assert D == C * kh * kw, (D, C, kh, kw)

    device = view_btspd.device
    dtype = view_btspd.dtype
    z = view_btspd.new_zeros(B, T, C, H_lat, W_lat)

    H_p, W_p = latent_patch_grid_hw(H_lat, W_lat, kh, kw)
    offsets: list[tuple[int, int]] = []
    for dh in range(-crop_radius, crop_radius + 1):
        for dw in range(-crop_radius, crop_radius + 1):
            offsets.append((dh, dw))
    assert len(offsets) == P, (len(offsets), P)

    centers = centers_bts2.long()
    for b in range(B):
        for t in range(T):
            for s in range(S):
                if subject_mask is not None and not bool(subject_mask[s].item()):
                    continue
                for p_idx, (dh, dw) in enumerate(offsets):
                    ph = int((centers[b, t, s, 0] + dh).clamp(0, H_p - 1).item())
                    pw = int((centers[b, t, s, 1] + dw).clamp(0, W_p - 1).item())
                    h0, w0 = ph * kh, pw * kw
                    micro = view_btspd[b, t, s, p_idx].view(C, kh, kw)
                    z[b, t, :, h0 : h0 + kh, w0 : w0 + kw] = micro
    return z


def flatten_view_tokens(btsp_d: torch.Tensor) -> torch.Tensor:
    """[B, T, S, P, D] -> [B, T*S*P, D] frame-major: (t0,s0,p*), (t0,s1,p*), ..."""
    B, T, S, P, D = btsp_d.shape
    return btsp_d.permute(0, 1, 2, 3, 4).reshape(B, T * S * P, D)


def unflatten_view_tokens(flat: torch.Tensor, T: int, S: int, P: int) -> torch.Tensor:
    B, L, D = flat.shape
    assert L == T * S * P
    return flat.view(B, T, S, P, D)
