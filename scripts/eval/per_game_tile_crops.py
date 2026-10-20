#!/usr/bin/env python3
"""
3×3 grid-cell crops in pixel space for per-game tile models.

Latent→pixel frame indices: ``[0] + [1 + 4*k for k in range(num_latent - 1)]`` (see ``get_latent_to_pixel``).

Edge crops pad the frame so every patch is exactly ``(3 * cell_px, 3 * cell_px)``.
**Default** is ``pad_mode="constant"`` with sentinel magenta ``DEFAULT_PAD_RGB``
(``(255, 0, 255)``) so OOB regions are visually distinct; use ``pad_mode="reflect"``
to mirror edge pixels instead.
"""
from __future__ import annotations

import numpy as np

def get_latent_to_pixel(num_latent: int) -> list[int]:
    if num_latent <= 0:
        return []
    return [0] + [1 + 4 * k for k in range(num_latent - 1)]


def coords_to_pixels(cx: float, cy: float, image_size: int) -> tuple[float, float]:
    """If coords look normalized, scale to pixel space."""
    m = max(float(cx), float(cy), 1.0)
    if m <= 1.0:
        return float(cx) * image_size, float(cy) * image_size
    return float(cx), float(cy)


def coords_to_cell_indices(cx: float, cy: float, cell_px: int, image_size: int) -> tuple[int, int]:
    px, py = coords_to_pixels(cx, cy, image_size)
    c = max(int(cell_px), 1)
    gi = int(np.floor(px / c))
    gj = int(np.floor(py / c))
    return gi, gj


def cell_delta_to_class(dgi: int, dgj: int) -> int:
    """Map (dgi, dgj) each in {-1,0,1} to 0..8 row-major with i (x) fast."""
    ii = int(dgi) + 1
    jj = int(dgj) + 1
    ii = max(0, min(2, ii))
    jj = max(0, min(2, jj))
    return jj * 3 + ii


def class_to_cell_delta(cls: int) -> tuple[int, int]:
    cls = int(cls) % 9
    jj, ii = divmod(cls, 3)
    return ii - 1, jj - 1


# Chroma-key style fill: rare in Melting Pot RGB; easier for a convnet than pure red (lava/UI).
DEFAULT_PAD_RGB: tuple[int, int, int] = (255, 0, 255)


def parse_pad_rgb(s: str) -> tuple[int, int, int]:
    """Parse ``'R,G,B'`` (comma-separated ints) for constant padding."""
    parts = [p.strip() for p in s.replace(" ", "").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"pad color must be R,G,B with three integers, got {s!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _spatial_pad_frame(
    frame: np.ndarray,
    pad: int,
    pad_mode: str,
    pad_fill: tuple[float, ...] | float | None,
) -> np.ndarray:
    """Return frame with ``pad`` pixels added on top/bottom/left/right (channels unchanged)."""
    if pad_mode == "reflect":
        return np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    if pad_mode != "constant":
        raise ValueError(f"pad_mode must be 'reflect' or 'constant', got {pad_mode!r}")

    h, w, c = frame.shape
    fill = pad_fill if pad_fill is not None else DEFAULT_PAD_RGB
    if isinstance(fill, (int, float, np.integer, np.floating)):
        v = float(fill)
        fill_arr = np.full(c, v, dtype=frame.dtype)
    else:
        fill_arr = np.asarray(fill, dtype=frame.dtype).reshape(-1)
        if fill_arr.size < c:
            raise ValueError(f"pad_fill needs at least {c} channel values, got {fill_arr!r}")
        fill_arr = fill_arr[:c]
    out = np.empty((h + 2 * pad, w + 2 * pad, c), dtype=frame.dtype)
    out[:] = fill_arr.reshape(1, 1, c)
    out[pad : pad + h, pad : pad + w, :] = frame
    return out


def extract_3x3_cell_patch(
    frame: np.ndarray,
    cx: float,
    cy: float,
    cell_px: int,
    image_size: int,
    pad_cells: int = 1,
    pad_mode: str = "constant",
    pad_fill: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """
    Crop a (3*cell_px, 3*cell_px) patch with the player's cell in the center
    of the 3×3 neighborhood (padded frame).

    Args:
        frame: (H, W, C) uint8/float; H,W should match image_size for in-bounds interior.
        cx, cy: player position (pixel or normalized — see coords_to_pixels).
        cell_px: grid step in pixels (e.g. 32 at 512).
        image_size: used only to detect normalized coords.
        pad_cells: per-side pad = pad_cells * cell_px (default 1 cell, sufficient for 3×3).
        pad_mode: ``"constant"`` (default, magenta OOB) or ``"reflect"`` (mirror edges).
        pad_fill: For ``constant``, per-channel fill (length C) or scalar; ``None`` → ``DEFAULT_PAD_RGB``.
    """
    if frame.ndim != 3:
        raise ValueError(f"frame must be (H,W,C), got {frame.shape}")
    c = max(int(cell_px), 1)
    patch = 3 * c
    pad = max(int(pad_cells), 1) * c
    gi, gj = coords_to_cell_indices(cx, cy, c, image_size)
    x0 = (gi - 1) * c
    y0 = (gj - 1) * c

    padded = _spatial_pad_frame(frame, pad, pad_mode, pad_fill)
    ys = pad + y0
    xs = pad + x0
    out = padded[ys : ys + patch, xs : xs + patch, :]
    if out.shape[0] != patch or out.shape[1] != patch:
        raise RuntimeError(f"bad crop shape {out.shape}, expected ({patch},{patch},*)")
    return np.ascontiguousarray(out)


def center_cell_hwc_from_3x3_patch(patch_hwc: np.ndarray, cell_px: int) -> np.ndarray:
    """
    Middle cell of a ``(3*cell_px, 3*cell_px, C)`` patch from ``extract_3x3_cell_patch``
    (player anchor is the center cell by construction).
    """
    c = max(int(cell_px), 1)
    h, w = patch_hwc.shape[:2]
    if h != 3 * c or w != 3 * c:
        raise ValueError(f"expected H=W=3*cell_px={3 * c}, got {h}x{w}")
    return np.ascontiguousarray(patch_hwc[c : 2 * c, c : 2 * c, :])


def extract_single_cell_patch_rgb(
    frame_hwc: np.ndarray,
    gi: int,
    gj: int,
    cell_px: int,
    *,
    pad_mode: str = "constant",
    pad_fill: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """
    Crop a **single** grid cell ``(gi, gj)`` as ``(cell_px, cell_px, C)`` uint8/float
    (same dtype as ``frame_hwc``).

    Convention matches ``coords_to_cell_indices`` / ``extract_3x3_cell_patch``:
    ``gi`` indexes the **horizontal** / width axis (second dimension of ``frame_hwc``),
    ``gj`` the **vertical** / height axis (first dimension).

    Pads the frame by ``cell_px`` on each side (same as one cell of the 3×3 path) so
    edge cells stay in-bounds.
    """
    if frame_hwc.ndim != 3:
        raise ValueError(f"frame_hwc must be (H,W,C), got {frame_hwc.shape}")
    c = max(int(cell_px), 1)
    pad = c
    padded = _spatial_pad_frame(frame_hwc, pad, pad_mode, pad_fill)
    r0 = pad + int(gj) * c
    c0 = pad + int(gi) * c
    out = padded[r0 : r0 + c, c0 : c0 + c, :]
    if out.shape[0] != c or out.shape[1] != c:
        raise RuntimeError(f"bad single-cell shape {out.shape}, expected ({c},{c},*)")
    return np.ascontiguousarray(out)


def cell_px_latent(cell_px: int, image_size: int, lat_h: int) -> int:
    """Latent grid step (in latent pixels) matching ``cell_px`` in image space (square H=W)."""
    return max(1, int(round(float(cell_px) * float(lat_h) / float(max(int(image_size), 1)))))


def extract_3x3_cell_patch_latent_chw(
    lat_chw: np.ndarray,
    cx: float,
    cy: float,
    cell_px: int,
    image_size: int,
    pad_cells: int = 1,
    pad_mode: str = "constant",
    pad_fill: tuple[float, ...] | float | None = None,
) -> np.ndarray:
    """
    Same 3×3 **logical** cell neighborhood as ``extract_3x3_cell_patch``, but on a
    VAE latent frame ``lat_chw`` with shape ``(C, Hlat, Wlat)`` (e.g. ``C=16``,
    ``Hlat=Wlat=image_size/8``). Uses ``cell_px_latent`` so cell boundaries align
    with the pixel grid when latents are uniform downsample of pixels.
    """
    if lat_chw.ndim != 3:
        raise ValueError(f"lat_chw must be (C,H,W), got {lat_chw.shape}")
    c_lat, h_lat, w_lat = lat_chw.shape
    if h_lat != w_lat:
        raise ValueError(f"lat_chw must be square H=W, got {h_lat}x{w_lat}")
    cell_lat = cell_px_latent(cell_px, image_size, h_lat)
    frame_hwc = np.transpose(lat_chw, (1, 2, 0))
    fill = pad_fill
    if pad_mode == "constant" and fill is None:
        fill = 0.0
    return extract_3x3_cell_patch(
        frame_hwc,
        cx,
        cy,
        cell_lat,
        h_lat,
        pad_cells=pad_cells,
        pad_mode=pad_mode,
        pad_fill=fill,
    )


def draw_cell_grid_overlay(
    patch: np.ndarray,
    cell_px: int,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Copy patch with grid lines at cell boundaries (BGR/RGB depending on frame)."""
    out = patch.copy()
    h, w = out.shape[:2]
    c = int(cell_px)
    for t in range(thickness):
        for i in range(0, w + 1, c):
            x = min(i + t, w - 1)
            out[:, x, :] = color
        for j in range(0, h + 1, c):
            y = min(j + t, h - 1)
            out[y, :, :] = color
    return out
