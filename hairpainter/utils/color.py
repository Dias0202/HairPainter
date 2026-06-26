"""Color constants and alpha compositing helpers."""
from __future__ import annotations

import numpy as np

# Primary fibril color: navy #1c3052
FIBRIL_COLOR_RGB: tuple[int, int, int] = (28, 48, 82)
FIBRIL_ALPHA_BASE: int = 128        # base opacity (0-255)
FIBRIL_ALPHA_OVERLAP_STEP: int = 32 # added per extra overlap layer
FIBRIL_ALPHA_MAX: int = 230

OVERLAY_ALPHA: float = 0.6          # for Deliverable 2 blend


def apply_fibril_color(
    canvas: np.ndarray,
    mask: np.ndarray,
    overlap_count: np.ndarray,
) -> np.ndarray:
    """
    Composite fibril color onto an RGBA canvas.

    canvas      : H x W x 4, uint8
    mask        : H x W, bool — pixels belonging to this fibril
    overlap_count: H x W, int — number of fibrils already drawn at each pixel

    Returns updated canvas (in-place).
    """
    r, g, b = FIBRIL_COLOR_RGB
    alpha = np.minimum(
        FIBRIL_ALPHA_MAX,
        FIBRIL_ALPHA_BASE + overlap_count[mask] * FIBRIL_ALPHA_OVERLAP_STEP,
    ).astype(np.uint8)

    canvas[mask, 0] = r
    canvas[mask, 1] = g
    canvas[mask, 2] = b
    canvas[mask, 3] = alpha
    return canvas


def blend_overlay(
    base_rgb: np.ndarray,
    fibril_rgba: np.ndarray,
) -> np.ndarray:
    """
    Alpha-blend fibril_rgba over base_rgb.

    base_rgb    : H x W x 3, uint8
    fibril_rgba : H x W x 4, uint8

    Returns H x W x 3 uint8 composite.
    """
    alpha = fibril_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = fibril_rgba[:, :, :3].astype(np.float32)
    bg = base_rgb.astype(np.float32)
    out = (fg * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return out
