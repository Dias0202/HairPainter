"""preprocess_variants.py — catalogue of image-treatment variants to evaluate.

Each variant is a function `fn(gray_uint8, ctx) -> uint8` registered by name in
`VARIANTS`.  `ctx` is a dict that may carry {"capsid": CapsidResult,
"image_data": ImageData} for variants that need the capsid geometry.

POLARITY CONTRACT (critical):
    SegmentService runs frangi(black_ridges=True), i.e. it detects DARK ridges
    (low-intensity valleys).  Therefore EVERY variant must return an image in
    which fibrils are DARK and the background is bright.  Variants whose natural
    response is bright-on-fibril (top-hat, background subtraction, ridge filters)
    are inverted before returning.

All variants end by stretching to the full [0,255] range via PreprocessService
._normalize, so they share the exact output contract of the production service.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import restoration
from skimage.filters import meijering, unsharp_mask

# Allow running as `python scripts/preprocess_variants.py` (not just -m).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hairpainter.services.preprocess.preprocess_service import PreprocessService

_normalize = PreprocessService._normalize

VARIANTS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {}


def variant(name: str) -> Callable:
    def deco(fn: Callable[[np.ndarray, dict], np.ndarray]) -> Callable:
        VARIANTS[name] = fn
        return fn

    return deco


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return c.apply(gray)


def _stretch_float(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary float array to uint8 [0,255]."""
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Baseline / CLAHE family
# ----------------------------------------------------------------------
@variant("production")
def _production(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Current production pipeline: normalize -> cond. equalize -> CLAHE 2.0/8."""
    g = _normalize(gray)
    if g.std() < 20:
        g = cv2.equalizeHist(g)
    return _clahe(g, 2.0, 8)


@variant("normalize_only")
def _normalize_only(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Pure histogram stretch — isolates whether CLAHE injects noise."""
    return _normalize(gray)


@variant("clahe_strong")
def _clahe_strong(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Aggressive local contrast (clip=4.0, tile=16) to recover faint distal tips."""
    return _clahe(_normalize(gray), 4.0, 16)


@variant("clahe_gentle_largetile")
def _clahe_gentle(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Gentle, large-tile CLAHE (clip=1.0, tile=32) — tile >= fibril length to
    avoid splitting a fibril across tile boundaries (anti-fragmentation)."""
    return _clahe(_normalize(gray), 1.0, 32)


# ----------------------------------------------------------------------
# Morphological / background-flattening family
# ----------------------------------------------------------------------
@variant("blackhat_tophat")
def _blackhat(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Black top-hat with elliptical SE ~21px (fibril width).  Highlights dark
    thin structures and removes low-frequency background in one step.
    Inverted so fibrils stay dark."""
    g = _normalize(gray)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)  # bright where dark fibrils
    return _normalize((255 - _stretch_float(bh)).astype(np.uint8))


@variant("bg_subtract_gauss")
def _bg_subtract_gauss(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """High-pass: gray - Gaussian(sigma=40).  Removes the slow global gradient so
    the local fibril contrast (otherwise ~0 against the dark zone) is exposed.
    Dark fibrils -> negative residual -> dark after stretch."""
    g = _normalize(gray).astype(np.float32)
    detrended = g - gaussian_filter(g, sigma=40)
    return _stretch_float(detrended)


@variant("rolling_ball")
def _rolling_ball(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Morphological rolling-ball background subtraction (microscopy standard).
    Operate on the inverted image (fibrils become bright peaks), subtract the
    estimated background, then invert back to keep fibrils dark."""
    g = _normalize(gray)
    inv = (255 - g).astype(np.float32)
    bg = restoration.rolling_ball(inv, radius=30)
    tophat = np.clip(inv - bg, 0, None)
    return _normalize((255 - _stretch_float(tophat)).astype(np.uint8))


@variant("bgsub_then_clahe")
def _bgsub_then_clahe(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Combo: flatten background (high-pass), then enhance local contrast."""
    flat = _bg_subtract_gauss(gray, ctx)
    return _clahe(flat, 2.0, 16)


# ----------------------------------------------------------------------
# Sharpening / band-pass family
# ----------------------------------------------------------------------
@variant("unsharp")
def _unsharp(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Unsharp masking (radius=7, amount=1.5): accentuates fibril-width ridges
    without the global noise boost of CLAHE.  Polarity preserved."""
    g = _normalize(gray)
    out = unsharp_mask(g, radius=7, amount=1.5)  # float [0,1]
    return _stretch_float(out)


@variant("dog_bandpass")
def _dog_bandpass(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Difference of Gaussians G(3)-G(10) tuned to the ~19px fibril width.
    Rejects low-freq background AND pixel noise.  Dark fibrils -> negative DoG
    -> dark after stretch."""
    g = _normalize(gray).astype(np.float32)
    dog = gaussian_filter(g, sigma=3) - gaussian_filter(g, sigma=10)
    return _stretch_float(dog)


# ----------------------------------------------------------------------
# Denoising family
# ----------------------------------------------------------------------
@variant("bilateral")
def _bilateral(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Edge-preserving bilateral filter: smooths background noise while keeping
    the fibril ridge, reducing noise-driven fragmentation.  Polarity preserved."""
    g = _normalize(gray)
    out = cv2.bilateralFilter(g, d=9, sigmaColor=20, sigmaSpace=7)
    return _normalize(out)


@variant("nlmeans")
def _nlmeans(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Non-local means denoising (h=10): exploits the repetitive fibril texture
    for cleaner continuity.  Polarity preserved."""
    g = _normalize(gray)
    out = cv2.fastNlMeansDenoising(g, h=10)
    return _normalize(out)


@variant("aniso_diffusion")
def _aniso_diffusion(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Perona-Malik anisotropic diffusion (~15 iter, kappa=20): smooths along
    ridges, not across them -> favours radial continuity.  Polarity preserved."""
    img = _normalize(gray).astype(np.float32)
    kappa, gamma, niter = 20.0, 0.15, 15
    out = img.copy()
    for _ in range(niter):
        d_n = np.roll(out, -1, axis=0) - out
        d_s = np.roll(out, 1, axis=0) - out
        d_e = np.roll(out, -1, axis=1) - out
        d_w = np.roll(out, 1, axis=1) - out
        c_n = np.exp(-((d_n / kappa) ** 2))
        c_s = np.exp(-((d_s / kappa) ** 2))
        c_e = np.exp(-((d_e / kappa) ** 2))
        c_w = np.exp(-((d_w / kappa) ** 2))
        out = out + gamma * (c_n * d_n + c_s * d_s + c_e * d_e + c_w * d_w)
    return _stretch_float(out)


@variant("bilateral_then_dog")
def _bilateral_then_dog(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Combo: denoise first, then band-pass -> a cleaner band-pass response."""
    den = _bilateral(gray, ctx)
    return _dog_bandpass(den, ctx)


# ----------------------------------------------------------------------
# Ridge-enhancement family
# ----------------------------------------------------------------------
@variant("meijering_ridge")
def _meijering_ridge(gray: np.ndarray, ctx: dict) -> np.ndarray:
    """Meijering neuriteness ridge filter (black_ridges=True) tuned to fibril
    width.  Produces a bright-on-fibril ridge map; inverted so fibrils stay dark.
    Suppresses non-ridge texture and helps reject tangential capsid-wall noise."""
    g = _normalize(gray).astype(np.float32) / 255.0
    ridge = meijering(g, sigmas=(2, 3, 4), black_ridges=True)
    return _normalize((255 - _stretch_float(ridge)).astype(np.uint8))


def list_variants() -> list[str]:
    return list(VARIANTS.keys())


if __name__ == "__main__":
    # Smoke test: run every variant on imagemL1 and report output stats.
    from pathlib import Path

    from hairpainter.services.io.io_service import IOService

    img = IOService().load(Path("Data/Raw/imagemL1.tif"))
    ctx: dict = {"image_data": img}
    print(f"{'variant':24s} {'min':>4s} {'max':>4s} {'mean':>6s} {'std':>6s}")
    for name, fn in VARIANTS.items():
        out = fn(img.array.copy(), ctx)
        assert out.dtype == np.uint8 and out.shape == img.array.shape, name
        print(f"{name:24s} {out.min():4d} {out.max():4d} {out.mean():6.1f} {out.std():6.1f}")
