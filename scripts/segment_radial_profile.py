"""segment_radial_profile.py — fragmentation-free segmentation by 1D radial runs.

Rationale (SDD section 13.4, item 2.2):
    The classic pipeline skeletonises a noisy Frangi mask and then tries to stitch
    the resulting disjoint crests back together (angular merge + radial connect).
    That stitching is the structural source of fragmentation.

This prototype removes that failure mode by construction:
    1. Frangi vesselness on the capsid-zeroed working image (same detector).
    2. Warp the vesselness map to polar space (rows = angle, cols = radius).
    3. For each angle, take the strongest contiguous radial RUN above threshold
       inside the band [inner_frac*r, outer_frac*r], after closing small radial
       gaps -> one continuous fibril per angle, no fragmentation by construction.
    4. Angular non-maximum suppression keeps only locally strongest angles, so
       the fibril count tracks the real ridge spacing instead of producing a
       solid filled ring (the failure of the earlier polar attempt, SDD v2.4).
    5. Each kept run -> a single continuous radial stroke in cartesian space,
       anchored at the capsid surface.

Returns a SegmentResult, so it is a drop-in alternative to SegmentService.segment
inside the experiment harness (`--segment radial_profile`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line as draw_line_pixels
from skimage.filters import frangi

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hairpainter.utils.types import CapsidResult, FibrilInstance, SegmentResult  # noqa: E402

_STRUCT_8 = np.ones((3, 3), dtype=int)


def _longest_run(row_bool: np.ndarray, weights: np.ndarray) -> tuple[int, int, float] | None:
    """Return the strongest contiguous True run as (start, end, strength).

    Strength = sum of `weights` over the run (favours long, high-vesselness runs).
    Indices are relative to the start of `row_bool`.
    """
    best = None
    i = 0
    n = row_bool.size
    while i < n:
        if not row_bool[i]:
            i += 1
            continue
        j = i
        while j < n and row_bool[j]:
            j += 1
        strength = float(weights[i:j].sum())
        if best is None or strength > best[2]:
            best = (i, j - 1, strength)
        i = j
    return best


def segment_radial_profile(
    enhanced: np.ndarray,
    capsid: CapsidResult,
    frangi_sigmas: tuple[float, ...] = (3, 5, 8),
    frangi_threshold: float = 0.05,
    min_fibril_px: int = 15,
    dilation_radius: int = 3,
    inner_frac: float = 0.9,
    outer_frac: float = 2.0,
    n_angles: int = 720,
    gap_close_px: int = 20,
    anchor_frac: float = 1.2,
    nms_half_window: int = 1,
) -> SegmentResult:
    h, w = enhanced.shape
    cx, cy = capsid.center
    r = capsid.radius
    scale_bar_y = int(h * 0.95)

    # 1. Frangi on capsid-zeroed working image (identical detector to classic).
    working = enhanced.astype(np.float32) / 255.0
    working[capsid.mask] = 0.0
    working[scale_bar_y:, :] = 0.0
    vesselness = frangi(working, sigmas=frangi_sigmas, black_ridges=True)
    mx = vesselness.max()
    if mx > 0:
        vesselness = vesselness / mx

    # 2. Polar transform: rows = angle (0..2pi), cols = radius (0..max_r).
    max_r = min(int(r * outer_frac) + 4, int(np.hypot(max(cx, w - cx), max(cy, h - cy))))
    polar = cv2.warpPolar(
        vesselness.astype(np.float32),
        dsize=(max_r, n_angles),
        center=(float(cx), float(cy)),
        maxRadius=float(max_r),
        flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
    )

    r_in = max(1, int(r * inner_frac))
    r_out = min(max_r - 1, int(r * outer_frac))
    band = polar[:, r_in:r_out] >= frangi_threshold

    # 3. Close small radial gaps per row (along the radius axis = columns).
    if gap_close_px > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_close_px, 1))
        band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    # 4. One strongest radial run per angle; record strength for angular NMS.
    runs: list[tuple | None] = [None] * n_angles
    strength = np.zeros(n_angles, dtype=np.float32)
    for a in range(n_angles):
        run = _longest_run(band[a], polar[a, r_in:r_out])
        if run is None:
            continue
        rs, re, st = run
        if (re - rs) < min_fibril_px:
            continue
        if (r_in + rs) > r * anchor_frac:  # inner end must be near surface
            continue
        runs[a] = (r_in + rs, r_in + re, st)
        strength[a] = st

    # 5. Angular non-maximum suppression: keep locally strongest angles so the
    #    count tracks real ridge spacing instead of filling a solid ring.
    if nms_half_window > 0:
        win = 2 * nms_half_window + 1
        local_max = ndi.maximum_filter1d(strength, size=win, mode="wrap")
        keep = (strength > 0) & (strength >= local_max)
    else:
        keep = strength > 0

    dil_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
    )
    two_pi = 2.0 * np.pi
    label_map = np.zeros((h, w), dtype=np.int32)
    fibrils: list[FibrilInstance] = []
    new_id = 1

    for a in range(n_angles):
        if not keep[a] or runs[a] is None:
            continue
        r_start, r_end, _ = runs[a]
        theta = a / n_angles * two_pi
        ux, uy = np.cos(theta), np.sin(theta)
        x0 = int(round(cx + r_start * ux))
        y0 = int(round(cy + r_start * uy))
        x1 = int(round(cx + r_end * ux))
        y1 = int(round(cy + r_end * uy))
        rr, cc = draw_line_pixels(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[valid], cc[valid]
        if rr.size == 0:
            continue

        skel = np.zeros((h, w), dtype=bool)
        skel[rr, cc] = True
        skel[capsid.mask] = False
        skel[scale_bar_y:, :] = False
        if skel.sum() < min_fibril_px:
            continue

        length_px = float(np.hypot(x1 - x0, y1 - y0))
        mask_i = cv2.dilate(skel.astype(np.uint8), dil_kernel).astype(bool)
        mask_i[capsid.mask] = False
        mask_i[scale_bar_y:, :] = False

        label_map[mask_i] = new_id
        fibrils.append(
            FibrilInstance(id=new_id, mask=mask_i, skeleton=skel, length_px=length_px)
        )
        new_id += 1

    return SegmentResult(label_map=label_map, fibrils=fibrils)


if __name__ == "__main__":
    from hairpainter.services.capsid.capsid_service import CapsidService
    from hairpainter.services.io.io_service import IOService
    from hairpainter.services.preprocess.preprocess_service import PreprocessService

    img = IOService().load(Path("Data/Raw/imagemL1.tif"))
    enh = PreprocessService().enhance(img)
    cap = CapsidService().detect(enh)
    seg = segment_radial_profile(enh, cap)
    lengths = [f.length_px for f in seg.fibrils]
    print(f"n_fibrils={seg.n_fibrils}")
    if lengths:
        print(f"length_px: min={min(lengths):.0f} mean={np.mean(lengths):.0f} max={max(lengths):.0f}")
        print(f"length_nm (px/1.36): mean={np.mean(lengths)/1.36:.0f}")
