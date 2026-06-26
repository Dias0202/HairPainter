"""unet.py — optional U-Net segmentation path for the production pipeline.

The U-Net beats the classical Frangi ceiling (F1@5px ~0.41 vs ~0.31, leave-one-out
on the 5 annotated images — see SDD section 11.4).  It is opt-in via
PipelineConfig.use_unet because it needs PyTorch and a trained checkpoint.

The model architecture here is byte-identical to scripts/train_unet.py so the
leave-one-out checkpoints load directly.  Torch is imported lazily so the rest of
the pipeline runs without it.
"""
from __future__ import annotations

import numpy as np
from skimage.draw import line as draw_line_pixels

from hairpainter.utils.types import CapsidResult, FibrilInstance, SegmentResult

PATCH = 256
STRIDE = 192
_STRUCT_8 = np.ones((3, 3), dtype=int)
_MODEL_CACHE: dict[str, object] = {}


def _build_model(base: int):
    import torch.nn as nn

    class _DoubleConv(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, base: int = 32) -> None:
            super().__init__()
            self.d1 = _DoubleConv(1, base)
            self.d2 = _DoubleConv(base, base * 2)
            self.d3 = _DoubleConv(base * 2, base * 4)
            self.bott = _DoubleConv(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.u3 = _DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.u2 = _DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.u1 = _DoubleConv(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            import torch
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            b = self.bott(self.pool(c3))
            x = self.u3(torch.cat([self.up3(b), c3], 1))
            x = self.u2(torch.cat([self.up2(x), c2], 1))
            x = self.u1(torch.cat([self.up1(x), c1], 1))
            return self.out(x)

    return UNet(base=base)


def _load(ckpt_path: str):
    if ckpt_path not in _MODEL_CACHE:
        import torch

        ck = torch.load(ckpt_path, map_location="cpu")
        model = _build_model(ck.get("base", 32))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _MODEL_CACHE[ckpt_path] = model
    return _MODEL_CACHE[ckpt_path]


def _predict_full(model, gray: np.ndarray) -> np.ndarray:
    """Overlapping-tile sigmoid probability map over the full image."""
    import torch

    h, w = gray.shape
    prob = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    ys = sorted(set(max(0, v) for v in list(range(0, max(1, h - PATCH + 1), STRIDE)) + [h - PATCH]))
    xs = sorted(set(max(0, v) for v in list(range(0, max(1, w - PATCH + 1), STRIDE)) + [w - PATCH]))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = gray[y:y + PATCH, x:x + PATCH]
                if tile.shape != (PATCH, PATCH):
                    continue
                t = torch.from_numpy(tile[None, None]).float()
                pr = torch.sigmoid(model(t))[0, 0].numpy()
                prob[y:y + PATCH, x:x + PATCH] += pr
                cnt[y:y + PATCH, x:x + PATCH] += 1.0
    cnt[cnt == 0] = 1.0
    return prob / cnt


def _longest_run(row_bool: np.ndarray, weights: np.ndarray):
    """Strongest contiguous True run as (start, end, strength=sum of weights)."""
    best = None
    i, n = 0, row_bool.size
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


def _radial_instances(
    response: np.ndarray,
    capsid: CapsidResult,
    threshold: float,
    min_fibril_px: int,
    dilation_radius: int,
    inner_frac: float,
    outer_frac: float,
    n_angles: int = 720,
    nms_half_window: int = 1,
) -> SegmentResult:
    """Decompose a dense response map (e.g. the U-Net probability) into individual
    continuous radial fibrils — one per locally-strongest angle — so the
    deliverable shows hundreds of strokes instead of a few skeletonised blobs."""
    import cv2
    from scipy import ndimage as ndi

    h, w = response.shape
    cx, cy = capsid.center
    r = capsid.radius
    scale_bar_y = int(h * 0.95)

    work = response.astype(np.float32).copy()
    work[capsid.mask] = 0.0
    work[scale_bar_y:, :] = 0.0

    max_r = min(int(r * outer_frac) + 4, int(np.hypot(max(cx, w - cx), max(cy, h - cy))))
    polar = cv2.warpPolar(
        work, dsize=(max_r, n_angles), center=(float(cx), float(cy)),
        maxRadius=float(max_r), flags=cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
    )
    r_in = max(1, int(r * inner_frac))
    r_out = min(max_r - 1, int(r * outer_frac))
    band = polar[:, r_in:r_out] >= threshold

    runs = [None] * n_angles
    strength = np.zeros(n_angles, dtype=np.float32)
    for a in range(n_angles):
        run = _longest_run(band[a], polar[a, r_in:r_out])
        if run is None:
            continue
        rs, re, st = run
        if (re - rs) < min_fibril_px:
            continue
        runs[a] = (r_in + rs, r_in + re)
        strength[a] = st

    if nms_half_window > 0:
        local_max = ndi.maximum_filter1d(strength, size=2 * nms_half_window + 1, mode="wrap")
        keep = (strength > 0) & (strength >= local_max)
    else:
        keep = strength > 0

    dil = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1)
    )
    two_pi = 2.0 * np.pi
    label_map = np.zeros((h, w), dtype=np.int32)
    fibrils: list[FibrilInstance] = []
    new_id = 1
    for a in range(n_angles):
        if not keep[a] or runs[a] is None:
            continue
        r0, r1 = runs[a]
        theta = a / n_angles * two_pi
        ux, uy = np.cos(theta), np.sin(theta)
        x0, y0 = int(round(cx + r0 * ux)), int(round(cy + r0 * uy))
        x1, y1 = int(round(cx + r1 * ux)), int(round(cy + r1 * uy))
        rr, cc = draw_line_pixels(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr, cc = rr[valid], cc[valid]
        if rr.size == 0:
            continue
        skel = np.zeros((h, w), dtype=bool)
        skel[rr, cc] = True
        skel[capsid.mask] = False
        skel[scale_bar_y:, :] = False
        if int(skel.sum()) < min_fibril_px:
            continue
        length_px = float(np.hypot(x1 - x0, y1 - y0))
        mask_i = cv2.dilate(skel.astype(np.uint8), dil).astype(bool)
        mask_i[capsid.mask] = False
        mask_i[scale_bar_y:, :] = False
        label_map[mask_i] = new_id
        fibrils.append(FibrilInstance(id=new_id, mask=mask_i, skeleton=skel, length_px=length_px))
        new_id += 1

    return SegmentResult(label_map=label_map, fibrils=fibrils)


def segment_unet(
    enhanced: np.ndarray,
    capsid: CapsidResult,
    ckpt_path: str,
    threshold: float = 0.45,
    min_fibril_px: int = 15,
    dilation_radius: int = 3,
    zone_inner_frac: float = 0.85,
    zone_outer_frac: float = 2.0,
) -> SegmentResult:
    """U-Net fibril segmentation → SegmentResult (drop-in for the classical
    SegmentService.segment).  The U-Net gives a dense probability map (superior
    pixel recall); it is decomposed into individual continuous radial fibrils so
    Measure/Render get hundreds of clean strokes, not a few skeletonised blobs."""
    model = _load(ckpt_path)
    prob = _predict_full(model, enhanced.astype(np.float32) / 255.0)
    return _radial_instances(
        prob, capsid, threshold, min_fibril_px, dilation_radius,
        zone_inner_frac, zone_outer_frac,
    )
