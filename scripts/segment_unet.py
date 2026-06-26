"""segment_unet.py — U-Net segmentation path, SegmentResult-compatible.

Drop-in alternative to SegmentService.segment / segment_radial_profile that runs
a trained U-Net checkpoint instead of Frangi.  The probability map is thresholded,
restricted to the fibril zone, then post-processed with the SAME classical steps
(skeletonize + label + arc-length) so downstream Measure/Render are unchanged.

The checkpoint is cached across calls (batch mode loads it once).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, skeletonize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from scripts.infer_unet import predict_full  # noqa: E402
from scripts.train_unet import UNet  # noqa: E402
from hairpainter.services.segment.segment_service import SegmentService  # noqa: E402
from hairpainter.utils.types import CapsidResult, FibrilInstance, SegmentResult  # noqa: E402

_STRUCT_8 = np.ones((3, 3), dtype=int)
_MODEL_CACHE: dict[str, tuple] = {}


def _load(ckpt_path: str):
    if ckpt_path not in _MODEL_CACHE:
        ck = torch.load(ckpt_path, map_location="cpu")
        model = UNet(base=ck.get("base", 32))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        _MODEL_CACHE[ckpt_path] = (model, ck.get("variant", "production"))
    return _MODEL_CACHE[ckpt_path]


def segment_unet(
    enhanced: np.ndarray,
    capsid: CapsidResult,
    ckpt_path: str,
    threshold: float = 0.5,
    min_fibril_px: int = 15,
    zone_inner_frac: float = 0.85,
    zone_outer_frac: float = 2.0,
) -> SegmentResult:
    h, w = enhanced.shape
    cx, cy = capsid.center
    r = capsid.radius
    scale_bar_y = int(h * 0.95)

    model, _ = _load(ckpt_path)
    prob = predict_full(model, enhanced.astype(np.float32) / 255.0)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    zone = (dist >= r * zone_inner_frac) & (dist <= r * zone_outer_frac)

    binary = (prob >= threshold) & zone
    binary[capsid.mask] = False
    binary[scale_bar_y:, :] = False
    binary = remove_small_objects(binary, max_size=7)

    skeleton = skeletonize(binary)
    lab, n = ndi.label(skeleton, structure=_STRUCT_8)
    if n == 0:
        return SegmentResult(label_map=np.zeros((h, w), dtype=np.int32), fibrils=[])

    label_map = np.zeros((h, w), dtype=np.int32)
    fibrils: list[FibrilInstance] = []
    new_id = 1
    import cv2

    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for lbl in range(1, n + 1):
        skel_i = lab == lbl
        if skel_i.sum() < min_fibril_px:
            continue
        length_px = SegmentService._corrected_skeleton_length(skel_i)
        mask_i = cv2.dilate(skel_i.astype(np.uint8), dil).astype(bool)
        mask_i[capsid.mask] = False
        mask_i[scale_bar_y:, :] = False
        label_map[mask_i] = new_id
        fibrils.append(FibrilInstance(id=new_id, mask=mask_i, skeleton=skel_i, length_px=length_px))
        new_id += 1

    return SegmentResult(label_map=label_map, fibrils=fibrils)
