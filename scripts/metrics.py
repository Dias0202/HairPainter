"""metrics.py — evaluation metrics for fibril detection experiments.

All metrics are restricted to the ANNOTATED QUADRANT (x < 1280, y < 720) because
the ground-truth SVGs only cover the top-left portion of the 1376x1070 image.
Evaluating outside that region would unfairly penalise precision.

The metrics are designed around the three reported failure modes:
  1. Capsid-as-fibril false positives  -> capsid_fp_fraction
  2. Low recall of faint fibrils        -> f1_tolerance / recall_centerline
  3. Fragmentation of fibrils           -> components_per_sector / frag_ratio

IoU is reported for reference but is physically capped at ~5% (local contrast
~0, see SDD section 11), so it must NOT dominate the selection score.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

# Annotated viewport of the ground-truth SVGs.
ANNOTATED_W, ANNOTATED_H = 1280, 720

_STRUCT_8 = np.ones((3, 3), dtype=int)


# ----------------------------------------------------------------------
# Domain restriction
# ----------------------------------------------------------------------
def annotated_quadrant_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask True inside the annotated SVG viewport (top-left)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[: min(ANNOTATED_H, h), : min(ANNOTATED_W, w)] = True
    return mask


def _restrict(mask: np.ndarray, quad: np.ndarray) -> np.ndarray:
    return mask & quad


# ----------------------------------------------------------------------
# Pixel-overlap metrics
# ----------------------------------------------------------------------
def iou(pred: np.ndarray, gt: np.ndarray, quad: np.ndarray | None = None) -> float:
    if quad is not None:
        pred, gt = pred & quad, gt & quad
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    return inter / union if union > 0 else 0.0


def _disk(radius: int) -> np.ndarray:
    k = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def f1_tolerance(
    pred: np.ndarray, gt: np.ndarray, tol: int = 5, quad: np.ndarray | None = None
) -> dict:
    """Boundary F1 with a tolerance band of `tol` pixels.

    recall    = fraction of GT pixels that have a pred pixel within `tol`
    precision = fraction of pred pixels that have a GT pixel within `tol`
    Restricting to the annotated quadrant keeps precision fair.
    """
    if quad is not None:
        pred, gt = pred & quad, gt & quad

    gt_sum = int(gt.sum())
    pred_sum = int(pred.sum())
    if gt_sum == 0 or pred_sum == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    kernel = _disk(tol)
    pred_dil = cv2.dilate(pred.astype(np.uint8), kernel).astype(bool)
    gt_dil = cv2.dilate(gt.astype(np.uint8), kernel).astype(bool)
    if quad is not None:
        pred_dil &= quad
        gt_dil &= quad

    recall = float((gt & pred_dil).sum()) / gt_sum
    precision = float((pred & gt_dil).sum()) / pred_sum
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def recall_centerline(
    pred: np.ndarray, gt: np.ndarray, tol: int = 5, quad: np.ndarray | None = None
) -> float:
    """Recall of the (undilated) GT centerline within `tol` px of any pred pixel."""
    return f1_tolerance(pred, gt, tol=tol, quad=quad)["recall"]


# ----------------------------------------------------------------------
# Problem 1: capsid false positives
# ----------------------------------------------------------------------
def capsid_fp_fraction(
    pred: np.ndarray,
    capsid_mask: np.ndarray,
    dist_from_center: np.ndarray,
    radius: int,
    inner_frac: float = 0.85,
    quad: np.ndarray | None = None,
) -> float:
    """Fraction of predicted pixels that fall on the capsid or its wall.

    A pred pixel is a capsid FP if it is inside the capsid mask OR within the
    inner buffer ring [inner_frac*r, 1.0*r] (the capsid wall, the usual source
    of edge false positives).
    """
    if quad is not None:
        pred = pred & quad
    pred_sum = int(pred.sum())
    if pred_sum == 0:
        return 0.0
    wall = (dist_from_center >= radius * inner_frac) & (dist_from_center <= radius)
    fp_region = capsid_mask | wall
    return float((pred & fp_region).sum()) / pred_sum


# ----------------------------------------------------------------------
# Problem 3: fragmentation
# ----------------------------------------------------------------------
def n_fibrils_gt(svg_path: Path) -> int:
    """Count annotated fibril paths in an SVG (skips the background rectangle)."""
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    count = 0
    for path_el in root.findall(".//svg:path", ns):
        d = path_el.get("d", "")
        if not d or d.strip().startswith("M0"):
            continue
        count += 1
    return count


def components_per_sector(
    pred: np.ndarray,
    center: tuple[int, int],
    radius: int,
    n_sectors: int = 36,
    quad: np.ndarray | None = None,
) -> dict:
    """Count connected pred components whose centroid falls in each angular sector.

    High fragmentation = many short components packed into the same sector.
    Returns {mean, max, total_components}.
    """
    if quad is not None:
        pred = pred & quad
    lab, n = ndi.label(pred, structure=_STRUCT_8)
    if n == 0:
        return {"mean": 0.0, "max": 0, "total_components": 0}

    cx, cy = center
    counts = np.zeros(n_sectors, dtype=int)
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = np.where(lab[sl] == i)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        theta = np.arctan2(ys.mean() - cy, xs.mean() - cx)  # [-pi, pi]
        sector = int((theta + np.pi) / (2 * np.pi) * n_sectors) % n_sectors
        counts[sector] += 1
    occupied = counts[counts > 0]
    return {
        "mean": float(occupied.mean()) if occupied.size else 0.0,
        "max": int(counts.max()),
        "total_components": int(n),
    }


def frag_ratio(n_pred: int, n_gt: int) -> float:
    """Ratio of predicted to ground-truth fibril counts (1.0 = ideal)."""
    return n_pred / n_gt if n_gt > 0 else 0.0


# ----------------------------------------------------------------------
# Composite score
# ----------------------------------------------------------------------
# Score is F1-centric with a recall nudge (recall is the weak axis: the pipeline
# misses the inner peri-capsid ring).  capsid_fp is NOT in the score: the GT
# legitimately occupies the [0.85r, 1.0r] ring (GT radial p10 ~0.23r), so
# penalising pred pixels there would punish correct detections.  capsid_fp is
# kept as a reported diagnostic for the radius-misdetection failure mode only.
DEFAULT_WEIGHTS = {"f1": 0.55, "recall": 0.35, "capsid_fp": 0.0, "length": 0.10}
# Measured from the ground-truth SVG paths: per-image mean fibril length is
# 74-98 nm (median 63-89 nm).  The earlier 140 nm assumption was wrong; the real
# fibrils are short and sit close to the capsid.  Target = GT mean (~80 nm).
TARGET_LEN_NM = 80.0


def composite_score(
    f1: float,
    recall: float,
    capsid_fp: float,
    mean_len_nm: float,
    weights: dict | None = None,
    target_len_nm: float = TARGET_LEN_NM,
) -> float:
    """Balanced score: reward F1/recall/length-match, penalise capsid FP.

        score = wf*f1 + wr*recall - wc*capsid_fp + wl*(1 - |len-target|/target)
    """
    w = weights or DEFAULT_WEIGHTS
    len_term = max(0.0, 1.0 - abs(mean_len_nm - target_len_nm) / target_len_nm)
    return (
        w["f1"] * f1
        + w["recall"] * recall
        - w["capsid_fp"] * capsid_fp
        + w["length"] * len_term
    )


# ----------------------------------------------------------------------
# Convenience: dist_from_center grid
# ----------------------------------------------------------------------
def dist_from_center_grid(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    h, w = shape
    cx, cy = center
    y_grid, x_grid = np.ogrid[:h, :w]
    return np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2).astype(np.float32)


# ----------------------------------------------------------------------
# Self-test: reproduce the current IoU (~0.058) against validation.json
# ----------------------------------------------------------------------
def _selftest() -> None:
    """Reproduce baseline IoU from the existing output/ vs Data/Manual_paint/."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from svg_to_mask import load_prediction_mask, svg_to_mask  # noqa: E402

    svg_dir = Path("Data/Manual_paint")
    pred_dir = Path("output")
    quad = None
    ious_full, ious_quad = [], []
    for svg_file in sorted(svg_dir.glob("*.svg")):
        gt = svg_to_mask(svg_file)
        pred = load_prediction_mask(pred_dir, svg_file.stem)
        if pred is None:
            print(f"  {svg_file.name}: no prediction")
            continue
        if pred.shape != gt.shape:
            print(f"  {svg_file.name}: shape mismatch {pred.shape} vs {gt.shape}")
            continue
        if quad is None:
            quad = annotated_quadrant_mask(gt.shape)
        i_full = iou(pred, gt)
        i_quad = iou(pred, gt, quad)
        f1 = f1_tolerance(pred, gt, tol=5, quad=quad)
        ious_full.append(i_full)
        ious_quad.append(i_quad)
        print(
            f"  {svg_file.name}: IoU(full)={i_full:.4f} IoU(quad)={i_quad:.4f} "
            f"F1@5px={f1['f1']:.3f} recall={f1['recall']:.3f} prec={f1['precision']:.3f} "
            f"n_gt_paths={n_fibrils_gt(svg_file)}"
        )
    if ious_full:
        print(f"\nmean IoU(full)={np.mean(ious_full):.4f}  (expected ~0.058)")
        print(f"mean IoU(quad)={np.mean(ious_quad):.4f}")


if __name__ == "__main__":
    _selftest()
