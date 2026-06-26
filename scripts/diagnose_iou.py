"""Detailed IoU diagnostic: check pixel counts, spatial overlap, and masks."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

from scripts.svg_to_mask import svg_to_mask, load_prediction_mask, _X_OFFSET, SCALE_X, SCALE_Y, IMG_W, IMG_H

svg_path = Path("Data/Manual_paint/imagem1_fios.svg")
pred_dir = Path("output")

print("=== IoU Diagnostic — imagemL1 ===\n")

# Load prediction
pred_mask = load_prediction_mask(pred_dir, "imagem1_fios")
print(f"Prediction mask: {pred_mask.shape if pred_mask is not None else 'NOT FOUND'}")
if pred_mask is not None:
    print(f"  nonzero pixels: {pred_mask.sum()}")
    ys, xs = np.where(pred_mask)
    print(f"  x range: {xs.min()}-{xs.max()}, y range: {ys.min()}-{ys.max()}")

print()

# Load GT
gt_mask = svg_to_mask(svg_path, out_w=IMG_W, out_h=IMG_H)
print(f"GT mask: {gt_mask.shape}")
print(f"  nonzero pixels: {gt_mask.sum()}")
ys, xs = np.where(gt_mask)
if len(ys) > 0:
    print(f"  x range: {xs.min()}-{xs.max()}, y range: {ys.min()}-{ys.max()}")

print()
if pred_mask is not None:
    # Resize pred if needed
    if pred_mask.shape != gt_mask.shape:
        img = Image.fromarray(pred_mask.astype(np.uint8)*255)
        img = img.resize((IMG_W, IMG_H), Image.NEAREST)
        pred_mask = np.array(img) > 127

    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    iou = intersection / union if union > 0 else 0
    print(f"IoU: {iou:.4f}")
    print(f"Intersection: {intersection} pixels")
    print(f"Union: {union} pixels")
    print(f"Pred-only: {(pred_mask & ~gt_mask).sum()} pixels")
    print(f"GT-only: {(~pred_mask & gt_mask).sum()} pixels")

    # Test with dilated GT (to account for 2px GT stroke vs 5px pred width)
    import cv2
    gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)
    pred_dilated = cv2.dilate(pred_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)

    int2 = (pred_dilated & gt_dilated).sum()
    uni2 = (pred_dilated | gt_dilated).sum()
    iou2 = int2/uni2 if uni2 > 0 else 0
    print(f"\nIoU with both masks dilated 5px: {iou2:.4f}")

    # Look at a small patch around a known fibril area
    # Fibril at SVG (696, 280) → image (696, 280)
    x0, y0 = 630, 220
    x1, y1 = 760, 340
    patch_pred = pred_mask[y0:y1, x0:x1]
    patch_gt = gt_mask[y0:y1, x0:x1]
    print(f"\nLocal patch around (630-760, 220-340) where first fibril starts:")
    print(f"  GT pixels in patch: {patch_gt.sum()}")
    print(f"  Pred pixels in patch: {patch_pred.sum()}")
    int_patch = (patch_pred & patch_gt).sum()
    print(f"  Intersection: {int_patch}")

    # Also check: where are the GT centroid and pred centroid?
    gy, gx = np.where(gt_mask)
    py, px = np.where(pred_mask)
    print(f"\nGT centroid: ({int(gx.mean())}, {int(gy.mean())})")
    print(f"Pred centroid: ({int(px.mean())}, {int(py.mean())})")
