"""
Analyze Frangi vesselness distribution inside vs outside GT mask.
This reveals whether real fibrils have higher vesselness than background.
"""
from pathlib import Path
import numpy as np

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from skimage.filters import frangi

from scripts.svg_to_mask import svg_to_mask, IMG_W, IMG_H

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img_data = io.load(Path("Data/Raw/imagemL1.tif"))
enhanced = pre.enhance(img_data)
capsid = cap_svc.detect(enhanced)
h, w = enhanced.shape
cx, cy = capsid.center
r = capsid.radius

print(f"Capsid: center=({cx},{cy}), r={r}")

# Build fibril zone and working image
working = enhanced.astype(np.float32) / 255.0
working[capsid.mask] = 0.0
working[int(h * 0.95):, :] = 0.0

# Compute Frangi on full image
print("Computing Frangi (may take 30s)...")
vesselness = frangi(working, sigmas=[1, 2, 3, 5], black_ridges=True)
v_max = vesselness.max()
if v_max > 0:
    vesselness /= v_max
print(f"Frangi max: {v_max:.6f}")

# Build fibril zone
y_grid, x_grid = np.ogrid[:h, :w]
dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
fibril_zone = (dist >= r * 0.85) & (dist <= r * 2.5)
fibril_zone[int(h * 0.95):, :] = False
fibril_zone[capsid.mask] = False
print(f"\nFibril zone: {fibril_zone.sum()} pixels ({fibril_zone.sum()/(h*w)*100:.1f}%)")

# Load GT mask
gt_mask = svg_to_mask(Path("Data/Manual_paint/imagem1_fios.svg"), out_w=IMG_W, out_h=IMG_H)
gt_in_zone = gt_mask & fibril_zone
bg_in_zone = fibril_zone & ~gt_mask & ~capsid.mask

print(f"GT pixels (total): {gt_mask.sum()}")
print(f"GT pixels in zone: {gt_in_zone.sum()}")
print(f"Background in zone: {bg_in_zone.sum()}")

v_gt = vesselness[gt_in_zone]
v_bg = vesselness[bg_in_zone]

print(f"\nFrangi response at GT fibril pixels:")
print(f"  mean={v_gt.mean():.4f} std={v_gt.std():.4f}")
print(f"  p50={np.percentile(v_gt, 50):.4f} p75={np.percentile(v_gt, 75):.4f}")
print(f"  p90={np.percentile(v_gt, 90):.4f} p95={np.percentile(v_gt, 95):.4f}")
print(f"  >0.05: {(v_gt > 0.05).mean()*100:.1f}%")
print(f"  >0.10: {(v_gt > 0.10).mean()*100:.1f}%")
print(f"  >0.20: {(v_gt > 0.20).mean()*100:.1f}%")

print(f"\nFrangi response at background (in zone, not GT):")
print(f"  mean={v_bg.mean():.4f} std={v_bg.std():.4f}")
print(f"  p50={np.percentile(v_bg, 50):.4f} p75={np.percentile(v_bg, 75):.4f}")
print(f"  p90={np.percentile(v_bg, 90):.4f} p95={np.percentile(v_bg, 95):.4f}")
print(f"  >0.05: {(v_bg > 0.05).mean()*100:.1f}%")
print(f"  >0.10: {(v_bg > 0.10).mean()*100:.1f}%")
print(f"  >0.20: {(v_bg > 0.20).mean()*100:.1f}%")

# Find threshold that gives same pixel count as GT
target_px = gt_in_zone.sum()
v_in_zone = vesselness[fibril_zone]
thresholds = np.linspace(0.0, 1.0, 1001)
for t in thresholds:
    count = (v_in_zone > t).sum()
    if count <= target_px * 1.5:
        print(f"\nThreshold that gives ~GT size (x1.5={target_px*1.5:.0f}px): {t:.3f} → {count} pixels")
        break

# IoU at different thresholds
print("\nIoU in zone at different thresholds (GT dilated 5px for fair width comparison):")
import cv2
gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)
gt_dilated_in_zone = gt_dilated & fibril_zone

for t in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    pred = (vesselness > t) & fibril_zone
    inter = (pred & gt_dilated_in_zone).sum()
    union = (pred | gt_dilated_in_zone).sum()
    iou = inter/union if union > 0 else 0
    recall = (pred & gt_mask).sum() / gt_mask.sum() if gt_mask.sum() > 0 else 0
    print(f"  t={t:.2f}: pred_px={pred.sum():6d} IoU(dil5)={iou:.3f}  recall={recall:.3f}")
