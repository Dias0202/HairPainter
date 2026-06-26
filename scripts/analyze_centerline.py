"""
Analyze intensity at fibril CENTERLINES (undilated GT) vs outer edges vs background.
Also test radial NMS (non-maximum suppression in tangential direction) as detector.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import cv2

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from scripts.svg_to_mask import svg_to_mask, IMG_W, IMG_H

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img_data = io.load(Path("Data/Raw/imagemL1.tif"))
raw = img_data.array
enhanced = pre.enhance(img_data)
h, w = raw.shape

capsid = cap_svc.detect(enhanced)
cx, cy = capsid.center
r = capsid.radius

y_grid, x_grid = np.mgrid[:h, :w]
dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).astype(np.float32)
fibril_zone = (dist >= r * 0.85) & (dist <= r * 2.5)
fibril_zone[int(h*0.95):, :] = False
fibril_zone[capsid.mask] = False

gt_mask = svg_to_mask(Path("Data/Manual_paint/imagem1_fios.svg"), out_w=IMG_W, out_h=IMG_H)
gt_d2 = cv2.dilate(gt_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)
gt_d5 = cv2.dilate(gt_mask.astype(np.uint8), np.ones((11,11), np.uint8)).astype(bool)

# Centerline (undilated), 2px ring, 5px ring
gt_center = gt_mask & fibril_zone               # centerline only
gt_edge = gt_d2 & ~gt_mask & fibril_zone        # 2-4px from centerline
gt_far = gt_d5 & ~gt_d2 & fibril_zone           # 5-6px from centerline
bg = fibril_zone & ~gt_d5                       # >6px from any GT annotation

raw_f = raw.astype(float)
print("=== Intensity at different distances from GT centerline (raw) ===")
print(f"Background  (>6px): mean={raw_f[bg].mean():.1f} std={raw_f[bg].std():.1f} n={bg.sum()}")
print(f"Far ring  (5-6px):  mean={raw_f[gt_far].mean():.1f} std={raw_f[gt_far].std():.1f} n={gt_far.sum()}")
print(f"Edge ring (2-4px):  mean={raw_f[gt_edge].mean():.1f} std={raw_f[gt_edge].std():.1f} n={gt_edge.sum()}")
print(f"Centerline (0-1px): mean={raw_f[gt_center].mean():.1f} std={raw_f[gt_center].std():.1f} n={gt_center.sum()}")
print(f"Contrast (BG - center): {raw_f[bg].mean()-raw_f[gt_center].mean():.1f}")

# Radial NMS: find local dark minima along tangential direction
print("\n=== Computing Radial NMS (tangential direction non-max suppression) ===")
dx = (x_grid - cx).astype(np.float32)
dy = (y_grid - cy).astype(np.float32)
dist_safe = np.maximum(dist, 1.0)
tx = -dy / dist_safe   # tangential x
ty =  dx / dist_safe   # tangential y (row direction)

img_f = raw_f
for step in [3, 5, 8]:
    coords_fwd = [
        (y_grid + ty * step).ravel().clip(0, h-1),
        (x_grid + tx * step).ravel().clip(0, w-1),
    ]
    coords_bwd = [
        (y_grid - ty * step).ravel().clip(0, h-1),
        (x_grid - tx * step).ravel().clip(0, w-1),
    ]
    v_fwd = ndi.map_coordinates(img_f, coords_fwd, order=1).reshape(h, w)
    v_bwd = ndi.map_coordinates(img_f, coords_bwd, order=1).reshape(h, w)
    local_min = (img_f <= v_fwd) & (img_f <= v_bwd) & fibril_zone

    tp = (local_min & gt_center).sum()
    fp_count = (local_min & ~gt_d5 & fibril_zone).sum()
    fn = (gt_center & ~local_min).sum()
    precision = tp / (tp + fp_count) if (tp+fp_count) > 0 else 0
    recall = tp / gt_center.sum() if gt_center.sum() > 0 else 0
    iou = tp / (tp + fp_count + fn) if (tp+fp_count+fn) > 0 else 0
    print(f"  step={step}: local_min_px={local_min.sum()} TP={tp} recall={recall:.2f} prec={precision:.3f} IoU={iou:.4f}")

# Contrast threshold within zone
print("\n=== Simple intensity threshold within fibril zone ===")
bg_mean = raw_f[bg].mean()
for thresh_offset in [5, 10, 15, 20, 25, 30]:
    thresh = bg_mean - thresh_offset
    pred = (raw_f < thresh) & fibril_zone
    tp = (pred & gt_center).sum()
    fp_count = (pred & ~gt_d5 & fibril_zone).sum()
    fn = (gt_center & ~pred).sum()
    iou = tp / (tp + fp_count + fn) if (tp+fp_count+fn) > 0 else 0
    recall = tp / gt_center.sum() if gt_center.sum() > 0 else 0
    print(f"  thresh=BG-{thresh_offset} ({thresh:.0f}): pred={pred.sum()} recall={recall:.2f} IoU={iou:.4f}")

# Combined: NMS + contrast
print("\n=== Combined: radial NMS (step=5) + contrast (BG-15) ===")
dx = (x_grid - cx).astype(np.float32)
dy = (y_grid - cy).astype(np.float32)
dist_safe = np.maximum(dist, 1.0)
tx = -dy / dist_safe
ty =  dx / dist_safe
v_fwd5 = ndi.map_coordinates(img_f, [(y_grid + ty*5).ravel().clip(0,h-1), (x_grid + tx*5).ravel().clip(0,w-1)], order=1).reshape(h,w)
v_bwd5 = ndi.map_coordinates(img_f, [(y_grid - ty*5).ravel().clip(0,h-1), (x_grid - tx*5).ravel().clip(0,w-1)], order=1).reshape(h,w)
nms = (img_f <= v_fwd5) & (img_f <= v_bwd5)
contrast_ok = raw_f < (bg_mean - 10)
combined = nms & contrast_ok & fibril_zone

tp = (combined & gt_center).sum()
fp_count = (combined & ~gt_d5 & fibril_zone).sum()
fn = (gt_center & ~combined).sum()
iou = tp / (tp + fp_count + fn) if (tp+fp_count+fn) > 0 else 0
recall = tp / gt_center.sum() if gt_center.sum() > 0 else 0
precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0
print(f"  pred={combined.sum()} TP={tp} recall={recall:.3f} prec={precision:.3f} IoU={iou:.4f}")
