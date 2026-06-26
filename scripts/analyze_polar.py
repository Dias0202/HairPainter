"""
Test polar coordinate transform approach for fibril detection.
Fibrils radiate from capsid center → appear as vertical dark bands in polar space.
Also tests background subtraction preprocessing.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates
import cv2

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from scripts.svg_to_mask import svg_to_mask, IMG_W, IMG_H
from PIL import Image

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img_data = io.load(Path("Data/Raw/imagemL1.tif"))
raw = img_data.array.astype(np.float32)
enhanced = pre.enhance(img_data).astype(np.float32)
h, w = raw.shape

capsid = cap_svc.detect(img_data.array)
cx, cy = capsid.center
r = capsid.radius
print(f"Capsid center=({cx},{cy}), r={r}")

# ---- Background subtraction (large-scale Gaussian blur) ----
bg_large = gaussian_filter(raw, sigma=40)
residual = bg_large - raw   # positive where pixel is locally darker than background
print(f"\nBackground subtraction (sigma=40):")
print(f"  Residual range: {residual.min():.1f} to {residual.max():.1f}")

# GT mask
gt_mask = svg_to_mask(Path("Data/Manual_paint/imagem1_fios.svg"), out_w=IMG_W, out_h=IMG_H)
gt_d5 = cv2.dilate(gt_mask.astype(np.uint8), np.ones((11,11), np.uint8)).astype(bool)
y_grid, x_grid = np.mgrid[:h, :w]
dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2).astype(np.float32)
fibril_zone = (dist >= r * 0.85) & (dist <= r * 2.5)
fibril_zone[int(h*0.95):, :] = False
fibril_zone[capsid.mask] = False

gt_center = gt_mask & fibril_zone
bg_zone = fibril_zone & ~gt_d5

# SNR of residual
print(f"  Residual at GT centerline: mean={residual[gt_center].mean():.2f} std={residual[gt_center].std():.2f}")
print(f"  Residual at background:    mean={residual[bg_zone].mean():.2f} std={residual[bg_zone].std():.2f}")
print(f"  Contrast:  {residual[gt_center].mean()-residual[bg_zone].mean():.2f}")

# ---- Polar transform ----
print("\n=== Polar Transform ===")
r_inner = int(r * 0.85)   # 166px
r_outer = int(r * 2.5)    # 488px
Nr = r_outer - r_inner
Ntheta = int(2 * np.pi * r_outer)   # angular circumference at outer radius
print(f"Polar grid: {Nr} radial x {Ntheta} angular pixels")

r_vals = np.linspace(r_inner, r_outer, Nr)
theta_vals = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
R_grid, T_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')
X_sample = cx + R_grid * np.cos(T_grid)
Y_sample = cy + R_grid * np.sin(T_grid)

# Sample image and residual in polar space
def sample_polar(img, X, Y, h, w):
    xs = np.clip(X.ravel(), 0, w-1)
    ys = np.clip(Y.ravel(), 0, h-1)
    return map_coordinates(img, [ys, xs], order=1, mode='nearest').reshape(img.shape[0] if img.ndim==2 else img.shape[0], *X.shape).reshape(X.shape)

polar_raw = map_coordinates(raw, [Y_sample.ravel().clip(0,h-1), X_sample.ravel().clip(0,w-1)], order=1).reshape(Nr, Ntheta)
polar_residual = map_coordinates(residual, [Y_sample.ravel().clip(0,h-1), X_sample.ravel().clip(0,w-1)], order=1).reshape(Nr, Ntheta)

# Also map the GT mask to polar
polar_gt = map_coordinates(gt_mask.astype(float), [Y_sample.ravel().clip(0,h-1), X_sample.ravel().clip(0,w-1)], order=0).reshape(Nr, Ntheta) > 0.5

print(f"GT pixels in polar space: {polar_gt.sum()} (original: {gt_center.sum()})")

# NMS in angular direction (find local dark minima in theta direction for each r)
# A fibril appears as local dark minimum in angular direction at each radius
def angular_nms(polar_img, half_width):
    """Find local minima in angular dimension."""
    # Compare with rolling windows of ±half_width
    pad = np.pad(polar_img, [(0,0), (half_width, half_width)], mode='wrap')
    is_min = polar_img < polar_img.mean()
    for d in range(1, half_width+1):
        shifted_left = pad[:, half_width-d: half_width-d+Ntheta]
        shifted_right = pad[:, half_width+d: half_width+d+Ntheta]
        is_min &= (polar_img <= shifted_left) & (polar_img <= shifted_right)
    return is_min

# Test residual (background subtracted) in polar space
# Large residual = darker than background = fibril candidate
for threshold in [5, 8, 10, 15, 20]:
    pred_polar = polar_residual > threshold
    # Convert back to Cartesian roughly: count overlap with GT
    # Map GT locations to polar... already done above
    tp = (pred_polar & polar_gt).sum()
    fp_count = (pred_polar & ~polar_gt).sum()
    fn = (polar_gt & ~pred_polar).sum()
    iou = tp / (tp + fp_count + fn) if (tp+fp_count+fn) > 0 else 0
    recall = tp / polar_gt.sum() if polar_gt.sum() > 0 else 0
    print(f"  residual > {threshold}: pred={pred_polar.sum()} TP={tp} recall={recall:.3f} IoU={iou:.4f}")

# NMS in angular direction on raw (polar)
print("\nAngular NMS on polar raw image:")
for half_w in [15, 25, 40]:
    nms = angular_nms(polar_raw, half_w)
    dark = polar_raw < np.percentile(polar_raw, 40)  # darkest 40%
    combined = nms & dark
    tp = (combined & polar_gt).sum()
    fp_count = (combined & ~polar_gt).sum()
    fn = (polar_gt & ~combined).sum()
    iou = tp / (tp + fp_count + fn) if (tp+fp_count+fn) > 0 else 0
    recall = tp / polar_gt.sum() if polar_gt.sum() > 0 else 0
    print(f"  half_w={half_w}: pred={combined.sum()} recall={recall:.3f} IoU={iou:.4f}")

# Save debug: polar raw and residual images
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)
polar_vis = ((polar_raw - polar_raw.min()) / (polar_raw.max() - polar_raw.min()) * 255).astype(np.uint8)
Image.fromarray(polar_vis).save(str(out_dir / "debug_polar_raw.png"))
polar_res_vis = np.clip(polar_residual * 4 + 128, 0, 255).astype(np.uint8)
Image.fromarray(polar_res_vis).save(str(out_dir / "debug_polar_residual.png"))
print(f"\nSaved polar images to output/debug_polar_*.png")
