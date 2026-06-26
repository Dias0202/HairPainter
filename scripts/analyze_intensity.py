"""Check raw pixel intensity at GT fibril locations vs background."""
from pathlib import Path
import numpy as np
import cv2
from skimage.filters import frangi

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from scripts.svg_to_mask import svg_to_mask, IMG_W, IMG_H

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img_data = io.load(Path("Data/Raw/imagemL1.tif"))
raw_gray = img_data.array          # uint8, original
enhanced = pre.enhance(img_data)   # CLAHE enhanced
h, w = raw_gray.shape

capsid = cap_svc.detect(enhanced)
cx, cy = capsid.center
r = capsid.radius

# Fibril zone
y_grid, x_grid = np.ogrid[:h, :w]
dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
fibril_zone = (dist >= r * 0.85) & (dist <= r * 2.5)
fibril_zone[int(h*0.95):, :] = False
fibril_zone[capsid.mask] = False

# GT mask (dilated slightly to get full fibril width)
gt_mask = svg_to_mask(Path("Data/Manual_paint/imagem1_fios.svg"), out_w=IMG_W, out_h=IMG_H)
gt_dilated = cv2.dilate(gt_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)

gt_in_zone = gt_dilated & fibril_zone
bg_in_zone = fibril_zone & ~gt_dilated

print("=== Raw pixel intensity (pre-CLAHE) ===")
print(f"GT (dilated 5px) pixels: {gt_in_zone.sum()}")
print(f"Background pixels: {bg_in_zone.sum()}")
v_gt_raw = raw_gray[gt_in_zone].astype(float)
v_bg_raw = raw_gray[bg_in_zone].astype(float)
print(f"GT intensity:  mean={v_gt_raw.mean():.1f} std={v_gt_raw.std():.1f}")
print(f"BG intensity:  mean={v_bg_raw.mean():.1f} std={v_bg_raw.std():.1f}")
print(f"Contrast (BG-GT): {v_bg_raw.mean()-v_gt_raw.mean():.1f} (positive=GT is darker)")

print("\n=== Enhanced (CLAHE) pixel intensity ===")
v_gt_enh = enhanced[gt_in_zone].astype(float)
v_bg_enh = enhanced[bg_in_zone].astype(float)
print(f"GT intensity:  mean={v_gt_enh.mean():.1f} std={v_gt_enh.std():.1f}")
print(f"BG intensity:  mean={v_bg_enh.mean():.1f} std={v_bg_enh.std():.1f}")
print(f"Contrast (BG-GT): {v_bg_enh.mean()-v_gt_enh.mean():.1f} (positive=GT is darker)")

# Frangi on RAW image vs ENHANCED
print("\n=== Frangi on RAW image ===")
working_raw = raw_gray.astype(np.float32) / 255.0
working_raw[capsid.mask] = 0.0

# Test both polarities
for polarity, br in [("black_ridges=True", True), ("black_ridges=False", False)]:
    for src_name, src in [("raw", working_raw), ("enhanced", enhanced.astype(np.float32)/255.0)]:
        v = frangi(src, sigmas=[1,2,3,5], black_ridges=br)
        if v.max() > 0: v /= v.max()
        v_gt_v = v[gt_in_zone]
        v_bg_v = v[bg_in_zone]
        snr = (v_gt_v.mean() - v_bg_v.mean()) / (v_bg_v.std() + 1e-9)
        print(f"  {src_name} {polarity}: GT_mean={v_gt_v.mean():.4f} BG_mean={v_bg_v.mean():.4f} SNR={snr:.2f}")

# Try different sigma scales — fibrils may be wider than expected
print("\n=== Frangi on enhanced, black_ridges=True, different sigmas ===")
working_enh = enhanced.astype(np.float32) / 255.0
working_enh[capsid.mask] = 0.0
for sigmas in [(1,2,3), (2,4,6), (3,5,8), (5,8,12), (1,2,3,5,8)]:
    v = frangi(working_enh, sigmas=sigmas, black_ridges=True)
    if v.max() > 0: v /= v.max()
    v_gt_v = v[gt_in_zone]
    v_bg_v = v[bg_in_zone]
    snr = (v_gt_v.mean() - v_bg_v.mean()) / (v_bg_v.std() + 1e-9)
    print(f"  sigmas={sigmas}: GT_mean={v_gt_v.mean():.4f} BG_mean={v_bg_v.mean():.4f} SNR={snr:.3f}")
