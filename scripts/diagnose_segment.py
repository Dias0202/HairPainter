"""Diagnostic for v2.3: where are fibrils being lost?"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from hairpainter.services.segment.segment_service import SegmentService

_STRUCT_8 = np.ones((3, 3), dtype=int)

img_path = ROOT / "Data" / "Raw" / "imagemL1.tif"
if not img_path.exists():
    tifs = sorted((ROOT / "Data" / "Raw").glob("*.tif"))
    img_path = tifs[0]

print(f"=== Diagnosing {img_path.name} ===")
image_data = IOService().load(img_path)
enhanced = PreprocessService().enhance(image_data)
capsid = CapsidService().detect(enhanced)
print(f"Capsid: center={capsid.center}, r={capsid.radius}")

h, w = enhanced.shape
cx, cy = capsid.center
r = capsid.radius
scale_bar_y = int(h * 0.95)

y_g, x_g = np.ogrid[:h, :w]
dist_from_center = np.sqrt((x_g - cx)**2 + (y_g - cy)**2).astype(np.float32)
fibril_zone = (dist_from_center >= r*0.85) & (dist_from_center <= r*2.0)
fibril_zone[scale_bar_y:] = False
fibril_zone[capsid.mask] = False
dist_from_surface = np.abs(dist_from_center - r)

print(f"Fibril zone pixels: {fibril_zone.sum()}")

# Run Frangi (natural, v2.3)
working = enhanced.astype(np.float32) / 255.0
v = frangi(working, sigmas=(3, 5, 8), black_ridges=True)
vmax = v.max()
print(f"\nFrangi(3,5,8) raw: max={vmax:.6f}")
if vmax > 0:
    v = v / vmax

zone_v = v[fibril_zone]
for thr in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    n = (zone_v >= thr).sum()
    print(f"  zone pixels >= {thr:.2f}: {n:6d}  ({100*n/fibril_zone.sum():.1f}%)")

# Choose threshold 0.05 (CLI default)
for thr in [0.05]:
    binary = (v >= thr).astype(bool)
    binary[capsid.mask] = False
    binary[~fibril_zone] = False
    binary[scale_bar_y:] = False
    binary = remove_small_objects(binary, max_size=7)
    print(f"\nWith thr={thr}: binary pixels={binary.sum()}")

    skeleton = skeletonize(binary)
    skel_labeled, n_raw = ndi.label(skeleton, structure=_STRUCT_8)
    print(f"  n_raw components: {n_raw}")

    sizes = np.bincount(skel_labeled.ravel()); sizes[0] = 0
    for min_px in [5, 10, 15, 20, 30]:
        n_sz = (sizes >= min_px).sum()
        print(f"  After size>={min_px}px: {n_sz}")

    print(f"\n  Anchoring breakdown (after size>=15):")
    for band in [15, 25, 40, 60, 80, 100, 200]:
        n_anch = 0
        for lbl in range(1, n_raw+1):
            if sizes[lbl] < 15: continue
            skel_i = skel_labeled == lbl
            if dist_from_surface[skel_i].min() <= band:
                n_anch += 1
        print(f"    anchor_band<={band:3d}px: {n_anch}")

    # Size distribution of raw components
    valid_sizes = sizes[sizes >= 5]
    if len(valid_sizes) > 0:
        print(f"\n  Size distribution (>=5px) of {len(valid_sizes)} components:")
        for pct in [25, 50, 75, 90, 99]:
            print(f"    p{pct}: {np.percentile(valid_sizes, pct):.0f}px")
        print(f"    max: {valid_sizes.max()}px")

    # Where are the large components?
    large_labels = np.where(sizes >= 15)[0]
    print(f"\n  Large components (>=15px): {len(large_labels)}")
    if len(large_labels) <= 20:
        for lbl in large_labels:
            skel_i = skel_labeled == lbl
            min_d = dist_from_surface[skel_i].min()
            mean_d = dist_from_surface[skel_i].mean()
            print(f"    label={lbl}: size={sizes[lbl]}px  min_dist_surf={min_d:.1f}  mean_dist={mean_d:.1f}")
    else:
        dists = []
        for lbl in large_labels:
            skel_i = skel_labeled == lbl
            dists.append(dist_from_surface[skel_i].min())
        dists = np.array(dists)
        print(f"    min_dist_surf distribution: p25={np.percentile(dists,25):.1f}  p50={np.percentile(dists,50):.1f}  p75={np.percentile(dists,75):.1f}  max={dists.max():.1f}")
        for band in [15, 25, 40, 60, 80, 100]:
            print(f"    within {band}px of surface: {(dists <= band).sum()}")

print("\n=== Current SegmentService result ===")
svc = SegmentService()
result = svc.segment(enhanced, capsid)
print(f"fibrils={result.n_fibrils}, sigmas={svc._sigmas}, threshold={svc._threshold}, anchor={svc._anchor_band_px}")
