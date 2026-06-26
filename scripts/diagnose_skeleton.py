"""Detailed skeleton analysis to tune branch-point cutting parameters."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import convolve2d
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img = io.load(Path("Data/Raw/imagemL1.tif"))
enhanced = pre.enhance(img)
capsid = cap_svc.detect(enhanced)
h, w = enhanced.shape

working = enhanced.copy().astype(np.float32) / 255.0
working[capsid.mask] = 0.0
working[int(h * 0.95):, :] = 0.0

vesselness = frangi(working, sigmas=[1,2,3,5], black_ridges=True)
v_max = vesselness.max()
if v_max > 0:
    vesselness /= v_max

binary = (vesselness >= 0.05).astype(bool)
binary[capsid.mask] = False
binary[int(h * 0.95):, :] = False
binary = remove_small_objects(binary, min_size=15)
skeleton = skeletonize(binary)

print(f"Binary mask pixels: {binary.sum()}")
print(f"Skeleton pixels: {skeleton.sum()}")

# Branch point analysis
nb_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
nb_count = convolve2d(skeleton.astype(np.int32), nb_kernel, mode="same")
branch_pts = skeleton & (nb_count >= 3)
end_pts = skeleton & (nb_count == 1)
mid_pts = skeleton & (nb_count == 2)

print(f"\nSkeleton topology:")
print(f"  Branch points (>=3 neighbors): {branch_pts.sum()}")
print(f"  End points (1 neighbor):       {end_pts.sum()}")
print(f"  Mid-line points (2 neighbors): {mid_pts.sum()}")
print(f"  Branch-point fraction: {branch_pts.sum()/skeleton.sum()*100:.1f}%")

# After cutting
skeleton_cut = skeleton & ~branch_pts
skel_labeled, n_segs = ndi.label(skeleton_cut, structure=np.ones((3,3)))
sizes = np.bincount(skel_labeled.ravel())
sizes = sizes[1:]  # remove background
print(f"\nAfter branch-point removal:")
print(f"  Connected components: {n_segs}")
if len(sizes) > 0:
    print(f"  Segment lengths: min={sizes.min()} mean={sizes.mean():.1f} max={sizes.max()} median={np.median(sizes):.0f}")
    for min_len in [5, 10, 15, 20, 30, 50, 100]:
        count = (sizes >= min_len).sum()
        print(f"  Components >= {min_len:3d}px: {count:4d}")

# What's the skeleton look like without any branch cutting?
skel_raw_labeled, n_raw = ndi.label(skeleton, structure=np.ones((3,3)))
sizes_raw = np.bincount(skel_raw_labeled.ravel())[1:]
print(f"\nWithout branch cutting:")
print(f"  Connected components: {n_raw}")
if len(sizes_raw) > 0:
    print(f"  Segment lengths: min={sizes_raw.min()} mean={sizes_raw.mean():.1f} max={sizes_raw.max()}")

# Ground truth: expected fibril count from SVGs
print("\nGround truth (from SVGs):")
print("  imagemL1_fios.svg: ~392 fibrils")
print("  Expected length ~100-300nm at ~1.36 px/nm = 136-408px per fibril skeleton")
