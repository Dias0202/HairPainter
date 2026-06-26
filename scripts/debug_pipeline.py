"""Debug pipeline - check segment service params and actual output."""
from pathlib import Path
import numpy as np
from hairpainter.orchestrator.pipeline import Pipeline, PipelineConfig
from hairpainter.utils.types import PipelineInput

cfg = PipelineConfig(frangi_threshold=0.05, min_fibril_px=15)
pipe = Pipeline(cfg)

print("SegmentService params:")
print(f"  sigmas: {pipe._segment._sigmas}")
print(f"  threshold: {pipe._segment._threshold}")
print(f"  min_px: {pipe._segment._min_px}")
print(f"  dilation_radius: {pipe._segment._dilation_radius}")

# Run segment manually to trace what happens
from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from scipy import ndimage as ndi
from skimage.filters import frangi
from skimage.morphology import remove_small_objects, skeletonize

io = IOService()
pre = PreprocessService()
cap_svc = CapsidService()

img_data = io.load(Path("Data/Raw/imagemL1.tif"))
enhanced = pre.enhance(img_data)
h, w = enhanced.shape
capsid = cap_svc.detect(enhanced)
cx, cy = capsid.center
r = capsid.radius
print(f"\nCapsid: center=({cx},{cy}), r={r}")

# Mirror the segment service logic
working = enhanced.copy().astype(np.float32) / 255.0
scale_bar_y = int(h * 0.95)
working[capsid.mask] = 0.0
working[scale_bar_y:, :] = 0.0

y_grid, x_grid = np.ogrid[:h, :w]
dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
fibril_zone = (dist >= r * 0.90) & (dist <= r * 2.0)
fibril_zone[scale_bar_y:, :] = False
print(f"Fibril zone pixels: {fibril_zone.sum()} ({fibril_zone.sum()/(h*w)*100:.1f}%)")

vesselness = frangi(working, sigmas=pipe._segment._sigmas, black_ridges=True)
v_max = vesselness.max()
print(f"Frangi v_max: {v_max:.6f}")
if v_max > 0:
    vesselness /= v_max

binary = (vesselness >= pipe._segment._threshold).astype(bool)
binary[capsid.mask] = False
binary[~fibril_zone] = False
binary[scale_bar_y:, :] = False
print(f"Binary pixels in zone: {binary.sum()}")

binary_clean = remove_small_objects(binary, max_size=7)
print(f"After remove_small (max_size=7): {binary_clean.sum()}")

skel = skeletonize(binary_clean)
print(f"Skeleton pixels: {skel.sum()}")

struct8 = np.ones((3,3), dtype=int)
labeled, n_raw = ndi.label(skel, structure=struct8)
print(f"Raw components: {n_raw}")

sizes = np.bincount(labeled.ravel())
sizes[0] = 0
print(f"Sizes: min={sizes[sizes>0].min() if any(sizes>0) else 0} max={sizes.max()} mean={sizes[sizes>0].mean():.1f}")
for min_len in [5, 10, 15, 20, 30]:
    count = (sizes >= min_len).sum()
    print(f"  >= {min_len}: {count}")
