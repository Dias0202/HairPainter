"""Quick pipeline check — runs on imagemL1.tif and reports stats at each stage."""
from pathlib import Path
import numpy as np

from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService
from hairpainter.services.segment.segment_service import SegmentService
from hairpainter.services.scale.scale_service import ScaleService

img_path = Path("Data/Raw/imagemL1.tif")

print("=== Hair Painter Quick Diagnostic ===\n")

# IO
io = IOService()
img = io.load(img_path)
print(f"[IO] Loaded: {img.array.shape}, dtype={img.array.dtype}")
print(f"     metadata keys: {list(img.metadata.keys())}")

# Preprocess
pre = PreprocessService()
enhanced = pre.enhance(img)
print(f"\n[PRE] Enhanced: min={enhanced.min()}, max={enhanced.max()}, mean={enhanced.mean():.1f}, std={enhanced.std():.1f}")

# Scale
scale_svc = ScaleService()
scale = scale_svc.detect(img)
print(f"\n[SCALE] source={scale.source}, text={scale.scale_text!r}, px_per_nm={scale.px_per_nm:.4f}, confidence={scale.confidence:.2f}")

# Capsid
cap_svc = CapsidService()
capsid = cap_svc.detect(enhanced)
print(f"\n[CAPSID] center={capsid.center}, radius={capsid.radius}px")
h, w = enhanced.shape
print(f"         mask covers {capsid.mask.sum() / capsid.mask.size * 100:.1f}% of image")

# Frangi + Segment
seg_svc = SegmentService(frangi_threshold=0.05)
from skimage.filters import frangi
working = enhanced.copy().astype(np.float32) / 255.0
working[capsid.mask] = 0.0
v = frangi(working, sigmas=[1,2,3,5], black_ridges=True)
v_max = v.max()
if v_max > 0:
    vn = v / v_max
    print(f"\n[FRANGI black_ridges=True] max={v_max:.4f}")
    for t in [0.01, 0.05, 0.1, 0.2]:
        px = (vn >= t).sum()
        print(f"  thresh={t}: {px:7d} px ({px/v.size*100:.2f}%)")

print("\n[SEGMENT] Running full segmentation (may take ~30s)...")
segment = seg_svc.segment(enhanced, capsid)
print(f"  n_fibrils={segment.n_fibrils}")
if segment.fibrils:
    lengths = [f.length_px for f in segment.fibrils]
    print(f"  skeleton length_px: min={min(lengths):.0f} mean={np.mean(lengths):.0f} max={max(lengths):.0f}")

# Measure
if segment.fibrils and scale.px_per_nm > 0:
    from hairpainter.services.measure.measure_service import MeasureService
    m = MeasureService().measure(segment, scale)
    print(f"\n[MEASURE] min={m.min_nm:.1f}nm mean={m.mean_nm:.1f}nm max={m.max_nm:.1f}nm")

print("\nDone.")
