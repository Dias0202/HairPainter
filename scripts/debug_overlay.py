"""
Render diagnostic overlay:
  - GT fibril paths in RED
  - Predicted fibril mask in BLUE
  - Overlap in WHITE
  - Raw image as background (50% alpha)
Saves to output/debug_alignment.png
"""
from pathlib import Path
import numpy as np
from PIL import Image

from scripts.svg_to_mask import svg_to_mask, IMG_W, IMG_H
from scripts.svg_to_mask import load_prediction_mask

svg_path = Path("Data/Manual_paint/imagem1_fios.svg")
raw_path = Path("Data/Raw/imagemL1.tif")
out_path = Path("output/debug_alignment.png")

# Load raw image as RGB
from hairpainter.services.io.io_service import IOService
io = IOService()
image_data = io.load(raw_path)
gray = image_data.array  # uint8 2D
raw_rgb = Image.fromarray(np.stack([gray, gray, gray], axis=-1), "RGB")
raw_rgb = raw_rgb.resize((IMG_W, IMG_H), Image.BILINEAR)

# Load masks
gt_mask = svg_to_mask(svg_path, out_w=IMG_W, out_h=IMG_H)
pred_mask = load_prediction_mask(Path("output"), "imagem1_fios")

print(f"GT pixels: {gt_mask.sum()}")
print(f"Pred pixels: {pred_mask.sum() if pred_mask is not None else 'N/A'}")

# Blend: start with raw at 40% brightness
base = np.array(raw_rgb).astype(np.float32) * 0.4

# GT = RED channel
gt_layer = np.zeros_like(base)
gt_layer[:, :, 0] = gt_mask.astype(np.float32) * 255

# Pred = BLUE channel (only within first 1280×720 to compare fairly)
pred_layer = np.zeros_like(base)
if pred_mask is not None:
    pred_layer[:, :, 2] = pred_mask.astype(np.float32) * 255

# Overlap = WHITE
overlap_mask = gt_mask & (pred_mask if pred_mask is not None else np.zeros_like(gt_mask))
overlap_layer = np.zeros_like(base)
overlap_layer[overlap_mask] = [255, 255, 255]

# Combine
composite = np.clip(base + gt_layer + pred_layer + overlap_layer, 0, 255).astype(np.uint8)

out_path.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(composite, "RGB").save(str(out_path))
print(f"Saved overlay to {out_path}")
print("Legend: RED=GT, BLUE=Pred, WHITE=overlap, dark=raw image")

# Also save a zoom into the top-left 640x640 patch
zoom = composite[:640, :640]
Image.fromarray(zoom, "RGB").save(str(out_path).replace(".png", "_zoom.png"))
print(f"Saved zoom to {str(out_path).replace('.png', '_zoom.png')}")
