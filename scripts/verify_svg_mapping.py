"""
Verify the correct SVG → image coordinate mapping by checking if fibril
start points land on the capsid surface (where they should begin anatomically).
"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from hairpainter.services.io.io_service import IOService
from hairpainter.services.preprocess.preprocess_service import PreprocessService
from hairpainter.services.capsid.capsid_service import CapsidService

# Known capsid from pipeline
io = IOService()
pre = PreprocessService()
img = io.load(Path("Data/Raw/imagemL1.tif"))
enhanced = pre.enhance(img)
capsid = CapsidService().detect(enhanced)
cx, cy = capsid.center
r = capsid.radius
print(f"Capsid: center=({cx},{cy}), radius={r}px\n")

# Extract first 20 fibril start points from SVG
tree = ET.parse("Data/Manual_paint/imagem1_fios.svg")
root = tree.getroot()
ns = {"svg": "http://www.w3.org/2000/svg"}

starts = []
for p in root.findall(".//svg:path", ns):
    stroke = p.get("stroke", "")
    if stroke != "#1c3052":
        continue
    d = p.get("d", "")
    m = re.match(r"m\s*([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)", d)
    if m:
        starts.append((float(m.group(1)), float(m.group(2))))
    if len(starts) >= 30:
        break

print("Testing 3 coordinate mapping hypotheses:\n")
print(f"  SVG viewport: 1280x720   Image: 1376x1070")

IMG_W, IMG_H = 1376, 1070
SVG_W, SVG_H = 1280.0, 720.0

for name, fn in [
    ("1:1 (no scaling)",
     lambda sx, sy: (sx, sy)),
    ("fit-height + letterbox (current)",
     lambda sx, sy: ((sx - (SVG_W - IMG_W * (SVG_H/IMG_H))/2) * (IMG_H/SVG_H), sy * (IMG_H/SVG_H))),
    ("non-uniform stretch",
     lambda sx, sy: (sx * IMG_W/SVG_W, sy * IMG_H/SVG_H)),
]:
    dists = []
    for sx, sy in starts:
        ix, iy = fn(sx, sy)
        d = np.sqrt((ix - cx)**2 + (iy - cy)**2)
        dists.append(d)
    d_arr = np.array(dists)
    in_range = ((d_arr >= r * 0.7) & (d_arr <= r * 1.5)).sum()
    print(f"\n  Hypothesis: {name}")
    print(f"    Distance to capsid center: min={d_arr.min():.0f} mean={d_arr.mean():.0f} max={d_arr.max():.0f}")
    print(f"    Expected: r={r}px.  Starts within [0.7r, 1.5r]: {in_range}/{len(dists)}")
