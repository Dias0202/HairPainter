"""Diagnose SVG coordinate system and mapping to image space."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

svg_path = Path("Data/Manual_paint/imagem1_fios.svg")
tree = ET.parse(str(svg_path))
root = tree.getroot()
ns = {"svg": "http://www.w3.org/2000/svg"}

# Check the main group transform
groups = root.findall(".//svg:g", ns)
print("Groups found:", len(groups))
for g in groups[:5]:
    print("  <g> attribs:", g.attrib)

# Look at actual fibril paths
paths = root.findall(".//svg:path", ns)
print(f"\nTotal paths: {len(paths)}")
print("\nFirst 5 paths summary:")
for p in paths[:5]:
    d = p.get("d", "")
    fill = p.get("fill", "none")
    stroke = p.get("stroke", "none")
    print(f"  fill={fill!r} stroke={stroke!r}")
    print(f"  d[:120]: {d[:120]}")
    print()

# Find a real fibril path and extract its coordinates
print("\nReal fibril path coordinates:")
for p in paths:
    d = p.get("d", "")
    stroke = p.get("stroke", "")
    if not stroke or stroke in ("none", "#000000"):
        continue
    # This is likely a fibril
    nums = re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", d)
    if len(nums) > 10:
        fns = [float(n) for n in nums[:60]]
        # Try as flat coordinate pairs
        xs = fns[0::2]
        ys = fns[1::2]
        print(f"  stroke={stroke!r}")
        print(f"  X range: {min(xs):.1f} to {max(xs):.1f}")
        print(f"  Y range: {min(ys):.1f} to {max(ys):.1f}")
        print(f"  First M command: {d[:50]}")
        break

# Look at the group with clip-path
for g in groups:
    print("\nGroup attribs:", g.attrib)
    for child in list(g)[:3]:
        print(f"  child tag={child.tag} attribs={child.attrib}")
        d = child.get("d", "")
        if d:
            print(f"    d[:80]={d[:80]}")
