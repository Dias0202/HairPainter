"""
svg_to_mask.py — Convert ground-truth SVG fibril paths to binary masks
and compute IoU / count / length metrics against model predictions.

Usage:
    python scripts/svg_to_mask.py \\
        --svg Data/Manual_paint/ \\
        --pred output/ \\
        --report validation.json

SVG coordinate space: viewport 1280×720
Target image space: 1376×1070
"""
from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Coordinate mapping: SVG viewport → image pixel space
#
# Empirically confirmed (see scripts/verify_svg_mapping.py):
# The drawing tool displays the image at its NATIVE PIXEL SIZE within the SVG
# canvas, without any scaling. The SVG viewport (1280×720) simply clips the
# top-left portion of the full image (1376×1070).
#
# Therefore the mapping is 1:1:
#   image_x = svg_x
#   image_y = svg_y
#
# The right-most 96px (x=1280..1376) and bottom 350px (y=720..1070) of the
# image fall outside the SVG viewport and were not annotated in the SVGs.
SVG_W, SVG_H = 1280.0, 720.0
IMG_W, IMG_H = 1376, 1070

SCALE_X = 1.0
SCALE_Y = 1.0
_X_OFFSET = 0.0


def _svg_path_to_points(d: str, steps: int = 50) -> list[tuple[float, float]]:
    """
    Parse SVG path 'd' attribute and sample points.
    Supports: M/m, L/l, C/c (cubic bezier), Z/z.
    Returns list of (x, y) in SVG coordinate space.
    """
    points: list[tuple[float, float]] = []
    tokens = re.findall(r"[MmLlCcZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)
    i = 0
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0  # start of current subpath

    while i < len(tokens):
        cmd = tokens[i]
        i += 1

        if cmd in ("M", "m"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "m":
                x, y = cx + x, cy + y
            cx, cy = x, y
            sx, sy = cx, cy
            points.append((cx, cy))

        elif cmd in ("L", "l"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            if cmd == "l":
                x, y = cx + x, cy + y
            cx, cy = x, y
            points.append((cx, cy))

        elif cmd in ("C", "c"):
            # Cubic bezier: (x1,y1) (x2,y2) (x,y)
            coords = [float(tokens[i + j]) for j in range(6)]
            i += 6
            if cmd == "c":
                coords = [
                    cx + coords[0], cy + coords[1],
                    cx + coords[2], cy + coords[3],
                    cx + coords[4], cy + coords[5],
                ]
            x1, y1, x2, y2, x, y = coords
            # Sample bezier
            for t in np.linspace(0, 1, steps):
                bx = (1 - t) ** 3 * cx + 3 * (1 - t) ** 2 * t * x1 + \
                     3 * (1 - t) * t ** 2 * x2 + t ** 3 * x
                by = (1 - t) ** 3 * cy + 3 * (1 - t) ** 2 * t * y1 + \
                     3 * (1 - t) * t ** 2 * y2 + t ** 3 * y
                points.append((bx, by))
            cx, cy = x, y

        elif cmd in ("Z", "z"):
            points.append((sx, sy))
            cx, cy = sx, sy

    return points


def svg_to_mask(svg_path: Path, out_w: int = IMG_W, out_h: int = IMG_H) -> np.ndarray:
    """
    Rasterize all fibril paths in an SVG file to a binary mask.
    Returns bool array of shape (out_h, out_w).
    """
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    mask = Image.new("L", (out_w, out_h), 0)
    draw = ImageDraw.Draw(mask)

    for path_el in root.findall(".//svg:path", ns):
        d = path_el.get("d", "")
        if not d or d.strip().startswith("M0"):
            continue  # skip background rectangle
        pts = _svg_path_to_points(d)
        if len(pts) < 2:
            continue
        # Scale from SVG viewport to image pixel coords
        # Apply horizontal offset (letterbox) then uniform scale
        scaled = [((x - _X_OFFSET) * SCALE_X, y * SCALE_Y) for x, y in pts]
        # Draw polyline with stroke_width≈2 to match visual thickness
        for j in range(len(scaled) - 1):
            x0, y0 = scaled[j]
            x1, y1 = scaled[j + 1]
            draw.line([(x0, y0), (x1, y1)], fill=255, width=2)

    return np.array(mask, dtype=bool)


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def load_prediction_mask(pred_dir: Path, stem: str) -> np.ndarray | None:
    """Load predicted fibril mask from Deliverable 1 image (black bg = no fibril)."""
    # Map SVG name (imagem1_fios) → prediction stem (imagemL1)
    number = re.search(r"\d+", stem)
    if not number:
        return None
    n = number.group()
    candidate = pred_dir / f"imagemL{n}" / f"imagemL{n}_fibrils_only.png"
    if not candidate.exists():
        return None
    img = Image.open(str(candidate)).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return arr > 10  # threshold: any non-black pixel = fibril


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Hair Painter vs. SVG ground truth")
    parser.add_argument("--svg", required=True, help="Folder with SVG ground truth files")
    parser.add_argument("--pred", required=True, help="Folder with model output")
    parser.add_argument("--report", default="validation.json", help="Output report path")
    args = parser.parse_args()

    svg_dir = Path(args.svg)
    pred_dir = Path(args.pred)
    report_path = Path(args.report)

    results = []
    for svg_file in sorted(svg_dir.glob("*.svg")):
        print(f"Processing {svg_file.name}...")
        gt_mask = svg_to_mask(svg_file)

        pred_mask = load_prediction_mask(pred_dir, svg_file.stem)
        if pred_mask is None:
            print(f"  [WARN] No prediction found for {svg_file.stem}")
            results.append({"svg": svg_file.name, "iou": None, "status": "no_prediction"})
            continue

        # Ensure same shape
        if pred_mask.shape != gt_mask.shape:
            pred_img = Image.fromarray(pred_mask.astype(np.uint8) * 255)
            pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
            pred_mask = np.array(pred_img) > 127

        iou = compute_iou(pred_mask, gt_mask)
        gt_fibril_px = gt_mask.sum()
        pred_fibril_px = pred_mask.sum()

        result = {
            "svg": svg_file.name,
            "iou": round(iou, 4),
            "gt_fibril_pixels": int(gt_fibril_px),
            "pred_fibril_pixels": int(pred_fibril_px),
            "pixel_coverage_ratio": round(pred_fibril_px / gt_fibril_px, 3) if gt_fibril_px > 0 else None,
            "status": "ok" if iou >= 0.70 else "below_threshold",
        }
        results.append(result)
        status_char = "OK" if iou >= 0.70 else "!!"
        print(f"  IoU: {iou:.4f} [{status_char}]")

    # Summary
    valid_ious = [r["iou"] for r in results if r["iou"] is not None]
    summary = {
        "n_images": len(results),
        "n_evaluated": len(valid_ious),
        "mean_iou": round(float(np.mean(valid_ious)), 4) if valid_ious else None,
        "images_above_070": sum(1 for v in valid_ious if v >= 0.70),
        "target_met": sum(1 for v in valid_ious if v >= 0.70) >= 3,
    }

    report = {"summary": summary, "per_image": results}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nValidation report saved to: {report_path}")
    print(f"Mean IoU: {summary['mean_iou']}")
    print(f"Images >= 0.70 IoU: {summary['images_above_070']}/5 -- Target met: {summary['target_met']}")


if __name__ == "__main__":
    main()
