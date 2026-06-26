"""report_table.py — render the preprocessing experiment report as a ranked
table and a side-by-side overlay montage for visual tie-breaking.

Usage:
    python scripts/report_table.py --report output/experiments/report_classic.json \
        --overlays output/experiments --montage output/experiments/montage.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_COLS = [
    ("score", "score", 7),
    ("f1", "f1", 6),
    ("recall", "rec", 6),
    ("precision", "prec", 6),
    ("capsid_fp", "capFP", 6),
    ("n_pred", "n_pred", 7),
    ("frag_ratio", "frag", 6),
    ("comps_mean", "cmp/sec", 8),
    ("mean_len_nm", "len_nm", 7),
    ("iou", "iou", 6),
]


def _recompute_scores(report: dict) -> None:
    """Recompute the composite score from stored raw metrics, so changes to the
    weights / target length in metrics.py take effect without re-running the
    (expensive) experiment."""
    import scripts.metrics as M

    for data in report["variants"].values():
        m = data["mean"]
        m["score"] = round(
            M.composite_score(m["f1"], m["recall"], m["capsid_fp"], m["mean_len_nm"]), 4
        )
        for pi in data["per_image"].values():
            pi["score"] = round(
                M.composite_score(pi["f1"], pi["recall"], pi["capsid_fp"], pi["mean_len_nm"]), 4
            )


def print_table(report: dict) -> list[str]:
    _recompute_scores(report)
    variants = report["variants"]
    ranked = sorted(variants.items(), key=lambda kv: kv[1]["mean"]["score"], reverse=True)

    header = f"{'variant':24s}" + "".join(f"{lbl:>{w}s}" for _, lbl, w in _COLS)
    print(header)
    print("-" * len(header))
    for name, data in ranked:
        m = data["mean"]
        row = f"{name:24s}"
        for key, _, w in _COLS:
            v = m.get(key, 0)
            cell = f"{v:.3f}" if isinstance(v, float) else str(v)
            row += f"{cell:>{w}s}"
        print(row)

    base = variants.get("production", {}).get("mean", {})
    if base:
        print(f"\nBaseline (production): score={base['score']:.3f} f1={base['f1']:.3f} "
              f"recall={base['recall']:.3f} capsid_fp={base['capsid_fp']:.3f} "
              f"len={base['mean_len_nm']:.0f}nm")
        best_name, best = ranked[0]
        if best_name != "production":
            print(f"Winner ({best_name}): score={best['mean']['score']:.3f} "
                  f"(+{best['mean']['score'] - base['score']:.3f} vs baseline)")
    return [n for n, _ in ranked]


def build_montage(report: dict, overlays_dir: Path, montage_path: Path,
                  order: list[str], thumb_w: int = 320) -> None:
    """Grid: one row per variant (ranked), one column per image."""
    variants = report["variants"]
    image_stems = sorted(next(iter(variants.values()))["per_image"].keys())
    if not image_stems:
        print("No per-image data for montage.")
        return

    # thumbnail height from first available overlay
    sample = None
    for v in order:
        for s in image_stems:
            p = overlays_dir / v / f"{s}.png"
            if p.exists():
                sample = Image.open(p)
                break
        if sample:
            break
    if sample is None:
        print("No overlay images found for montage.")
        return
    ratio = sample.height / sample.width
    thumb_h = int(thumb_w * ratio)
    label_w = 170
    pad = 4

    n_rows, n_cols = len(order), len(image_stems)
    W = label_w + n_cols * (thumb_w + pad)
    H = 24 + n_rows * (thumb_h + pad)
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for ci, s in enumerate(image_stems):
        draw.text((label_w + ci * (thumb_w + pad) + 4, 6), s, fill=(230, 230, 230))

    for ri, vname in enumerate(order):
        y = 24 + ri * (thumb_h + pad)
        m = variants[vname]["mean"]
        draw.text((4, y + 4), vname, fill=(255, 220, 120))
        draw.text((4, y + 20), f"sc={m['score']:.2f}", fill=(200, 200, 200))
        draw.text((4, y + 36), f"f1={m['f1']:.2f}", fill=(160, 200, 255))
        draw.text((4, y + 52), f"cFP={m['capsid_fp']:.2f}", fill=(255, 160, 160))
        for ci, s in enumerate(image_stems):
            p = overlays_dir / vname / f"{s}.png"
            if not p.exists():
                continue
            thumb = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.BILINEAR)
            canvas.paste(thumb, (label_w + ci * (thumb_w + pad), y))

    montage_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(montage_path))
    print(f"\nMontage saved to {montage_path}  ({W}x{H})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="output/experiments/report_classic.json")
    p.add_argument("--overlays", default="output/experiments")
    p.add_argument("--montage", default="output/experiments/montage.png")
    p.add_argument("--no-montage", action="store_true")
    args = p.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    order = print_table(report)
    if not args.no_montage:
        build_montage(report, Path(args.overlays), Path(args.montage), order)


if __name__ == "__main__":
    main()
