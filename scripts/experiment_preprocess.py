"""experiment_preprocess.py — run every preprocessing variant through the same
segmentation and score the result against ground truth.

For each image the capsid is detected ONCE on the production-enhanced image and
shared across all variants, so the experiment isolates the effect of the image
treatment on SEGMENTATION (and keeps the capsid_fp metric comparable).

Usage:
    python scripts/experiment_preprocess.py \
        --variants all \
        --images Data/Raw --gt Data/Manual_paint \
        --segment classic \
        --out output/experiments \
        --report output/experiments/report.json

    # quick subset
    python scripts/experiment_preprocess.py --variants production,bg_subtract_gauss,dog_bandpass
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.metrics as M  # noqa: E402
from scripts.preprocess_variants import VARIANTS, list_variants  # noqa: E402
from scripts.svg_to_mask import svg_to_mask  # noqa: E402

from hairpainter.services.capsid.capsid_service import CapsidService  # noqa: E402
from hairpainter.services.io.io_service import IOService  # noqa: E402
from hairpainter.services.measure.measure_service import MeasureService  # noqa: E402
from hairpainter.services.scale.scale_service import ScaleService  # noqa: E402
from hairpainter.services.segment.segment_service import SegmentService  # noqa: E402

_FALLBACK_PX_PER_NM = 1.36  # ~136 px / 100 nm at 98,000x (SDD section 2)


def _svg_for_image(gt_dir: Path, stem: str) -> Path | None:
    """Map imagemL{N}.tif -> imagem{N}_fios.svg."""
    m = re.search(r"\d+", stem)
    if not m:
        return None
    cand = gt_dir / f"imagem{m.group()}_fios.svg"
    return cand if cand.exists() else None


def _detect_scale(scale_svc: ScaleService, img) -> float:
    try:
        scale = scale_svc.detect(img)
        if scale.px_per_nm and scale.px_per_nm > 0:
            return float(scale.px_per_nm)
    except Exception as exc:  # noqa: BLE001
        print(f"    [scale fallback: {exc}]")
    return _FALLBACK_PX_PER_NM


def _segment_classic(treated: np.ndarray, capsid, cfg: dict):
    svc = SegmentService(
        frangi_threshold=cfg["frangi_threshold"],
        min_fibril_px=cfg["min_fibril_px"],
        zone_inner_frac=cfg["zone_inner_frac"],
        zone_outer_frac=cfg["zone_outer_frac"],
        capsid_mask_frac=cfg["capsid_mask_frac"],
        extend_inward_to_frac=cfg["extend_inward_to_frac"],
    )
    return svc.segment(treated, capsid)


def _segment_radial(treated: np.ndarray, capsid, cfg: dict):
    from scripts.segment_radial_profile import segment_radial_profile

    return segment_radial_profile(
        treated,
        capsid,
        frangi_threshold=cfg["frangi_threshold"],
        min_fibril_px=cfg["min_fibril_px"],
        inner_frac=cfg["zone_inner_frac"],
        outer_frac=cfg["zone_outer_frac"],
    )


def _save_overlay(
    out_path: Path, raw_gray: np.ndarray, gt: np.ndarray, pred: np.ndarray
) -> None:
    """RED=GT, BLUE=pred, WHITE=overlap, over a dim raw background."""
    h, w = gt.shape
    g = raw_gray
    if g.shape != (h, w):
        g = np.array(Image.fromarray(g).resize((w, h), Image.BILINEAR))
    base = np.stack([g, g, g], axis=-1).astype(np.float32) * 0.4
    base[..., 0] += gt.astype(np.float32) * 255
    base[..., 2] += pred.astype(np.float32) * 255
    overlap = gt & pred
    base[overlap] = [255, 255, 255]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").save(str(out_path))


def run(args: argparse.Namespace) -> None:
    images_dir = Path(args.images)
    gt_dir = Path(args.gt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.variants == "all":
        variant_names = list_variants()
    else:
        variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in variant_names if v not in VARIANTS]
        if unknown:
            raise SystemExit(f"Unknown variants: {unknown}. Available: {list_variants()}")

    seg_fn = _segment_radial if args.segment == "radial_profile" else _segment_classic
    cfg = {
        "frangi_threshold": args.frangi_threshold,
        "min_fibril_px": args.min_fibril_px,
        "zone_inner_frac": args.zone_inner,
        "zone_outer_frac": args.zone_outer,
        "capsid_mask_frac": args.capsid_mask_frac,
        "extend_inward_to_frac": args.extend_inward,
    }

    image_paths = sorted(images_dir.glob("*.tif"))
    print(f"Images: {[p.stem for p in image_paths]}")
    print(f"Variants ({len(variant_names)}): {variant_names}")
    print(f"Segment: {args.segment}  cfg={cfg}\n")

    io = IOService()
    scale_svc = ScaleService()
    capsid_svc = CapsidService()

    # Per-image shared context (capsid, scale, GT) computed once.
    shared: dict[str, dict] = {}
    for path in image_paths:
        img = io.load(path)
        svg = _svg_for_image(gt_dir, path.stem)
        if svg is None:
            print(f"[skip] no GT svg for {path.stem}")
            continue
        gt = svg_to_mask(svg)
        quad = M.annotated_quadrant_mask(gt.shape)
        enhanced_prod = VARIANTS["production"](img.array.copy(), {"image_data": img})
        capsid = capsid_svc.detect(enhanced_prod)
        dist = M.dist_from_center_grid(gt.shape, capsid.center)
        px_per_nm = _detect_scale(scale_svc, img)
        shared[path.stem] = {
            "img": img,
            "gt": gt,
            "quad": quad,
            "capsid": capsid,
            "dist": dist,
            "n_gt": M.n_fibrils_gt(svg),
            "px_per_nm": px_per_nm,
        }
        print(
            f"[{path.stem}] capsid center={capsid.center} r={capsid.radius} "
            f"px/nm={px_per_nm:.3f} n_gt_paths={shared[path.stem]['n_gt']}"
        )

    report: dict = {"weights": M.DEFAULT_WEIGHTS, "config": cfg, "segment": args.segment,
                    "variants": {}}

    for vname in variant_names:
        fn = VARIANTS[vname]
        per_image: dict[str, dict] = {}
        print(f"\n=== variant: {vname} ===")
        for stem, ctx in shared.items():
            t0 = time.time()
            treated = fn(ctx["img"].array.copy(), {"image_data": ctx["img"], "capsid": ctx["capsid"]})
            segment = seg_fn(treated, ctx["capsid"], cfg)
            pred = segment.label_map > 0

            # measure (mean length nm)
            mean_len_nm = 0.0
            if segment.fibrils:
                lengths = np.array([f.length_px for f in segment.fibrils]) / ctx["px_per_nm"]
                mean_len_nm = float(lengths.mean())

            quad = ctx["quad"]
            gt = ctx["gt"]
            f1d = M.f1_tolerance(pred, gt, tol=5, quad=quad)
            capsid_fp = M.capsid_fp_fraction(
                pred, ctx["capsid"].mask, ctx["dist"], ctx["capsid"].radius, quad=quad
            )
            comps = M.components_per_sector(pred, ctx["capsid"].center, ctx["capsid"].radius, quad=quad)
            n_pred = segment.n_fibrils
            score = M.composite_score(f1d["f1"], f1d["recall"], capsid_fp, mean_len_nm)

            per_image[stem] = {
                "iou": round(M.iou(pred, gt, quad), 4),
                "f1": round(f1d["f1"], 4),
                "recall": round(f1d["recall"], 4),
                "precision": round(f1d["precision"], 4),
                "capsid_fp": round(capsid_fp, 4),
                "n_pred": n_pred,
                "n_gt": ctx["n_gt"],
                "frag_ratio": round(M.frag_ratio(n_pred, ctx["n_gt"]), 3),
                "comps_mean": round(comps["mean"], 2),
                "comps_max": comps["max"],
                "mean_len_nm": round(mean_len_nm, 1),
                "score": round(score, 4),
            }
            _save_overlay(out_dir / vname / f"{stem}.png", ctx["img"].array, gt, pred)
            dt = time.time() - t0
            print(
                f"  {stem}: f1={f1d['f1']:.3f} rec={f1d['recall']:.3f} prec={f1d['precision']:.3f} "
                f"capsid_fp={capsid_fp:.3f} n_pred={n_pred} len={mean_len_nm:.0f}nm "
                f"score={score:.3f} ({dt:.0f}s)"
            )

        # aggregate
        keys = ["iou", "f1", "recall", "precision", "capsid_fp", "n_pred",
                "frag_ratio", "comps_mean", "comps_max", "mean_len_nm", "score"]
        mean = {k: round(float(np.mean([per_image[s][k] for s in per_image])), 4) for k in keys}
        report["variants"][vname] = {"per_image": per_image, "mean": mean}
        print(f"  MEAN score={mean['score']:.3f} f1={mean['f1']:.3f} recall={mean['recall']:.3f} "
              f"capsid_fp={mean['capsid_fp']:.3f} n_pred={mean['n_pred']:.0f} len={mean['mean_len_nm']:.0f}nm")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved to {report_path}")

    ranked = sorted(report["variants"].items(), key=lambda kv: kv[1]["mean"]["score"], reverse=True)
    print("\n--- Ranking by composite score ---")
    for name, data in ranked:
        m = data["mean"]
        print(f"  {name:24s} score={m['score']:.3f}  f1={m['f1']:.3f} rec={m['recall']:.3f} "
              f"capsid_fp={m['capsid_fp']:.3f} len={m['mean_len_nm']:.0f}nm")


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocessing experiment harness")
    p.add_argument("--variants", default="all", help="'all' or comma-separated names")
    p.add_argument("--images", default="Data/Raw")
    p.add_argument("--gt", default="Data/Manual_paint")
    p.add_argument("--segment", default="classic", choices=["classic", "radial_profile"])
    p.add_argument("--frangi-threshold", type=float, default=0.05, dest="frangi_threshold")
    p.add_argument("--min-fibril-px", type=int, default=15, dest="min_fibril_px")
    p.add_argument("--zone-inner", type=float, default=0.85, dest="zone_inner")
    p.add_argument("--zone-outer", type=float, default=2.0, dest="zone_outer")
    p.add_argument("--capsid-mask-frac", type=float, default=1.0, dest="capsid_mask_frac")
    p.add_argument("--extend-inward", type=float, default=0.0, dest="extend_inward")
    p.add_argument("--out", default="output/experiments")
    p.add_argument("--report", default="output/experiments/report.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
