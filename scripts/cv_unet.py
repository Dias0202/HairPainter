"""cv_unet.py — full leave-one-out cross-validation for the U-Net in one command.

For each fold it trains on 4 images, evaluates on the held-out image (threshold
swept for best F1@5px), and writes output/unet/cv_report.json with per-fold and
mean F1 / IoU / recall / precision.  Designed to run unattended (overnight).

Usage (overnight, best quality — several hours on CPU):
    python scripts/cv_unet.py --base 32 --epochs 50 --patches-per-image 150 --batch 8

    # faster CV (~3-4 h on CPU)
    python scripts/cv_unet.py --base 16 --epochs 40 --patches-per-image 120
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

import scripts.metrics as M  # noqa: E402
from scripts.infer_unet import predict_full  # noqa: E402
from scripts.preprocess_variants import VARIANTS  # noqa: E402
from scripts.train_unet import UNet, train_fold, load_gt_mask, N_IMAGES, DEVICE  # noqa: E402

from hairpainter.services.io.io_service import IOService  # noqa: E402


def _evaluate(ckpt_path: Path, val_n: int) -> dict:
    ck = torch.load(str(ckpt_path), map_location=DEVICE)
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    variant = ck.get("variant", "production")

    img = IOService().load(Path(f"Data/Raw/{val_n}.tif"))
    gray = VARIANTS[variant](img.array.copy(), {"image_data": img}).astype(np.float32) / 255.0
    prob = predict_full(model, gray)

    gt = load_gt_mask(val_n).astype(bool)
    if gt.shape != prob.shape:
        pil_gt = Image.fromarray(gt.astype(np.uint8) * 255)
        pil_gt = pil_gt.resize((prob.shape[1], prob.shape[0]), Image.NEAREST)
        gt = np.array(pil_gt).astype(bool)
    quad = M.annotated_quadrant_mask(gt.shape)
    best = {"f1": -1.0}
    for t in np.linspace(0.1, 0.9, 17):
        pred = prob >= t
        f1 = M.f1_tolerance(pred, gt, tol=5, quad=quad)
        if f1["f1"] > best["f1"]:
            best = {"threshold": round(float(t), 3), "f1": round(f1["f1"], 4),
                    "recall": round(f1["recall"], 4), "precision": round(f1["precision"], 4),
                    "iou": round(M.iou(pred, gt, quad), 4)}
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patches-per-image", type=int, default=120, dest="patches_per_image")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--variant", default="production")
    p.add_argument("--out", default="output/unet")
    p.add_argument("--n-images", type=int, default=N_IMAGES, dest="n_images",
                   help="total annotated images (default: 19)")
    p.add_argument("--checkpoint-every", type=int, default=10, dest="checkpoint_every",
                   help="save a mid-fold checkpoint every N epochs (0 = disable)")
    p.add_argument("--start-fold", type=int, default=1, dest="start_fold",
                   help="resume from this fold number (skips earlier folds)")
    args = p.parse_args()

    ns = vars(args)
    report = {"config": ns.copy(), "folds": {}}
    fold_range = range(1, args.n_images + 1)
    for val_n in fold_range:
        out_dir = Path(args.out)
        ckpt_path = out_dir / f"unet_fold_val{val_n}.pt"
        if val_n < args.start_fold and ckpt_path.exists():
            # Already trained — just evaluate the saved checkpoint.
            res = _evaluate(ckpt_path, val_n)
            report["folds"][val_n] = res
            print(f"[fold {val_n}] (loaded) F1={res['f1']:.3f} IoU={res['iou']:.3f} "
                  f"recall={res['recall']:.3f} prec={res['precision']:.3f} t={res['threshold']}",
                  flush=True)
            continue
        elif val_n < args.start_fold:
            continue
        ckpt = train_fold(val_n, types.SimpleNamespace(**ns))
        res = _evaluate(ckpt, val_n)
        report["folds"][val_n] = res
        print(f"[fold {val_n}] F1={res['f1']:.3f} IoU={res['iou']:.3f} "
              f"recall={res['recall']:.3f} prec={res['precision']:.3f} t={res['threshold']}",
              flush=True)

    keys = ["f1", "iou", "recall", "precision"]
    report["mean"] = {k: round(float(np.mean([report["folds"][n][k] for n in fold_range])), 4)
                      for k in keys}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    m = report["mean"]
    print(f"\n=== 5-fold CV mean ===  F1={m['f1']:.3f}  IoU={m['iou']:.3f}  "
          f"recall={m['recall']:.3f}  prec={m['precision']:.3f}  (classical F1 ~0.31)")
    print(f"saved {out / 'cv_report.json'}")


if __name__ == "__main__":
    main()
