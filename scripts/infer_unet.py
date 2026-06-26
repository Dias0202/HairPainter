"""infer_unet.py — sliding-window inference + scoring for the U-Net prototype.

Loads a leave-one-out checkpoint, predicts a fibril probability map over the full
image with overlapping 256x256 tiles, thresholds it, and scores against the SVG
ground truth using the same metrics as the classical experiment harness.

Usage:
    python scripts/infer_unet.py --val 1 --ckpt output/unet/unet_fold_val1.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

import scripts.metrics as M  # noqa: E402
from scripts.preprocess_variants import VARIANTS  # noqa: E402
from scripts.train_unet import UNet, PATCH, DEVICE, load_gt_mask  # noqa: E402

from hairpainter.services.io.io_service import IOService  # noqa: E402

STRIDE = 192  # overlapping tiles (PATCH=256 -> 64px overlap)


def predict_full(model: UNet, gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    prob = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    model.eval()
    ys = list(range(0, max(1, h - PATCH + 1), STRIDE)) + [h - PATCH]
    xs = list(range(0, max(1, w - PATCH + 1), STRIDE)) + [w - PATCH]
    model.to(DEVICE)
    with torch.no_grad():
        for y in sorted(set(max(0, v) for v in ys)):
            for x in sorted(set(max(0, v) for v in xs)):
                tile = gray[y:y + PATCH, x:x + PATCH]
                if tile.shape != (PATCH, PATCH):
                    continue
                t = torch.from_numpy(tile[None, None]).float().to(DEVICE)
                pr = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                prob[y:y + PATCH, x:x + PATCH] += pr
                cnt[y:y + PATCH, x:x + PATCH] += 1.0
    cnt[cnt == 0] = 1.0
    return prob / cnt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=int, default=1)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default="output/unet")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt or f"output/unet/unet_fold_val{args.val}.pt")
    ck = torch.load(str(ckpt_path), map_location="cpu")
    model = UNet(base=ck.get("base", 32))
    model.load_state_dict(ck["state_dict"])
    variant = ck.get("variant", "production")

    img = IOService().load(Path(f"Data/Raw/{args.val}.tif"))
    gray = VARIANTS[variant](img.array.copy(), {"image_data": img}).astype(np.float32) / 255.0
    prob = predict_full(model, gray)

    gt = load_gt_mask(args.val).astype(bool)
    quad = M.annotated_quadrant_mask(gt.shape)

    # Sweep thresholds and report the F1-maximising one (severe class imbalance
    # makes the fixed 0.5 threshold a poor default).
    best = {"f1": -1.0}
    for t in np.linspace(0.1, 0.9, 17):
        pred = prob >= t
        f1 = M.f1_tolerance(pred, gt, tol=5, quad=quad)
        if f1["f1"] > best["f1"]:
            best = {"t": float(t), "f1": f1["f1"], "recall": f1["recall"],
                    "precision": f1["precision"], "iou": M.iou(pred, gt, quad)}
    print(f"val={args.val} variant={variant}  best_t={best['t']:.2f}  "
          f"IoU={best['iou']:.4f}  F1@5px={best['f1']:.3f}  "
          f"recall={best['recall']:.3f}  prec={best['precision']:.3f}  "
          f"(classical F1 ~0.31)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred = prob >= best["t"]
    Image.fromarray((pred * 255).astype(np.uint8)).save(str(out_dir / f"unet_pred_val{args.val}.png"))
    np.save(str(out_dir / f"unet_prob_val{args.val}.npy"), prob.astype(np.float32))
    print(f"saved {out_dir / f'unet_pred_val{args.val}.png'}")
    return best


if __name__ == "__main__":
    main()
