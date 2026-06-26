"""train_unet.py — leave-one-out U-Net prototype for fibril segmentation.

Self-contained: a small U-Net in pure PyTorch (no segmentation_models_pytorch or
albumentations dependency), so it runs on a CPU-only install.

Data: annotated image pairs in Data/Raw/{n}.tif + Data/Manual_paint/{n}.svg or
Data/Manual_paint/{n}.png.  Images 1-5 use SVG ground truth; images 6+ use PNG.
Training samples 256x256 patches from the ANNOTATED quadrant.
Augmentation (flips, rot90, noise, brightness) is done in numpy.

Usage (one fold, validate on image 1, all 19 images available):
    python scripts/train_unet.py --val 1 --epochs 30 --patches-per-image 200

    # all leave-one-out folds
    python scripts/train_unet.py --all-folds --epochs 30

WARNING: CPU training is slow.  Start with --epochs 5 --patches-per-image 50 to
verify the loop, then scale up.
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.svg_to_mask import svg_to_mask  # noqa: E402
from scripts.preprocess_variants import VARIANTS  # noqa: E402
from scripts.metrics import ANNOTATED_W, ANNOTATED_H  # noqa: E402

from hairpainter.services.io.io_service import IOService  # noqa: E402

N_IMAGES = 19  # total annotated images available

PATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Small U-Net
# ----------------------------------------------------------------------
class _DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):  # noqa: ANN001
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.d1 = _DoubleConv(1, base)
        self.d2 = _DoubleConv(base, base * 2)
        self.d3 = _DoubleConv(base * 2, base * 4)
        self.bott = _DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):  # noqa: ANN001
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        x = self.u3(torch.cat([self.up3(b), c3], 1))
        x = self.u2(torch.cat([self.up2(x), c2], 1))
        x = self.u1(torch.cat([self.up1(x), c1], 1))
        return self.out(x)


def dice_bce_loss(logits, target):  # noqa: ANN001
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum() + 1.0
    den = prob.sum() + target.sum() + 1.0
    return bce + (1 - num / den)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def _png_to_mask(path: Path) -> np.ndarray:
    """Convert a manually-painted PNG annotation to a binary fibril mask.

    Supports:
    - RGBA with transparent background (alpha > 0 = fibril)
    - RGB/RGBA on black background (any non-black pixel = fibril)
    - RGB/RGBA on white background (any non-white pixel = fibril)
    Navy color #1c3052 = (28, 48, 82) is detected explicitly as fallback.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return (arr > 0).astype(np.float32)

    has_alpha = arr.shape[2] == 4
    if has_alpha and arr[:, :, 3].min() < 100:
        return (arr[:, :, 3] > 50).astype(np.float32)

    rgb = arr[:, :, :3].astype(np.int32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Navy #1c3052 with tolerance
    navy = (r < 70) & (g < 100) & (b > 40) & (b > r + 15)
    if navy.sum() > 100:
        return navy.astype(np.float32)

    # Generic: marks on light or dark background
    luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    if luminance.mean() > 128:
        return (luminance < 128).astype(np.float32)
    return (luminance > 30).astype(np.float32)


def load_gt_mask(n: int) -> np.ndarray:
    """Load ground-truth fibril mask for image n (SVG preferred, PNG fallback)."""
    svg_path = Path(f"Data/Manual_paint/{n}.svg")
    if svg_path.exists():
        return svg_to_mask(svg_path).astype(np.float32)
    png_path = Path(f"Data/Manual_paint/{n}.png")
    if png_path.exists():
        return _png_to_mask(png_path)
    raise FileNotFoundError(f"No GT annotation found for image {n} (.svg or .png)")


def _load_pair(n: int, variant: str) -> tuple[np.ndarray, np.ndarray]:
    img = IOService().load(Path(f"Data/Raw/{n}.tif"))
    gray = VARIANTS[variant](img.array.copy(), {"image_data": img}).astype(np.float32) / 255.0
    gt = load_gt_mask(n)
    if gt.shape != gray.shape:
        # PNG GT was drawn on a canvas with different dimensions; resize to raw.
        pil_gt = Image.fromarray((gt * 255).astype(np.uint8))
        pil_gt = pil_gt.resize((gray.shape[1], gray.shape[0]), Image.NEAREST)
        gt = np.array(pil_gt).astype(np.float32) / 255.0
    return gray, gt


def _augment(p_img: np.ndarray, p_gt: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        p_img, p_gt = p_img[:, ::-1], p_gt[:, ::-1]
    if rng.random() < 0.5:
        p_img, p_gt = p_img[::-1], p_gt[::-1]
    k = int(rng.integers(0, 4))
    if k:
        p_img, p_gt = np.rot90(p_img, k), np.rot90(p_gt, k)
    p_img = p_img + rng.normal(0, 0.03, p_img.shape).astype(np.float32)  # noise
    p_img = p_img * float(rng.uniform(0.9, 1.1))  # brightness
    return np.ascontiguousarray(p_img.clip(0, 1)), np.ascontiguousarray(p_gt)


def _sample_patches(imgs, gts, n_per_image, rng):  # noqa: ANN001
    """Sample PATCH x PATCH patches from the annotated quadrant, biased to fibrils."""
    xs, ys = [], []
    for gray, gt in zip(imgs, gts):
        H = min(ANNOTATED_H, gray.shape[0]) - PATCH
        W = min(ANNOTATED_W, gray.shape[1]) - PATCH
        for _ in range(n_per_image):
            # 70% positive-biased: center near a fibril pixel
            if rng.random() < 0.7 and gt.sum() > 0:
                fy, fx = np.where(gt[:H + PATCH, :W + PATCH] > 0)
                k = int(rng.integers(0, fy.size))
                y = int(np.clip(fy[k] - PATCH // 2, 0, H))
                x = int(np.clip(fx[k] - PATCH // 2, 0, W))
            else:
                y, x = int(rng.integers(0, H + 1)), int(rng.integers(0, W + 1))
            pi = gray[y:y + PATCH, x:x + PATCH]
            pg = gt[y:y + PATCH, x:x + PATCH]
            pi, pg = _augment(pi, pg, rng)
            xs.append(pi)
            ys.append(pg)
    X = torch.from_numpy(np.stack(xs)[:, None]).float()
    Y = torch.from_numpy(np.stack(ys)[:, None]).float()
    return X, Y


def _save_checkpoint(out_dir: Path, val_n: int, ep: int, model: "UNet",
                     opt: "torch.optim.Optimizer", args: argparse.Namespace,
                     suffix: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"unet_fold_val{val_n}{suffix}.pt"
    path = out_dir / name
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": ep,
        "base": args.base,
        "variant": args.variant,
    }, path)
    return path


def train_fold(val_n: int, args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(42 + val_n)
    n_images = getattr(args, "n_images", N_IMAGES)
    checkpoint_every = getattr(args, "checkpoint_every", 10)
    train_ns = [n for n in range(1, n_images + 1) if n != val_n]
    print(f"\n=== Fold val={val_n}  train={train_ns}  variant={args.variant} ===")
    imgs, gts = zip(*[_load_pair(n, args.variant) for n in train_ns])

    out_dir = Path(args.out)
    print(f"  device={DEVICE}")
    model = UNet(base=args.base).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Resume from latest periodic checkpoint if it exists.
    start_ep = 0
    for ep_check in range(args.epochs - 1, 0, -checkpoint_every):
        ep_check = (ep_check // checkpoint_every) * checkpoint_every
        resume_path = out_dir / f"unet_fold_val{val_n}_ep{ep_check}.pt"
        if resume_path.exists():
            ck = torch.load(str(resume_path), map_location=DEVICE)
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(ck["optimizer"])
            start_ep = ck["epoch"] + 1
            print(f"  resumed from {resume_path.name} (epoch {start_ep})")
            break

    model.train()
    for ep in range(start_ep, args.epochs):
        X, Y = _sample_patches(imgs, gts, args.patches_per_image, rng)
        perm = torch.randperm(X.shape[0])
        total = 0.0
        for i in range(0, X.shape[0], args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = dice_bce_loss(model(X[idx].to(DEVICE)), Y[idx].to(DEVICE))
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        print(f"  epoch {ep + 1}/{args.epochs}  loss={total / X.shape[0]:.4f}", flush=True)

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and (ep + 1) < args.epochs:
            p = _save_checkpoint(out_dir, val_n, ep, model, opt, args, suffix=f"_ep{ep + 1}")
            print(f"  checkpoint saved: {p.name}")

    # Final checkpoint (no epoch suffix = the "official" one used by cv/infer).
    ckpt = _save_checkpoint(out_dir, val_n, args.epochs - 1, model, opt, args)
    print(f"  saved {ckpt}")
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=int, default=1, help="validation image number")
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--n-images", type=int, default=N_IMAGES, dest="n_images",
                   help="total annotated images (default: 19)")
    p.add_argument("--checkpoint-every", type=int, default=10, dest="checkpoint_every",
                   help="save a mid-fold checkpoint every N epochs (0 = disable)")
    p.add_argument("--variant", default="production", help="preprocessing variant for input")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patches-per-image", type=int, default=200, dest="patches_per_image")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="output/unet")
    args = p.parse_args()

    if args.all_folds:
        for n in range(1, args.n_images + 1):
            train_fold(n, args)
    else:
        train_fold(args.val, args)


if __name__ == "__main__":
    main()
