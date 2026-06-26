"""train_kaggle.py — GPU-optimised leave-one-out training for Kaggle (T4x2 / P100 / TPU).

Extends train_unet.py with:
  - nn.DataParallel for multi-GPU (T4 x2 → 2x throughput)
  - torch.cuda.amp  automatic mixed precision (2-3x on T4/P100)
  - Optional TPU via torch_xla
  - CosineAnnealingLR scheduler
  - Larger GPU-friendly defaults

Kaggle usage (run ALL folds, overnight):
    python scripts/train_kaggle.py --all-folds --epochs 150 --base 64 --batch 32

Test one fold:
    python scripts/train_kaggle.py --val 1 --epochs 10 --base 32 --batch 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn

from scripts.train_unet import (
    UNet, dice_bce_loss, _load_pair, _sample_patches,
    load_gt_mask, N_IMAGES, PATCH,
)

# ── Accelerator detection ──────────────────────────────────────────────────
_USE_TPU = False
try:
    import torch_xla.core.xla_model as xm  # type: ignore
    DEVICE = xm.xla_device()
    _USE_TPU = True
    print("TPU detected — using torch_xla")
except ImportError:
    pass

if not _USE_TPU:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_GPU = torch.cuda.device_count()
    if N_GPU > 0:
        names = [torch.cuda.get_device_name(i) for i in range(N_GPU)]
        print(f"GPU(s): {N_GPU}x  [{', '.join(names)}]")
    else:
        print("WARNING: no GPU — CPU training will be very slow")
else:
    N_GPU = 0


# ── Model + scaler factory ─────────────────────────────────────────────────
def _build_model(base: int) -> Tuple[nn.Module, torch.cuda.amp.GradScaler]:
    model = UNet(base=base).to(DEVICE)
    if not _USE_TPU and N_GPU > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel across {N_GPU} GPUs")
    use_amp = (not _USE_TPU) and (DEVICE.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return model, scaler


def _raw_model(model: nn.Module) -> UNet:
    """Unwrap DataParallel to get the underlying UNet (needed for state_dict save)."""
    return model.module if isinstance(model, nn.DataParallel) else model  # type: ignore[return-value]


def _save_checkpoint(out_dir: Path, val_n: int, ep: int, model: nn.Module,
                     opt: torch.optim.Optimizer, args: argparse.Namespace,
                     suffix: str = "") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"unet_fold_val{val_n}{suffix}.pt"
    path = out_dir / name
    torch.save({
        "state_dict": _raw_model(model).state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": ep,
        "base": args.base,
        "variant": args.variant,
    }, path)
    return path


# ── Training fold ──────────────────────────────────────────────────────────
def train_fold_gpu(val_n: int, args: argparse.Namespace) -> Path:
    rng = np.random.default_rng(42 + val_n)
    n_images = args.n_images
    checkpoint_every = args.checkpoint_every
    train_ns = [n for n in range(1, n_images + 1) if n != val_n]
    print(f"\n=== Fold val={val_n}  train={train_ns}  device={DEVICE} ===")

    imgs, gts = zip(*[_load_pair(n, args.variant) for n in train_ns])

    out_dir = Path(args.out)
    model, scaler = _build_model(args.base)
    use_amp = scaler.is_enabled()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume from latest periodic checkpoint if available
    start_ep = 0
    if checkpoint_every > 0:
        for ep_try in range(args.epochs - checkpoint_every, 0, -checkpoint_every):
            ep_try = (ep_try // checkpoint_every) * checkpoint_every
            resume_path = out_dir / f"unet_fold_val{val_n}_ep{ep_try}.pt"
            if resume_path.exists():
                ck = torch.load(str(resume_path), map_location=DEVICE)
                _raw_model(model).load_state_dict(ck["state_dict"])
                opt.load_state_dict(ck["optimizer"])
                start_ep = ck["epoch"] + 1
                print(f"  resumed from {resume_path.name} (starting epoch {start_ep + 1})")
                break

    model.train()
    for ep in range(start_ep, args.epochs):
        X, Y = _sample_patches(imgs, gts, args.patches_per_image, rng)
        perm = torch.randperm(X.shape[0])
        total_loss = 0.0

        for i in range(0, X.shape[0], args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = X[idx].to(DEVICE), Y[idx].to(DEVICE)
            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = dice_bce_loss(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            elif _USE_TPU:
                loss = dice_bce_loss(model(xb), yb)
                loss.backward()
                xm.optimizer_step(opt)
                xm.mark_step()
            else:
                loss = dice_bce_loss(model(xb), yb)
                loss.backward()
                opt.step()

            total_loss += loss.item() * idx.numel()

        scheduler.step()
        avg_loss = total_loss / X.shape[0]
        lr_now = scheduler.get_last_lr()[0]
        print(f"  ep {ep + 1:03d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}",
              flush=True)

        if checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0 and (ep + 1) < args.epochs:
            p = _save_checkpoint(out_dir, val_n, ep, model, opt, args, suffix=f"_ep{ep + 1}")
            print(f"  ↳ checkpoint saved: {p.name}")

    ckpt = _save_checkpoint(out_dir, val_n, args.epochs - 1, model, opt, args)
    print(f"  ✓ fold {val_n} done → {ckpt}")
    return ckpt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=int, default=1)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--n-images", type=int, default=N_IMAGES, dest="n_images")
    p.add_argument("--checkpoint-every", type=int, default=10, dest="checkpoint_every")
    p.add_argument("--variant", default="production")
    p.add_argument("--epochs", type=int, default=150,
                   help="training epochs per fold (default 150 — suited for T4/P100)")
    p.add_argument("--patches-per-image", type=int, default=400, dest="patches_per_image",
                   help="patches sampled per training image per epoch")
    p.add_argument("--batch", type=int, default=32,
                   help="batch size (32 for T4, 64 for T4x2 DataParallel)")
    p.add_argument("--base", type=int, default=64,
                   help="UNet base channels (32=small/fast, 64=better quality)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--out", default="output/unet")
    args = p.parse_args()

    if args.all_folds:
        for n in range(1, args.n_images + 1):
            train_fold_gpu(n, args)
    else:
        train_fold_gpu(args.val, args)


if __name__ == "__main__":
    main()
