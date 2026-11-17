"""
レイアウト解析（座標AI）の学習。

    python -m image_layout.train --train image_classifier/data/train --val image_classifier/data/val

損失は画素ごとのクロスエントロピー。画素の大半は背景なので、背景の重みを下げる（全部「背景」と答える近道を防ぐ）。
それに加えて、ブロックのふち（dataset.boundary_mask）を当てる2値の損失を足す（--no-boundary で外す）。
best.pt は、背景以外のクラスの IoU（予測と正解の塗り分けの重なり）の平均で選ぶ。
ブロック単位の採点（四角の重なりでのF1）は evaluate.py で行う。
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_layout.dataset import CLASSES, LayoutDataset
from image_layout.model import LayoutUNet, ModelConfig

BACKGROUND_WEIGHT = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("image_classifier/data/train"))
    parser.add_argument("--val", type=Path, default=Path("image_classifier/data/val"))
    parser.add_argument("--out", type=Path, default=Path("image_layout/checkpoints/unet"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="学習データの枚数を絞る（お試し用）")
    parser.add_argument("--resume", action="store_true", help="out/last.pt から再開する")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-boundary", action="store_true", help="ふちの出力を使わない（前のモデルと同じ）")
    parser.add_argument("--no-figure-aug", action="store_true", help="図の枠の中をランダムな図に描き替える水増しをしない")
    return parser.parse_args()


@torch.no_grad()
def class_iou(model: nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    """クラスごとの IoU = (予測∩正解の画素数) / (予測∪正解の画素数)"""
    model.eval()
    inter = torch.zeros(len(CLASSES))
    union = torch.zeros(len(CLASSES))
    for images, masks, _ in loader:
        logits = model(images.to(device)).cpu()
        pred = logits[:, : len(CLASSES)].argmax(1)
        if logits.shape[1] > len(CLASSES):
            pred[logits[:, len(CLASSES)] > 0] = 0  # postprocess.logits_to_classes と同じ
        for c in range(len(CLASSES)):
            p, t = pred == c, masks == c
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
    model.train()
    return (inter / union.clamp(min=1)).tolist()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = LayoutDataset(args.train, args.limit, augment_figures=not args.no_figure_aug), LayoutDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    config = ModelConfig(num_classes=len(CLASSES), boundary=not args.no_boundary)
    model = LayoutUNet(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  パラメータ数={params:,}  学習 {len(train_ds)} 枚 / 検証 {len(val_ds)} 枚", flush=True)

    weights = torch.ones(len(CLASSES))
    weights[0] = BACKGROUND_WEIGHT
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    boundary_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )

    start_epoch, best = 1, -1.0
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        print(f"last.pt から再開: epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started, total = time.time(), 0.0
        for images, masks, boundaries in train_loader:
            logits = model(images.to(device))
            loss = criterion(logits[:, : len(CLASSES)], masks.to(device))
            if config.boundary:
                loss = loss + boundary_criterion(logits[:, len(CLASSES)], boundaries.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()

        ious = class_iou(model, val_loader, device)
        present = [iou for iou in ious[1:] if iou > 0]
        mean_iou = sum(present) / max(len(present), 1)
        print(f"[epoch {epoch}] {time.time() - started:.0f}s  loss {total / len(train_loader):.4f}  val mIoU {mean_iou:.4f}", flush=True)
        if mean_iou > best:
            best = mean_iou
            torch.save({"model": model.state_dict(), "config": asdict(config), "classes": CLASSES}, args.out / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "best": best},
            args.out / "last.pt",
        )  # fmt: skip

    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device)["model"])
    ious = class_iou(model, val_loader, device)
    print(f"\n検証データでのクラスごとの IoU（best.pt、背景以外の平均 {best:.4f}）")
    for name, iou in zip(CLASSES, ious):
        print(f"  {name:<12} {iou:.3f}")


if __name__ == "__main__":
    main()
