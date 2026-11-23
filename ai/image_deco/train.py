"""
装飾AIの学習。

    python -m image_deco.train --train image_deco/data/train --val image_deco/data/val

損失は、位置ごと・装飾ごとの2値クロスエントロピー（埋めた右側は数えない）。装飾の付いた位置は少ないので、付いた位置の重みを上げる。
best.pt は、検証データで「装飾の付いた位置の F1」（太字・斜体・下線の平均）で選ぶ。
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_deco.dataset import DecoDataset, collate
from image_deco.model import MARKS, DecorationTagger, ModelConfig

POSITIVE_WEIGHT = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("image_deco/data/train"))
    parser.add_argument("--val", type=Path, default=Path("image_deco/data/val"))
    parser.add_argument("--out", type=Path, default=Path("image_deco/checkpoints/tagger"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def position_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """装飾ごとの、位置単位の F1"""
    model.eval()
    tp, fp, fn = torch.zeros(len(MARKS)), torch.zeros(len(MARKS)), torch.zeros(len(MARKS))
    for images, targets, valid in loader:
        pred = (model(images.to(device)).cpu() > 0) & valid[:, None]
        gold = (targets > 0.5) & valid[:, None]
        tp += (pred & gold).sum((0, 2))
        fp += (pred & ~gold).sum((0, 2))
        fn += (~pred & gold).sum((0, 2))
    model.train()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return {m: float(v) for m, v in zip(MARKS, f1)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = DecoDataset(args.train, args.limit, augment=True), DecoDataset(args.val)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers, collate_fn=collate)

    config = ModelConfig()
    model = DecorationTagger(config).to(device)
    print(f"device={device}  パラメータ数={sum(p.numel() for p in model.parameters()):,}  学習 {len(train_ds)} 行 / 検証 {len(val_ds)} 行", flush=True)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.full((len(MARKS), 1), POSITIVE_WEIGHT, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)

    start_epoch, best = 1, -1.0
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]

    for epoch in range(start_epoch, args.epochs + 1):
        started, total = time.time(), 0.0
        for images, targets, valid in train_loader:
            images, targets, valid = images.to(device), targets.to(device), valid.to(device)
            loss = criterion(model(images), targets)
            loss = (loss * valid[:, None]).sum() / (valid.sum() * len(MARKS)).clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        f1 = position_f1(model, val_loader, device)
        mean = sum(f1.values()) / len(f1)
        print(f"[epoch {epoch}] {time.time() - started:.0f}s  loss {total / len(train_loader):.4f}  val F1 "
              + "  ".join(f"{m} {v:.3f}" for m, v in f1.items()) + f"  平均 {mean:.4f}", flush=True)  # fmt: skip
        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(), "config": asdict(config)}, args.out / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best": best}, args.out / "last.pt")  # fmt: skip
    print(f"best: 平均 F1 {best:.4f}")


if __name__ == "__main__":
    main()
