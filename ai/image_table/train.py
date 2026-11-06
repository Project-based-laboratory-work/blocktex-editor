"""
表AIの学習。

    python -m image_table.train --train image_table/data/train --val image_table/data/val

損失は、行・列それぞれの位置ごとの2値クロスエントロピー（「外」か「中」か）。
best.pt は、検証データで「形が合った表」（行・列が全部正しく区切れた表）の割合で選ぶ。
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from image_table.dataset import TableDataset
from image_table.evaluate import matched
from image_table.model import ModelConfig, TableSplitter
from image_table.postprocess import spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("image_table/data/train"))
    parser.add_argument("--val", type=Path, default=Path("image_table/data/val"))
    parser.add_argument("--out", type=Path, default=Path("image_table/checkpoints/split"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="学習データの数を絞る（お試し用）")
    parser.add_argument("--resume", action="store_true", help="out/last.pt から再開する")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def structure_accuracy(model: nn.Module, ds: TableDataset, loader: DataLoader, device: torch.device) -> float:
    """検証データで、行・列が全部正しく区切れた表の割合"""
    model.eval()
    ok, k = 0, 0
    for images, _, _ in loader:
        rows, cols = model(images.to(device))
        for r, c in zip(rows.cpu().numpy(), cols.cpu().numpy()):
            label = json.loads(Path(ds.label_paths[k]).read_text(encoding="utf-8"))
            scale = min(images.shape[2] / label["height"], images.shape[3] / label["width"])
            pr, pc = spans(r, scale), spans(c, scale)
            ok += int(len(pr) == len(label["rows"]) == matched(pr, label["rows"])
                      and len(pc) == len(label["cols"]) == matched(pc, label["cols"]))  # fmt: skip
            k += 1
    model.train()
    return ok / max(k, 1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = TableDataset(args.train, args.limit), TableDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    config = ModelConfig()
    model = TableSplitter(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  パラメータ数={params:,}  学習 {len(train_ds)} 個 / 検証 {len(val_ds)} 個", flush=True)

    criterion = nn.BCEWithLogitsLoss()
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
        for images, rows, cols in train_loader:
            pred_rows, pred_cols = model(images.to(device))
            loss = criterion(pred_rows, rows.to(device)) + criterion(pred_cols, cols.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()

        accuracy = structure_accuracy(model, val_ds, val_loader, device)
        print(f"[epoch {epoch}] {time.time() - started:.0f}s  loss {total / len(train_loader):.4f}  val 形が合った表 {accuracy:.4f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"model": model.state_dict(), "config": asdict(config)}, args.out / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "best": best},
            args.out / "last.pt",
        )  # fmt: skip
    print(f"best: 形が合った表 {best:.4f}")


if __name__ == "__main__":
    main()
