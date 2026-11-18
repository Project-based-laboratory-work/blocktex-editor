"""
画像の構造化器（分類）の学習。学習ループの形は block_classifier/train.py と同じ。

    python -m image_classifier.train                 # 画像も見る版
    python -m image_classifier.train --no-image      # 数値の特徴だけの版（画像を見る効果を比べる用）

正解率だけだと数の多い「段落」の出来に引っ張られるので、best.pt は macro F1（クラスごとのF1の平均）で選ぶ。
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.ir import LABELS
from image_classifier.dataset import IGNORE, PageDataset, collate
from image_classifier.features import NUM_FEATURES
from image_classifier.model import ModelConfig, PageClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("image_classifier/data/train"))
    parser.add_argument("--val", type=Path, default=Path("image_classifier/data/val"))
    parser.add_argument("--out", type=Path, default=None, help="省略時は image_classifier/checkpoints/<image|noimage>")
    parser.add_argument("--no-image", action="store_true", help="要素の画像を使わない（数値の特徴だけ）")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def report(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, dict[str, float]]:
    """クラスごとの適合率・再現率・F1 と、全体の正解率・macro F1"""
    result = {}
    for i, label in enumerate(LABELS):
        tp = ((predictions == i) & (targets == i)).sum().item()
        predicted, actual = (predictions == i).sum().item(), (targets == i).sum().item()
        if actual == 0 and predicted == 0:
            continue
        precision, recall = tp / max(predicted, 1), tp / max(actual, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": actual}
    present = [r["f1"] for r in result.values() if r["support"] > 0]
    result["_overall"] = {
        "accuracy": (predictions == targets).float().mean().item(),
        "macro_f1": sum(present) / len(present),
    }
    return result


def print_report(result: dict[str, dict[str, float]]) -> None:
    print(f"  {'ラベル':<10} {'適合率':>6} {'再現率':>6} {'F1':>6} {'件数':>7}")
    for label, r in result.items():
        if label != "_overall":
            print(f"  {label:<12} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} {r['support']:>8}")
    overall = result["_overall"]
    print(f"  正解率 {overall['accuracy']:.4f}   macro F1 {overall['macro_f1']:.4f}")


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets = [], []
    for features, crops, labels, padding_mask in loader:
        logits = model(features.to(device), crops.to(device), padding_mask.to(device)).cpu()
        keep = labels != IGNORE
        predictions.append(logits.argmax(-1)[keep])
        targets.append(labels[keep])
    model.train()
    return torch.cat(predictions), torch.cat(targets)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or Path(f"image_classifier/checkpoints/{'noimage' if args.no_image else 'image'}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = PageDataset(args.train), PageDataset(args.val)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers)

    config = ModelConfig(num_features=NUM_FEATURES, num_classes=len(LABELS), use_image=not args.no_image)
    model = PageClassifier(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  画像={config.use_image}  パラメータ数={params:,}  学習 {len(train_ds)} ページ / 検証 {len(val_ds)} ページ")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started, total_loss = time.time(), 0.0
        for features, crops, labels, padding_mask in train_loader:
            features, crops, labels, padding_mask = (t.to(device) for t in (features, crops, labels, padding_mask))
            logits = model(features, crops, padding_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        overall = report(*predict_all(model, val_loader, device))["_overall"]
        print(
            f"[epoch {epoch}] {time.time() - started:.0f}s  loss {total_loss / len(train_loader):.4f}  "
            f"val 正解率 {overall['accuracy']:.4f}  macro F1 {overall['macro_f1']:.4f}",
            flush=True,
        )
        if overall["macro_f1"] > best_f1:
            best_f1 = overall["macro_f1"]
            torch.save({"model": model.state_dict(), "config": asdict(config), "labels": LABELS}, out / "best.pt")

    print(f"\n検証データでの最終結果（best.pt, macro F1 {best_f1:.4f}）:")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    print_report(report(*predict_all(model, val_loader, device)))


if __name__ == "__main__":
    main()
