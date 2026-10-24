"""
ブロック分類モデルの学習と評価。

    python -m block_classifier.train --model transformer
    python -m block_classifier.train --model mlp

学習ループの形は数式OCR（math_ocr/train.py）と同じ。違いは、1ページの全ブロックを同時に分類することと、
評価指標にクラスごとの適合率・再現率・F1を使うこと（クラスごとのブロック数の差が大きく、
正解率だけだと数の多い「段落」の出来に引っ張られるため）。
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from block_classifier.dataset import IGNORE, LABELS, PageDataset, collate
from block_classifier.features import NUM_FEATURES
from block_classifier.model import ModelConfig, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("block_classifier/data/train"))
    parser.add_argument("--val", type=Path, default=Path("block_classifier/data/val"))
    parser.add_argument("--out", type=Path, default=None, help="省略時は block_classifier/checkpoints/<model>")
    parser.add_argument("--model", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def report(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, dict[str, float]]:
    """クラスごとの適合率（precision）・再現率（recall）・F1 と、全体の正解率。

    適合率: 「見出し」と予測したうち、本当に見出しだった割合
    再現率: 本当の見出しのうち、「見出し」と予測できた割合
    F1:     両者の調和平均。どちらかが低いと低くなる
    """
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
        "macro_f1": sum(present) / len(present),  # クラスごとのF1の平均（数の少ないクラスも同じ重み）
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
def predict_all(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets = [], []
    for features, labels, padding_mask in loader:
        logits = model(features, padding_mask)
        keep = labels != IGNORE
        predictions.append(logits.argmax(-1)[keep])
        targets.append(labels[keep])
    model.train()
    return torch.cat(predictions), torch.cat(targets)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out = args.out or Path(f"block_classifier/checkpoints/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = PageDataset(args.train), PageDataset(args.val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate)

    config = ModelConfig(num_features=NUM_FEATURES, num_classes=len(LABELS), kind=args.model)
    model = build_model(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  パラメータ数={params:,}  学習 {len(train_ds)} ページ / 検証 {len(val_ds)} ページ")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started, total_loss = time.time(), 0.0
        for features, labels, padding_mask in train_loader:
            logits = model(features, padding_mask)  # (B, N, クラス数)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        result = report(*predict_all(model, val_loader))
        overall = result["_overall"]
        print(
            f"[epoch {epoch}] {time.time() - started:.0f}s  loss {total_loss / len(train_loader):.4f}  "
            f"val 正解率 {overall['accuracy']:.4f}  macro F1 {overall['macro_f1']:.4f}"
        )
        if overall["macro_f1"] > best_f1:
            best_f1 = overall["macro_f1"]
            torch.save({"model": model.state_dict(), "config": asdict(config), "labels": LABELS}, out / "best.pt")

    print(f"\n検証データでの最終結果（best.pt, macro F1 {best_f1:.4f}）:")
    model.load_state_dict(torch.load(out / "best.pt")["model"])
    print_report(report(*predict_all(model, val_loader)))


if __name__ == "__main__":
    main()
