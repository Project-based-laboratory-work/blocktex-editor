"""
学習済みモデルと、チームのルールベース分類を、同じ検証データで比べる。

    python -m block_classifier.evaluate block_classifier/checkpoints/transformer/best.pt

ルールベースは 見出し/段落/キャプション/表/図 しか出さないので、チームのラベル定義の
粗いラベル（rule_based/labels.py の to_coarse: 箇条書き・数式は段落にまとめる）でも比べる。
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from block_classifier.dataset import LABEL_TO_ID, LABELS, PageDataset, collate
from block_classifier.model import ModelConfig, build_model
from block_classifier.train import predict_all, print_report, report
from rule_based.labels import to_coarse

# 細かいラベルのID → 粗いラベルの代表のID（list, math → paragraph）
TO_COARSE = torch.tensor([LABEL_TO_ID[to_coarse(label)] for label in LABELS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("block_classifier/data/val"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    model = build_model(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])

    dataset = PageDataset(args.data)
    predictions, targets = predict_all(model, DataLoader(dataset, batch_size=32, collate_fn=collate))
    rule_based = torch.cat([page["rule_based"] for page in dataset.pages])

    print(f"=== モデル（{checkpoint['config']['kind']}）: 細かいラベル ===")
    print_report(report(predictions, targets))
    print("\n=== モデル: 粗いラベル（箇条書き・数式 → 段落）===")
    print_report(report(TO_COARSE[predictions], TO_COARSE[targets]))
    print("\n=== ルールベース: 粗いラベル ===")
    print_report(report(TO_COARSE[rule_based], TO_COARSE[targets]))


if __name__ == "__main__":
    main()
