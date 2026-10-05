"""
学習済みモデルをデータセット全体で評価し、間違えた例を表示する。

    python -m math_ocr.evaluate checkpoints/math_ocr_d192/best.pt --data data/math_latex --split val

学習時の検証（train.py の validate）は時間節約のため一部のサンプルしか生成していない。
こちらは全件を生成して評価するので、モデル同士・データ同士の比較にはこちらの数字を使う。
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from math_ocr.dataset import Collator, MathFormulaDataset
from math_ocr.metrics import edit_distance, evaluate
from math_ocr.model import count_parameters
from math_ocr.predict import load_model
from math_ocr.tokenizer import detokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--show-errors", type=int, default=10, help="間違えた例を何件表示するか")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint)
    dataset = MathFormulaDataset(args.data, args.split, vocab, limit=args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=Collator(vocab.pad_id))

    predictions, references = [], []
    started = time.time()
    for images, seqs in loader:
        generated = model.generate(images)
        predictions += [vocab.ids_to_tokens(row.tolist()[1:]) for row in generated]
        references += [vocab.ids_to_tokens(row.tolist()[1:]) for row in seqs]
    elapsed = time.time() - started

    metrics = evaluate(predictions, references)
    print(f"モデル: {args.checkpoint}（{count_parameters(model):,} パラメータ）")
    print(f"データ: {args.data} / {args.split}（{len(references)} 件）")
    print(f"完全一致率 {metrics['exact_match']:.3f}   トークン誤り率 {metrics['token_error_rate']:.3f}")
    print(f"推論時間 {elapsed / len(references) * 1000:.0f} ms/件（CPU、バッチサイズ {args.batch_size}）")

    # 編集距離の大きい順ではなく出現順に見せる（ひどい例だけ見ると全体の傾向を見誤るため）
    errors = [(p, r) for p, r in zip(predictions, references) if p != r]
    for pred, ref in errors[: args.show_errors]:
        print(f"  正解: {detokenize(ref)}")
        print(f"  予測: {detokenize(pred)}   （編集距離 {edit_distance(pred, ref)}）")


if __name__ == "__main__":
    main()
