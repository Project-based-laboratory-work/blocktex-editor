"""ハイパーパラメータの比較（区切りとラベルの専門家、MLP）。

いつもの設定（隠れ層 64、学習率 1e-3、バッチ 256、最大 60 エポック＋早期終了、pos_weight あり）から、
1つの値だけを変えて学習し、乱数の種 3つの平均とぶれ（標準偏差）を出す（種だけで F1 が 0.006 ぶれると分かったため）。
良い設定は検証用データで選び、テスト用データの値は参考に出す。重みは保存しない（いつもの重みはそのまま）。

    python -m hybrid_extractor.sweep boundary    # 約30分
    python -m hybrid_extractor.sweep labels      # 約30分
    python -m hybrid_extractor.sweep inline      # 文中の数式（保存してある学習の記録からエポックごとの成績。計算なし）
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from hybrid_extractor.nn import binary_scores, predict_binary, train_binary
from hybrid_extractor.train import (
    CHECKPOINTS,
    DATA,
    LABEL_TARGETS,
    boundary_examples,
    label_examples,
    load_docs,
    operator_decide,
)

SEEDS = [0, 1, 2]
BASE = {"hidden": 64, "lr": 1e-3, "batch": 256, "epochs": 60, "patience": 6, "use_pos_weight": True}
# 1つずつ変える値（名前, 変える項目と値）
VARIANTS = {
    "boundary": [
        *[(f"最大エポック {e}", {"epochs": e}) for e in (1, 3, 5, 10, 20)],
        ("早期終了なし（60エポック全部）", {"patience": 10_000}),
        *[(f"隠れ層 {h}", {"hidden": h}) for h in (8, 16, 32, 128, 256)],
        *[(f"学習率 {lr:g}", {"lr": lr}) for lr in (1e-4, 3e-4, 3e-3, 1e-2)],
        *[(f"バッチ {b}", {"batch": b}) for b in (32, 128, 1024, 4096)],
        ("pos_weight なし", {"use_pos_weight": False}),
    ],
    "labels": [
        *[(f"最大エポック {e}", {"epochs": e}) for e in (3, 10, 20)],
        ("早期終了なし（60エポック全部）", {"patience": 10_000}),
        *[(f"隠れ層 {h}", {"hidden": h}) for h in (16, 256)],
        ("pos_weight なし", {"use_pos_weight": False}),
    ],
}


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def run_boundary(params: dict, data: dict, seed: int) -> dict:
    (x, y), (xv, yv), (xt, yt) = data["train"], data["val"], data["test"]
    t = time.perf_counter()
    model, std, history = train_binary(x, y, xv, yv, seed=seed, log=False, **params)
    seconds = time.perf_counter() - t
    val = binary_scores(predict_binary(model, std, xv), yv)
    test = binary_scores(predict_binary(model, std, xt), yt)
    return {"val_acc": val["acc"], "val_f1": val["f1"], "test_acc": test["acc"], "epochs": len(history),
            "seconds": seconds}


def run_labels(params: dict, data: dict, seed: int) -> dict:
    train, val, test = data["train"], data["val"], data["test"]
    t = time.perf_counter()
    val_probs, test_probs, epochs = {}, {}, []
    for target in LABEL_TARGETS:
        y = (train["labels"] == target).astype(np.float32)
        yv = (val["labels"] == target).astype(np.float32)
        model, std, history = train_binary(train["x"], y, val["x"], yv, seed=seed, log=False, **params)
        val_probs[target] = predict_binary(model, std, val["x"])
        test_probs[target] = predict_binary(model, std, test["x"])
        epochs.append(len(history))
    seconds = time.perf_counter() - t
    val_acc = float((operator_decide(val_probs, val) == val["labels"]).mean())
    test_acc = float((operator_decide(test_probs, test) == test["labels"]).mean())
    return {"val_acc": val_acc, "test_acc": test_acc, "epochs": statistics.mean(epochs), "seconds": seconds}


def sweep(kind: str) -> dict:
    print(f"{kind}: 特徴を作る")
    if kind == "boundary":
        data = {s: boundary_examples(load_docs(s))[:2] for s in ("train", "val", "test")}
        run = run_boundary
    else:
        data = {s: label_examples(load_docs(s)) for s in ("train", "val", "test")}
        run = run_labels
    results = {}
    for name, change in [("いつもの設定", {}), *VARIANTS[kind]]:
        params = {**BASE, **change}
        runs = [run(params, data, seed) for seed in SEEDS]
        row = {k: mean_std([r[k] for r in runs]) for k in runs[0]}
        results[name] = {"params": params, **row}
        extra = f"  検証F1 {row['val_f1'][0]:.4f}" if "val_f1" in row else ""
        print(f"  {name:28} 検証 {row['val_acc'][0]:.4f} ±{row['val_acc'][1]:.4f}{extra}"
              f"  テスト {row['test_acc'][0]:.4f}  エポック {row['epochs'][0]:5.1f}  {row['seconds'][0]:5.1f}秒", flush=True)
    return results


def inline_curves() -> dict:
    """文中の数式の専門家の、エポックごとの検証の成績（重みと一緒に保存してある学習の記録）"""
    import torch

    out = {}
    for name in ("inline_math.pt", "inline_teacher_s1.pt", "inline_teacher_s2.pt", "inline_student.pt"):
        path = CHECKPOINTS / name
        if not path.exists():
            continue
        history = torch.load(path, map_location="cpu", weights_only=False)["extra"]["history"]
        out[name] = [{"epoch": h["epoch"], "val_loss": h["val_loss"], "val_f1": h["f1"]} for h in history]
        print(f"  {name:22} " + "  ".join(f"{h['epoch']}:{h['f1']:.3f}" for h in history))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ハイパーパラメータの比較")
    parser.add_argument("kind", choices=["boundary", "labels", "inline"])
    args = parser.parse_args(argv)
    result = inline_curves() if args.kind == "inline" else sweep(args.kind)
    out = DATA / f"sweep_{args.kind}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"保存した: {out}")


if __name__ == "__main__":
    main()
