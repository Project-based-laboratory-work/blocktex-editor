"""見たことのない見た目・中身への強さを測る実験（AI.md の 7.7）。

学習データと同じ作り方の文書で評価すると、AI が作り方を覚えたぶん点数が高めに出る。そこで、
  - 見た目: 学習データからある見た目（書体・文書クラス・箇条書きの記号）を抜いて学習し、抜いた見た目だけの文書で測る
  - 中身:   学習に使っていない材料（別の単語・本物の文・本物の数式。content.py）の文書で測る
ルール（hybrid_operator）、全部で学習した AI（いつもの重み）、抜いて学習した AI（checkpoints/<名前>/）を、同じデータで採点する。

    python -m hybrid_extractor.generalization --dataset evaluation/data/gen_only_times --models rule ai ai:xtimes

models: rule（ルールのオペレータ）/ ai（いつもの重み）/ ai:<名前>（checkpoints/<名前>/ の重み）
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from evaluation.metrics import label_scores, summarize
from evaluation.models import load_registry
from evaluation.run import evaluate_dataset

SHOW = ["f1", "label_acc", "level_acc", "code_indent_acc", "sec_per_page"]
SHOW_LABELS = ["paragraph", "heading", "list", "caption", "code", "math", "table", "other"]


def model_spec(name: str):
    registry = load_registry()
    if name == "rule":
        return registry["hybrid_operator"]
    if name == "ai":
        return registry["hybrid_ai"]
    if name.startswith("ai:"):
        base = registry["hybrid_ai"]
        return replace(base, id=f"hybrid_ai_{name[3:]}", params={**base.params, "checkpoints": name[3:]})
    raise SystemExit(f"知らないモデル: {name}（rule / ai / ai:<名前>）")


def run(dataset: Path, names: list[str]) -> dict:
    results = {}
    for name in names:
        result = evaluate_dataset(model_spec(name), dataset)
        summary = summarize(result.total)
        by_label = label_scores(result.total)
        results[name] = {"summary": summary, "by_label": {k: v.get("f1") for k, v in by_label.items()}}
        row = "  ".join(f"{k} {summary[k]:.3f}" for k in SHOW if summary.get(k) is not None)
        labels = "  ".join(f"{k} {by_label[k]['f1']:.2f}" for k in SHOW_LABELS
                           if k in by_label and by_label[k].get("f1") is not None)
        print(f"  {name:16} {row}")
        print(f"  {'':16} ラベル別 F1: {labels}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="見たことのない見た目・中身への強さを測る")
    parser.add_argument("--dataset", required=True, action="append", help="評価サイトの形のデータセット（何度でも指定できる）")
    parser.add_argument("--models", nargs="+", default=["rule", "ai"], help="rule / ai / ai:<checkpoints の中の名前>")
    parser.add_argument("--out", help="結果を JSON で保存する")
    args = parser.parse_args(argv)
    report = {}
    for dataset in args.dataset:
        print(f"== {dataset}")
        report[dataset] = run(Path(dataset), args.models)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
