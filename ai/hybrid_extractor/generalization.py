"""見たことのない見た目・中身への強さを測る実験（AI.md の 7.7）。

学習データと同じ作り方の文書で評価すると、AI が作り方を覚えたぶん点数が高めに出る。そこで、
  - 見た目: 学習データからある見た目（書体・文書クラス・箇条書きの記号）を抜いて学習し、抜いた見た目だけの文書で測る
  - 中身:   学習に使っていない材料（別の単語・本物の文・本物の数式。content.py）の文書で測る
ルール（hybrid_operator）、全部で学習した AI（いつもの重み）、抜いて学習した AI（checkpoints/<名前>/）を、同じデータで採点する。

    python -m hybrid_extractor.generalization --dataset evaluation/data/gen_only_times --models rule ai ai:xtimes

models: rule（ルールのオペレータ）/ ai（いつもの重み）/ ai:<名前>（checkpoints/<名前>/ の重み）

数式は担当が別なので、数式の精度で全体の点数が動かないように、**数式を外した点数**をメインに出す:
正解の数式ブロックと、それに重なる予測（IoU 0.5 以上か、中心が正解の数式の中）を、どちらも数えない。
数式の点数（数式ブロックの F1・LaTeX の誤り率）は、外さずに採点したものを別の行に出す。評価サイトの採点は変えない。
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from common.ir import IRDocument, load_ir
from evaluation.metrics import Counts, count_document, iou, label_scores, summarize
from evaluation.models import load_registry, predict

SHOW = ["f1", "label_acc", "level_acc", "code_indent_acc"]
SHOW_LABELS = ["paragraph", "heading", "list", "caption", "code", "table", "other"]


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


def without_math(pred: IRDocument, gt: IRDocument) -> tuple[IRDocument, IRDocument]:
    """正解の数式ブロックと、それに重なる予測を外した (予測, 正解)。数式でない正解を数式と答えた予測は残す（間違いとして数える）"""
    new_pred, new_gt = [], []
    gt_pages = {p.page: p for p in gt.pages}
    for page in gt.pages:
        new_gt.append(replace(page, elements=[e for e in page.elements if e.label != "math"]))
    for page in pred.pages:
        maths = [e.bbox for e in gt_pages[page.page].elements if e.label == "math"] if page.page in gt_pages else []

        def on_math(bbox) -> bool:
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            return any(iou(bbox, m) >= 0.5 or (m[0] <= cx <= m[2] and m[1] <= cy <= m[3]) for m in maths)

        new_pred.append(replace(page, elements=[e for e in page.elements if not on_math(e.bbox)]))
    return replace(pred, pages=new_pred), replace(gt, pages=new_gt)


def run(dataset: Path, names: list[str]) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for name in names:
        spec = model_spec(name)
        full, nomath, seconds, pages = Counts(), Counts(), 0.0, 0
        for case in manifest["cases"]:
            case_dir = dataset / "cases" / case["id"]
            gt = load_ir(str(case_dir / "gt.labeled.json"))
            started = time.perf_counter()
            try:
                pred = predict(spec, str(case_dir / "input.pdf"))
            except Exception as e:  # 評価サイトと同じく、失敗したケースは何も見つけられなかったとして数える
                print(f"    {case['id']}: {type(e).__name__}: {e}")
                pred = IRDocument("input.pdf", spec.id, [])
            seconds += time.perf_counter() - started
            pages += len(gt.pages)
            count_document(pred, gt, full)
            count_document(*without_math(pred, gt), nomath)
        summary = summarize(nomath)
        by_label = label_scores(nomath)
        full_summary, full_labels = summarize(full), label_scores(full)
        math = {"math_f1": full_labels.get("math", {}).get("f1"), "math_ter": full_summary.get("math_ter")}
        results[name] = {"summary": summary, "by_label": {k: v.get("f1") for k, v in by_label.items()},
                         "with_math": full_summary, "math": math, "sec_per_page": seconds / max(1, pages)}
        row = "  ".join(f"{k} {summary[k]:.3f}" for k in SHOW if summary.get(k) is not None)
        labels = "  ".join(f"{k} {by_label[k]['f1']:.2f}" for k in SHOW_LABELS
                           if k in by_label and by_label[k].get("f1") is not None)
        math_row = "  ".join(f"{k} {v:.3f}" for k, v in math.items() if v is not None)
        print(f"  {name:16} [数式を除く] {row}", flush=True)
        print(f"  {'':16} ラベル別 F1: {labels}")
        print(f"  {'':16} [数式] {math_row}  （数式も入れた検出F1 {full_summary['f1']:.3f}・ラベル {full_summary['label_acc']:.3f}）")
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
