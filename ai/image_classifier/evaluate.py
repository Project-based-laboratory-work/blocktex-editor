"""
同じOCR結果・同じ正解・同じ物差しで、ルールベース（image_rule_based）と自作AI（image_classifier）を比べる。

    python -m image_classifier.evaluate image_classifier/data/val
    python -m image_classifier.evaluate image_classifier/data/val --checkpoint a.pt --checkpoint b.pt

採点は evaluation/metrics.py（チーム共通）。どちらも入力の段落の区切り方は同じなので、
差が出るのは主にラベルの正しさ（label_acc）。ルールベースは heading / paragraph / caption しか出さないので、
正解がその3種類の要素だけに絞った label_acc も一緒に出す（ルールベースに不利にならない比べ方）。
"""

import argparse
import collections
import glob
import os
import time
from pathlib import Path

from common.ir import load_ir
from evaluation.metrics import Counts, count_document, match, summarize
from image_classifier.ir_adapter import DEFAULT_CHECKPOINT, classify_ir, load_model
from image_rule_based.ir_adapter import classify_ir as rule_classify_ir

RULE_LABELS = {"heading", "paragraph", "caption"}


def label_pairs(pred, truth) -> list[tuple[str, str]]:
    """IoUで対応が取れた (正解, 予測) のラベルの組"""
    pairs = []
    pred_pages = {p.page: p for p in pred.pages}
    for gt_page in truth.pages:
        pred_page = pred_pages.get(gt_page.page)
        if pred_page is None:
            continue
        for i, j in match([e.bbox for e in pred_page.elements], [e.bbox for e in gt_page.elements]):
            pairs.append((gt_page.elements[j].label, pred_page.elements[i].label))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="ルールベースと自作AIを、同じOCR結果・同じ正解で比べる")
    parser.add_argument("dir", help="images/ truth/ ocr/ があるディレクトリ（ocr/ は run_ocr.py で作る）")
    parser.add_argument("--checkpoint", type=Path, action="append", help="自作AIのモデル（複数指定可）")
    args = parser.parse_args()
    checkpoints = args.checkpoint or [DEFAULT_CHECKPOINT]

    truth_paths = sorted(glob.glob(os.path.join(args.dir, "truth", "*.ir.json")))
    systems = {"ルールベース": None} | {f"自作AI（{c.parent.name}）": load_model(c) for c in checkpoints}
    counts = {name: Counts() for name in systems}
    by_style = {name: collections.defaultdict(Counts) for name in systems}
    pairs = {name: [] for name in systems}
    seconds = collections.Counter()

    used = 0
    for truth_path in truth_paths:
        ocr_path = os.path.join(args.dir, "ocr", os.path.basename(truth_path))
        if not os.path.exists(ocr_path):
            continue
        used += 1
        truth, ocr = load_ir(truth_path), load_ir(ocr_path)
        image = os.path.join(args.dir, "images", truth.source)
        for name, model in systems.items():
            started = time.perf_counter()
            pred = rule_classify_ir(ocr) if model is None else classify_ir(ocr, image, *model)
            seconds[name] += time.perf_counter() - started
            count_document(pred, truth, counts[name])
            count_document(pred, truth, by_style[name][truth.params.get("style", "-")])
            pairs[name] += label_pairs(pred, truth)

    print(f"{args.dir}: {used} 枚（OCR結果があるもの）\n")
    print(f"{'':<22} {'label_acc':>9} {'3種のみ':>8} {'f1':>6} {'ms/枚':>7}")
    for name in systems:
        s = summarize(counts[name])
        rule_only = [(g, p) for g, p in pairs[name] if g in RULE_LABELS]
        acc3 = sum(g == p for g, p in rule_only) / max(len(rule_only), 1)
        print(f"{name:<20} {s['label_acc'] or 0:>9.3f} {acc3:>8.3f} {s['f1']:>6.3f} {seconds[name] / max(used, 1) * 1000:>7.1f}")
        for style, c in sorted(by_style[name].items()):
            print(f"  {style:<18} label_acc {summarize(c)['label_acc'] or 0:.3f}")

    print("\nラベルごとの再現率（正解がそのラベルのうち、当てられた割合）")
    labels = sorted({g for p in pairs.values() for g, _ in p})
    print(f"{'正解':<10}" + "".join(f"{name[:10]:>14}" for name in systems) + f"{'件数':>7}")
    for label in labels:
        row = []
        for name in systems:
            hits = [p == label for g, p in pairs[name] if g == label]
            row.append(sum(hits) / max(len(hits), 1))
        support = sum(1 for g, _ in pairs["ルールベース"] if g == label)
        print(f"{label:<10}" + "".join(f"{r:>14.3f}" for r in row) + f"{support:>7}")


if __name__ == "__main__":
    main()
