"""
image_dataset の正解で、画像の構造化器（EasyOCR → grouping → image_rule_based）を採点する。

使い方（ai/ で）:
    python -m image_dataset.build --out image_rule_based/data/val --docs 30 --seed 1
    python -m image_rule_based.evaluate image_rule_based/data/val
      → 段落の切り方（precision / recall / f1）、ラベルの正しさ（label_acc）、見出しの適合率・再現率、style ごとの成績

EasyOCR は1枚に数秒〜十数秒かかるので、OCRの結果（段落単位のIR）を <dir>/ocr/ に保存し、2回目からはそれを使う。
"""

import argparse
import collections
import glob
import os

from common.ir import IRDocument, load_ir, save_ir
from evaluation.metrics import Counts, count_document, match, summarize
from image_rule_based.ir_adapter import classify_ir


def load_or_run_ocr(root: str, truth_path: str, truth: IRDocument) -> IRDocument:
    """保存済みのOCR結果を読む。無ければ EasyOCR（base）で読み、段落にまとめて保存する。"""
    path = os.path.join(root, "ocr", os.path.basename(truth_path))
    if os.path.exists(path):
        return load_ir(path)
    from easyocr_ocr.grouping import group_document  # EasyOCR の読み込みは重いので、必要なときだけ
    from easyocr_ocr.ir_adapter import extract_ir

    doc = group_document(extract_ir(os.path.join(root, "images", truth.source)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_ir(doc, path)
    return doc


def count_headings(pred: IRDocument, truth: IRDocument, counts: collections.Counter) -> None:
    """対応が取れた箱のうち、見出しの当たり（tp）・誤検出（fp）・取りこぼし（fn）を数える。"""
    pred_pages = {p.page: p for p in pred.pages}
    for gt_page in truth.pages:
        pred_page = pred_pages.get(gt_page.page)
        if pred_page is None:
            continue
        for i, j in match([e.bbox for e in pred_page.elements], [e.bbox for e in gt_page.elements]):
            p, g = pred_page.elements[i].label, gt_page.elements[j].label
            if p == g == "heading":
                counts["tp"] += 1
            elif p == "heading":
                counts["fp"] += 1
            elif g == "heading":
                counts["fn"] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="image_dataset の正解で画像の構造化器を採点する")
    parser.add_argument("dir", help="image_dataset.build の出力先（images/ と truth/ があるディレクトリ）")
    args = parser.parse_args()

    truth_paths = sorted(glob.glob(os.path.join(args.dir, "truth", "*.ir.json")))
    if not truth_paths:
        raise SystemExit(f"エラー: {args.dir}/truth/ に正解がありません。")

    counts = Counts()
    by_style: dict[str, Counts] = collections.defaultdict(Counts)
    headings: collections.Counter = collections.Counter()
    for n, truth_path in enumerate(truth_paths, 1):
        truth = load_ir(truth_path)
        pred = classify_ir(load_or_run_ocr(args.dir, truth_path, truth))
        count_document(pred, truth, counts)
        count_document(pred, truth, by_style[truth.params.get("style", "-")])
        count_headings(pred, truth, headings)
        print(f"\r{n}/{len(truth_paths)}", end="", flush=True)
    print()

    scores = summarize(counts)
    print("全体:", ", ".join(f"{k} {v:.3f}" for k, v in scores.items() if v is not None))
    tp, fp, fn = headings["tp"], headings["fp"], headings["fn"]
    print(f"見出し: 適合率 {tp / max(1, tp + fp):.3f}  再現率 {tp / max(1, tp + fn):.3f}  （当たり {tp}・誤検出 {fp}・取りこぼし {fn}）")
    for style, c in sorted(by_style.items()):
        s = summarize(c)
        print(f"  {style}: f1 {s['f1']:.3f}  label_acc {s['label_acc'] or 0:.3f}")


if __name__ == "__main__":
    main()
