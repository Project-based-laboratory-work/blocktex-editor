"""
座標AI（image_layout）と、ルールベース側（EasyOCR → 段落まとめ → image_rule_based）を、
同じ画像・同じ正解・同じ物差し（evaluation/metrics.py）で比べる。

    python -m image_layout.evaluate image_classifier/data/val --checkpoint image_layout/checkpoints/unet/best.pt

  - 検出の f1: 正解のブロックと四角が IoU 0.5 以上で重なった割合（ブロックの見つけ方・区切り方の正しさ）
  - label_acc: 対応が取れたブロックのうち、ラベルも合っていた割合
  - 表・図: ルールベース側は表・図の位置を出さない（OCRは文字しか見つけない）ので、ラベルごとの再現率も出す

ルールベース側のOCR結果は <dir>/ocr/（image_classifier/run_ocr.py で作ったもの）を使う。
座標AIの文字は、同じOCR結果から、四角に入る段落の文字をつなげて入れる（CERの比較用）。
"""

import argparse
import collections
import glob
import os
import time
from pathlib import Path

import torch

from common.ir import load_ir
from evaluation.metrics import Counts, count_document, match, summarize
from image_layout.dataset import load_canvas
from image_layout.model import LayoutUNet, ModelConfig
from image_layout.postprocess import logits_to_classes, to_ir
from image_rule_based.ir_adapter import classify_ir as rule_classify_ir


def load_model(checkpoint: Path) -> LayoutUNet:
    state = torch.load(checkpoint, map_location="cpu")
    model = LayoutUNet(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: LayoutUNet, image_path: str, width: int, height: int, source: str, texts=None):
    canvas, scale = load_canvas(image_path)
    classes = logits_to_classes(model(torch.from_numpy(canvas)[None, None])[0].numpy())
    return to_ir(classes, source, width, height, scale, texts)


def main() -> None:
    parser = argparse.ArgumentParser(description="座標AIとルールベース側を、同じ正解で比べる")
    parser.add_argument("dir", help="images/ truth/ ocr/ があるディレクトリ")
    parser.add_argument("--checkpoint", type=Path, default=Path("image_layout/checkpoints/unet/best.pt"))
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    names = ["ルールベース（OCR＋ルール）", "座標AI"]
    counts = {n: Counts() for n in names}
    by_style = {n: collections.defaultdict(Counts) for n in names}
    found = {n: collections.Counter() for n in names}  # ラベルごとの、対応が取れた正解の数
    total = collections.Counter()
    seconds = collections.Counter()

    used = 0
    for truth_path in sorted(glob.glob(os.path.join(args.dir, "truth", "*.ir.json"))):
        ocr_path = os.path.join(args.dir, "ocr", os.path.basename(truth_path))
        if not os.path.exists(ocr_path):
            continue
        used += 1
        truth, ocr = load_ir(truth_path), load_ir(ocr_path)
        page = truth.pages[0]
        texts = [(el.bbox, el.text) for el in ocr.pages[0].elements if el.text]
        style = truth.params.get("style", "-")
        total.update(el.label for el in page.elements)

        started = time.perf_counter()
        preds = {names[0]: rule_classify_ir(ocr)}
        seconds[names[0]] += time.perf_counter() - started
        started = time.perf_counter()
        preds[names[1]] = predict(model, os.path.join(args.dir, "images", truth.source), page.width, page.height, truth.source, texts)
        seconds[names[1]] += time.perf_counter() - started

        for name, pred in preds.items():
            count_document(pred, truth, counts[name])
            count_document(pred, truth, by_style[name][style])
            for _, j in match([e.bbox for e in pred.pages[0].elements], [e.bbox for e in page.elements]):
                found[name][page.elements[j].label] += 1

    print(f"{args.dir}: {used} 枚\n")
    print(f"{'':<26} {'f1':>6} {'適合率':>6} {'再現率':>6} {'mIoU':>6} {'label_acc':>9} {'CER':>6} {'ms/枚':>7}")
    for name in names:
        s = summarize(counts[name])
        print(
            f"{name:<24} {s['f1']:>6.3f} {s['precision']:>7.3f} {s['recall']:>7.3f} {s['miou'] or 0:>6.3f} "
            f"{s['label_acc'] or 0:>9.3f} {s['cer'] or 0:>6.3f} {seconds[name] / max(used, 1) * 1000:>7.1f}"
        )
        for style, c in sorted(by_style[name].items()):
            st = summarize(c)
            print(f"  {style:<22} f1 {st['f1']:.3f}  label_acc {st['label_acc'] or 0:.3f}")
    print("\nラベルごとの検出の再現率（正解のブロックのうち、IoU 0.5 以上の四角で見つけられた割合。ラベルは問わない）")
    print(f"{'正解':<10}" + "".join(f"{n[:12]:>16}" for n in names) + f"{'件数':>7}")
    for label, n in total.most_common():
        print(f"{label:<10}" + "".join(f"{found[name][label] / n:>16.3f}" for name in names) + f"{n:>7}")
    print("\n注: ルールベース側の時間はOCRを含まない（保存済みのOCR結果を使用）。座標AIの時間はCPUでの推論")


if __name__ == "__main__":
    main()
