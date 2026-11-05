"""
画像のIR（OCRの結果、段落単位）を学習済みモデルで分類し、ラベル付きIR・エディタ用JSONにするアダプタ。
image_rule_based/ir_adapter.py の AI版。入力・出力の形は同じ。

    python -m image_classifier.ir_adapter <IR> --image <元の画像> -o out/
      → out/<名前>.labeled.json（ラベル付きIR）, out/<名前>.blocks.json（エディタ用JSON）

要素の画像を見るので、IRに加えて元の画像が要る。
"""

import argparse
import json
import os
from pathlib import Path

import torch

from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page, load_ir, save_ir, to_dict, validate
from image_classifier.features import page_inputs
from image_classifier.model import ModelConfig, PageClassifier

NAME = "image_classifier"
DEFAULT_CHECKPOINT = Path("image_classifier/checkpoints/image/best.pt")


def load_model(checkpoint: Path = DEFAULT_CHECKPOINT) -> tuple[PageClassifier, list[str]]:
    state = torch.load(checkpoint, map_location="cpu")
    model = PageClassifier(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model, state["labels"]


@torch.no_grad()
def classify_ir(doc: IRDocument, image_path: str, model: PageClassifier, labels: list[str]) -> IRDocument:
    """1ページ1画像のIRを分類し、ラベル付きIRにする。要素の順番・位置は入力のまま。"""
    pages = []
    for src in doc.pages:
        page = Page(page=src.page, width=src.width, height=src.height, unit=src.unit)
        pages.append(page)
        if not src.elements:
            continue
        features, crops = page_inputs(src, image_path)
        padding_mask = torch.zeros(1, len(src.elements), dtype=torch.bool)
        logits = model(features[None], crops[None].float() / 255.0, padding_mask)[0]
        for el, label_id in zip(src.elements, logits.argmax(-1).tolist()):
            page.elements.append(
                Element(
                    kind="text", bbox=el.bbox, text=el.text, font_size=el.font_size, bold=el.bold,
                    confidence=el.confidence, label=labels[label_id],
                )
            )  # fmt: skip
    return IRDocument(source=doc.source, extractor=doc.extractor, structurer=NAME, pages=pages, params=doc.params)


def main() -> None:
    parser = argparse.ArgumentParser(description="画像のIR → ラベル付きIR・エディタ用JSON（自作AI）")
    parser.add_argument("ir", help="入力のIR（段落単位。EasyOCR は --group を付けて出したもの）")
    parser.add_argument("--image", required=True, help="IRの元の画像")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument("--stdout", action="store_true", help='{"labeled": …, "document": …} を標準出力に出す')
    args = parser.parse_args()

    doc = load_ir(args.ir)
    problems = validate(doc)
    if problems:
        raise SystemExit("入力のIRの形式に問題があります:\n" + "\n".join(problems))
    model, labels = load_model(args.checkpoint)
    labeled = classify_ir(doc, args.image, model, labels)
    document = ir_to_document(labeled)

    if args.stdout:
        print(json.dumps({"labeled": to_dict(labeled), "document": document}, ensure_ascii=False))
        return
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.basename(args.ir).removesuffix(".json").removesuffix(".ir")
    save_ir(labeled, os.path.join(args.out_dir, f"{stem}.labeled.json"))
    with open(os.path.join(args.out_dir, f"{stem}.blocks.json"), "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out_dir}/{stem}.labeled.json, {stem}.blocks.json")


if __name__ == "__main__":
    main()
