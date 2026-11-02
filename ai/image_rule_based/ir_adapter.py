"""
画像のIR（OCRの結果）を image_classifier で分類し、ラベル付きIR・エディタ用JSONにするアダプタ（パイプライン IMG-A の構造化）。
入力は段落（ブロック）単位のIR。EasyOCR は行単位で返すので、--group を付けて段落にまとめたものを渡す。

使い方（ai/ で）:
    python -m easyocr_ocr.ir_adapter ../ocr/sample/eng/eng_001.jpg --group -o image_rule_based/data/out/
    python -m image_rule_based.ir_adapter image_rule_based/data/out/eng_001.base.group.ir.json -o image_rule_based/data/out/
      → image_rule_based/data/out/eng_001.base.group.labeled.json  分類結果（ラベル付きIR）
        image_rule_based/data/out/eng_001.base.group.blocks.json   エディタ用JSON
    python -m image_rule_based.ir_adapter <IR> --stdout
      → {"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す（rule_based.ir_adapter と同じ形）
"""

import argparse
import json
import os

from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page, denormalize_bbox, load_ir, normalize_bbox, save_ir, to_dict, validate
from image_rule_based.image_classifier import classify_page

NAME = "image_rule_based"


def classify_ir(doc: IRDocument) -> IRDocument:
    """抽出しただけのIRを分類し、ラベル付きIRにする。要素は分類器が並べた読み順。"""
    pages = []
    for src in doc.pages:
        page = Page(page=src.page, width=src.width, height=src.height, unit=src.unit)
        pages.append(page)
        # 分類結果には文字サイズ・表のセルなどが無いので、同じbboxの元の要素から引く
        by_bbox = {denormalize_bbox(el.bbox, src.width, src.height): el for el in src.elements}
        for cb in classify_page(src):
            source = by_bbox.get(cb.bbox)
            page.elements.append(
                Element(
                    kind=cb.kind if cb.kind in ("table", "figure") else "text",
                    bbox=normalize_bbox(cb.bbox, page.width, page.height),
                    text=cb.text,
                    cells=source.cells if source and cb.kind == "table" else None,
                    font_size=source.font_size if source else None,
                    bold=source.bold if source else None,
                    confidence=source.confidence if source else None,
                    label=cb.kind,
                )
            )
    return IRDocument(source=doc.source, extractor=doc.extractor, structurer=NAME, pages=pages, params=doc.params)


def main() -> None:
    parser = argparse.ArgumentParser(description="画像のIR → ラベル付きIR・エディタ用JSON（ルールベース）")
    parser.add_argument("ir", help="入力のIR（段落単位。EasyOCR は --group を付けて出したもの）")
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help='ファイルに書かず、{"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す',
    )
    parser.add_argument("--title", help="エディタ用JSONの文書名（省略時は元の画像のファイル名）")
    args = parser.parse_args()

    if not os.path.isfile(args.ir):
        raise SystemExit(f"エラー: {args.ir} が見つかりません。")
    doc = load_ir(args.ir)
    problems = validate(doc)
    if problems:
        raise SystemExit("入力のIRの形式に問題があります:\n" + "\n".join(problems))

    labeled = classify_ir(doc)
    problems = validate(labeled)
    if problems:
        raise SystemExit("labeled の形式に問題があります:\n" + "\n".join(problems))
    document = ir_to_document(labeled, args.title)

    if args.stdout:
        print(json.dumps({"labeled": to_dict(labeled), "document": document}, ensure_ascii=False))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(args.ir)
    name = name[: -len(".ir.json")] if name.endswith(".ir.json") else os.path.splitext(name)[0]
    stem = os.path.join(args.out_dir, name)

    save_ir(labeled, f"{stem}.labeled.json")
    print(f"{stem}.labeled.json")
    with open(f"{stem}.blocks.json", "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"{stem}.blocks.json（{len(document['blocks'])}ブロック）")


if __name__ == "__main__":
    main()
