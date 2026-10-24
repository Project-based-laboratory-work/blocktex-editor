"""
rule_based の分類結果・学習ラベルを、共通のIR形式（common/ir.py）に変換するアダプタ（パイプライン PDF-A）。
抽出結果のIR化は pdf_extract.extract_ir が行う。rule_based_classifier.py / labels.py の出力形式は変えず、
違い（0始まりのページ、ptの座標、表・図が別リスト）はここで吸収する。

使い方（ai/ で）:
    python -m rule_based.ir_adapter rule_based/sample_data/sample.pdf -o out/
      → out/sample.ir.json       抽出結果（IR）
        out/sample.labeled.json  分類結果（ラベル付きIR）
        out/sample.blocks.json   エディタ用JSON
    python -m rule_based.ir_adapter rule_based/sample_data/sample.pdf --stdout
      → {"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す（server/ の /api/extract が使う）
"""

import argparse
import json
import os

import pymupdf

from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, to_dict, validate
from rule_based.labels import load_labels
from rule_based.pdf_extract import NAME, extract, extract_ir
from rule_based.rule_based_classifier import classify_page


def _empty_pages(pdf_path: str) -> list[Page]:
    with pymupdf.open(pdf_path) as doc:
        return [Page(page=i + 1, width=p.rect.width, height=p.rect.height, unit="pt") for i, p in enumerate(doc)]


def classify_ir(pdf_path: str) -> IRDocument:
    """rule_based_classifier の分類結果をラベル付きIRにする。要素は分類器が並べた読み順。"""
    pages = []
    for page_blocks in extract(pdf_path):
        page = Page(page=page_blocks.page + 1, width=page_blocks.width, height=page_blocks.height, unit="pt")
        pages.append(page)
        cells_by_bbox = {t.bbox: t.cells for t in page_blocks.table_blocks}
        # 分類結果には文字サイズ・太字が無いので、同じbboxの抽出結果から引く（見出しレベルの推定に使う）
        text_by_bbox = {b.bbox: b for b in page_blocks.text_blocks}
        for cb in classify_page(page_blocks):
            bbox = normalize_bbox(cb.bbox, page.width, page.height)
            if cb.kind in ("table", "figure"):
                table_cells = cells_by_bbox.get(cb.bbox) if cb.kind == "table" else None
                page.elements.append(Element(kind=cb.kind, bbox=bbox, cells=table_cells, label=cb.kind))
                continue
            source = text_by_bbox.get(cb.bbox)
            page.elements.append(
                Element(
                    kind="text",
                    bbox=bbox,
                    text=cb.text,
                    font_size=source.font_size if source else None,
                    bold=source.bold if source else None,
                    label=cb.kind,
                )
            )
    return IRDocument(source=os.path.basename(pdf_path), extractor=NAME, structurer=NAME, pages=pages)


def labels_to_ir(label_json: str, pdf_path: str) -> IRDocument:
    """labels.py 形式の学習ラベルをラベル付きIRにする。ページサイズを知るために元のPDFも要る。"""
    _, blocks = load_labels(label_json)
    pages = _empty_pages(pdf_path)
    for b in blocks:
        page = pages[b.page]
        kind = b.label if b.label in ("table", "figure") else "text"
        page.elements.append(
            Element(
                kind=kind,
                bbox=normalize_bbox(b.bbox, page.width, page.height),
                text=b.text if kind == "text" else None,
                label=b.label,
                label_source=b.source,
            )
        )
    return IRDocument(source=os.path.basename(pdf_path), extractor=NAME, structurer="labels", pages=pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → IR・ラベル付きIR・エディタ用JSON（ルールベース）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help='ファイルに書かず、{"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す（server/ の /api/extract 用）',
    )
    parser.add_argument("--title", help="エディタ用JSONの文書名（省略時は入力ファイル名）")
    args = parser.parse_args()

    if args.stdout:
        labeled = classify_ir(args.pdf)
        problems = validate(labeled)
        if problems:
            raise SystemExit("labeled の形式に問題があります:\n" + "\n".join(problems))
        print(json.dumps({"labeled": to_dict(labeled), "document": ir_to_document(labeled, args.title)}, ensure_ascii=False))
        return

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.pdf))[0])

    labeled = classify_ir(args.pdf)
    for suffix, doc in (("ir", extract_ir(args.pdf)), ("labeled", labeled)):
        problems = validate(doc)
        if problems:
            raise SystemExit(f"{suffix} の形式に問題があります:\n" + "\n".join(problems))
        save_ir(doc, f"{stem}.{suffix}.json")
        print(f"{stem}.{suffix}.json")

    document = ir_to_document(labeled, args.title)
    with open(f"{stem}.blocks.json", "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"{stem}.blocks.json（{len(document['blocks'])}ブロック）")


if __name__ == "__main__":
    main()
