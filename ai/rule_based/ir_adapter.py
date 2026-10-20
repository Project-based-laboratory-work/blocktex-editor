"""
rule_based の抽出結果・分類結果・学習ラベルを、共通のIR形式（common/ir.py）に変換するアダプタ（パイプライン PDF-A）。
pdf_extract.py / rule_based_classifier.py / labels.py の出力形式は変えず、違い（0始まりのページ、
ptの座標、表・図が別リスト）はここで吸収する。

使い方（ai/ で）:
    python -m rule_based.ir_adapter rule_based/sample_data/sample.pdf -o out/
      → out/sample.ir.json       抽出結果（IR）
        out/sample.labeled.json  分類結果（ラベル付きIR）
        out/sample.blocks.json   エディタ用JSON
"""

import argparse
import json
import os

import pdfplumber
import pymupdf

from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from rule_based.labels import load_labels
from rule_based.pdf_extract import extract
from rule_based.rule_based_classifier import classify_page

NAME = "rule_based"


def _empty_pages(pdf_path: str) -> list[Page]:
    with pymupdf.open(pdf_path) as doc:
        return [Page(page=i + 1, width=p.rect.width, height=p.rect.height, unit="pt") for i, p in enumerate(doc)]


def _table_cells(pdf_path: str) -> dict[tuple[int, tuple], list[list[str]]]:
    """表の中身を {(ページ, bbox): 行×列の文字列} で返す。pdf_extract は bbox しか持たないため、ここで読み直す。"""
    cells = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages):
            for table in page.find_tables():
                cells[(page_no, table.bbox)] = [[c or "" for c in row] for row in table.extract()]
    return cells


def extract_ir(pdf_path: str) -> IRDocument:
    """pdf_extract.extract の結果をIRにする。要素の順は 文字 → 表 → 図（読み順ではない）。"""
    pages = _empty_pages(pdf_path)
    cells = _table_cells(pdf_path)
    for page_blocks in extract(pdf_path):
        page = pages[page_blocks.page]
        w, h = page.width, page.height
        page.elements += [
            Element(kind="text", bbox=normalize_bbox(b.bbox, w, h), text=b.text, font_size=b.font_size, bold=b.bold)
            for b in page_blocks.text_blocks
        ]
        page.elements += [
            Element(kind="table", bbox=normalize_bbox(b.bbox, w, h), cells=cells.get((page_blocks.page, b.bbox)))
            for b in page_blocks.table_blocks
        ]
        page.elements += [Element(kind="figure", bbox=normalize_bbox(b.bbox, w, h)) for b in page_blocks.figure_blocks]
    return IRDocument(source=os.path.basename(pdf_path), extractor=NAME, pages=pages)


def classify_ir(pdf_path: str) -> IRDocument:
    """rule_based_classifier の分類結果をラベル付きIRにする。要素は分類器が並べた読み順。"""
    pages = _empty_pages(pdf_path)
    cells = _table_cells(pdf_path)
    for page_blocks in extract(pdf_path):
        page = pages[page_blocks.page]
        # 分類結果には文字サイズ・太字が無いので、同じbboxの抽出結果から引く（見出しレベルの推定に使う）
        text_by_bbox = {b.bbox: b for b in page_blocks.text_blocks}
        for cb in classify_page(page_blocks):
            bbox = normalize_bbox(cb.bbox, page.width, page.height)
            if cb.kind in ("table", "figure"):
                table_cells = cells.get((page_blocks.page, cb.bbox)) if cb.kind == "table" else None
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.pdf))[0])

    labeled = classify_ir(args.pdf)
    for suffix, doc in (("ir", extract_ir(args.pdf)), ("labeled", labeled)):
        problems = validate(doc)
        if problems:
            raise SystemExit(f"{suffix} の形式に問題があります:\n" + "\n".join(problems))
        save_ir(doc, f"{stem}.{suffix}.json")
        print(f"{stem}.{suffix}.json")

    document = ir_to_document(labeled)
    with open(f"{stem}.blocks.json", "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"{stem}.blocks.json（{len(document['blocks'])}ブロック）")


if __name__ == "__main__":
    main()
