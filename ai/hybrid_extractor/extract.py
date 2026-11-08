"""PDFから文字を取り出して、共通のIR（common/README.md の①）にする抽出器。

使い方（ai/ で実行）:
    python -m hybrid_extractor.extract hybrid_extractor/sample_data/sample.pdf
"""

import sys

import pymupdf

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir


def page_lines(page: pymupdf.Page) -> list[dict]:
    """1ページの行を、{"text", "bbox", "size"} の辞書のリストにして返す。"""
    lines = []
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = ""
            size = 0.0
            for span in line["spans"]:
                text += span["text"]                              # ① text に span の文字をつなげる
                size = max(size, span["size"])                       # ② 今の size と span のサイズの、大きい方
            lines.append({"text": text, "bbox": line["bbox"], "size": size})
    return lines


def extract(pdf_path: str) -> IRDocument:
    """PDFの全ページを読んで、IR（①）を返す。今は1行 = 1要素。"""
    pages = []
    doc = pymupdf.open(pdf_path)
    for page in doc:
        width = page.rect.width
        height = page.rect.height
        elements = []
        for line in page_lines(page):
            elements.append(Element(
                kind="text",
                bbox=normalize_bbox(line["bbox"], width, height),
                text=line["text"],
                font_size=line["size"],
            ))
        pages.append(Page(
            page=page.number + 1,
            width=width,
            height=height,
            unit="pt",
            elements=elements,
        ))
    return IRDocument(source=pdf_path, extractor="hybrid_extractor", pages=pages)

def main() -> None:
    ir = extract(sys.argv[1])
    save_ir(ir, sys.argv[2])
    print("保存しました:", sys.argv[2])


if __name__ == "__main__":
    main()
