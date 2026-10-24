"""
4-1-1: PDFからテキスト・座標・フォントサイズを抽出する。

- テキスト: PyMuPDF (pymupdf) の get_text("dict") でブロック単位のテキスト＋bbox＋フォント情報を取得
- 表: pdfplumber の find_tables() でbboxを取得（罫線・列位置の検出精度がPyMuPDFより高いため）
- 図: PyMuPDF の get_images() / get_drawings() でbboxを取得し、罫線・ページ枠・表内の図形を除いたうえで近接するものを1つにまとめる

extract() は分類器向けの内部形式（PageBlocks。0始まりのページ・ptの座標）を返す。
外に出すときは extract_ir() で共通のIR形式（common/ir.py、仕様は common/README.md）にする。

選定理由・比較結果は README.md の「4-1-1 調査結果」を参照。

使い方（ai/ で）:
    python -m rule_based.pdf_extract rule_based/sample_data/sample.pdf               # IRを標準出力に出す
    python -m rule_based.pdf_extract rule_based/sample_data/sample.pdf -o out/sample.ir.json
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import pdfplumber
import pymupdf

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, to_dict, validate

NAME = "rule_based"


@dataclass
class TextBlock:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    font_name: str
    bold: bool


@dataclass
class TableBlock:
    page: int
    bbox: tuple[float, float, float, float]
    cells: list[list[str]] = field(default_factory=list)  # 行 × 列の文字列。空のセルは ""


@dataclass
class FigureBlock:
    page: int
    bbox: tuple[float, float, float, float]
    source: str 


@dataclass
class PageBlocks:
    page: int  # 0始まり（IRにするときに+1する）
    width: float  # pt
    height: float
    text_blocks: list[TextBlock] = field(default_factory=list)
    table_blocks: list[TableBlock] = field(default_factory=list)
    figure_blocks: list[FigureBlock] = field(default_factory=list)


def _extract_text_blocks(page: pymupdf.Page, page_no: int) -> list[TextBlock]:
    blocks = []
    for raw_block in page.get_text("dict")["blocks"]:
        if raw_block.get("type") != 0:  # 0 = テキストブロック（画像ブロックは1）
            continue

        spans = []
        for line in raw_block["lines"]:
            for span in line["spans"]:
                spans.append(span)
        if not spans:
            continue

        sorted_lines = sorted(raw_block["lines"], key=lambda line: (line["bbox"][1], line["bbox"][0]))
        groups = []
        for line in sorted_lines:
            top = line["bbox"][1]
            height = (line["bbox"][3] - line["bbox"][1]) / 2

            if groups and abs(top - groups[-1][0]["bbox"][1]) <= height:
                groups[-1].append(line)
            else:
                groups.append([line])
        
        rows = []
        for group in groups:
            parts = ["".join(span["text"] for span in line["spans"]) for line in group]
            rows.append(" ".join(parts))

        text = "\n".join(rows).strip()
        if not text:
            continue

        # ブロック内で最も文字数の多いフォントサイズ・フォント名を代表値とする
        # （見出しの記号などで一部だけフォントが違うケースを吸収するため）
        by_font: dict[tuple[float, str], int] = {}
        bold_chars = 0
        total_chars = 0
        for span in spans:
            key = (round(span["size"], 1), span["font"])
            by_font[key] = by_font.get(key, 0) + len(span["text"])
            total_chars += len(span["text"])
            if span["flags"] & (1 << 4):  # bit 4 = bold
                bold_chars += len(span["text"])

        font_size, font_name = max(by_font, key=lambda k: by_font[k])

        blocks.append(
            TextBlock(
                page=page_no,
                text=text,
                bbox=tuple(raw_block["bbox"]),
                font_size=font_size,
                font_name=font_name,
                bold=bold_chars > total_chars / 2,
            )
        )
    return blocks


def _is_inside(inner: tuple, outer: tuple, tol: float = 2.0) -> bool:
    """innerがouterの内側に収まっているか（tolだけはみ出しを許容）。"""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 - tol and iy0 >= oy0 - tol and ix1 <= ox1 + tol and iy1 <= oy1 + tol




def _extract_table_blocks(page: pdfplumber.page.Page, page_no: int) -> list[TableBlock]:
    result = []
    for table in page.find_tables():
        cells = [[cell or "" for cell in row] for row in table.extract()]
        result.append(TableBlock(page=page_no, bbox=table.bbox, cells=cells))
    return result

def _extract_figure_blocks(
    page: pymupdf.Page,
    page_no: int,
    table_bboxes: list[tuple[float, float, float, float]],
    min_size: float = 3.0,
    page_ratio: float = 0.8,
) -> list[FigureBlock]:
    figures = []
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            figures.append(FigureBlock(page=page_no, bbox=tuple(rect), source="image"))

    page_area = page.rect.width * page.rect.height
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width < min_size and rect.height < min_size:
            continue  # 罫線・下線
        if rect.width * rect.height >= page_area * page_ratio:
            continue  # ページ枠・背景
        if any(_is_inside(tuple(rect), table_bbox) for table_bbox in table_bboxes):
            continue  # 表の罫線・背景
        figures.append(FigureBlock(page=page_no, bbox=tuple(rect), source="drawing"))
    return figures


def _union_bbox(bboxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )

def _merge_figure_blocks(figure_blocks: list[FigureBlock],margin: float = 5.0,min_size: float = 5.0,) -> list[FigureBlock]:
    def _is_near(bbox_a, bbox_b, margin: float) -> bool:
        ax0,ay0,ax1,ay1 = bbox_a
        bx0,by0,bx1,by1 = bbox_b
        ax0, ay0, ax1, ay1 = ax0 - margin, ay0 - margin, ax1 + margin, ay1 + margin

        overlap_w = max(0.0,min(ax1,bx1) - max(ax0,bx0))
        overlap_h = max(0.0,min(ay1,by1) - max(ay0,by0))

        return overlap_h * overlap_w > 0.0
    
    remaining = list(figure_blocks)
    merged = []

    while remaining:
        group = [remaining.pop(0)]
        group_bbox = group[0].bbox

        while True:
            near = [block for block in remaining if _is_near(group_bbox, block.bbox, margin)]
            if not near:
                break
            
            group.extend(near)
            near_ids = {id(block) for block in near}
            remaining = [block for block in remaining if id(block) not in near_ids]
            group_bbox = _union_bbox([block.bbox for block in group])
        
        width = group_bbox[2] - group_bbox[0]
        height = group_bbox[3] - group_bbox[1]
        if width <= min_size or height <= min_size:
            continue
        
        sources = {block.source for block in group}
        merged.append(
            FigureBlock(
                page = group[0].page,
                bbox = group_bbox,
                source = sources.pop() if len(sources) == 1 else "mixed",
            )
        )
    return merged
        


def extract(pdf_path: str) -> list[PageBlocks]:
    pages = []
    with pymupdf.open(pdf_path) as doc, pdfplumber.open(pdf_path) as pdf:
        for page_no, (mu_page, plumber_page) in enumerate(zip(doc, pdf.pages)):
            text_blocks = _extract_text_blocks(mu_page, page_no)
            table_blocks = _extract_table_blocks(plumber_page, page_no)
            table_bboxes = [t.bbox for t in table_blocks]
            figure_blocks = _extract_figure_blocks(mu_page, page_no, table_bboxes)

            pages.append(
                PageBlocks(
                    page=page_no,
                    width=mu_page.rect.width,
                    height=mu_page.rect.height,
                    text_blocks=text_blocks,
                    table_blocks=table_blocks,
                    figure_blocks=_merge_figure_blocks(figure_blocks),
                )
            )
    return pages


def to_ir(pages: list[PageBlocks], source: str) -> IRDocument:
    """extract() の結果をIRにする。要素の順は 文字 → 表 → 図（読み順ではない）。"""
    ir_pages = []
    for page_blocks in pages:
        w, h = page_blocks.width, page_blocks.height
        page = Page(page=page_blocks.page + 1, width=w, height=h, unit="pt")
        page.elements += [
            Element(kind="text", bbox=normalize_bbox(b.bbox, w, h), text=b.text, font_size=b.font_size, bold=b.bold)
            for b in page_blocks.text_blocks
        ]
        page.elements += [
            Element(kind="table", bbox=normalize_bbox(b.bbox, w, h), cells=b.cells) for b in page_blocks.table_blocks
        ]
        page.elements += [Element(kind="figure", bbox=normalize_bbox(b.bbox, w, h)) for b in page_blocks.figure_blocks]
        ir_pages.append(page)
    return IRDocument(source=source, extractor=NAME, pages=ir_pages)


def extract_ir(pdf_path: str) -> IRDocument:
    return to_ir(extract(pdf_path), os.path.basename(pdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → IR（ルールベースの抽出）")
    parser.add_argument("pdf", nargs="?", default="rule_based/sample_data/sample.pdf", help="入力PDF")
    parser.add_argument("-o", "--out", help="出力するIRのJSON。省略すると標準出力に出す")
    args = parser.parse_args()

    doc = extract_ir(args.pdf)
    problems = validate(doc)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))

    if args.out:
        save_ir(doc, args.out)
        print(args.out)
    else:
        print(json.dumps(to_dict(doc), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
