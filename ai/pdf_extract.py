"""
4-1-1: PDFからテキスト・座標・フォントサイズを抽出する。

- テキスト: PyMuPDF (pymupdf) の get_text("dict") でブロック単位のテキスト＋bbox＋フォント情報を取得
- 表: pdfplumber の find_tables() でbboxを取得（罫線・列位置の検出精度がPyMuPDFより高いため）
- 図: PyMuPDF の get_images() / get_drawings() でラスター画像・ベクター図形のbboxを取得

選定理由・比較結果は README.md の「4-1-1 調査結果」を参照。

使い方:
    python pdf_extract.py sample_data/sample.pdf
"""

import sys
from dataclasses import dataclass, field

import pdfplumber
import pymupdf


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


@dataclass
class FigureBlock:
    page: int
    bbox: tuple[float, float, float, float]
    source: str  # "image" (ラスター画像) or "drawing" (ベクター図形)


@dataclass
class PageBlocks:
    page: int
    text_blocks: list[TextBlock] = field(default_factory=list)
    table_blocks: list[TableBlock] = field(default_factory=list)
    figure_blocks: list[FigureBlock] = field(default_factory=list)


def _extract_text_blocks(page: pymupdf.Page, page_no: int) -> list[TextBlock]:
    blocks = []
    for raw_block in page.get_text("dict")["blocks"]:
        if raw_block.get("type") != 0:  # 0 = テキストブロック（画像ブロックは1）
            continue

        spans = [span for line in raw_block["lines"] for span in line["spans"]]
        if not spans:
            continue

        text = "".join(span["text"] for span in spans).strip()
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


def _extract_table_blocks(pdf_path: str, page_no: int) -> list[TableBlock]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no]
        return [TableBlock(page=page_no, bbox=table.bbox) for table in page.find_tables()]


def _extract_figure_blocks(page: pymupdf.Page, page_no: int) -> list[FigureBlock]:
    figures = []
    for image in page.get_images(full=True):
        xref = image[0]
        for rect in page.get_image_rects(xref):
            figures.append(FigureBlock(page=page_no, bbox=tuple(rect), source="image"))

    for drawing in page.get_drawings():
        rect = drawing["rect"]
        # 罫線1本などの細長い矩形は表の枠線であることが多く図として扱わないため除外
        if rect.width > 5 and rect.height > 5:
            figures.append(FigureBlock(page=page_no, bbox=tuple(rect), source="drawing"))

    return figures


def extract(pdf_path: str) -> list[PageBlocks]:
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_no, page in enumerate(doc):
        pages.append(
            PageBlocks(
                page=page_no,
                text_blocks=_extract_text_blocks(page, page_no),
                table_blocks=_extract_table_blocks(pdf_path, page_no),
                figure_blocks=_extract_figure_blocks(page, page_no),
            )
        )
    return pages


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample.pdf"
    for page in extract(pdf_path):
        print(f"=== page {page.page} ===")
        for block in page.text_blocks:
            print(f"[text] size={block.font_size} bold={block.bold} font={block.font_name!r} text={block.text[:40]!r}")
        for block in page.table_blocks:
            print(f"[table] bbox={block.bbox}")
        for block in page.figure_blocks:
            print(f"[figure:{block.source}] bbox={block.bbox}")


if __name__ == "__main__":
    main()
