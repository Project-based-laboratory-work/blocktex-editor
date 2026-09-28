"""
4-1-1: PDFからテキスト・座標・フォントサイズを抽出する。

- テキスト: PyMuPDF (pymupdf) の get_text("dict") でブロック単位のテキスト＋bbox＋フォント情報を取得
- 表: pdfplumber の find_tables() でbboxを取得（罫線・列位置の検出精度がPyMuPDFより高いため）
- 図: PyMuPDF の get_images() / get_drawings() でbboxを取得し、罫線・ページ枠・表内の図形を除いたうえで近接するものを1つにまとめる

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
    source: str 


@dataclass
class PageBlocks:
    page: int
    width: float
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

        lines = ["".join(span["text"] for span in line["spans"]) for line in raw_block["lines"]]
        text = "\n".join(lines).strip()
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
        result.append(TableBlock(page=page_no, bbox=table.bbox))
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
                    text_blocks=text_blocks,
                    table_blocks=table_blocks,
                    figure_blocks=_merge_figure_blocks(figure_blocks),
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
