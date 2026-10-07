import pymupdf

from labels import Bbox, LabeledBlock, save_labels
from pdf_extract import  extract
from rule_based_classifier import OVERLAP_RATIO_THRESHOLD, _overlap_ratio

BLACK = 0x000000

# TeX側の \definecolor と値を一致させること。
# 表・図は検出結果からラベルが決まるので、ここには含めない。
COLOR_TO_LABEL: dict[int, str] = {
    0x000000: "paragraph",
    0xFF0000: "heading",
    0x00AA00: "caption",
    0x0000FF: "list",
    0xAA00AA: "math",
    0x888888: "other",
}


def color_to_label(color: int) -> str:
    """文字色からラベルを返す。"""
    if color not in COLOR_TO_LABEL:
        known = ", ".join(f"#{c:06X}" for c in COLOR_TO_LABEL)
        raise ValueError(f"不明な色です: #{color:06X}（使えるのは {knows}）")
    return COLOR_TO_LABEL[color]


def _is_inside(inner: tuple, outer: tuple, tol: float = 2.0) -> bool:
    """innerがouterの内側に収まっているか（tolだけはみ出しを許容）。"""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 - tol and iy0 >= oy0 - tol and ix1 <= ox1 + tol and iy1 <= oy1 + tol


def dominant_label(page: pymupdf.Page, bbox: Bbox) -> str:
    """bbox内のspanの色からラベルを決める。"""

    by_color: dict[int, int] = {}

    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if _is_inside(span["bbox"], bbox):
                    color = span["color"]
                    by_color[color] = by_color.get(color, 0) + len(span["text"])             

    colored = {}
    for c, n in by_color.items():
        if c != BLACK:
            colored[c] = n

    if colored:
        max_key = max(colored, key = colored.get)
        return color_to_label(max_key)
    
    return "paragraph"


def label_colored_pdf(colored_pdf: str) -> list[LabeledBlock]:
    """色付きPDFから抽出したブロックにラベルを付ける。"""
    result = []
    pages = extract(colored_pdf)

    with pymupdf.open(colored_pdf) as doc:
        for page_blocks in pages:
            page = doc[page_blocks.page]

            for table in page_blocks.table_blocks:
                result.append(
                    LabeledBlock(page=page_blocks.page, bbox=table.bbox, label="table", text=None, source="auto")
                )

            for figure in page_blocks.figure_blocks:
                result.append(
                    LabeledBlock(page=page_blocks.page, bbox=figure.bbox, label="figure", text=None, source="auto")
                )

            absorbed_bboxes = [t.bbox for t in page_blocks.table_blocks] + [
                f.bbox for f in page_blocks.figure_blocks
            ]

            for block in page_blocks.text_blocks:
                if any(
                    _overlap_ratio(block.bbox, absorbed) >= OVERLAP_RATIO_THRESHOLD
                    for absorbed in absorbed_bboxes
                ):
                    continue

                result.append(
                    LabeledBlock(
                        page=page_blocks.page,
                        bbox=block.bbox,
                        label=dominant_label(page, block.bbox),
                        text=block.text,
                        source="auto",
                    )
                )

    return result
