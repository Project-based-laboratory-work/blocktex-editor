"""
TeXで作ったPDFから、正解のラベル付きIRを作る。

目印（marks.py）から各ブロックの縦の範囲（1行目〜最後の行のベースライン）が分かるので、
その範囲にある文字の枠を全部包む枠を、そのブロックの正解の枠にする。
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from common.ir import Element, IRDocument, Page, normalize_bbox
from evaluation.marks import read_marks
from evaluation.synth_doc import SourceBlock

# 上付き・下付きの文字は行のベースラインから少しずれるので、その分だけ範囲を広げる（pt）。
# 行と行の間は16pt前後あるので、5ptなら隣の行の文字は入らない
BASELINE_TOLERANCE = 5.0

PtBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)。pt、上からの距離（正規化する前）


@dataclass
class Glyph:
    """PDFに印刷された1文字"""
    page: int        # 1始まり
    bbox: PtBox
    baseline: float  # 文字が乗っている線の、上からの距離（pt）
    c: str = ""      # その文字。空白も入る


def union_bbox(boxes: list[PtBox]) -> PtBox:
    """全部の枠を包む、一番小さい枠"""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def glyphs_in_block(glyphs: list[Glyph], page: int, top: float, bottom: float) -> list[Glyph]:
    """page の、ベースラインが top〜bottom（前後に BASELINE_TOLERANCE の余裕）にある文字"""
    return [g for g in glyphs if g.page == page and top - BASELINE_TOLERANCE <= g.baseline <= bottom + BASELINE_TOLERANCE]


def block_bbox(glyphs: list[Glyph], page: int, top: float, bottom: float) -> PtBox | None:
    """ブロックの文字を包む枠。空白は枠を広げてしまうので使わない。文字が無ければ None"""
    boxes = [g.bbox for g in glyphs_in_block(glyphs, page, top, bottom) if not g.c.isspace()]
    if not boxes:
        return None
    return union_bbox(boxes)


def read_glyphs(pdf_path: str) -> list[Glyph]:
    """PDFに印刷された文字を1文字ずつ読む。空白も含める（枠を作るときは block_bbox が空白を使わない）。"""
    glyphs = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for block in page.get_text("rawdict")["blocks"]:
                for line in block.get("lines", []):  # 画像のブロックには lines が無い
                    for span in line["spans"]:
                        for char in span["chars"]:
                            glyphs.append(Glyph(page=page_index + 1, bbox=tuple(char["bbox"]), baseline=char["origin"][1], c=char["c"]))
    return glyphs


def block_text(glyphs: list[Glyph], page: int, top: float, bottom: float) -> str:
    """ブロックの文字列。ベースラインが変わったところで改行（\\n）を入れ、各行の前後の空白は取る"""
    text = ""
    prev_baseline = None
    for g in glyphs_in_block(glyphs, page, top, bottom):
        if prev_baseline is not None and abs(g.baseline - prev_baseline) > BASELINE_TOLERANCE:
            text += "\n"
        text += g.c
        prev_baseline = g.baseline
    return "\n".join(line.strip() for line in text.split("\n"))


# 目印（pdf:dest）を消さずに残すため、dvipdfmx に -C 0x0010 を付ける
COMPILE_COMMANDS = [
    ["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
    ["dvipdfmx", "-q", "-C", "0x0010", "doc.dvi"],
]


def compile_tex(tex: str, work_dir: Path) -> Path:
    """TeXを work_dir/doc.tex に書いて PDF にし、PDF のパスを返す。失敗したら subprocess.CalledProcessError"""
    (work_dir / "doc.tex").write_text(tex, encoding="utf-8")
    for command in COMPILE_COMMANDS:
        subprocess.run(command, cwd=work_dir, check=True, capture_output=True, timeout=120)
    return work_dir / "doc.pdf"


def build_ground_truth(pdf_path: str, blocks: list[SourceBlock], source: str) -> IRDocument:
    """目印付きのPDFから、正解のラベル付きIRを作る。要素はブロックを置いた順（＝読み順）に並ぶ。
    ページをまたぐブロックは、ページごとの要素に分ける。"""
    marks = read_marks(pdf_path)
    glyphs = read_glyphs(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        pages = [Page(page=i + 1, width=p.rect.width, height=p.rect.height, unit="pt") for i, p in enumerate(doc)]

    for block in blocks:
        start, end = marks.get(f"b{block.id}s"), marks.get(f"b{block.id}e")
        if start is None or end is None:  # 目印がPDFに出なかった（ページの外に流れたなど）
            continue
        for page_no in range(start.page, end.page + 1):
            page = pages[page_no - 1]
            top = start.y if page_no == start.page else 0.0
            bottom = end.y if page_no == end.page else page.height
            box = block_bbox(glyphs, page_no, top, bottom)
            if box is None:
                continue
            page.elements.append(Element(
                kind="text",
                bbox=normalize_bbox(box, page.width, page.height),
                text=block_text(glyphs, page_no, top, bottom),
                label=block.label,
                level=block.level,
                label_source="auto",  # TeXから自動で作った正解
            ))
    return IRDocument(source=source, extractor="ground_truth", structurer="ground_truth", pages=pages)
