"""
TeXで作ったPDFから、正解のラベル付きIRを作る。

目印（marks.py）から各ブロックの縦の範囲（1行目〜最後の行のベースライン）が分かるので、
その範囲にある文字の枠を全部包む枠を、そのブロックの正解の枠にする。
文中の装飾（runs）も、装飾した語の前後の目印の間にある文字を、その装飾の文字とする。
"""
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from common.ir import Element, IRDocument, Page, Run, normalize_bbox
from evaluation.marks import Mark, read_marks
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


def block_lines(glyphs: list[Glyph], page: int, top: float, bottom: float) -> list[list[Glyph]]:
    """ブロックの文字を行ごとに分ける。ベースラインが変わったところで次の行にし、各行の前後の空白は取る"""
    lines: list[list[Glyph]] = []
    prev_baseline = None
    for g in glyphs_in_block(glyphs, page, top, bottom):
        if prev_baseline is None or abs(g.baseline - prev_baseline) > BASELINE_TOLERANCE:
            lines.append([])
        lines[-1].append(g)
        prev_baseline = g.baseline

    def strip(line: list[Glyph]) -> list[Glyph]:
        start, end = 0, len(line)
        while start < end and line[start].c.isspace():
            start += 1
        while end > start and line[end - 1].c.isspace():
            end -= 1
        return line[start:end]

    return [strip(line) for line in lines]


def block_text(glyphs: list[Glyph], page: int, top: float, bottom: float) -> str:
    """ブロックの文字列。ベースラインが変わったところで改行（\\n）を入れ、各行の前後の空白は取る"""
    return "\n".join("".join(g.c for g in line) for line in block_lines(glyphs, page, top, bottom))


def _at_or_after(g: Glyph, m: Mark) -> bool:
    """文字 g が、読む順で目印 m の位置かそれより後ろにあるか。同じ行なら文字の中心の x で比べる"""
    if g.page != m.page:
        return g.page > m.page
    if abs(g.baseline - m.y) <= BASELINE_TOLERANCE:
        return (g.bbox[0] + g.bbox[2]) / 2 >= m.x
    return g.baseline > m.y


def block_runs(lines: list[list[Glyph]], decorations: list[tuple[Mark, Mark, str]]) -> list[Run] | None:
    """block_lines の文字に、装飾（(始まりの目印, 終わりの目印, "bold" など)）を付けて Run の並びにする。
    つなげると block_text と同じ文字列になる。装飾が1つも無ければ None（IRで runs を省く決まりに合わせる）"""
    chars: list[tuple[str, frozenset[str]]] = []
    for k, line in enumerate(lines):
        if k > 0:
            chars.append(("\n", frozenset()))  # 改行の装飾は、前後の文字を見てあとで決める
        for g in line:
            marks = frozenset(kind for start, end, kind in decorations if _at_or_after(g, start) and not _at_or_after(g, end))
            chars.append((g.c, marks))
    # 装飾した語が行をまたぐとき、間の改行も同じ装飾の中に入れる（前後どちらにも付いている装飾だけ）
    for k, (c, _) in enumerate(chars):
        if c == "\n" and 0 < k < len(chars) - 1:
            chars[k] = (c, chars[k - 1][1] & chars[k + 1][1])

    if not any(marks for _, marks in chars):
        return None
    runs: list[Run] = []
    prev_marks = None
    for c, marks in chars:
        if marks == prev_marks:
            runs[-1].text += c
            continue
        runs.append(Run(
            text=c,
            bold=True if "bold" in marks else None,
            italic=True if "italic" in marks else None,
            underline=True if "underline" in marks else None,
        ))
        prev_marks = marks
    return runs


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
        decorations = [
            (marks[f"{name}s"], marks[f"{name}e"], kind)
            for name, kind in block.decorations
            if f"{name}s" in marks and f"{name}e" in marks
        ]
        for page_no in range(start.page, end.page + 1):
            page = pages[page_no - 1]
            top = start.y if page_no == start.page else 0.0
            bottom = end.y if page_no == end.page else page.height
            box = block_bbox(glyphs, page_no, top, bottom)
            if box is None:
                continue
            lines = block_lines(glyphs, page_no, top, bottom)
            page.elements.append(Element(
                kind="text",
                bbox=normalize_bbox(box, page.width, page.height),
                text="\n".join("".join(g.c for g in line) for line in lines),
                runs=block_runs(lines, decorations),
                label=block.label,
                level=block.level,
                label_source="auto",  # TeXから自動で作った正解
            ))
    return IRDocument(source=source, extractor="ground_truth", structurer="ground_truth", pages=pages)
