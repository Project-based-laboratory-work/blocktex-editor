"""
TeXで作ったPDFから、正解のラベル付きIRを作る。

目印（marks.py）から各ブロックの縦の範囲（1行目〜最後の行のベースライン）が分かるので、
その範囲にある文字の枠を全部包む枠を、そのブロックの正解の枠にする。
文中の装飾（runs）も、装飾した語の前後の目印の間にある文字を、その装飾の文字とする。

ブロックの種類ごとの決まり:
- 表・図: 文字に加えて、範囲の中の罫線・画像（Graphic）も枠に含める。表の cells・揃え・罫線は、文書を作ったときの値を使う
- コード: 文字はPDFからではなく、ソースから取る（行頭の空白＝字下げを正しく持つため）
- 数式: 分数の分子・分母はベースラインが大きくずれるので、範囲を広めにとる。正解のLaTeX（tex）も持つ
- 2段組: 範囲を段ごとに分け、その段の中の文字だけを集める（タイトルなど段をまたぐものは分けない）
- ページ番号・ヘッダー（柱）: 目印 pn{ページ}s / pn{ページ}e、hd{ページ}s / hd{ページ}e から label="other" の要素にし、
  ほかのブロックの範囲からは外す
"""
import re
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
# 別行立ての数式は、分数の分子・分母などがベースラインから大きくずれるので広くとる。
# 数式の前後には段落より広い空き（\abovedisplayskip など）があるので、隣の段落の文字は入らない
MATH_TOLERANCE = 14.0

PtBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)。pt、上からの距離（正規化する前）


@dataclass
class Glyph:
    """PDFに印刷された1文字"""
    page: int        # 1始まり
    bbox: PtBox
    baseline: float  # 文字が乗っている線の、上からの距離（pt）
    c: str = ""      # その文字。空白も入る


@dataclass
class Graphic:
    """PDFに描かれた画像・線・図形（表の罫線や図）"""
    page: int
    bbox: PtBox


@dataclass
class Region:
    """ブロックが占める、1ページ（2段組なら1つの段）の中の縦の範囲"""
    page: int
    top: float      # 1行目のベースライン（上からの距離、pt）
    bottom: float   # 最後の行のベースライン
    tolerance: float = BASELINE_TOLERANCE  # 上下に広げる幅
    column: int | None = None  # 2段組の段（0: 左、1: 右）。None は段で絞らない
    mid_x: float = 0.0         # 2段組の段の境目の x（ページの幅の半分）

    def in_column(self, bbox: PtBox) -> bool:
        return self.column is None or int((bbox[0] + bbox[2]) / 2 >= self.mid_x) == self.column

    def contains(self, g: Glyph) -> bool:
        return (
            g.page == self.page
            and self.top - self.tolerance <= g.baseline <= self.bottom + self.tolerance
            and self.in_column(g.bbox)
        )

    def contains_graphic(self, graphic: Graphic) -> bool:
        center_y = (graphic.bbox[1] + graphic.bbox[3]) / 2
        return graphic.page == self.page and self.top <= center_y <= self.bottom and self.in_column(graphic.bbox)


def union_bbox(boxes: list[PtBox]) -> PtBox:
    """全部の枠を包む、一番小さい枠"""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def region_glyphs(glyphs: list[Glyph], region: Region) -> list[Glyph]:
    return [g for g in glyphs if region.contains(g)]


def glyphs_in_block(glyphs: list[Glyph], page: int, top: float, bottom: float) -> list[Glyph]:
    """page の、ベースラインが top〜bottom（前後に BASELINE_TOLERANCE の余裕）にある文字"""
    return region_glyphs(glyphs, Region(page, top, bottom))


def region_bbox(glyphs: list[Glyph], region: Region, graphics: list[Graphic] | None = None) -> PtBox | None:
    """範囲の文字（と graphics を渡せば、範囲の中の罫線・画像）を包む枠。空白は枠を広げてしまうので使わない。
    何も無ければ None"""
    boxes = [g.bbox for g in region_glyphs(glyphs, region) if not g.c.isspace()]
    boxes += [x.bbox for x in graphics or [] if region.contains_graphic(x)]
    if not boxes:
        return None
    return union_bbox(boxes)


def block_bbox(glyphs: list[Glyph], page: int, top: float, bottom: float) -> PtBox | None:
    """ブロックの文字を包む枠。空白は枠を広げてしまうので使わない。文字が無ければ None"""
    return region_bbox(glyphs, Region(page, top, bottom))


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


def read_graphics(pdf_path: str) -> list[Graphic]:
    """PDFの画像と、線・図形（表の罫線や枠）の位置を読む"""
    graphics = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            for info in page.get_image_info():
                graphics.append(Graphic(page_index + 1, tuple(info["bbox"])))
            for drawing in page.get_drawings():
                graphics.append(Graphic(page_index + 1, tuple(drawing["rect"])))
    return graphics


def region_lines(glyphs: list[Glyph], region: Region) -> list[list[Glyph]]:
    """範囲の文字を行ごとに分ける。ベースラインが変わったところで次の行にし、各行の前後の空白は取る"""
    lines: list[list[Glyph]] = []
    prev_baseline = None
    for g in region_glyphs(glyphs, region):
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


def block_lines(glyphs: list[Glyph], page: int, top: float, bottom: float) -> list[list[Glyph]]:
    """ブロックの文字を行ごとに分ける。ベースラインが変わったところで次の行にし、各行の前後の空白は取る"""
    return region_lines(glyphs, Region(page, top, bottom))


def lines_text(lines: list[list[Glyph]]) -> str:
    return "\n".join("".join(g.c for g in line) for line in lines)


def block_text(glyphs: list[Glyph], page: int, top: float, bottom: float) -> str:
    """ブロックの文字列。ベースラインが変わったところで改行（\\n）を入れ、各行の前後の空白は取る"""
    return lines_text(block_lines(glyphs, page, top, bottom))


def _at_or_after(g: Glyph, m: Mark, mid_x: float | None = None) -> bool:
    """文字 g が、読む順で目印 m の位置かそれより後ろにあるか。同じ行なら文字の中心の x で比べる。
    2段組（mid_x を渡す）では、左の段を読んでから右の段を読む"""
    if g.page != m.page:
        return g.page > m.page
    center_x = (g.bbox[0] + g.bbox[2]) / 2
    if mid_x is not None:
        g_column, m_column = int(center_x >= mid_x), int(m.x >= mid_x)
        if g_column != m_column:
            return g_column > m_column
    if abs(g.baseline - m.y) <= BASELINE_TOLERANCE:
        return center_x >= m.x
    return g.baseline > m.y


def block_runs(
    lines: list[list[Glyph]], decorations: list[tuple[Mark, Mark, str]], mid_x: float | None = None
) -> list[Run] | None:
    """block_lines の文字に、装飾（(始まりの目印, 終わりの目印, "bold" など)）を付けて Run の並びにする。
    つなげると block_text と同じ文字列になる。装飾が1つも無ければ None（IRで runs を省く決まりに合わせる）"""
    chars: list[tuple[str, frozenset[str]]] = []
    for k, line in enumerate(lines):
        if k > 0:
            chars.append(("\n", frozenset()))  # 改行の装飾は、前後の文字を見てあとで決める
        for g in line:
            marks = frozenset(
                kind for start, end, kind in decorations if _at_or_after(g, start, mid_x) and not _at_or_after(g, end, mid_x)
            )
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


def compile_tex(tex: str, work_dir: Path, files: dict[str, bytes] | None = None) -> Path:
    """TeXを work_dir/doc.tex に書いて PDF にし、PDF のパスを返す。失敗したら subprocess.CalledProcessError。
    files（図の画像など）も work_dir に置く。画像は dvipdfmx が大きさを知れるように extractbb で .xbb を作る"""
    for name, data in (files or {}).items():
        (work_dir / name).write_bytes(data)
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            subprocess.run(["extractbb", name], cwd=work_dir, check=True, capture_output=True, timeout=60)
    (work_dir / "doc.tex").write_text(tex, encoding="utf-8")
    for command in COMPILE_COMMANDS:
        subprocess.run(command, cwd=work_dir, check=True, capture_output=True, timeout=120)
    return work_dir / "doc.pdf"


def block_regions(block: SourceBlock, start: Mark, end: Mark, pages: list[Page], two_column: bool) -> list[Region]:
    """ブロックの範囲を、ページ（2段組なら段）ごとに分ける。始まりの段は目印の高さから下の端まで、
    終わりの段は上の端から目印の高さまで、間の段は丸ごと"""
    if block.own_line_marks:
        tolerance = 0.0   # 目印はブロックの上と下の別の行にあるので、広げると隣のキャプションなどが入る
    elif block.label == "math":
        tolerance = MATH_TOLERANCE
    else:
        tolerance = BASELINE_TOLERANCE

    if not two_column or block.full_width:
        return [
            Region(
                page_no,
                start.y if page_no == start.page else 0.0,
                end.y if page_no == end.page else pages[page_no - 1].height,
                tolerance,
            )
            for page_no in range(start.page, end.page + 1)
        ]

    def position(m: Mark) -> tuple[int, int]:
        return m.page, int(m.x >= pages[m.page - 1].width / 2)

    first, last = position(start), position(end)
    regions = []
    for page_no in range(start.page, end.page + 1):
        for column in (0, 1):
            if not first <= (page_no, column) <= last:
                continue
            page = pages[page_no - 1]
            regions.append(Region(
                page_no,
                start.y if (page_no, column) == first else 0.0,
                end.y if (page_no, column) == last else page.height,
                tolerance,
                column,
                page.width / 2,
            ))
    return regions


# ページ番号（pn）とヘッダー（hd）の目印。どちらも label="other" にする
PAGE_MARK = re.compile(r"^(pn|hd)(\d+)s$")


def build_ground_truth(pdf_path: str, blocks: list[SourceBlock], source: str, two_column: bool = False) -> IRDocument:
    """目印付きのPDFから、正解のラベル付きIRを作る。要素はブロックを置いた順（＝読み順）に並ぶ。
    ページ（2段組なら段）をまたぐブロックは、ページ・段ごとの要素に分ける。ページ番号は各ページの最後に置く。"""
    marks = read_marks(pdf_path)
    glyphs = read_glyphs(pdf_path)
    graphics = read_graphics(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        pages = [Page(page=i + 1, width=p.rect.width, height=p.rect.height, unit="pt") for i, p in enumerate(doc)]

    # ページ番号・ヘッダーの文字は、ほかのブロックに入らないように先に取り分ける（ヘッダーを先に置く）
    page_numbers: list[tuple[Page, str, list[list[Glyph]]]] = []  # (ページ, "pn" か "hd", 文字の行)
    for name, start in sorted(marks.items(), key=lambda item: (item[0].startswith("pn"), item[0])):
        m = PAGE_MARK.match(name)
        end = marks.get(f"{m.group(1)}{m.group(2)}e") if m else None
        if m is None or end is None:
            continue
        lines = region_lines(glyphs, Region(start.page, start.y, end.y))
        page_numbers.append((pages[start.page - 1], m.group(1), lines))
    reserved = {id(g) for _, _, lines in page_numbers for line in lines for g in line}
    # 段をまたぐブロック（タイトルなど）の文字も先に取り分ける。2段組で、ブロックが右の段の上から続くとき、
    # 右の段の範囲（ページの上の端から）にタイトルの右半分が入ってしまわないように
    for block in blocks:
        start, end = marks.get(f"b{block.id}s"), marks.get(f"b{block.id}e")
        if block.full_width and start is not None and end is not None:
            for region in block_regions(block, start, end, pages, two_column):
                reserved |= {id(g) for g in region_glyphs(glyphs, region)}
    body_glyphs = [g for g in glyphs if id(g) not in reserved]
    full_width_glyphs = glyphs  # 段をまたぐブロック自身は、取り分ける前の文字から集める

    for block in blocks:
        start, end = marks.get(f"b{block.id}s"), marks.get(f"b{block.id}e")
        if start is None or end is None:  # 目印がPDFに出なかった（ページの外に流れたなど）
            continue
        decorations = [
            (marks[f"{name}s"], marks[f"{name}e"], kind)
            for name, kind in block.decorations
            if f"{name}s" in marks and f"{name}e" in marks
        ]
        uses_graphics = block.kind in ("table", "figure")
        source_glyphs = full_width_glyphs if block.full_width else body_glyphs
        # 範囲ごとの枠。何も無い範囲（段の切れ目に目印だけがあるなど）は捨てる
        boxed = [
            (region, box)
            for region in block_regions(block, start, end, pages, two_column)
            if (box := region_bbox(source_glyphs, region, graphics if uses_graphics else None)) is not None
        ]
        for region, box in boxed:
            page = pages[region.page - 1]
            lines = region_lines(source_glyphs, region)
            # ソースから文字を取るブロック（コード・表）は、要素が1つのときだけその文字を使う（分かれたら分けられない）
            text = block.text if block.text is not None and len(boxed) == 1 else lines_text(lines)
            page.elements.append(Element(
                kind=block.kind,
                bbox=normalize_bbox(box, page.width, page.height),
                text=None if block.kind == "figure" else text,
                runs=block_runs(lines, decorations, region.mid_x if two_column and not block.full_width else None)
                if block.kind == "text" and block.text is None else None,
                cells=block.cells,
                aligns=block.aligns,
                col_rules=block.col_rules,
                row_rules=block.row_rules,
                col_widths=block.col_widths,
                label=block.label,
                level=block.level,
                tex=block.tex_value,
                label_source="auto",  # TeXから自動で作った正解
            ))

    for page, prefix, lines in page_numbers:
        boxes = [g.bbox for line in lines for g in line if not g.c.isspace()]
        if not boxes:
            continue
        element = Element(
            kind="text",
            bbox=normalize_bbox(union_bbox(boxes), page.width, page.height),
            text=lines_text(lines),
            label="other",
            label_source="auto",
        )
        # 読み順: ヘッダーはページの先頭、ページ番号はページの最後
        if prefix == "hd":
            page.elements.insert(0, element)
        else:
            page.elements.append(element)
    return IRDocument(source=source, extractor="ground_truth", structurer="ground_truth", pages=pages)
