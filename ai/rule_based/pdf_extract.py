"""
4-1-1: PDFからテキスト・座標・フォントサイズ・文字の装飾・表の構造を抽出する。

- テキスト: PyMuPDF (pymupdf) の get_text("rawdict") でブロック単位のテキスト＋bbox＋フォント情報を取得。
  1文字ずつの位置も取れるので、太字・斜体（フォント）と下線（文字のすぐ下の横線）を文字の範囲ごとに判定する
- 表: pdfplumber の find_tables() で罫線から探す（罫線・列位置の検出精度がPyMuPDFより高いため）。
  縦線の無い表（booktabs・罫線なし）は罫線からは見つからないので、「表1」などのキャプションの下の文字の並びから探す。
  見つけた表ごとに、列の揃え・罫線の有無・折り返している列の幅も判定する
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
import re
from dataclasses import dataclass, field
from itertools import groupby

import pdfplumber
import pymupdf

from common.ir import Element, IRDocument, Page, Run, normalize_bbox, save_ir, to_dict, validate

NAME = "rule_based"

Bbox = tuple[float, float, float, float]
# 1文字と、その文字に付いている装飾（"bold" / "italic" / "underline"）
Char = tuple[str, tuple[str, ...]]

# 太字・斜体のフォント名。フォントのフラグが立っていないPDFも多いので名前でも見る
BOLD_FONT_PATTERN = re.compile(r"Bold|Black|Heavy|CMBX|CMB\d", re.IGNORECASE)
# 和文のゴシック体。和文の太字は（jsarticle の \textbf など）ゴシック体になるので、
# 明朝体が多いブロックの中のゴシック体は太字とみなす（本文がゴシック体の文書で全部を太字にしないため）
GOTHIC_FONT_PATTERN = re.compile(r"Gothic|KakuGo|Kaku-Go|Meiryo|SansCJK|Sans-?JP|SourceHanSans", re.IGNORECASE)
ITALIC_FONT_PATTERN = re.compile(r"Italic|Oblique|CMTI|CMSL", re.IGNORECASE)
# 数式のフォント。斜体のフラグが立つが、文字の装飾ではないので斜体にしない
MATH_FONT_PATTERN = re.compile(r"CMMI|CMSY|CMEX", re.IGNORECASE)
# 下線とみなす横線の、ベースラインからの下がり幅の上限（文字サイズに対する比率）。
# 下線は 0.3 程度、表の罫線（行の下の線）は 0.55 以上だった
UNDERLINE_MAX_DROP = 0.45
# 下線が文字のブロックからはみ出してよい幅(pt)。これより広い線は表の罫線など
UNDERLINE_OVERHANG = 2.0

# 表のキャプション。本文の「表1の結果から…」と文字だけでは区別できないので、
# 1行で短いこと・下に2行×2列以上の文字の並びがあることも合わせて見る
TABLE_CAPTION_PATTERN = re.compile(r"(表|Table)\s*\d+")
TABLE_CAPTION_MAX_CHARS = 100
# キャプションの下の文字を表の行として集めるとき、間がこの値（キャプションの文字サイズに対する比率）より空いたら表の終わり
TABLE_ROW_GAP_RATIO = 2.0
# キャプションと表の1行目の間の上限（同じ比率）。キャプションの下の余白の分、行の間より広く取る
TABLE_CAPTION_GAP_RATIO = 3.5
# 2段組みの判定。段の境目（ページの中央）からこの値(pt)までのはみ出しは同じ段とみなし、
# 右の段だけにある幅の広い（ページの幅のこの割合以上の）文字があれば2段組みのページとみなす
COLUMN_GUTTER = 10.0
COLUMN_TEXT_MIN_WIDTH_RATIO = 0.3
# キャプションの下で集めた文字の範囲を、表を探すときに広げる幅(pt)
TABLE_REGION_PADDING = 2.0
# 罫線とみなす線の長さ（表の幅・高さに対する割合）。セルの中の下線などを罫線と間違えないため
TABLE_RULE_MIN_COVERAGE = 0.5
# 列の揃えの判定で、セルの文字の幅の差がこの値(pt)以下なら、どの揃えとも決められない（左寄せにする）
ALIGN_TOLERANCE = 1.0


@dataclass
class TextBlock:
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    font_size: float
    font_name: str
    bold: bool
    runs: list[Run] | None = None  # 文字の範囲ごとの装飾。装飾が1つも無いブロックは None


@dataclass
class TableBlock:
    page: int
    bbox: tuple[float, float, float, float]
    cells: list[list[str]] = field(default_factory=list)  # 行 × 列の文字列。空のセルは ""
    aligns: list[str] | None = None  # 列ごとの揃え（"l" / "c" / "r"）
    col_rules: list[bool] | None = None  # 縦線の有無。列の境目（左端・列の間・右端）ごとで、列数+1個
    row_rules: list[bool] | None = None  # 横線の有無。行の境目（上端・行の間・下端）ごとで、行数+1個
    col_widths: list[float | None] | None = None  # 折り返している列の幅(pt)。ほかの列は None


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


def _span_marks(span: dict, gothic_is_bold: bool) -> tuple[str, ...]:
    """フォントから分かる装飾（太字・斜体）。gothic_is_bold なら和文のゴシック体も太字にする。"""
    font = span["font"]
    marks = []
    if span["flags"] & (1 << 4) or BOLD_FONT_PATTERN.search(font) or (gothic_is_bold and GOTHIC_FONT_PATTERN.search(font)):  # bit 4 = bold
        marks.append("bold")
    if (span["flags"] & (1 << 1) or ITALIC_FONT_PATTERN.search(font)) and not MATH_FONT_PATTERN.search(font):  # bit 1 = italic
        marks.append("italic")
    return tuple(marks)


def _horizontal_lines(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """ページの横線（細い長方形も含む）を (x0, x1, y) で返す。下線の判定に使う。"""
    lines = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < 0.5 and abs(item[1].x - item[2].x) > 1.0:
                lines.append((min(item[1].x, item[2].x), max(item[1].x, item[2].x), item[1].y))
            elif item[0] == "re" and item[1].height < 1.5 and item[1].width > 1.0:
                lines.append((item[1].x0, item[1].x1, (item[1].y0 + item[1].y1) / 2))
    return lines


def _is_underlined(char_bbox: Bbox, baseline: float, size: float, lines: list[tuple[float, float, float]]) -> bool:
    """文字の中央の真下で、ベースラインの少し下に横線があるか。"""
    center = (char_bbox[0] + char_bbox[2]) / 2
    return any(x0 <= center <= x1 and 0 < y - baseline <= size * UNDERLINE_MAX_DROP for x0, x1, y in lines)


def _strip_chars(chars: list[Char]) -> list[Char]:
    start, end = 0, len(chars)
    while start < end and chars[start][0].isspace():
        start += 1
    while end > start and chars[end - 1][0].isspace():
        end -= 1
    return chars[start:end]


def _runs(chars: list[Char]) -> list[Run] | None:
    """装飾が同じで隣り合う文字を1つの Run にする。装飾が1つも無ければ None（IRで runs を省く）。"""
    if not any(marks for _, marks in chars):
        return None
    return [
        Run("".join(c for c, _ in group), **{mark: True for mark in marks})
        for marks, group in groupby(chars, key=lambda ch: ch[1])
    ]


def _extract_text_blocks(page: pymupdf.Page, page_no: int) -> list[TextBlock]:
    all_lines = _horizontal_lines(page)
    blocks = []
    for raw_block in page.get_text("rawdict")["blocks"]:
        if raw_block.get("type") != 0:  # 0 = テキストブロック（画像ブロックは1）
            continue

        spans = []
        for line in raw_block["lines"]:
            for span in line["spans"]:
                spans.append(span)
        if not spans:
            continue

        # 下線の候補は、このブロックの範囲にある横線だけ。ブロックより広い線（表の罫線など）は除く
        x0, y0, x1, y1 = raw_block["bbox"]
        max_size = max(span["size"] for span in spans)
        underlines = [
            (lx0, lx1, ly)
            for lx0, lx1, ly in all_lines
            if y0 <= ly <= y1 + max_size and lx0 >= x0 - UNDERLINE_OVERHANG and lx1 <= x1 + UNDERLINE_OVERHANG
        ]

        sorted_lines = sorted(raw_block["lines"], key=lambda line: (line["bbox"][1], line["bbox"][0]))
        groups = []
        for line in sorted_lines:
            top = line["bbox"][1]
            height = (line["bbox"][3] - line["bbox"][1]) / 2

            if groups and abs(top - groups[-1][0]["bbox"][1]) <= height:
                groups[-1].append(line)
            else:
                groups.append([line])

        gothic_chars = sum(len(span["chars"]) for span in spans if GOTHIC_FONT_PATTERN.search(span["font"]))
        gothic_is_bold = gothic_chars <= sum(len(span["chars"]) for span in spans) / 2

        # 行は "\n"、同じ行に並ぶ line どうしは " " でつなぐ
        chars: list[Char] = []
        for g, group in enumerate(groups):
            group.sort(key=lambda line: line["bbox"][0])  # 同じ行の中は、左から順に並べる
            if g > 0:
                chars.append(("\n", ()))
            for k, line in enumerate(group):
                if k > 0:
                    chars.append((" ", ()))
                for span in line["spans"]:
                    marks = _span_marks(span, gothic_is_bold)
                    baseline = span["origin"][1]
                    for ch in span["chars"]:
                        underlined = _is_underlined(ch["bbox"], baseline, span["size"], underlines)
                        chars.append((ch["c"], marks + ("underline",) if underlined else marks))

        chars = _strip_chars(chars)
        if not chars:
            continue
        text = "".join(c for c, _ in chars)

        # ブロック内で最も文字数の多いフォントサイズ・フォント名を代表値とする
        # （見出しの記号などで一部だけフォントが違うケースを吸収するため）
        by_font: dict[tuple[float, str], int] = {}
        bold_chars = 0
        total_chars = 0
        for span in spans:
            n_chars = len(span["chars"])
            key = (round(span["size"], 1), span["font"])
            by_font[key] = by_font.get(key, 0) + n_chars
            total_chars += n_chars
            if "bold" in _span_marks(span, gothic_is_bold):
                bold_chars += n_chars

        font_size, font_name = max(by_font, key=lambda k: by_font[k])

        blocks.append(
            TextBlock(
                page=page_no,
                text=text,
                bbox=tuple(raw_block["bbox"]),
                font_size=font_size,
                font_name=font_name,
                bold=bold_chars > total_chars / 2,
                runs=_runs(chars),
            )
        )
    return blocks


def _is_inside(inner: tuple, outer: tuple, tol: float = 2.0) -> bool:
    """innerがouterの内側に収まっているか（tolだけはみ出しを許容）。"""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return ix0 >= ox0 - tol and iy0 >= oy0 - tol and ix1 <= ox1 + tol and iy1 <= oy1 + tol




def _column_align(extents: list[tuple[float, float]]) -> str:
    """
    列の各セルの文字の (左端, 右端) から揃えを決める。左端・中央・右端のうち、セルどうしのずれが一番小さいもの。
    セルが1つしか無い・どのセルも同じ幅で決められないときは左寄せ。
    """
    if len(extents) < 2:
        return "l"
    widths = [x1 - x0 for x0, x1 in extents]
    if max(widths) - min(widths) <= ALIGN_TOLERANCE:
        return "l"

    def spread(values: list[float]) -> float:
        return max(values) - min(values)

    candidates = {
        "l": spread([x0 for x0, _ in extents]),
        "c": spread([(x0 + x1) / 2 for x0, x1 in extents]),
        "r": spread([x1 for _, x1 in extents]),
    }
    return min(candidates, key=lambda align: candidates[align])  # 同じずれなら l → c → r の順


def _rule_flags(bounds: list[tuple[float, float]], segments: list[tuple[float, float, float]], span: tuple[float, float]) -> list[bool]:
    """
    行（または列）の境目ごとに、罫線があるかを返す（len(bounds) + 1 個）。
    bounds: 行の (上端, 下端)。segments: 線の (位置, 始点, 終点)。span: 表の (始点, 終点)。
    境目の線は、前後の行の中央どうしの間で探す。外側の線は、端の行の高さの分だけ外まで探す
    （罫線から見つけた表でなければ、行の範囲は文字にぴったりで、外側の線は少し離れているため）。
    同じ境目にある線の長さを足して、表の幅（高さ）の TABLE_RULE_MIN_COVERAGE 以上あれば罫線とみなす。
    """
    first, last = bounds[0], bounds[-1]
    limits = [first[0] - (first[1] - first[0])] + [(a + b) / 2 for a, b in bounds] + [last[1] + (last[1] - last[0])]
    length = span[1] - span[0]
    flags = []
    for lo, hi in zip(limits, limits[1:]):
        covered = sum(max(0.0, min(end, span[1]) - max(start, span[0])) for pos, start, end in segments if lo <= pos <= hi)
        flags.append(length > 0 and covered >= length * TABLE_RULE_MIN_COVERAGE)
    return flags


def _table_block(page: pdfplumber.page.Page, page_no: int, bbox: Bbox, rows: list) -> TableBlock:
    """
    pdfplumber の表の行（Row と、その行のセルの文字の組）から TableBlock を作り、
    列の揃え・罫線・折り返している列の幅も判定する。
    """
    cells = [texts for _, texts in rows]
    n_cols = max(len(texts) for texts in cells)
    chars = page.chars

    aligns = []
    col_bounds: list[tuple[float, float] | None] = []
    for c in range(n_cols):
        cell_bboxes = [row.cells[c] for row, _ in rows if c < len(row.cells) and row.cells[c] is not None]
        col_bounds.append((min(b[0] for b in cell_bboxes), max(b[2] for b in cell_bboxes)) if cell_bboxes else None)
        extents = []
        for x0, top, x1, bottom in cell_bboxes:
            inside = [ch for ch in chars if x0 <= (ch["x0"] + ch["x1"]) / 2 <= x1 and top <= (ch["top"] + ch["bottom"]) / 2 <= bottom]
            visible = [ch for ch in inside if ch["text"].strip()]
            if visible:
                extents.append((min(ch["x0"] for ch in visible), max(ch["x1"] for ch in visible)))
        aligns.append(_column_align(extents))

    table = TableBlock(page=page_no, bbox=bbox, cells=cells, aligns=aligns)
    if any(bounds is None for bounds in col_bounds):
        return table  # 結合したセルなどで列の範囲が分からないときは、罫線・幅を判定しない

    row_bounds = [(row.bbox[1], row.bbox[3]) for row, _ in rows]
    x_span = (col_bounds[0][0], col_bounds[-1][1])
    y_span = (row_bounds[0][0], row_bounds[-1][1])
    horizontal = [(e["top"], e["x0"], e["x1"]) for e in page.horizontal_edges]
    vertical = [(e["x0"], e["top"], e["bottom"]) for e in page.vertical_edges]
    table.row_rules = _rule_flags(row_bounds, horizontal, x_span)
    table.col_rules = _rule_flags(col_bounds, vertical, y_span)
    # 折り返している（セルの文字に改行がある）列だけ幅を残す。ほかの列はエディタで文字に合わせる
    table.col_widths = [
        bounds[1] - bounds[0] if any(c < len(texts) and "\n" in texts[c] for texts in cells) else None
        for c, bounds in enumerate(col_bounds)
    ]
    return table


def _extract_table_blocks(page: pdfplumber.page.Page, page_no: int) -> list[TableBlock]:
    result = []
    for table in page.find_tables():
        rows = [(row, [cell or "" for cell in texts]) for row, texts in zip(table.rows, table.extract())]
        result.append(_table_block(page, page_no, table.bbox, rows))
    return result


def _overlaps(a: Bbox, b: Bbox) -> bool:
    """2つのbboxが少しでも重なっているか。"""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _side(bbox: Bbox, page_width: float) -> str | None:
    """2段組みのどちらの段にあるか。段をまたいでいれば None。"""
    center = page_width / 2
    if bbox[2] <= center + COLUMN_GUTTER:
        return "left"
    if bbox[0] >= center - COLUMN_GUTTER:
        return "right"
    return None


def _extract_tables_near_captions(
    page: pdfplumber.page.Page, page_no: int, text_blocks: list[TextBlock], found_bboxes: list[Bbox]
) -> list[TableBlock]:
    """
    罫線から見つからなかった表（縦線の無い booktabs の表・罫線の無い表）を、キャプションの下の文字から探す。
    「表1」などで始まる短い1行の下に続く文字のブロックを、間が空くところまで集め、その範囲だけで
    文字の位置から行・列を決める（pdfplumber の text 方式）。2行×2列に満たないものは表とみなさない。
    キャプションが表の上にある前提。2段組みのページでは、キャプションと同じ段の文字だけを集める。
    """
    result = []
    blocks = sorted(text_blocks, key=lambda b: b.bbox[1])
    min_width = page.width * COLUMN_TEXT_MIN_WIDTH_RATIO
    two_column = any(_side(b.bbox, page.width) == "right" and b.bbox[2] - b.bbox[0] >= min_width for b in blocks)

    for i, caption in enumerate(blocks):
        is_caption = "\n" not in caption.text and len(caption.text) <= TABLE_CAPTION_MAX_CHARS
        if not is_caption or not TABLE_CAPTION_PATTERN.match(caption.text):
            continue

        side = _side(caption.bbox, page.width) if two_column else None
        candidates = []
        bottom = caption.bbox[3]
        for block in blocks[i + 1 :]:
            if side is not None and _side(block.bbox, page.width) != side:
                continue  # もう一方の段の文字
            ratio = TABLE_ROW_GAP_RATIO if candidates else TABLE_CAPTION_GAP_RATIO
            if block.bbox[1] - bottom > caption.font_size * ratio:
                break
            candidates.append(block)
            bottom = max(bottom, block.bbox[3])  # 同じ行のセルは上端がほぼ同じなので、下端の一番下から測る
        if not candidates:
            continue

        x0, y0, x1, y1 = _union_bbox([b.bbox for b in candidates])
        pad = TABLE_REGION_PADDING
        region = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        if any(_overlaps(region, found) for found in found_bboxes):
            continue  # 罫線から見つかっている表

        crop = page.crop(region, strict=False)
        for table in crop.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"}):
            # 行の間にできる空の行は捨てる
            rows = [(row, [cell or "" for cell in texts]) for row, texts in zip(table.rows, table.extract())]
            rows = [(row, texts) for row, texts in rows if any(texts)]
            if len(rows) < 2 or max(len(texts) for _, texts in rows) < 2:
                continue  # 1行・1列しかないものは表とみなさない
            result.append(_table_block(page, page_no, table.bbox, rows))
            found_bboxes = [*found_bboxes, table.bbox]

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
            table_blocks += _extract_tables_near_captions(plumber_page, page_no, text_blocks, [t.bbox for t in table_blocks])
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


def table_element(table: TableBlock, bbox: Bbox, page_width: float, label: str | None = None) -> Element:
    """TableBlock を表の要素にする。列幅(pt)はページの幅で割る。"""
    widths = None if table.col_widths is None else [None if w is None else round(w / page_width, 5) for w in table.col_widths]
    return Element(
        kind="table",
        bbox=bbox,
        cells=table.cells,
        aligns=table.aligns,
        col_rules=table.col_rules,
        row_rules=table.row_rules,
        col_widths=widths,
        label=label,
    )


def to_ir(pages: list[PageBlocks], source: str) -> IRDocument:
    """extract() の結果をIRにする。要素の順は 文字 → 表 → 図（読み順ではない）。"""
    ir_pages = []
    for page_blocks in pages:
        w, h = page_blocks.width, page_blocks.height
        page = Page(page=page_blocks.page + 1, width=w, height=h, unit="pt")
        page.elements += [
            Element(kind="text", bbox=normalize_bbox(b.bbox, w, h), text=b.text, runs=b.runs, font_size=b.font_size, bold=b.bold)
            for b in page_blocks.text_blocks
        ]
        page.elements += [table_element(b, normalize_bbox(b.bbox, w, h), w) for b in page_blocks.table_blocks]
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
