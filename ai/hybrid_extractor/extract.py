"""PDFから文字・表・図を取り出して、共通のIR（common/README.md の①）にする抽出器（ルールベース）。

流れ:
    PDF ─ page_lines ─▶ 行（PyMuPDF の line。1文字ずつの位置・フォントつき）
        ─ merge_rows ─▶ 横の行（同じ高さに並ぶ line をつなぐ。「1」＋「はじめに」）
        ─ find_tables / find_figures ─▶ 表・図の領域（その中の文字は表・図のもの）
        ─ group_rows ─▶ ブロック（段落の続きの行・数式の断片・コードの行を縦につなぐ）
        ─ to_ir ─▶ IR

分類器（rules.py や自作AI）の手がかりになるように、IRに無い情報（フォント名、等幅・数式の文字の割合、行数など）も
Block に持たせている。IRにするときはそれを落とす。

使い方（ai/ で実行）:
    python -m hybrid_extractor.extract hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.ir.json
"""

import argparse
import os
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import pymupdf

from common.ir import Element, IRDocument, Page, Run, normalize_bbox, save_ir, validate

NAME = "hybrid_extractor"

PtBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)。pt、左上が原点

# ---- フォント名から分かること ----
MONO_FONT = re.compile(r"CMTT|Typewriter|Mono|Courier|lmtt|Consola|Menlo|Inconsolata", re.IGNORECASE)
MATH_FONT = re.compile(r"CMMI|CMSY|CMEX|CMBSY|MSAM|MSBM|EUFM|EUSM|RSFS|Math", re.IGNORECASE)
BOLD_FONT = re.compile(r"Bold|Black|Heavy|Semibold|Demi|CMBX|CMB\d", re.IGNORECASE)
# 和文の太字（\textbf）はゴシック体になるので、本文が明朝体のページではゴシック体を太字とみなす
GOTHIC_FONT = re.compile(r"Gothic|KakuGo|Kaku-Go|Meiryo|SansCJK|Sans-?JP|SourceHanSans", re.IGNORECASE)
ITALIC_FONT = re.compile(r"Italic|Oblique|CMTI|CMSL", re.IGNORECASE)

# 箇条書きの頭（• や 1. や (a)）
LIST_HEAD = re.compile(r"^([•◦▪‣∙·∗]|[-–—*]\s|\(?\d{1,2}[.)]\s|\(?[a-z][.)]\s|\([ivx]+\)\s)")
# 表のキャプション（罫線の無い表を、キャプションの下の文字の並びから探す）
TABLE_CAPTION = re.compile(r"^(表|Table)\s*\d+")

# ---- しきい値（本文の文字サイズ body に対する比率） ----
LINE_GAP = 0.8       # 段落の行と行の間のすき間の上限（jsarticle は 16pt 送りで、10pt の文字なら すき間 6pt くらい）
MATH_GAP = 1.2       # 数式の断片どうしのすき間の上限（分数・添字で間が少し空く）
LIST_GAP = 1.3       # 箇条書きの項目と項目の間のすき間の上限
INDENT = 0.5         # 段落の1行目の字下げとみなす、左端のずれ
SIZE_RATIO = 1.15    # これより文字の大きさが違う行は、別のブロック（見出しと本文など）
MATH_RATIO = 0.3     # 数式のフォントの文字がこの割合以上の行は、数式の行
MONO_RATIO = 0.6     # 等幅フォントの文字がこの割合以上の行は、コードの行
COLUMN_GUTTER = 10.0  # 2段組の段の境目（ページの中央）からこの幅(pt)まではみ出しても、同じ段とみなす
# 表
TABLE_LINE_TOLERANCE = 3.0   # 同じ表の罫線とみなす、左端・右端のずれ(pt)
TABLE_MAX_LINE_GAP = 200.0   # 同じ表の横線どうしの間の上限(pt)。折り返すセルがあると行が高くなる
TABLE_COL_GAP = 0.7          # 列の境目とみなす、文字の無いすき間の下限（文字の高さに対する比率）
RULE_COVERAGE = 0.5          # 罫線とみなす線の長さ（表の幅・高さに対する割合）
# 図
FIGURE_MIN_SIZE = 15.0       # 図とみなす大きさの下限(pt)。下線・分数の線を図にしないため
FIGURE_MAX_TEXT = 0.25       # 中の文字がこの割合以上を覆う囲みは、図ではなく文字の枠（枠付きのコードなど）
FIGURE_NEAR = 4.0            # この距離(pt)までの図形は、1つの図にまとめる
# 下線: 文字のベースラインの少し下（文字サイズのこの割合まで）にある横線
UNDERLINE_MAX_DROP = 0.45


# =====================================================================
# 小さい部品（bbox の計算）
# =====================================================================


def same_row(a: dict, b: dict) -> bool:
    """行 a と b の縦の範囲が、低い方の行の高さの半分より多く重なっていれば True。"""
    top = max(a["bbox"][1], b["bbox"][1])       # 重なりの上端（下にある方）
    bottom = min(a["bbox"][3], b["bbox"][3])
    overlap = bottom - top
    height_a = a["bbox"][3] - a["bbox"][1]
    height_b = b["bbox"][3] - b["bbox"][1]
    return overlap > min(height_a, height_b) / 2


def union_bbox(a: tuple, b: tuple) -> tuple:
    """2つの bbox をどちらも包む、一番小さい bbox。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))


def union_all(boxes: list[tuple]) -> PtBox:
    box = boxes[0]
    for other in boxes[1:]:
        box = union_bbox(box, other)
    return box


def center(box: tuple) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def inside(point: tuple[float, float], box: tuple, margin: float = 0.0) -> bool:
    x, y = point
    return box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin


def overlaps(a: tuple, b: tuple, margin: float = 0.0) -> bool:
    return min(a[2], b[2]) + margin > max(a[0], b[0]) and min(a[3], b[3]) + margin > max(a[1], b[1])


def area(box: tuple) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


# =====================================================================
# 行（PyMuPDF の line）
# =====================================================================


def page_lines(page: pymupdf.Page) -> list[dict]:
    """1ページの行を、{"text", "bbox", "size", "chars"} の辞書のリストにして返す。

    chars は1文字ずつの {"c", "bbox", "font", "size", "flags", "baseline"}。装飾（runs）や字下げの判定に使う。
    """
    lines = []
    d = page.get_text("rawdict")
    for block in d["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = ""
            size = 0.0
            chars = []
            for span in line["spans"]:
                for ch in span["chars"]:
                    text += ch["c"]                      # ① text に文字をつなげる
                    chars.append({
                        "c": ch["c"], "bbox": tuple(ch["bbox"]), "font": span["font"], "size": span["size"],
                        "flags": span["flags"], "baseline": span["origin"][1],
                    })
                size = max(size, span["size"])           # ② 今の size と span のサイズの、大きい方
            if text.strip():
                lines.append({"text": text, "bbox": tuple(line["bbox"]), "size": size, "chars": chars})
    return lines


# =====================================================================
# 段組みと読み順
# =====================================================================


def column_of(box: tuple, page_width: float) -> int:
    """2段組のどの段か。0: 段をまたぐ、1: 左の段、2: 右の段"""
    mid = page_width / 2
    if box[2] <= mid + COLUMN_GUTTER:
        return 1
    if box[0] >= mid - COLUMN_GUTTER:
        return 2
    return 0


def is_two_column(boxes: list[tuple], page_width: float) -> bool:
    """2段組のページか。ある程度の幅（ページの15%以上）の行が左右どちらの段にもあり、
    中央をまたぐ行が、片方の段に収まる行の3分の1以下なら2段組（1段組の本文の行はたいてい中央をまたぐ）"""
    wide = [b for b in boxes if b[2] - b[0] >= page_width * 0.15]
    columns = [column_of(b, page_width) for b in wide]
    left, right, cross = columns.count(1), columns.count(2), columns.count(0)
    return left >= 1 and right >= 1 and left + right >= 4 and cross * 3 <= left + right


def reading_order(boxes: list[tuple], page_width: float, two_column: bool) -> list[int]:
    """boxes を読む順に並べたときの添字の並び。
    1段組: 上から。2段組: 段をまたぐもので帯に区切り、帯ごとに 左の段 → 右の段 の順で上から"""
    if not two_column:
        return sorted(range(len(boxes)), key=lambda i: (round(boxes[i][1]), boxes[i][0]))
    columns = [column_of(b, page_width) for b in boxes]
    spanner_tops = sorted(b[1] for b, col in zip(boxes, columns) if col == 0)

    def key(i: int) -> tuple:
        band = sum(1 for top in spanner_tops if top <= boxes[i][1])
        return (band, columns[i], boxes[i][1], boxes[i][0])

    return sorted(range(len(boxes)), key=key)


# =====================================================================
# 横の行（同じ高さに並ぶ line をつなぐ）
# =====================================================================


def merge_rows(lines: list[dict], page_width: float, two_column: bool) -> list[dict]:
    """同じ高さに並ぶ行を、横に1つにつなぐ。返す行は {"bbox", "size", "parts"}（parts は左から並べた line）。
    2段組では、左右の段の行はつながない"""
    rows: list[dict] = []
    for line in lines:
        target = None
        for row in rows[-3:]:  # 直前の数行だけ見る（同じ高さの line は続けて出てくる）
            if not same_row(row, line):
                continue
            if two_column and {column_of(row["bbox"], page_width), column_of(line["bbox"], page_width)} == {1, 2}:
                continue
            gap = max(line["bbox"][0] - row["bbox"][2], row["bbox"][0] - line["bbox"][2])
            if gap <= 3 * max(row["size"], line["size"]):
                target = row
                break
        if target is not None:
            target["parts"].append(line)
            target["bbox"] = union_bbox(target["bbox"], line["bbox"])
            target["size"] = max(target["size"], line["size"])
        else:
            rows.append({"bbox": line["bbox"], "size": line["size"], "parts": [line]})
    for row in rows:
        row["parts"].sort(key=lambda part: part["bbox"][0])
        row["chars"] = row_chars(row["parts"])
        row["text"] = "".join(ch["c"] for ch in row["chars"])
    return rows


def row_chars(parts: list[dict]) -> list[dict]:
    """左から並べた line の文字を、間に空白を1つはさんでつなぐ"""
    chars: list[dict] = []
    for k, part in enumerate(parts):
        if k > 0:
            chars.append(space_char())
        chars.extend(part["chars"])
    return chars


def space_char() -> dict:
    """つなぎ目に入れる空白（位置もフォントも無い）"""
    return {"c": " ", "bbox": None, "font": "", "size": 0.0, "flags": 0, "baseline": 0.0}


def visible(chars: list[dict]) -> list[dict]:
    return [ch for ch in chars if ch["bbox"] is not None and not ch["c"].isspace()]


def ratio(chars: list[dict], pattern: re.Pattern) -> float:
    shown = visible(chars)
    return sum(bool(pattern.search(ch["font"])) for ch in shown) / len(shown) if shown else 0.0


# =====================================================================
# 表
# =====================================================================


@dataclass
class Segments:
    """ページの線。横線は (x0, x1, y)、縦線は (x, y0, y1)"""
    horizontal: list[tuple[float, float, float]] = field(default_factory=list)
    vertical: list[tuple[float, float, float]] = field(default_factory=list)


def page_segments(page: pymupdf.Page) -> Segments:
    """罫線・下線の候補になる、細い線を集める"""
    seg = Segments()
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                p, q = item[1], item[2]
                if abs(p.y - q.y) < 1 and abs(p.x - q.x) > 2:
                    seg.horizontal.append((min(p.x, q.x), max(p.x, q.x), (p.y + q.y) / 2))
                elif abs(p.x - q.x) < 1 and abs(p.y - q.y) > 2:
                    seg.vertical.append(((p.x + q.x) / 2, min(p.y, q.y), max(p.y, q.y)))
            elif item[0] == "re":
                r = item[1]
                if r.height < 2 and r.width > 2:
                    seg.horizontal.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
                elif r.width < 2 and r.height > 2:
                    seg.vertical.append(((r.x0 + r.x1) / 2, r.y0, r.y1))
    return seg


@dataclass
class Table:
    bbox: PtBox
    cells: list[list[str]]
    aligns: list[str]
    col_rules: list[bool]
    row_rules: list[bool]
    col_widths: list[float | None] | None  # ページの幅で割った値。折り返す列だけ


def join_words(a: str, b: str) -> str:
    """語をつなぐ。英数字どうしなら空白をはさみ、日本語ならそのまま"""
    if a and b and a[-1].isascii() and b[0].isascii():
        return f"{a} {b}"
    return a + b


def column_align(extents: list[tuple[float, float]], left: float, right: float) -> str:
    """列の各セルの文字の (左端, 右端) から揃えを決める。左端・中央・右端のうち、ずれが一番小さいもの。
    どのセルも同じ幅で決められないときは、列の境目（left, right）からの左右の余白で決める"""
    if not extents:
        return "l"
    widths = [x1 - x0 for x0, x1 in extents]
    if len(extents) < 2 or max(widths) - min(widths) <= 1.0:
        pad_left = statistics.mean(x0 - left for x0, _ in extents)
        pad_right = statistics.mean(right - x1 for _, x1 in extents)
        if abs(pad_left - pad_right) <= 2.0:
            return "c"
        return "l" if pad_left < pad_right else "r"

    def spread(values: list[float]) -> float:
        return max(values) - min(values)

    candidates = {
        "l": spread([x0 for x0, _ in extents]),
        "c": spread([(x0 + x1) / 2 for x0, x1 in extents]),
        "r": spread([x1 for _, x1 in extents]),
    }
    return min(candidates, key=lambda a: candidates[a])


def build_table(words: list[tuple], region: PtBox, seg: Segments, body: float, page_width: float,
                ruled_edges: bool) -> Table | None:
    """表の範囲の語（PyMuPDF の words）から、セル・揃え・罫線・列幅を決める。2行×2列に満たなければ None。
    ruled_edges: 範囲の左右の端が罫線から分かっている（横線の長さ = 表の幅）"""
    words = [w for w in words if inside(center(w[:4]), region, 2.0)]
    if len(words) < 4:
        return None
    height_med = statistics.median(w[3] - w[1] for w in words)
    tabcolsep = 0.6 * body  # \tabcolsep（6pt）。文字サイズに合わせて \mag で大きくなる分も含める

    # --- 列の境目: 縦線があればそれを、無ければ「どの行にも文字の無いすき間」を使う
    x_lo, x_hi = region[0], region[2]
    top, bottom = region[1], region[3]
    v_lines = sorted(x for x, y0, y1 in seg.vertical
                     if x_lo - 3 <= x <= x_hi + 3 and min(y1, bottom) - max(y0, top) >= (bottom - top) * RULE_COVERAGE)
    v_dedup: list[float] = []
    for x in v_lines:
        if not v_dedup or x - v_dedup[-1] > 2:
            v_dedup.append(x)
    if len(v_dedup) >= 3:
        bounds = v_dedup
    else:
        spans = sorted((w[0], w[2]) for w in words)
        bounds = [x_lo if ruled_edges else spans[0][0] - tabcolsep]
        right = spans[0][1]
        for x0, x1 in spans[1:]:
            if x0 - right >= height_med * TABLE_COL_GAP:
                bounds.append((right + x0) / 2)
            right = max(right, x1)
        bounds.append(x_hi if ruled_edges else right + tabcolsep)
    n_cols = len(bounds) - 1
    if n_cols < 2:
        return None

    def col_of(w: tuple) -> int:
        cx = (w[0] + w[2]) / 2
        for c in range(n_cols):
            if cx <= bounds[c + 1]:
                return c
        return n_cols - 1

    # --- 行: 縦の位置が重なる語をまとめて、文字の行にする
    text_rows: list[dict] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        row = text_rows[-1] if text_rows else None
        if row is not None and same_row({"bbox": row["bbox"]}, {"bbox": w[:4]}):
            row["words"].append(w)
            row["bbox"] = union_bbox(row["bbox"], w[:4])
        else:
            text_rows.append({"bbox": tuple(w[:4]), "words": [w]})

    # 表の行の間の横線（grid の表）。文字の行の間にこれがあれば、そこで表の行が分かれる
    def has_h_line(y_lo: float, y_hi: float) -> bool:
        width = x_hi - x_lo
        covered = sum(max(0.0, min(x1, x_hi) - max(x0, x_lo)) for x0, x1, y in seg.horizontal if y_lo <= y <= y_hi)
        return width > 0 and covered >= width * RULE_COVERAGE

    # --- 文字の行を表の行にまとめる。空のセルがある文字の行は、上の行のセルの折り返しとみなす
    table_rows: list[dict] = []
    for row in text_rows:
        filled = {col_of(w) for w in row["words"]}
        prev = table_rows[-1] if table_rows else None
        continues = (
            prev is not None
            and len(filled) < n_cols
            and not has_h_line(prev["bbox"][3] - 1, row["bbox"][1] + 1)
        )
        if continues:
            prev["lines"].append(row)
            prev["bbox"] = union_bbox(prev["bbox"], row["bbox"])
        else:
            table_rows.append({"bbox": row["bbox"], "lines": [row]})
    if len(table_rows) < 2:
        return None

    cells: list[list[str]] = []
    wrapped = [False] * n_cols
    extents: list[list[tuple[float, float]]] = [[] for _ in range(n_cols)]
    for trow in table_rows:
        texts = [""] * n_cols
        boxes: list[list[tuple]] = [[] for _ in range(n_cols)]
        lines_in_cell = [0] * n_cols
        for line in trow["lines"]:
            seen = set()
            for w in sorted(line["words"], key=lambda w: w[0]):
                c = col_of(w)
                texts[c] = join_words(texts[c], w[4])
                boxes[c].append(w[:4])
                seen.add(c)
            for c in seen:
                lines_in_cell[c] += 1
        for c in range(n_cols):
            if lines_in_cell[c] > 1:
                wrapped[c] = True
            if boxes[c]:
                extents[c].append((min(b[0] for b in boxes[c]), max(b[2] for b in boxes[c])))
        cells.append(texts)

    aligns = ["l" if wrapped[c] else column_align(extents[c], bounds[c], bounds[c + 1]) for c in range(n_cols)]

    # --- 罫線: 縦線は列の境目ごと、横線は行の境目ごと
    t_top = min(r["bbox"][1] for r in table_rows)
    t_bottom = max(r["bbox"][3] for r in table_rows)
    col_rules = []
    for x in bounds:
        covered = sum(max(0.0, min(y1, t_bottom) - max(y0, t_top)) for vx, y0, y1 in seg.vertical if abs(vx - x) <= 3)
        col_rules.append(covered >= (t_bottom - t_top) * RULE_COVERAGE)
    row_rules = [has_h_line(t_top - 1.5 * body, t_top + 1)]
    for a, b in zip(table_rows, table_rows[1:]):
        row_rules.append(has_h_line(a["bbox"][3] - 1, b["bbox"][1] + 1))
    row_rules.append(has_h_line(t_bottom - 1, t_bottom + 1.5 * body))

    col_widths = None
    if any(wrapped):
        col_widths = [round((bounds[c + 1] - bounds[c]) / page_width, 4) if wrapped[c] else None for c in range(n_cols)]

    word_box = union_all([w[:4] for w in words])
    bbox = union_bbox(word_box, region) if ruled_edges else word_box
    return Table(bbox=bbox, cells=cells, aligns=aligns, col_rules=col_rules, row_rules=row_rules, col_widths=col_widths)


def find_tables(page: pymupdf.Page, rows: list[dict], seg: Segments, body: float, two_column: bool) -> list[Table]:
    """表を探す。
    1. 罫線のある表: 左右の端がそろった横線が2本以上、縦に並んでいるところ（grid / hline / booktabs）
    2. 罫線の無い表: 「表1」などのキャプションの行の下に続く、列に分かれた文字の並び"""
    words = page.get_text("words")
    tables: list[Table] = []

    # 1. 横線のまとまり
    groups: list[list[tuple[float, float, float]]] = []
    for line in sorted(seg.horizontal, key=lambda h: h[2]):
        if line[1] - line[0] < 3 * body:
            continue  # 下線・分数の線など、短い線
        for group in groups:
            x0, x1, y = group[-1]
            if (abs(line[0] - x0) <= TABLE_LINE_TOLERANCE and abs(line[1] - x1) <= TABLE_LINE_TOLERANCE
                    and line[2] - y <= TABLE_MAX_LINE_GAP):
                group.append(line)
                break
        else:
            groups.append([line])
    for group in groups:
        if len(group) < 2:
            continue
        region = (min(h[0] for h in group), group[0][2], max(h[1] for h in group), group[-1][2])
        table = build_table(words, region, seg, body, page.rect.width, ruled_edges=True)
        if table is not None:
            tables.append(table)

    # 2. キャプションの下の、罫線の無い表
    for i, row in enumerate(rows):
        if not TABLE_CAPTION.match(row["text"].strip()) or len(row["text"]) > 100:
            continue
        side = column_of(row["bbox"], page.rect.width) if two_column else None
        picked: list[dict] = []
        bottom = row["bbox"][3]
        for nxt in rows[i + 1:]:
            if side is not None and column_of(nxt["bbox"], page.rect.width) not in (side, 0):
                continue
            limit = (3.5 if not picked else 1.2) * body
            if nxt["bbox"][1] - bottom > limit:
                break
            picked.append(nxt)
            bottom = max(bottom, nxt["bbox"][3])
        if not picked:
            continue
        region = union_all([r["bbox"] for r in picked])
        if any(overlaps(region, t.bbox) for t in tables):
            continue  # 罫線から見つけた表
        table = build_table(words, region, seg, body, page.rect.width, ruled_edges=False)
        if table is not None:
            tables.append(table)
    return tables


# =====================================================================
# 図
# =====================================================================


def find_figures(page: pymupdf.Page, rows: list[dict], tables: list[Table]) -> list[PtBox]:
    """図の範囲を探す。画像と、ベクターの図形（表の罫線・下線などの細い線は除く）を近いものどうしでまとめる。
    図形のまとまりで、中の文字が多いもの（枠付きのコードなど）は図にしない"""
    page_area = page.rect.width * page.rect.height
    items: list[tuple[PtBox, bool]] = []  # (範囲, 画像か)
    for info in page.get_image_info():
        items.append((tuple(info["bbox"]), True))
    for drawing in page.get_drawings():
        box = tuple(drawing["rect"])
        if area(box) >= page_area * 0.8:
            continue  # ページ全体の枠・背景
        if any(overlaps(box, t.bbox, 3.0) and inside(center(box), t.bbox, 3.0) for t in tables):
            continue  # 表の罫線
        items.append((box, False))

    # 近い図形を1つにまとめる（くっつかなくなるまで繰り返す）。[範囲, 画像を含むか, 直線だけか]
    def is_line(box: PtBox) -> bool:
        return box[2] - box[0] < 2.5 or box[3] - box[1] < 2.5

    groups = [[box, is_image, not is_image and is_line(box)] for box, is_image in items]
    merged = True
    while merged:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if overlaps(groups[a][0], groups[b][0], FIGURE_NEAR):
                    groups[a] = [union_bbox(groups[a][0], groups[b][0]), groups[a][1] or groups[b][1],
                                 groups[a][2] and groups[b][2]]
                    del groups[b]
                    merged = True
                    break
            if merged:
                break

    figures = []
    for box, has_image, lines_only in groups:
        if not has_image:
            if box[2] - box[0] < FIGURE_MIN_SIZE or box[3] - box[1] < FIGURE_MIN_SIZE:
                continue
            inner = [ch for row in rows for ch in visible(row["chars"]) if inside(center(ch["bbox"]), box)]
            if lines_only and inner:
                continue  # 直線だけで、中に文字がある → 文字を囲む枠（枠付きのコードなど）
            if sum(area(ch["bbox"]) for ch in inner) > area(box) * FIGURE_MAX_TEXT:
                continue  # 中の大部分が文字
        figures.append(box)
    return figures


# =====================================================================
# ブロック（縦につなぐ）
# =====================================================================


@dataclass
class Block:
    """抽出したブロック1つ。IRの項目に加えて、分類の手がかりになる情報も持つ"""
    page: int                   # 1始まり
    kind: str                   # "text" / "table" / "figure"
    bbox: PtBox                 # pt
    text: str | None = None
    runs: list[Run] | None = None
    font_size: float | None = None
    bold: bool | None = None
    table: Table | None = None
    # ↓ IRには出さない、分類の手がかり
    font: str = ""              # 一番多く使われているフォント名
    mono_ratio: float = 0.0     # 等幅フォントの文字の割合
    math_ratio: float = 0.0     # 数式フォントの文字の割合
    n_rows: int = 0             # 行数
    first_indent: float = 0.0   # 1行目の左端の、2行目以降の左端からのずれ(pt)
    list_head: bool = False     # 箇条書きの頭（• や 1.）で始まる


def row_kind(row: dict) -> str:
    """行の種類。"math"（数式の行）/ "code"（等幅フォントの行）/ "text" """
    if ratio(row["chars"], MATH_FONT) >= MATH_RATIO:
        return "math"
    if ratio(row["chars"], MONO_FONT) >= MONO_RATIO:
        return "code"
    return "text"


def row_font(row: dict) -> str:
    """行の文字（数字・記号・数式・コードを除く）で一番多い書体。見出しと本文の書体の違いを見るのに使う"""
    fonts: dict[str, int] = {}
    for ch in visible(row["chars"]):
        if ch["c"].isalpha() and not MATH_FONT.search(ch["font"]):
            fonts[ch["font"]] = fonts.get(ch["font"], 0) + 1
    return max(fonts, key=lambda f: fonts[f]) if fonts else ""


def is_tiny(row: dict) -> bool:
    """数式の断片になりやすい、2〜3文字だけの英数字の行（∫ の上下の 0 や 1、分母の 3 など）"""
    shown = visible(row["chars"])
    return len(shown) <= 3 and all(ch["c"].isascii() or MATH_FONT.search(ch["font"]) for ch in shown)


@dataclass
class PageContext:
    """ページ全体から分かる情報。区切りの専門家が、行と行をつなぐかを決めるときに使う"""
    body: float                   # 本文の文字サイズ
    width: float                  # ページの幅
    height: float                 # ページの高さ
    two_column: bool
    margins: dict[int, float]     # 段ごとの左端（一番多くの行が始まる位置）。0: 1段組 / 1: 左の段 / 2: 右の段

    def column(self, box: tuple) -> int:
        return column_of(box, self.width) if self.two_column else 0

    def indented(self, row: dict) -> bool:
        """段の左端より字下げされている行（段落の1行目など）"""
        return row["bbox"][0] > self.margins.get(self.column(row["bbox"]), row["bbox"][0]) + INDENT * self.body


def page_context(rows: list[dict], width: float, height: float, two_column: bool) -> PageContext:
    starts: dict[int, dict[int, int]] = {}
    for row in rows:
        col = column_of(row["bbox"], width) if two_column else 0
        x = round(row["bbox"][0])
        starts.setdefault(col, {})[x] = starts.setdefault(col, {}).get(x, 0) + 1
    margins = {col: float(max(xs, key=lambda x: xs[x])) for col, xs in starts.items()}
    return PageContext(body_size(rows), width, height, two_column, margins)


def group_kind(row: dict) -> str:
    """ブロックの1行目から決める、ブロックの種類（"text" / "list" / "math" / "code"）。つなぎ方の決まりが変わる"""
    kind = row_kind(row)
    return "list" if kind == "text" and LIST_HEAD.match(row["text"].lstrip()) else kind


def rule_continues(group: list[dict], gkind: str, row: dict, ctx: PageContext) -> float:
    """区切りの専門家（ルール版）: row が、今のブロック group の続きである自信（0〜1）。ルールなので 1.0 か 0.0。

    つなぐ条件（直前の行とのすき間が小さい、が前提）:
    - 数式: 数式の行・小さい断片の行どうし
    - コード: 等幅フォントの行どうし
    - 箇条書き: 次の項目（• で始まる）か、字下げされた続きの行
    - 段落: 文字の大きさ・書体がほぼ同じで、字下げされていない（字下げは新しい段落の1行目）
    """
    body = ctx.body
    kind = row_kind(row)
    is_list_head = bool(LIST_HEAD.match(row["text"].lstrip()))
    prev = group[-1]
    box = union_all([r["bbox"] for r in group])
    gap = row["bbox"][1] - prev["bbox"][3]
    # 横の範囲が重なっているか。数式は ∫ や添字が左右に少しはみ出すので、文字3つ分まで離れていてよい
    reach = 3 * body if gkind == "math" else 0.0
    same_column = row["bbox"][0] < box[2] + reach and row["bbox"][2] > box[0] - reach
    if ctx.two_column and {column_of(row["bbox"], ctx.width), column_of(box, ctx.width)} == {1, 2}:
        same_column = False
    # すき間の上限は、本文と、その行の文字の大きい方で測る（コードの等幅フォントは本文より少し大きい）
    unit = max(body, prev["size"], row["size"]) if gkind != "math" else body
    ok = False
    if gkind == "math" and same_column and -2 * body <= gap:
        ok = gap <= MATH_GAP * body and (kind == "math" or is_tiny(row))
    elif same_column and -0.5 * body <= gap:
        # 書体が変わったら別のブロック（本文と同じくらいの大きさの見出しを、段落とつながないため）
        fonts = {row_font(row), row_font(prev)}
        same_font = len(fonts) == 1 or "" in fonts
        size_ok = max(row["size"], prev["size"]) <= SIZE_RATIO * min(row["size"], prev["size"])
        if gkind == "math":
            ok = False
        elif gkind == "code":
            ok = gap <= LINE_GAP * unit and kind == "code"
        elif gkind == "list" and kind == "text":
            # 箇条書きの項目の間は、行の間より少し広い（\itemsep）
            limit = (LIST_GAP if is_list_head else LINE_GAP) * unit
            continues = is_list_head or row["bbox"][0] > group[0]["bbox"][0] + 0.3 * body
            ok = gap <= limit and size_ok and same_font and continues
        elif kind == "text" and gap <= LINE_GAP * unit:
            # 字下げされた行は新しい段落の1行目。段の左端と、このブロックの左端の両方と比べる
            left = min(r["bbox"][0] for r in group)
            new_paragraph = ctx.indented(row) or row["bbox"][0] > left + INDENT * body
            ok = size_ok and same_font and not is_list_head and not new_paragraph
    return 1.0 if ok else 0.0


# 区切りの専門家の形: (今のブロック, ブロックの種類, 次の行, ページの情報) → 続きである自信 0〜1
Continues = Callable[[list[dict], str, dict, PageContext], float]


def group_rows(rows: list[dict], ctx: PageContext, continues: Continues = rule_continues,
               threshold: float = 0.5) -> list[list[dict]]:
    """読み順に並んだ行を、上から順に縦につないでブロックにする。
    つなぐかどうかは区切りの専門家（continues）に聞き、自信が threshold 以上ならつなぐ"""
    groups: list[list[dict]] = []
    kinds: list[str] = []
    for row in rows:
        if groups and continues(groups[-1], kinds[-1], row, ctx) >= threshold:
            groups[-1].append(row)
            continue
        groups.append([row])
        kinds.append(group_kind(row))
    return groups


# ---- ブロックの文字と装飾 ----


def char_decorations(ch: dict, gothic_is_bold: bool, underlines: list[tuple[float, float, float]]) -> tuple[str, ...]:
    """1文字の装飾（"bold" / "italic" / "underline"）"""
    if ch["bbox"] is None or ch["c"].isspace():
        return ()
    font = ch["font"]
    marks = []
    if ch["flags"] & 16 or BOLD_FONT.search(font) or (gothic_is_bold and GOTHIC_FONT.search(font)):
        marks.append("bold")
    if (ch["flags"] & 2 or ITALIC_FONT.search(font)) and not MATH_FONT.search(font):
        marks.append("italic")
    cx = (ch["bbox"][0] + ch["bbox"][2]) / 2
    if any(x0 <= cx <= x1 and 0 < y - ch["baseline"] <= ch["size"] * UNDERLINE_MAX_DROP for x0, x1, y in underlines):
        marks.append("underline")
    return tuple(marks)


def block_chars(group: list[dict], code: bool) -> list[dict]:
    """ブロックの文字を、行の間に改行をはさんでつなぐ。コードは行頭に字下げの空白を足す"""
    chars: list[dict] = []
    left = min(r["bbox"][0] for r in group)
    widths = [ch["bbox"][2] - ch["bbox"][0] for r in group for ch in visible(r["chars"])]
    char_width = statistics.median(widths) if widths else 1.0
    for k, row in enumerate(group):
        if k > 0:
            chars.append({**space_char(), "c": "\n"})
        if code:
            indent = round((row["bbox"][0] - left) / char_width)
            chars.extend(space_char() for _ in range(indent))
        chars.extend(row["chars"])
    return chars


def to_runs(chars: list[dict], decorations: list[tuple[str, ...]]) -> list[Run] | None:
    """装飾が同じで隣り合う文字を1つの Run にする。装飾が1つも無い・全部同じ装飾なら None"""
    shown = {d for ch, d in zip(chars, decorations) if ch["bbox"] is not None and not ch["c"].isspace()}
    if shown <= {()} or len(shown) == 1:
        return None
    runs: list[Run] = []
    current: tuple[str, ...] | None = None
    for ch, deco in zip(chars, decorations):
        if ch["c"].isspace() and runs:
            deco = current  # 空白は前の文字の装飾に含める（装飾の区切りを増やさない）
        if runs and deco == current:
            runs[-1].text += ch["c"]
        else:
            runs.append(Run(ch["c"], **{m: True for m in deco or ()}))
            current = deco
    return runs


def make_block(page_no: int, group: list[dict], gkind: str, gothic_is_bold: bool,
               underlines: list[tuple[float, float, float]]) -> Block:
    chars = block_chars(group, code=gkind == "code")
    text = "".join(ch["c"] for ch in chars)
    decorations = [char_decorations(ch, gothic_is_bold, underlines) for ch in chars]
    if gkind in ("math", "code"):
        decorations = [()] * len(chars)  # 数式の分数の線を下線と間違えない。コードも装飾は見ない
    shown = [(ch, d) for ch, d in zip(chars, decorations) if ch["bbox"] is not None and not ch["c"].isspace()]
    sizes: dict[float, int] = {}
    fonts: dict[str, int] = {}
    for ch, _ in shown:
        sizes[round(ch["size"], 1)] = sizes.get(round(ch["size"], 1), 0) + 1
        fonts[ch["font"]] = fonts.get(ch["font"], 0) + 1
    rest_left = min((r["bbox"][0] for r in group[1:]), default=group[0]["bbox"][0])
    # 太字かどうかは、数字・記号を除いた文字で数える（見出しの番号「1.1.1」は太字の書体にならないことがある）
    letters = [(ch, d) for ch, d in shown if ch["c"].isalpha()] or shown
    return Block(
        page=page_no,
        kind="text",
        bbox=union_all([r["bbox"] for r in group]),
        text=text,
        runs=to_runs(chars, decorations),
        font_size=max(sizes, key=lambda s: sizes[s]) if sizes else None,
        bold=sum("bold" in d for _, d in letters) > len(letters) / 2 if letters else None,
        font=max(fonts, key=lambda f: fonts[f]) if fonts else "",
        mono_ratio=round(sum(bool(MONO_FONT.search(ch["font"])) for ch, _ in shown) / max(1, len(shown)), 3),
        math_ratio=round(sum(bool(MATH_FONT.search(ch["font"])) for ch, _ in shown) / max(1, len(shown)), 3),
        n_rows=len(group),
        first_indent=round(group[0]["bbox"][0] - rest_left, 2),
        list_head=bool(LIST_HEAD.match(text.lstrip())),
    )


def body_size(rows: list[dict]) -> float:
    """ページの本文の文字サイズ（一番多くの文字に使われているサイズ）"""
    counts: dict[float, int] = {}
    for row in rows:
        for ch in visible(row["chars"]):
            counts[round(ch["size"], 1)] = counts.get(round(ch["size"], 1), 0) + 1
    return max(counts, key=lambda s: counts[s]) if counts else 10.0


def page_is_gothic(rows: list[dict]) -> bool:
    """本文（数式・コード以外で一番多い書体）がゴシック体か"""
    fonts: dict[str, int] = {}
    for row in rows:
        for ch in visible(row["chars"]):
            if not MATH_FONT.search(ch["font"]) and not MONO_FONT.search(ch["font"]):
                fonts[ch["font"]] = fonts.get(ch["font"], 0) + 1
    return bool(fonts) and bool(GOTHIC_FONT.search(max(fonts, key=lambda f: fonts[f])))


# =====================================================================
# 1ページ・文書全体
# =====================================================================


@dataclass
class PageRows:
    """表・図を取り除いたあとの、1ページの行（読み順）と、表・図"""
    rows: list[dict]
    ctx: PageContext
    tables: list[Table]
    figures: list[PtBox]
    underlines: list[tuple[float, float, float]]


def page_rows(page: pymupdf.Page) -> PageRows:
    """1ページの行（横につないだもの、読み順）。表・図の中の行は除く。区切りの専門家の学習データにも使う"""
    width = page.rect.width
    lines = page_lines(page)
    two_column = is_two_column([line["bbox"] for line in lines], width)
    order = reading_order([line["bbox"] for line in lines], width, two_column)
    rows = merge_rows([lines[i] for i in order], width, two_column)
    rows = [rows[i] for i in reading_order([r["bbox"] for r in rows], width, two_column)]
    body = body_size(rows)
    seg = page_segments(page)

    tables = find_tables(page, rows, seg, body, two_column)
    figures = find_figures(page, rows, tables)
    taken = [t.bbox for t in tables] + figures
    rows = [r for r in rows if not any(inside(center(r["bbox"]), box, 1.0) for box in taken)]
    ctx = page_context(rows, width, page.rect.height, two_column)
    ctx.body = body  # 本文の文字サイズは、表・図の中の文字も含めて数える（取り除く前と同じ値にする）
    return PageRows(rows, ctx, tables, figures, seg.horizontal)


def extract_page(page: pymupdf.Page, continues: Continues = rule_continues) -> list[Block]:
    """1ページ分のブロック（読み順）。continues は区切りの専門家（行と行をつなぐかを決める）"""
    page_no = page.number + 1
    pr = page_rows(page)
    rows, ctx, tables, figures = pr.rows, pr.ctx, pr.tables, pr.figures
    width, two_column = ctx.width, ctx.two_column

    gothic_is_bold = not page_is_gothic(rows)
    blocks: list[Block] = []
    for group in group_rows(rows, ctx, continues):
        gkind = row_kind(group[0])
        blocks.append(make_block(page_no, group, gkind, gothic_is_bold, pr.underlines))
    for t in tables:
        text = "\n".join(" ".join(c for c in row if c) for row in t.cells)
        blocks.append(Block(page=page_no, kind="table", bbox=t.bbox, text=text, table=t, n_rows=len(t.cells)))
    for box in figures:
        blocks.append(Block(page=page_no, kind="figure", bbox=box))

    order = reading_order([b.bbox for b in blocks], width, two_column)
    return [blocks[i] for i in order]


@dataclass
class PageBlocks:
    page: int       # 1始まり
    width: float
    height: float
    blocks: list[Block]


def extract_blocks(pdf_path: str, continues: Continues = rule_continues) -> list[PageBlocks]:
    """PDFの全ページを読んで、ページごとのブロックを返す（分類器はこれを使う）。
    continues で区切りの専門家を差し替えられる（既定はルール版）"""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pages.append(PageBlocks(page.number + 1, page.rect.width, page.rect.height, extract_page(page, continues)))
    return pages


def to_element(block: Block, width: float, height: float, label: str | None = None, level: int | None = None) -> Element:
    """Block を IR の要素にする（IRに無い手がかりは落とす）"""
    t = block.table
    return Element(
        kind=block.kind,
        bbox=normalize_bbox(block.bbox, width, height),
        text=block.text,
        runs=block.runs,
        font_size=block.font_size,
        bold=block.bold,
        cells=t.cells if t else None,
        aligns=t.aligns if t else None,
        col_rules=t.col_rules if t else None,
        row_rules=t.row_rules if t else None,
        col_widths=t.col_widths if t else None,
        label=label,
        level=level,
    )


def to_ir(pages: list[PageBlocks], source: str) -> IRDocument:
    ir_pages = [
        Page(page=p.page, width=p.width, height=p.height, unit="pt",
             elements=[to_element(b, p.width, p.height) for b in p.blocks])
        for p in pages
    ]
    return IRDocument(source=source, extractor=NAME, pages=ir_pages)


def extract(pdf_path: str) -> IRDocument:
    """PDFの全ページを読んで、IR（①）を返す"""
    return to_ir(extract_blocks(pdf_path), os.path.basename(pdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → IR（hybrid_extractor の抽出器）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("out", help="出力するIRのJSON")
    args = parser.parse_args()
    ir = extract(args.pdf)
    problems = validate(ir)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(ir, args.out)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()
