"""AIの専門家が読む「特徴」（数字の並び）を作る。

    区切りの専門家   : 隣り合う2行の組   → pair_features   （PAIR_FEATURE_NAMES の数）
    ラベルの専門家   : 1ブロック         → block_features  （BLOCK_FEATURE_NAMES の数）
    文中の数式の専門家: 段落の1文字ずつ   → char_features   （1文字あたり CHAR_FEATURE_NAMES の数）＋ 文字そのもの

学習データ（build_dataset.py・inline_math_data.py が保存する JSON）と、実際に使うとき（抽出器・オペレータ）で、
必ずこのファイルの同じ関数を通す。入力は「記録」（record）という辞書にそろえてある:

    row_record(行, ページの情報)    → 行の記録     （座標はページの幅・高さで割った 0〜1）
    ctx_record(ページの情報)        → ページの記録 （本文の文字サイズ・段の左端と右端）
    block_record(ブロック, ページ)  → ブロックの記録

特徴の値は、長さを本文の文字サイズで割るなど、文書の大きさ（10pt / 12pt、A4 など）によらない数にしている。
"""

import math
import re
import statistics
import unicodedata

from common.ir import normalize_bbox
from hybrid_extractor.extract import (
    LIST_HEAD,
    MATH_FONT,
    MONO_FONT,
    Block,
    PageContext,
    row_font,
    row_kind,
    visible,
)

# =====================================================================
# 記録（学習データと、実際に使うときで共通の形）
# =====================================================================


def ctx_record(ctx: PageContext) -> dict:
    """ページの情報。段の左端・右端は 0〜1 にする（キーは段の番号の文字列。JSON にするため）"""
    return {
        "width": ctx.width,
        "height": ctx.height,
        "body": ctx.body,
        "two_column": ctx.two_column,
        "margins": {str(col): x / ctx.width for col, x in ctx.margins.items()},
        "rights": {str(col): x / ctx.width for col, x in ctx.rights.items()},
    }


def row_record(row: dict, ctx: PageContext) -> dict:
    """1行（横につないだもの）の記録"""
    return {
        "bbox": list(normalize_bbox(row["bbox"], ctx.width, ctx.height)),
        "text": row["text"],
        "size": round(row["size"], 2),
        "font": row_font(row),
        "kind": row_kind(row),          # "text" / "math" / "code"
        "list_head": bool(LIST_HEAD.match(row["text"].lstrip())),
        "indented": ctx.indented(row),  # 段の左端より字下げされている
        "column": ctx.column(row["bbox"]),
        "n_chars": len(visible(row["chars"])),
    }


def block_record(block: Block, width: float, height: float) -> dict:
    """1ブロックの記録（build_dataset.py の docs/<id>.json の blocks と同じ形。label などは足さない）"""
    return {
        "kind": block.kind,
        "bbox": list(normalize_bbox(block.bbox, width, height)),
        "text": block.text,
        "font_size": block.font_size,
        "bold": block.bold,
        "font": block.font,
        "mono_ratio": block.mono_ratio,
        "math_ratio": block.math_ratio,
        "n_rows": block.n_rows,
        "first_indent": block.first_indent,
        "list_head": block.list_head,
    }


# =====================================================================
# 小さい道具
# =====================================================================

ROW_KINDS = ["text", "math", "code"]
GROUP_KINDS = ["text", "list", "math", "code"]
BLOCK_KINDS = ["text", "table", "figure"]

SENTENCE_END = set("。．.!?！？:：")
COMMA = set("、，,")
BULLETS = set("•◦▪‣∙·∗★☆■□◆◇―–—-*・>▶►✓")
SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\.?\s")
CAPTION_HEAD = re.compile(r"^(図|表|Figure|Fig\.|Table)\s*\d")
ENUM_HEAD = re.compile(r"^(\(?\d{1,2}[.)]|\(?[a-z][.)]|\([ivx]+\))\s")
DIGITS_ONLY = re.compile(r"^[\s\-–—]*[0-9ivxlc]+[\s\-–—]*$", re.IGNORECASE)


def one_hot(value: str, choices: list[str]) -> list[float]:
    return [float(value == c) for c in choices]


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def is_cjk(c: str) -> bool:
    return ord(c) >= 0x3000


def char_mix(text: str) -> list[float]:
    """文字の種類の割合: 数字・英字・日本語など・記号"""
    shown = [c for c in text if not c.isspace()]
    if not shown:
        return [0.0, 0.0, 0.0, 0.0]
    n = len(shown)
    digits = sum(c.isdigit() for c in shown)
    ascii_letters = sum(c.isascii() and c.isalpha() for c in shown)
    cjk = sum(is_cjk(c) for c in shown)
    return [digits / n, ascii_letters / n, cjk / n, (n - digits - ascii_letters - cjk) / n]


# =====================================================================
# 区切りの専門家: 隣り合う2行
# =====================================================================


def group_kind_of(first: dict) -> str:
    """ブロックの1行目の記録から決める、ブロックの種類（extract.group_kind と同じ決め方）"""
    return "list" if first["kind"] == "text" and first["list_head"] else first["kind"]


PAIR_FEATURE_NAMES = [
    "gap", "dx_prev", "dx_first", "dright", "prev_short", "prev_indent", "next_indent",
    "prev_size", "next_size", "size_ratio", "prev_height", "next_height", "same_font", "same_column", "x_overlap",
    *[f"prev_{k}" for k in ROW_KINDS], *[f"next_{k}" for k in ROW_KINDS], *[f"group_{k}" for k in GROUP_KINDS],
    "prev_list_head", "next_list_head", "prev_end_sentence", "prev_end_comma", "next_tiny", "log_next_chars",
]


def pair_features(first: dict, prev: dict, nxt: dict, ctx: dict) -> list[float]:
    """「今のブロック（1行目 first、最後の行 prev）」に、次の行 nxt が続くか、を判断する手がかり。

    長さは本文の文字サイズ（body）で割る。例: gap = 1.0 は「本文の文字1つ分すき間がある」"""
    w, h, body = ctx["width"], ctx["height"], ctx["body"]
    p, n, f = prev["bbox"], nxt["bbox"], first["bbox"]

    def margin(rec: dict) -> float:
        return ctx["margins"].get(str(rec["column"]), rec["bbox"][0])

    def right(rec: dict) -> float:
        return ctx["rights"].get(str(rec["column"]), rec["bbox"][2])

    overlap = min(p[2], n[2]) - max(p[0], n[0])
    narrow = max(1e-6, min(p[2] - p[0], n[2] - n[0]))
    text_p, text_n = prev["text"].strip(), nxt["text"].strip()
    return [
        clip((n[1] - p[3]) * h / body, -5, 10),             # 行の間のすき間
        clip((n[0] - p[0]) * w / body, -20, 20),             # 前の行との左端のずれ（字下げ）
        clip((n[0] - f[0]) * w / body, -20, 20),             # ブロックの1行目との左端のずれ（箇条書きの続きの行）
        clip((n[2] - p[2]) * w / body, -40, 40),             # 右端のずれ
        clip((right(prev) - p[2]) * w / body, 0, 40),        # 前の行が段の右端より手前で終わっている（段落の最後の行）
        clip((p[0] - margin(prev)) * w / body, -5, 20),      # 前の行の、段の左端からの字下げ
        clip((n[0] - margin(nxt)) * w / body, -5, 20),       # 次の行の、段の左端からの字下げ
        prev["size"] / body,
        nxt["size"] / body,
        nxt["size"] / max(prev["size"], 1e-6),
        (p[3] - p[1]) * h / body,
        (n[3] - n[1]) * h / body,
        float(prev["font"] == nxt["font"] or not prev["font"] or not nxt["font"]),
        float(prev["column"] == nxt["column"]),
        clip(overlap / narrow, -1, 1),                       # 横の重なり（2段組の別の段なら負）
        *one_hot(prev["kind"], ROW_KINDS),
        *one_hot(nxt["kind"], ROW_KINDS),
        *one_hot(group_kind_of(first), GROUP_KINDS),
        float(prev["list_head"]),
        float(nxt["list_head"]),
        float(bool(text_p) and text_p[-1] in SENTENCE_END),  # 前の行が文の終わりで終わる
        float(bool(text_p) and text_p[-1] in COMMA),
        float(nxt["n_chars"] <= 3),                          # 数式の断片のような短い行
        math.log1p(nxt["n_chars"]),
    ]


# =====================================================================
# ラベルの専門家: 1ブロック
# =====================================================================


def doc_body(pages: list[dict]) -> tuple[float, str]:
    """文書の本文の文字サイズと書体（文字数で重みをつけた一番多いもの）。pages は {"blocks": [記録, ...]} の並び"""
    sizes: dict[float, int] = {}
    fonts: dict[str, int] = {}
    for page in pages:
        for b in page["blocks"]:
            if b["kind"] != "text" or not b.get("text"):
                continue
            n = len(b["text"])
            if b.get("font_size"):
                sizes[b["font_size"]] = sizes.get(b["font_size"], 0) + n
            if b.get("font"):
                fonts[b["font"]] = fonts.get(b["font"], 0) + n
    body = max(sizes, key=lambda s: sizes[s]) if sizes else 10.0
    font = max(fonts, key=lambda f: fonts[f]) if fonts else ""
    return body, font


BLOCK_FEATURE_NAMES = [
    "size_ratio", "bold", "body_font", "mono", "math", "log_rows", "first_indent", "list_head",
    "x0", "y0", "x1", "y1", "width", "height", "off_center", "log_len",
    "first_digit", "first_upper", "first_lower", "first_cjk", "first_bullet", "section_number", "caption_head",
    "enum_head", "ends_sentence", "ends_colon", "digits_only",
    "mix_digit", "mix_ascii", "mix_cjk", "mix_symbol",
    *[f"kind_{k}" for k in BLOCK_KINDS], "first_page", "top_zone", "bottom_zone",
    "gap_above", "gap_below", "prev_size_ratio", "next_size_ratio", "prev_same_font", "next_same_font",
    "prev_mono", "next_mono", "prev_list_head", "next_list_head", "prev_is_table", "next_is_table",
    "prev_is_figure", "next_is_figure", "repeats",
]


def block_features(blocks: list[dict], i: int, page: dict, doc: dict) -> list[float]:
    """ページの i 番目のブロック（読み順）の手がかり。前後のブロックも少し見る。
    page は {"page", "width", "height"}、doc は {"body", "font", "repeated"}（repeated: 複数ページの上に出る文字）"""
    b = blocks[i]
    body, h = doc["body"], page["height"]
    x0, y0, x1, y1 = b["bbox"]
    text = (b.get("text") or "").strip()
    first = text[:1]
    prev = blocks[i - 1] if i > 0 else None
    nxt = blocks[i + 1] if i + 1 < len(blocks) else None

    def size_ratio(other: dict | None) -> float:
        if other is None or not other.get("font_size") or not b.get("font_size"):
            return 1.0
        return other["font_size"] / b["font_size"]

    return [
        (b.get("font_size") or body) / body,
        float(bool(b.get("bold"))),
        float(b.get("font") == doc["font"]),
        b.get("mono_ratio", 0.0),
        b.get("math_ratio", 0.0),
        math.log1p(b.get("n_rows", 0)),
        clip(b.get("first_indent", 0.0) / body, -10, 10),
        float(bool(b.get("list_head"))),
        x0, y0, x1, y1, x1 - x0,
        clip((y1 - y0) * h / body, 0, 60),
        abs((x0 + x1) / 2 - 0.5),
        math.log1p(len(text)),
        float(first.isdigit()),
        float(first.isascii() and first.isupper()),
        float(first.isascii() and first.islower()),
        float(bool(first) and is_cjk(first)),
        float(first in BULLETS),
        float(bool(SECTION_NUMBER.match(text))),
        float(bool(CAPTION_HEAD.match(text))),
        float(bool(ENUM_HEAD.match(text))),
        float(bool(text) and text[-1] in SENTENCE_END),
        float(bool(text) and text[-1] in ":："),
        float(bool(DIGITS_ONLY.match(text))),
        *char_mix(text),
        *one_hot(b["kind"], BLOCK_KINDS),
        float(page["page"] == 1),
        float(y1 < 0.15),
        float(y0 > 0.88),
        clip((y0 - prev["bbox"][3]) * h / body, -5, 20) if prev else 20.0,
        clip((nxt["bbox"][1] - y1) * h / body, -5, 20) if nxt else 20.0,
        size_ratio(prev),
        size_ratio(nxt),
        float(prev is not None and prev.get("font") == b.get("font")),
        float(nxt is not None and nxt.get("font") == b.get("font")),
        prev.get("mono_ratio", 0.0) if prev else 0.0,
        nxt.get("mono_ratio", 0.0) if nxt else 0.0,
        float(bool(prev and prev.get("list_head"))),
        float(bool(nxt and nxt.get("list_head"))),
        float(bool(prev and prev["kind"] == "table")),
        float(bool(nxt and nxt["kind"] == "table")),
        float(bool(prev and prev["kind"] == "figure")),     # キャプションは図・表のすぐ隣にある
        float(bool(nxt and nxt["kind"] == "figure")),
        float(text in doc["repeated"]),
    ]


def doc_info(pages: list[dict]) -> dict:
    """block_features に渡す、文書全体の情報"""
    body, font = doc_body(pages)
    seen: dict[str, set[int]] = {}
    for page in pages:
        for b in page["blocks"]:
            text = (b.get("text") or "").strip()
            if b["kind"] == "text" and text and len(text) <= 60 and b["bbox"][3] <= 0.15:
                seen.setdefault(text, set()).add(page["page"])
    return {"body": body, "font": font, "repeated": {t for t, on in seen.items() if len(on) >= 2}}


# =====================================================================
# 文中の数式の専門家: 1文字ずつ
# =====================================================================

BB_FONT = re.compile(r"MSBM|BBold|DoubleStruck", re.IGNORECASE)
BF_FONT = re.compile(r"CMBX|CMMIB|Bold", re.IGNORECASE)
CJK_FONT = re.compile(r"Mincho|Gothic|HaranoAji|IPA|Hiragino|CJK|Ryumin|Meiryo|MS-", re.IGNORECASE)
OPERATORS = set("=+-−×·÷±<>≤≥≠≈≡∼∝/|")
ACCENT_CHARS = set("¯ˉˆ⃗˜˙")

CHAR_FEATURE_NAMES = [
    "space", "newline", "math_font", "mono_font", "bold_font", "bb_font", "cjk_font",
    "ascii_lower", "ascii_upper", "digit", "cjk", "greek", "math_symbol", "operator", "bracket", "punct", "accent",
    "size_ratio", "small", "base_shift", "italic_flag", "bold_flag", "gap_before", "gap_after", "x_in_line",
    "prev_math_font", "next_math_font",
]


def char_category_flags(c: str) -> list[float]:
    o = ord(c[0]) if c else 0
    return [
        float(c.isascii() and c.islower()),
        float(c.isascii() and c.isupper()),
        float(c.isdigit()),
        float(is_cjk(c)),
        float(0x370 <= o <= 0x3FF or c in "ℓ∂"),
        float(0x2190 <= o <= 0x22FF or 0x2A00 <= o <= 0x2AFF),
        float(c in OPERATORS),
        float(c in "()[]{}"),
        float(bool(c) and unicodedata.category(c[0]).startswith("P") and c not in "()[]{}"),
        float(c in ACCENT_CHARS),
    ]


def line_baselines(chars: list[dict]) -> list[float | None]:
    """文字ごとの「その行のベースライン」（行の大きい文字のベースラインの中央値）。行は改行で区切る"""
    out: list[float | None] = [None] * len(chars)
    start = 0
    for k in range(len(chars) + 1):
        if k == len(chars) or chars[k]["c"] == "\n":
            line = [ch for ch in chars[start:k] if ch["bbox"] is not None and not ch["c"].isspace()]
            if line:
                big = max(ch["size"] for ch in line)
                base = statistics.median(ch["baseline"] for ch in line if ch["size"] >= big * 0.9)
                for j in range(start, k):
                    out[j] = base
            start = k + 1
    return out


def char_features(chars: list[dict], block_size: float, width: float, height: float) -> list[list[float]]:
    """ブロックの文字の並びの、1文字ずつの手がかり。
    chars は inline_math_data.py の記録（bbox・baseline は 0〜1）か、抽出器の文字（pt）を to_char_records で直したもの"""
    bases = line_baselines(chars)
    size = block_size or 10.0
    feats = []
    for k, ch in enumerate(chars):
        synthetic = ch["bbox"] is None
        font = ch["font"]
        prev = chars[k - 1] if k > 0 else None
        nxt = chars[k + 1] if k + 1 < len(chars) else None
        if synthetic:
            gap_before = gap_after = 0.0
            shift = 0.0
            x_in_line = 0.0
        else:
            gap_before = clip((ch["bbox"][0] - prev["bbox"][2]) * width / size, -3, 5) if prev and prev["bbox"] else 0.0
            gap_after = clip((nxt["bbox"][0] - ch["bbox"][2]) * width / size, -3, 5) if nxt and nxt["bbox"] else 0.0
            shift = clip((ch["baseline"] - bases[k]) * height / size, -1.5, 1.5) if bases[k] is not None else 0.0
            x_in_line = ch["bbox"][0]
        feats.append([
            float(synthetic and ch["c"] != "\n"),
            float(ch["c"] == "\n"),
            float(bool(MATH_FONT.search(font))),
            float(bool(MONO_FONT.search(font))),
            float(bool(BF_FONT.search(font))),
            float(bool(BB_FONT.search(font))),
            float(bool(CJK_FONT.search(font))),
            *char_category_flags(ch["c"]),
            (ch["size"] / size) if not synthetic else 1.0,
            float(not synthetic and ch["size"] < size * 0.85),
            shift,                                           # ベースラインのずれ（上付きは負、下付きは正）
            float(bool(ch.get("flags", 0) & 2)),
            float(bool(ch.get("flags", 0) & 16)),
            gap_before,
            gap_after,
            x_in_line,
            float(bool(prev and MATH_FONT.search(prev["font"]))),
            float(bool(nxt and MATH_FONT.search(nxt["font"]))),
        ])
    return feats


def to_char_records(chars: list[dict], width: float, height: float) -> list[dict]:
    """抽出器の文字（pt）を、inline_math_data.py の記録と同じ形（bbox・baseline は 0〜1）にする"""
    out = []
    for ch in chars:
        synthetic = ch["bbox"] is None
        out.append({
            "c": ch["c"], "font": ch["font"], "size": ch["size"], "flags": ch.get("flags", 0),
            "bbox": None if synthetic else list(normalize_bbox(ch["bbox"], width, height)),
            "baseline": None if synthetic else ch["baseline"] / height,
        })
    return out


def block_char_size(chars: list[dict]) -> float:
    """ブロックの文字サイズ（見えている文字で一番多い大きさ）。extract.make_block の font_size と同じ決め方"""
    counts: dict[float, int] = {}
    for ch in chars:
        if ch["bbox"] is not None and not ch["c"].isspace():
            counts[round(ch["size"], 1)] = counts.get(round(ch["size"], 1), 0) + 1
    return max(counts, key=lambda s: counts[s]) if counts else 10.0
