"""
PDFから取り出したテキストブロック1つを、モデルに入れる数値の列（特徴ベクトル）にする。

数式OCRでは画像そのものを入力にしたが、こちらは pdf_extract が取り出した情報（位置・文字の大きさ・
フォント名・文字列）から、人が「見出しっぽい」「表っぽい」と判断する手がかりを数値にして渡す。

値の大きさを 0〜1 程度にそろえているのは、特徴ごとに桁が違う（座標は数百、比率は0〜1）と、
桁の大きい特徴ばかりが学習に効いてしまうため。
"""

import math
import re

import torch

from rule_based.pdf_extract import PageBlocks, TextBlock

# フォント名の種類。実際のPDFではフォントが様々なので、名前の一部で大まかに分ける。
# 「ゴシック体か」「明朝体か」は特徴に入れない。文書によって本文がゴシック体のこともあり、
# 「ゴシック体なら見出し」のような学習データの偏りをモデルが覚えてしまうため（ショートカット学習）。
# 代わりに「本文と同じフォントか」を入れる（_font_features）。
FONT_KINDS = {
    "mono": re.compile(r"CMTT|Courier|Mono|Consolas|Inconsolata|Menlo|LMMono|Typewriter", re.I),
    "math": re.compile(r"CMMI|CMSY|CMEX|MSAM|MSBM|Math|Symbol|EUFM|rsfs|STIX", re.I),
    "bold": re.compile(r"Gothic|Goth|Kaku|Sans|Bold|BX|Heavy|Black|Medium|CMSS", re.I),
}

SECTION_NUMBER = re.compile(r"^\d+(\.\d+)*\s")  # 「1 」「1.2 」
CAPTION_START = re.compile(r"^([図表]\s*\d|(Figure|Fig\.|Table)\s*\d)")
LIST_MARKER = re.compile(r"^([・•∙\-–—]|\(\d+\)|\d+\.\s|[a-z]\)\s)")
JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
MATH_SYMBOL = re.compile(r"[=+\-−×·÷≤≥≠≈∑∏∫√∞∂∇∈⊂∪∩→←⇒^_()\[\]{}|/<>!′αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]")


def _font_features(name: str, body_name: str) -> list[float]:
    """フォントを、本文のフォントとの比較で表す。

    [本文と同じフォント, 本文と同じ種類, 等幅, 数式用, 本文より太い種類]
    """
    def kind(n: str) -> str:
        return next((k for k, pattern in FONT_KINDS.items() if pattern.search(n)), "plain")

    this, body = kind(name), kind(body_name)
    return [
        float(name == body_name),
        float(this == body),
        float(this == "mono"),
        float(this == "math"),
        float(this == "bold" and body != "bold"),
    ]


def _ratio(pattern: re.Pattern, text: str) -> float:
    chars = text.replace(" ", "").replace("\n", "")
    return len(pattern.findall(chars)) / max(len(chars), 1)


def body_font_size(blocks: list[TextBlock]) -> float:
    """ページの本文の文字サイズ = 一番多くの文字に使われているサイズ"""
    counts: dict[float, int] = {}
    for block in blocks:
        counts[block.font_size] = counts.get(block.font_size, 0) + len(block.text)
    return max(counts, key=counts.get) if counts else 10.0


def body_font_name(blocks: list[TextBlock]) -> str:
    """ページの本文のフォント = 一番多くの文字に使われているフォント"""
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.font_name] = counts.get(block.font_name, 0) + len(block.text)
    return max(counts, key=counts.get) if counts else ""


def block_features(
    block: TextBlock, page_width: float, page_height: float, body_size: float, body_name: str, left_margin: float
) -> list[float]:
    x0, y0, x1, y1 = block.bbox
    text = block.text
    first_line = text.split("\n", 1)[0]
    return [
        # 位置と大きさ（ページの幅・高さに対する割合）
        x0 / page_width, y0 / page_height, x1 / page_width, y1 / page_height,
        (x1 - x0) / page_width, (y1 - y0) / page_height,
        (x0 + x1) / 2 / page_width,
        (x0 - left_margin) / page_width,  # 字下げ・中央寄せの手がかり
        # 文字の大きさ
        block.font_size / body_size,
        block.font_size / 12.0,
        # 文字列の形
        min(text.count("\n") + 1, 10) / 10,  # 行数
        math.log1p(len(text)) / 6,  # 文字数（対数）
        float(bool(SECTION_NUMBER.match(first_line))),
        float(bool(CAPTION_START.match(first_line))),
        float(bool(LIST_MARKER.match(first_line))),
        float(text.rstrip().endswith(("。", "．", "."))),
        _ratio(re.compile(r"\d"), text),
        _ratio(re.compile(r"[A-Za-z]"), text),
        _ratio(JAPANESE, text),
        _ratio(MATH_SYMBOL, text),
        # フォント
        float(block.bold),
        *_font_features(block.font_name, body_name),
    ]  # fmt: skip


NUM_FEATURES = 21 + 5


def page_features(page: PageBlocks, page_height: float) -> torch.Tensor:
    """ページ内の全テキストブロックの特徴 → (ブロック数, NUM_FEATURES)"""
    blocks = page.text_blocks
    body_size, body_name = body_font_size(blocks), body_font_name(blocks)
    left_margin = min((b.bbox[0] for b in blocks), default=0.0)
    rows = [block_features(b, page.width, page_height, body_size, body_name, left_margin) for b in blocks]
    return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), NUM_FEATURES)
