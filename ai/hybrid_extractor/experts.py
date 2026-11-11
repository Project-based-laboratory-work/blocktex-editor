"""ラベルの専門家。1ブロックを見て「自分のラベルである自信（0〜1）」を答える。

最初は全員ルール版（自信は 1.0 か 0.0）。Step 8 で、1人ずつ自作AIの専門家に入れ替える。
入れ替えるときは、同じ形の Expert を作って operator_rules.py の LABEL_EXPERTS に入れるだけでよい。

    Expert(name="list_rule", label="list", judge=関数)
        judge(ブロック, 文書の情報) → 自信 0〜1
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from hybrid_extractor.extract import Block

SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
HEADING_SIZE_RATIO = 1.15   # 本文の文字サイズのこの倍以上なら見出し
HEADING_MAX_CHARS = 80
CODE_RATIO = 0.6
MATH_RATIO = 0.3


@dataclass
class DocInfo:
    """文書全体から分かる情報（専門家が「本文と比べてどうか」を見るのに使う）"""
    body_size: float  # 本文の文字サイズ
    body_font: str    # 本文の書体


@dataclass
class Expert:
    name: str                                   # 専門家の名前（どの専門家が決めたかを記録するため）
    label: str                                  # この専門家が答えるラベル
    judge: Callable[[Block, DocInfo], float]    # 自分のラベルである自信 0〜1


def text_of(block: Block) -> str:
    return (block.text or "").strip()


# ---- ルール版の専門家 ----


def code_rule(block: Block, info: DocInfo) -> float:
    """等幅フォントの文字が多ければコード"""
    return 1.0 if block.mono_ratio >= CODE_RATIO else 0.0


def math_rule(block: Block, info: DocInfo) -> float:
    """数式のフォントの文字が多ければ数式"""
    return 1.0 if block.math_ratio >= MATH_RATIO else 0.0


def list_rule(block: Block, info: DocInfo) -> float:
    """• や 1. で始まれば箇条書き"""
    return 1.0 if block.list_head else 0.0


def heading_rule(block: Block, info: DocInfo) -> float:
    """短くて、本文より大きいか、番号から始まる1行で太字か本文と違う書体なら見出し"""
    text = text_of(block)
    short = len(text) <= HEADING_MAX_CHARS and block.n_rows <= 2
    big = (block.font_size or 0) >= info.body_size * HEADING_SIZE_RATIO
    numbered = block.n_rows == 1 and bool(SECTION_NUMBER.match(text)) and (bool(block.bold) or block.font != info.body_font)
    return 1.0 if short and (big or numbered) else 0.0


RULE_EXPERTS = {
    "code": Expert("code_rule", "code", code_rule),
    "math": Expert("math_rule", "math", math_rule),
    "list": Expert("list_rule", "list", list_rule),
    "heading": Expert("heading_rule", "heading", heading_rule),
}
