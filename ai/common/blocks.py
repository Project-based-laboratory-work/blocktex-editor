"""
ラベル付きIR → エディタ用JSON（BlockDocument）への変換。どの構造化器の出力にもこの1本を使う。

出力の形は web/src/model.ts の型に合わせている。model.ts の SCHEMA_VERSION を上げたら、
ここの EDITOR_SCHEMA_VERSION と変換も合わせて直すこと（tests/test_blocks.py が版数の食い違いを検出する）。

前提: ラベル付きIRの elements は、ページ内の読み順に並んでいること（並べるのは構造化器の役目）。

使い方:
    python -m common.blocks out/sample.labeled.json -o out/sample.blocks.json
"""

import argparse
import json
import re
import uuid
from itertools import groupby
from typing import Any

from common.ir import Element, IRDocument, load_ir

EDITOR_SCHEMA_VERSION = 3

MAX_HEADING_LEVEL = 3
TABLE_CAPTION_PATTERN = re.compile(r"(表|Table)\s*\d")

# 文字の装飾（ir.py の Run と model.ts の TextMarks で同じ名前）
MARKS = ("bold", "italic", "underline")
# 1文字と、その文字に付いている装飾（MARKS のうち付いているもの）
Char = tuple[str, tuple[str, ...]]

# \maketitle のタイトル・著者・日付。まとめて文書の先頭のタイトルブロック1つにする
TITLE_LABELS = ("title", "author", "date")

# 箇条書きの行頭記号 → model.ts の ListMarker
LIST_MARKER_PATTERNS = [
    (re.compile(r"^[(（]\d+[)）]\s*"), "paren"),
    (re.compile(r"^\d+[.．)）]\s*"), "number"),
    (re.compile(r"^[・]\s*"), "dot"),
    (re.compile(r"^[•●◦▪*]\s*"), "bullet"),
    (re.compile(r"^[-–—−‐]\s+"), "dash"),
]


def _new_id() -> str:
    return str(uuid.uuid4())


def _label_of(el: Element) -> str:
    """ラベルが無い要素（抽出しただけのIR）は kind から決める。"""
    if el.label is not None:
        return el.label
    return "paragraph" if el.kind == "text" else el.kind


def _chars(el: Element) -> list[Char]:
    """要素の文字を、1文字ずつ装飾と組にする。runs が無い要素は装飾なし。"""
    if el.runs is None:
        return [(c, ()) for c in el.text or ""]
    return [(c, tuple(mark for mark in MARKS if getattr(run, mark))) for run in el.runs for c in run.text]


def _lines(chars: list[Char]) -> list[list[Char]]:
    """改行で行に分け、行の前後の空白を除く。空の行は捨てる。"""
    lines: list[list[Char]] = [[]]
    for ch in chars:
        if ch[0] == "\n":
            lines.append([])
        else:
            lines[-1].append(ch)
    stripped = []
    for line in lines:
        start, end = 0, len(line)
        while start < end and line[start][0].isspace():
            start += 1
        while end > start and line[end - 1][0].isspace():
            end -= 1
        if start < end:
            stripped.append(line[start:end])
    return stripped


def _is_ascii_alnum(c: str) -> bool:
    return c.isascii() and c.isalnum()


def _join_char_lines(lines: list[list[Char]]) -> list[Char]:
    """
    PDFの折り返しで入った改行をつなぐ。英数字どうしの間だけ空白を入れる（日本語は詰める）。
    入れた空白には、前後の文字の両方に付いている装飾を付ける（太字の語どうしの間の空白は太字のまま）。
    """
    result: list[Char] = []
    for line in lines:
        if not line:
            continue
        if result and _is_ascii_alnum(result[-1][0]) and _is_ascii_alnum(line[0][0]):
            result.append((" ", tuple(mark for mark in result[-1][1] if mark in line[0][1])))
        result += line
    return result


def _plain(chars: list[Char]) -> str:
    return "".join(c for c, _ in chars)


def _join_lines(text: str) -> str:
    """文字列の改行をつなぐ（装飾を使わない見出し・キャプションなど）。"""
    return _plain(_join_char_lines(_lines([(c, ()) for c in text])))


def _rich(chars: list[Char]) -> list[dict[str, Any]]:
    """
    装飾が同じで隣り合う文字を1つの text にする。付いている装飾だけキーを持たせる
    （model.ts の TextMarks・normalizeRich と同じ形）。
    """
    return [
        {"type": "text", "text": _plain(list(group)), **{mark: True for mark in marks}}
        for marks, group in groupby(chars, key=lambda ch: ch[1])
    ]


def _heading_levels(elements: list[Element]) -> dict[int, int]:
    """level が無い見出しに、文字サイズの大きい順で 1, 2, 3 を割り当てる（4番目以降の大きさは3）。"""
    sizes = sorted({round(el.font_size * 2) / 2 for el in elements if el.font_size is not None}, reverse=True)
    rank = {size: min(i + 1, MAX_HEADING_LEVEL) for i, size in enumerate(sizes)}
    levels = {}
    for el in elements:
        if el.level is not None:
            levels[id(el)] = el.level
        elif el.font_size is not None:
            levels[id(el)] = rank[round(el.font_size * 2) / 2]
        else:
            levels[id(el)] = 1
    return levels


def _attach_captions(items: list[tuple[int, Element, str]]) -> tuple[dict[int, str], set[int]]:
    """
    キャプションを同じページの隣の表・図に付ける。
    戻り値は（{表・図の位置: キャプション文字列}, 付け先が見つかったキャプションの位置）。
    表のキャプションは上、図のキャプションは下に置かれることが多いので、
    「表」で始まるものは次の要素、それ以外は前の要素を先に見る。
    """
    captions: dict[int, str] = {}
    used: set[int] = set()
    for i, (page, el, label) in enumerate(items):
        if label != "caption":
            continue
        text = _join_lines(el.text or "")
        order = [i + 1, i - 1] if TABLE_CAPTION_PATTERN.match(text) else [i - 1, i + 1]
        for j in order:
            if 0 <= j < len(items) and items[j][0] == page and items[j][2] in ("table", "figure") and j not in captions:
                captions[j] = text
                used.add(i)
                break
    return captions, used


def _list_items(elements: list[Element]) -> tuple[str, list[dict[str, Any]]]:
    """連続した list 要素を1つの箇条書きにまとめる。行頭記号のある行で項目を区切る。"""
    marker = None
    items: list[list[list[Char]]] = []  # 項目ごとの行（行は装飾付きの文字の並び）
    for el in elements:
        lines = _lines(_chars(el))
        texts = [_plain(line) for line in lines]
        has_marker = any(pattern.match(text) for text in texts for pattern, _ in LIST_MARKER_PATTERNS)
        for k, (line, text) in enumerate(zip(lines, texts)):
            for pattern, name in LIST_MARKER_PATTERNS:
                if match := pattern.match(text):
                    marker = marker or name
                    items.append([line[match.end() :]])
                    break
            else:
                # 記号の無い行は前の項目の折り返し。要素内に記号が1つも無ければ要素ごとに1項目
                if items and (has_marker or k > 0):
                    items[-1].append(line)
                else:
                    items.append([line])
    return marker or "dot", [{"content": _rich(_join_char_lines(item))} for item in items]


def _table_block(el: Element, caption: str | None) -> dict[str, Any]:
    rows = el.cells or [[""]]
    n_cols = max(len(row) for row in rows) or 1
    # 揃えが分からない（または列数と合わない）ときは左寄せ
    aligns = el.aligns if el.aligns and len(el.aligns) == n_cols else ["l"] * n_cols
    block: dict[str, Any] = {
        "id": _new_id(),
        "type": "table",
        "columns": [{"align": align} for align in aligns],
        "rows": [{"cells": row + [""] * (n_cols - len(row))} for row in rows],
    }
    if caption:
        block["caption"] = caption
    return block


def _image_block(el: Element, caption: str | None, text_area: tuple[float, float]) -> dict[str, Any]:
    """画像ファイルはIRに無いので src は空（エディタで選び直す）。幅は本文の幅に対する比率で再現する。"""
    left, right = text_area
    width_ratio = (el.bbox[2] - el.bbox[0]) / (right - left) if right > left else 1.0
    block: dict[str, Any] = {"id": _new_id(), "type": "image", "src": "", "widthRatio": round(min(1.0, max(0.05, width_ratio)), 3)}
    if caption:
        block["caption"] = caption
    return block


def _title_block(items: list[tuple[int, Element, str]]) -> dict[str, Any] | None:
    """
    title / author / date の要素から、タイトルブロック（\\maketitle）を1つ作る。どれも無ければ None。
    タイトルが折り返して複数の要素になったときは段落と同じくつなぎ、著者・日付が複数あれば空白で区切る。
    """
    parts = {label: [_join_lines(el.text or "") for _, el, item_label in items if item_label == label] for label in TITLE_LABELS}
    if not any(parts.values()):
        return None
    return {
        "id": _new_id(),
        "type": "title",
        "text": _join_lines("\n".join(parts["title"])),
        "author": " ".join(parts["author"]),
        "date": " ".join(parts["date"]),
    }


def ir_to_document(ir: IRDocument, title: str | None = None) -> dict[str, Any]:
    items = [(page.page, el, _label_of(el)) for page in ir.pages for el in page.elements]
    texts = [el for _, el, label in items if el.kind == "text" and label != "other"]
    # 本文の左右の端。図の幅を「本文の幅に対する比率」に直すのに使う
    text_area = (min((el.bbox[0] for el in texts), default=0.0), max((el.bbox[2] for el in texts), default=1.0))
    levels = _heading_levels([el for _, el, label in items if label == "heading"])
    captions, used_captions = _attach_captions(items)

    blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(items):
        _, el, label = items[i]
        if label == "list":
            run = []
            while i < len(items) and items[i][2] == "list":
                run.append(items[i][1])
                i += 1
            marker, list_items = _list_items(run)
            blocks.append({"id": _new_id(), "type": "list", "marker": marker, "items": list_items})
            continue

        if label == "code":
            # コードはPDFで1行ずつ別の要素になりやすいので、連続した code 要素を1つにまとめる。
            # 改行や字下げに意味があるので、段落のように行をつながない
            lines = []
            while i < len(items) and items[i][2] == "code":
                lines.append((items[i][1].text or "").rstrip("\n"))
                i += 1
            blocks.append({"id": _new_id(), "type": "code", "language": "plain", "code": "\n".join(lines)})
            continue

        if label == "heading":
            blocks.append({"id": _new_id(), "type": "heading", "level": levels[id(el)], "text": _join_lines(el.text or "")})
        elif label == "math":
            # tex が無いときは仮にPDFの文字をそのまま入れる（数式OCRで tex を埋めるまで）
            blocks.append({"id": _new_id(), "type": "math", "displayMode": True, "tex": el.tex or (el.text or "").strip()})
        elif label == "table":
            blocks.append(_table_block(el, captions.get(i)))
        elif label == "figure":
            blocks.append(_image_block(el, captions.get(i), text_area))
        elif label == "paragraph" or (label == "caption" and i not in used_captions):
            # 付け先の無いキャプションは段落として残す
            chars = _join_char_lines(_lines(_chars(el)))
            if chars:
                blocks.append({"id": _new_id(), "type": "paragraph", "content": _rich(chars)})
        # 表・図に付けたキャプション、"other"（ページ番号・ヘッダーなど）、タイトル・著者・日付（先頭にまとめる）はここでは作らない
        i += 1

    # タイトルブロックは文書に1つだけで、常に先頭（model.ts の TitleBlock）
    title_block = _title_block(items)
    if title_block:
        blocks.insert(0, title_block)
    return {"schemaVersion": EDITOR_SCHEMA_VERSION, "id": _new_id(), "title": title or ir.source, "blocks": blocks}


def main() -> None:
    parser = argparse.ArgumentParser(description="ラベル付きIR → エディタ用JSON")
    parser.add_argument("labeled_ir", help="ラベル付きIRのJSON")
    parser.add_argument("-o", "--out", required=True, help="出力するエディタ用JSONのパス")
    parser.add_argument("--title", help="文書の名前（省略時は入力ファイル名）")
    args = parser.parse_args()

    document = ir_to_document(load_ir(args.labeled_ir, check=True), args.title)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"{args.out}: {len(document['blocks'])}ブロック")


if __name__ == "__main__":
    main()
