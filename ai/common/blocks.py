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
from typing import Any

from common.ir import Element, IRDocument, load_ir

EDITOR_SCHEMA_VERSION = 3

MAX_HEADING_LEVEL = 3
TABLE_CAPTION_PATTERN = re.compile(r"(表|Table)\s*\d")

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


def _join_lines(text: str) -> str:
    """PDFの折り返しで入った改行をつなぐ。英数字どうしの間だけ空白を入れる（日本語は詰める）。"""
    result = ""
    for line in (line.strip() for line in text.split("\n")):
        if not line:
            continue
        if result and result[-1].isascii() and result[-1].isalnum() and line[0].isascii() and line[0].isalnum():
            result += " "
        result += line
    return result


def _rich(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


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
    items: list[str] = []
    for el in elements:
        lines = [line.strip() for line in (el.text or "").split("\n") if line.strip()]
        has_marker = any(pattern.match(line) for line in lines for pattern, _ in LIST_MARKER_PATTERNS)
        for k, line in enumerate(lines):
            for pattern, name in LIST_MARKER_PATTERNS:
                if pattern.match(line):
                    marker = marker or name
                    items.append(pattern.sub("", line, count=1))
                    break
            else:
                # 記号の無い行は前の項目の折り返し。要素内に記号が1つも無ければ要素ごとに1項目
                if items and (has_marker or k > 0):
                    items[-1] = _join_lines(items[-1] + "\n" + line)
                else:
                    items.append(line)
    return marker or "dot", [{"content": _rich(item)} for item in items]


def _table_block(el: Element, caption: str | None) -> dict[str, Any]:
    rows = el.cells or [[""]]
    n_cols = max(len(row) for row in rows) or 1
    block: dict[str, Any] = {
        "id": _new_id(),
        "type": "table",
        "columns": [{"align": "l"} for _ in range(n_cols)],
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
            text = _join_lines(el.text or "")
            if text:
                blocks.append({"id": _new_id(), "type": "paragraph", "content": _rich(text)})
        # 表・図に付けたキャプションと "other"（ページ番号・ヘッダーなど）はブロックにしない
        i += 1

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
