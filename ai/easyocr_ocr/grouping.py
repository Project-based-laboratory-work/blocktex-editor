"""
行（または行の一部）単位の要素を、段落（ブロック）単位にまとめる。

EasyOCRは1行ずつ（ときには1行を途中で分けて）箱を返すが、チームの評価（ai/evaluation）の正解や、
構造化器（rule_based）が受け取るのは段落単位のブロック。その差を埋める。
IRだけを見て動き、EasyOCRには依存しない（Tesseract など他の抽出器の結果にも使える）。

まとめ方は2段階:
1. 同じ高さに並んだ、横に近い箱を1行にする（1行が途中で割れた箱をつなぐ）
2. 縦に近く、文字の大きさが近く、横の範囲が重なる行を1つのブロックにする（行の区切りは "\\n"）。
   本のページなどは段落の間に空行がないので、次の行が字下げされていたら、または前の行が
   途中で終わっていたら（段落の最後の行や見出し）、そこで新しいブロックにする
縦書きの箱（高さが幅の VERTICAL_RATIO 倍以上）はまとめずにそのまま残す。
距離は横と縦で同じ物差しにするため、ページの px（または pt）に戻してから比べる。
"""

import statistics
from dataclasses import replace

from common.ir import Element, IRDocument, Page, denormalize_bbox, normalize_bbox

# 1. 同じ行とみなす条件：縦の重なりが低いほうの箱の高さのこの割合以上で、横の隙間が行の高さのこの倍以下
SAME_LINE_OVERLAP = 0.5
SAME_LINE_GAP = 1.5
# 2. 同じブロックとみなす条件：行の間の隙間が行の高さのこの倍以下、文字の大きさの比がこれ以下、横の範囲が重なる
BLOCK_LINE_GAP = 0.8
BLOCK_SIZE_RATIO = 1.3
# 字下げ：ブロックの左端より、行の高さのこの倍以上右から始まる行は、新しい段落
INDENT = 0.8
# 前の行が、次の行の右端より行の高さのこの倍以上手前で終わっていたら、前の行で段落（見出し）が終わっている
SHORT_LINE = 2.0
VERTICAL_RATIO = 1.5


def _height(el: Element) -> float:
    return el.bbox[3] - el.bbox[1]


def _union(elements: list[Element]) -> tuple[float, float, float, float]:
    return (
        min(el.bbox[0] for el in elements),
        min(el.bbox[1] for el in elements),
        max(el.bbox[2] for el in elements),
        max(el.bbox[3] for el in elements),
    )


def _join_pieces(left: str, right: str) -> str:
    """同じ行の2つの文字列をつなぐ。英数字・記号どうしの間だけ空白を入れる（日本語の間には入れない）。"""
    if left and right and left[-1].isascii() and not left[-1].isspace() and right[0].isascii() and right[0].isalnum():
        return f"{left} {right}"
    return left + right


def _font_size(elements: list[Element]) -> float | None:
    sizes = [el.font_size for el in elements if el.font_size is not None]
    return round(statistics.median(sizes), 1) if sizes else None


def _merge(elements: list[Element], text: str) -> Element:
    confidences = [el.confidence for el in elements if el.confidence is not None]
    return Element(
        kind="text",
        bbox=_union(elements),
        text=text,
        font_size=_font_size(elements),
        confidence=round(min(confidences), 4) if confidences else None,
    )


def _is_vertical(el: Element) -> bool:
    return _height(el) >= (el.bbox[2] - el.bbox[0]) * VERTICAL_RATIO


def _same_line(line: list[Element], el: Element) -> bool:
    last = line[-1]
    overlap = min(last.bbox[3], el.bbox[3]) - max(last.bbox[1], el.bbox[1])
    gap = el.bbox[0] - last.bbox[2]
    return overlap >= min(_height(last), _height(el)) * SAME_LINE_OVERLAP and gap <= _height(last) * SAME_LINE_GAP


def group_lines(elements: list[Element]) -> list[Element]:
    """1. 横に並んだ箱を行にまとめる。行は上から、行の中は左から並ぶ。"""
    lines: list[list[Element]] = []
    for el in sorted(elements, key=lambda e: (e.bbox[0], e.bbox[1])):
        for line in lines:
            if _same_line(line, el):
                line.append(el)
                break
        else:
            lines.append([el])
    merged = []
    for line in lines:
        text = ""
        for el in line:
            text = _join_pieces(text, el.text or "")
        merged.append(_merge(line, text))
    return sorted(merged, key=lambda e: (e.bbox[1], e.bbox[0]))


def _same_block(block: list[Element], line: Element) -> bool:
    last = block[-1]
    height = max(_height(last), _height(line))
    gap = line.bbox[1] - last.bbox[3]
    a, b = last.font_size or _height(last), line.font_size or _height(line)
    overlaps = min(last.bbox[2], line.bbox[2]) > max(last.bbox[0], line.bbox[0])
    if not (gap <= height * BLOCK_LINE_GAP and max(a, b) / min(a, b) <= BLOCK_SIZE_RATIO and overlaps):
        return False
    indented = line.bbox[0] - min(el.bbox[0] for el in block) >= height * INDENT
    previous_ended = line.bbox[2] - last.bbox[2] >= height * SHORT_LINE
    return not indented and not previous_ended


def group_blocks(lines: list[Element]) -> list[Element]:
    """2. 上から順に見て、近い行を1つのブロックにする。"""
    blocks: list[list[Element]] = []
    for line in lines:
        for block in reversed(blocks):
            if _same_block(block, line):
                block.append(line)
                break
        else:
            blocks.append([line])
    return [_merge(block, "\n".join(el.text or "" for el in block)) for block in blocks]


def group_page(page: Page) -> Page:
    """1ページ分の text 要素を段落にまとめる。縦書きの箱・表・図はそのまま残す。"""
    w, h = page.width, page.height
    # 0〜1 の割合のままだと横と縦で物差しが違う（横長の画像では横の距離が小さく出る）ので、px に戻して比べる
    to_page = lambda el: replace(el, bbox=denormalize_bbox(el.bbox, w, h))  # noqa: E731
    horizontal = [to_page(el) for el in page.elements if el.kind == "text" and not _is_vertical(to_page(el))]
    others = [el for el in page.elements if not (el.kind == "text" and not _is_vertical(to_page(el)))]
    blocks = [replace(el, bbox=normalize_bbox(el.bbox, w, h)) for el in group_blocks(group_lines(horizontal))]
    return Page(page=page.page, width=w, height=h, unit=page.unit, elements=blocks + others)


def group_document(doc: IRDocument) -> IRDocument:
    """IR全体の要素を段落にまとめる。まとめたことは params の "group" に記録する。"""
    return IRDocument(
        source=doc.source,
        extractor=doc.extractor,
        pages=[group_page(page) for page in doc.pages],
        params={**doc.params, "group": "lines_to_blocks"},
        structurer=doc.structurer,
        ir_version=doc.ir_version,
    )
