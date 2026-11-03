"""
画像のIRに、見出し・段落・キャプションのラベルを付けるルールベースの構造化器。
入力は段落（ブロック）単位のIR（easyocr_ocr の --group、または tesseract_ocr の出力）。

判定の流れは PDF版（rule_based/rule_based_classifier.py）と同じにして、画像では成り立たない前提だけを変えている。
  - 本文の文字の大きさ: PDF版は「同じ大きさの文字数が最も多い値」。OCRで推定した大きさは 18.3, 18.7, … と
    少しずつばらつき、同じ値がほとんど出ないので、文字数で重みを付けた中央値にする
  - 長さの単位: PDF版は pt の固定値。画像は解像度がばらばらなので、ページの幅に対する割合にする
  - 文字の大きさが無い要素（EasyOCRの paragraph=True など）: 箱の高さ ÷ 行数を文字の大きさとみなす
  - 見出し: 大きさ・太字に加えて「番号で始まる短い1行」も見出しにする（OCRの大きさでは差が出にくく、太字も取れないため）
"""

import re

from common.ir import Page, denormalize_bbox
from rule_based.rule_based_classifier import ClassifiedBlock

Bbox = tuple[float, float, float, float]

# しきい値は PDF版と同じ値から始める（画像向けの調整は評価データで確かめてから）
HEADING_SIZE_RATIO = 1.3  # 本文の文字の大きさの何倍から見出しとみなすか
HEADING_MAX_CHARS = 50
OVERLAP_RATIO_THRESHOLD = 0.5  # テキストの箱の何割が表・図と重なったら、表・図の一部とみなすか
CAPTION_MAX_CHARS = 100
CAPTION_PATTERN = re.compile(r"(図|表|Figure|Fig\.|Table)\s*\d+([.\-]\d+)*")
# 番号で始まる1行（「3 実験」「4.1 CPUの評価」「1. Introduction」）も見出しとみなす。
# OCRで推定した文字の大きさでは見出しと本文の差がほとんど出ない（image_dataset で、見出しの大きさの比の中央値は 1.02）ため。
# 「1,1」のようにOCRが「.」を「,」と読み違えることがあるので、区切りには「,」も許す
NUMBERED_HEADING_PATTERN = re.compile(r"\d{1,2}(\s*[.,．]\s*\d{1,2}){0,2}\.?\s*[^\d\s.,)]")
NUMBERED_HEADING_MAX_CHARS = 30
SENTENCE_END_PATTERN = re.compile(r"[。．.,、，;:：]$")  # 文の終わり方をしていれば、番号で始まっても本文（番号付きの箇条書きなど）
# 段組みの判定で、ページの中央線からどれだけはみ出しても片側の段とみなすか（ページの幅に対する割合）。
# PDF版の 10pt を A4 の幅（595pt）で割った値
COLUMN_GUTTER_RATIO = 10.0 / 595.0


def _area(bbox: Bbox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _overlap_ratio(inner: Bbox, outer: Bbox) -> float:
    """innerの面積のうち、outerと重なっている割合（0〜1）。"""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    overlap = _area((max(ix0, ox0), max(iy0, oy0), min(ix1, ox1), min(iy1, oy1)))
    inner_area = _area(inner)
    return overlap / inner_area if inner_area > 0 else 0.0


def font_size_of(text: str, bbox: Bbox, font_size: float | None) -> float:
    """要素の文字の大きさ。OCRが推定していなければ、箱の高さを行数で割った値を使う。"""
    if font_size is not None:
        return font_size
    lines = max(1, len([line for line in text.split("\n") if line.strip()]))
    return (bbox[3] - bbox[1]) / lines


def body_font_size(sizes: list[tuple[float, int]]) -> float | None:
    """(文字の大きさ, 文字数) の並びから、本文の文字の大きさを決める。文字数で重みを付けた中央値。"""
    sizes = sorted((size, n) for size, n in sizes if n > 0)
    total = sum(n for _, n in sizes)
    if total == 0:
        return None
    count = 0
    for size, n in sizes:
        count += n
        if count * 2 >= total:
            return size
    return sizes[-1][0]


def is_numbered_heading(text: str) -> bool:
    """番号で始まる、文の終わり方をしていない短い1行か。"""
    return (
        "\n" not in text
        and len(text) <= NUMBERED_HEADING_MAX_CHARS
        and NUMBERED_HEADING_PATTERN.match(text) is not None
        and SENTENCE_END_PATTERN.search(text) is None
    )


def _column_of(bbox: Bbox, page_width: float) -> int:
    x0, _, x1, _ = bbox
    center = page_width / 2
    gutter = page_width * COLUMN_GUTTER_RATIO
    if x1 <= center + gutter:
        return 1
    if x0 >= center - gutter:
        return 2
    return 0


def _sort_reading_order(blocks: list[ClassifiedBlock], page_width: float) -> list[ClassifiedBlock]:
    """読み順に並べる。両方の段にまたがる要素で区切った区間ごとに、左の段 → 右の段、それぞれ上から。"""
    columns = [_column_of(block.bbox, page_width) for block in blocks]
    spanner_tops = sorted(block.bbox[1] for block, col in zip(blocks, columns) if col == 0)

    def key(pair):
        block, col = pair
        top = block.bbox[1]
        section = sum(1 for spanner_top in spanner_tops if spanner_top <= top)
        return (section, col, top)

    return [block for block, _ in sorted(zip(blocks, columns), key=key)]


def classify_page(page: Page) -> list[ClassifiedBlock]:
    """IRの1ページを分類し、読み順に並べて返す。bbox はページの単位（px）に戻した値。"""
    page_no = page.page - 1  # PDF版の ClassifiedBlock に合わせて0始まり
    results = []
    absorbed_bboxes = []
    texts = []
    for el in page.elements:
        bbox = denormalize_bbox(el.bbox, page.width, page.height)
        if el.kind in ("table", "figure"):
            results.append(ClassifiedBlock(page_no, el.kind, bbox, None))
            absorbed_bboxes.append(bbox)
        elif el.text and el.text.strip():
            texts.append((el, bbox, font_size_of(el.text, bbox, el.font_size)))

    body_size = body_font_size([(size, len(el.text.strip())) for el, _, size in texts])

    for el, bbox, size in texts:
        if any(_overlap_ratio(bbox, absorbed) >= OVERLAP_RATIO_THRESHOLD for absorbed in absorbed_bboxes):
            continue

        text = el.text.strip()
        is_large = body_size is not None and size >= body_size * HEADING_SIZE_RATIO
        is_short = len(text) <= HEADING_MAX_CHARS

        if CAPTION_PATTERN.match(text) and len(text) <= CAPTION_MAX_CHARS:
            kind = "caption"
        elif ((is_large or bool(el.bold)) and is_short) or is_numbered_heading(text):
            kind = "heading"
        else:
            kind = "paragraph"

        results.append(ClassifiedBlock(page_no, kind, bbox, el.text))
    return _sort_reading_order(results, page.width)
