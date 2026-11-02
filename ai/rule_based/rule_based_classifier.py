"""
4-1-2: ルールベースでブロックを タイトル/著者/日付/見出し/段落/キャプション/表/図 に分類するベースライン実装。

ルール:
  1. 表として検出された領域と重なるテキストは "table" として扱う（テキスト自体は重複させない）
  2. 図として検出された領域と重なるテキストは "figure" として扱う
  3. 1ページ目の上の、中央に置かれた短い行の並び（\\maketitle）のうち、一番大きい文字を "title"、
     日付の形のものを "date"、それ以外を "author"（_title_blocks を参照）
  4. 残ったテキストのうち、「図1」「Table 2」などで始まるものを "caption"
  5. 本文フォントサイズより明らかに大きい／太字で、短いものを "heading"
  6. それ以外を "paragraph"

本文フォントサイズは、ページ内で最も多く使われているフォントサイズ（最頻値）を採用する。
分類後のブロックは、2段組みを考慮した読み順（段をまたぐブロックで区切り、左の段→右の段）に並べる。
Transformerによる分類（4-3）の精度比較用ベースラインとして使う。
"""

import sys
import re
from dataclasses import dataclass

from rule_based.pdf_extract import PageBlocks, TextBlock, extract

Bbox = tuple[float, float, float, float]

HEADING_SIZE_RATIO = 1.3  # 本文フォントサイズの何倍から見出しとみなすか
HEADING_MAX_CHARS = 50
OVERLAP_RATIO_THRESHOLD = 0.5  # テキストbboxの何割が表/図と重なったら吸収するか
CAPTION_MAX_CHARS = 100
CAPTION_PATTERN = re.compile(r"(図|表|Figure|Fig\.|Table)\s*\d+([.\-]\d+)*")
COLUMN_GUTTER = 10.0
# タイトル部分の行とみなす条件。中央からのずれ（ページの幅に対する割合）と、幅の上限（同じ割合）
TITLE_CENTER_TOLERANCE = 0.05
TITLE_MAX_WIDTH_RATIO = 0.7
# タイトル部分の行どうしの間の上限（タイトルの文字サイズに対する比率）。離れた行（中央揃えの見出しなど）を含めないため
TITLE_MAX_GAP_RATIO = 2.0
DATE_PATTERN = re.compile(
    r"\d{4}\s*年|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|(令和|平成|昭和)\s*(\d+|元)\s*年"
    r"|(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d"
)


@dataclass
class ClassifiedBlock:
    page: int
    kind: str  # "title" | "author" | "date" | "heading" | "paragraph" | "table" | "figure" | "caption"
    bbox: Bbox
    text: str | None


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


def _body_font_size(text_blocks: list[TextBlock]) -> float:
    if not text_blocks:
        return 10.0
    counts: dict[float, int] = {}
    for block in text_blocks:
        counts[block.font_size] = counts.get(block.font_size, 0) + len(block.text)
    return max(counts, key=lambda size: counts[size])


def _column_of(bbox: Bbox, page_width: float) -> int:
    x0, _, x1, _ = bbox
    center = page_width / 2
    if x1 <= center + COLUMN_GUTTER:
        return 1
    if x0 >= center - COLUMN_GUTTER:
        return 2
    return 0


def _sort_reading_order(blocks: list[ClassifiedBlock], page_width: float) -> list[ClassifiedBlock]:
    columns = [_column_of(block.bbox, page_width) for block in blocks]
    spanner_tops = sorted(block.bbox[1] for block, col in zip(blocks, columns) if col == 0)

    def key(pair):
        block, col = pair
        top = block.bbox[1]
        section = sum(1 for spanner_top in spanner_tops if spanner_top <= top)
        return (section, col, top)

    return [block for block, _ in sorted(zip(blocks, columns), key=key)]


def _title_blocks(page: PageBlocks, text_blocks: list[TextBlock], body_size: float) -> dict[int, str]:
    """
    1ページ目のタイトル部分（\\maketitle のタイトル・著者・日付）を探し、{id(ブロック): ラベル} で返す。
    上から順に「中央に置かれた・幅の狭い・短い1行」が近い間隔で続く間を候補にし、
    その中に本文より明らかに大きい文字があればタイトル部分とみなす。無ければ空。
    """
    if page.page != 0:
        return {}
    candidates: list[TextBlock] = []
    for block in sorted(text_blocks, key=lambda b: b.bbox[1]):
        x0, _, x1, _ = block.bbox
        centered = abs((x0 + x1) / 2 - page.width / 2) <= page.width * TITLE_CENTER_TOLERANCE
        narrow = x1 - x0 <= page.width * TITLE_MAX_WIDTH_RATIO
        short = "\n" not in block.text and len(block.text) <= HEADING_MAX_CHARS
        if not (centered and narrow and short):
            if candidates:
                break
            continue  # タイトルより上のヘッダーなどは飛ばす
        if candidates:
            gap_limit = max(b.font_size for b in candidates) * TITLE_MAX_GAP_RATIO
            if block.bbox[1] - candidates[-1].bbox[3] > gap_limit:
                break
        candidates.append(block)

    title_size = max((b.font_size for b in candidates), default=0.0)
    if title_size < body_size * HEADING_SIZE_RATIO:
        return {}
    labels = {}
    # タイトルより上の候補（ページのヘッダーなど）はタイトル部分に入れない
    first = next(k for k, b in enumerate(candidates) if b.font_size == title_size)
    for block in candidates[first:]:
        if block.font_size == title_size:
            labels[id(block)] = "title"
        elif DATE_PATTERN.search(block.text):
            labels[id(block)] = "date"
        else:
            labels[id(block)] = "author"
    return labels


def classify_page(page: PageBlocks) -> list[ClassifiedBlock]:
    absorbed_bboxes = [t.bbox for t in page.table_blocks] + [f.bbox for f in page.figure_blocks]
    body_size = _body_font_size(page.text_blocks)

    results = [ClassifiedBlock(page.page, "table", t.bbox, None) for t in page.table_blocks]
    results += [ClassifiedBlock(page.page, "figure", f.bbox, None) for f in page.figure_blocks]

    text_blocks = [
        block
        for block in page.text_blocks
        if not any(_overlap_ratio(block.bbox, absorbed) >= OVERLAP_RATIO_THRESHOLD for absorbed in absorbed_bboxes)
    ]
    title_labels = _title_blocks(page, text_blocks, body_size)

    for block in text_blocks:
        if id(block) in title_labels:
            results.append(ClassifiedBlock(page.page, title_labels[id(block)], block.bbox, block.text))
            continue

        is_large = block.font_size >= body_size * HEADING_SIZE_RATIO
        is_short = len(block.text) <= HEADING_MAX_CHARS

        if CAPTION_PATTERN.match(block.text) and len(block.text) <= CAPTION_MAX_CHARS:
            kind = "caption"
        elif (is_large or block.bold) and is_short:
            kind = "heading"
        else:
            kind = "paragraph"
        
        results.append(ClassifiedBlock(page.page, kind, block.bbox, block.text))
    return _sort_reading_order(results, page.width)


def classify(pdf_path: str) -> list[ClassifiedBlock]:
    blocks = []
    for page in extract(pdf_path):
        blocks += classify_page(page)
    return blocks


def main() -> None:
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "rule_based/sample_data/sample.pdf"
    for block in classify(pdf_path):
        preview = f" text={block.text[:30]!r}" if block.text else ""
        print(f"[{block.kind}] page={block.page} bbox={block.bbox}{preview}")


if __name__ == "__main__":
    main()
