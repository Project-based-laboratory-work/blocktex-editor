"""ルール分類器: 抽出器（extract.py）のブロックに、決まりでラベルを付けてラベル付きIR（common/README.md の②）にする。

自作AIと比べるためのベースラインで、ハイブリッド（ルール＋AI）のときは「確実に分かるもの」をここで決める。

決まり（上から順に調べて、最初に当てはまったもの）:
    1. 表・図の領域             → table / figure
    2. 表・図のすぐ上か下の「表1」「図1」で始まる短いもの → caption（表・図1つにつき1つ）
    3. ページの上端・下端の短いもの（ページ番号・ヘッダー） → other
    4. 1ページ目の上の、中央に置かれた短い行の並びのうち、一番大きい文字 → title、日付の形 → date、残り → author
    5. 等幅フォントが多い       → code
    6. 数式のフォントが多い     → math
    7. • や 1. で始まる         → list
    8. 本文より大きい、または番号から始まる1行で太字か本文と違う書体のもの → heading（番号の深さ、無ければ文字の大きさの順でレベル）
    9. それ以外                 → paragraph

使い方（ai/ で実行）:
    python -m hybrid_extractor.rules hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.labeled.json
"""

import argparse
import os
import re

from common.ir import IRDocument, Page, save_ir, validate
from hybrid_extractor.extract import Block, PageBlocks, extract_blocks, to_element

NAME = "hybrid_rules"

CAPTION = re.compile(r"^(図|表|Figure|Fig\.|Table)\s*\d+")
PAGE_NUMBER = re.compile(r"^[-–—\s]*(\d+|[ivxlc]+)[-–—\s]*$", re.IGNORECASE)
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
DATE = re.compile(
    r"\d{4}\s*年|\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|(令和|平成|昭和)\s*(\d+|元)\s*年"
    r"|(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d"
)

HEADING_SIZE_RATIO = 1.15   # 本文の文字サイズのこの倍以上なら見出し
HEADING_MAX_CHARS = 80
CAPTION_MAX_CHARS = 150
CAPTION_MAX_GAP = 4.0       # キャプションと表・図の間の上限（本文の文字サイズに対する比率）
EDGE = 0.07                 # ページの上端・下端のこの割合の中にあるものは、ページ番号・ヘッダーの候補
HEADER_ZONE = 0.15          # ヘッダー（柱）を探す、ページの上からの範囲。jsarticle の柱は上から 10〜13% にある
TITLE_CENTER = 0.05         # タイトル部分の行の、中央からのずれの上限（ページの幅に対する割合）
TITLE_SIZE_RATIO = 1.2      # タイトルは本文のこの倍以上
CODE_RATIO = 0.6
MATH_RATIO = 0.3


def text_of(block: Block) -> str:
    return (block.text or "").strip()


def body_size(pages: list[PageBlocks]) -> float:
    """文書の本文の文字サイズ（文字数で重みをつけた最頻値）"""
    counts: dict[float, int] = {}
    for page in pages:
        for b in page.blocks:
            if b.kind == "text" and b.font_size is not None:
                counts[b.font_size] = counts.get(b.font_size, 0) + len(text_of(b))
    return max(counts, key=lambda s: counts[s]) if counts else 10.0


def find_captions(page: PageBlocks, body: float) -> set[int]:
    """表・図ごとに、すぐ上か下にある「表1」「図1」で始まる短いブロックを1つ選ぶ（添字の集合を返す）"""
    chosen: set[int] = set()
    for target in page.blocks:
        if target.kind not in ("table", "figure"):
            continue
        best, best_gap = None, CAPTION_MAX_GAP * body
        for i, b in enumerate(page.blocks):
            if b.kind != "text" or i in chosen:
                continue
            text = text_of(b)
            if not CAPTION.match(text) or len(text) > CAPTION_MAX_CHARS:
                continue
            gap = max(target.bbox[1] - b.bbox[3], b.bbox[1] - target.bbox[3])  # 上にあっても下にあっても、間の距離
            overlap_x = min(b.bbox[2], target.bbox[2]) > max(b.bbox[0], target.bbox[0])
            if overlap_x and -2 <= gap < best_gap:
                best, best_gap = i, gap
        if best is not None:
            chosen.add(best)
    return chosen


def title_labels(page: PageBlocks, body: float, headers: set[str]) -> dict[int, str]:
    """1ページ目の上の、中央に置かれた短い行の並び（\\maketitle）に title / author / date を付ける"""
    if page.page != 1:
        return {}
    candidates: list[int] = []
    for i, b in enumerate(page.blocks):
        if b.kind != "text":
            break
        if is_edge_block(page, b, headers):
            continue  # ヘッダー
        mid = (b.bbox[0] + b.bbox[2]) / 2
        centered = abs(mid - page.width / 2) <= page.width * TITLE_CENTER
        if not (centered and b.n_rows <= 2 and len(text_of(b)) <= HEADING_MAX_CHARS):
            break
        candidates.append(i)
    if not candidates:
        return {}
    title = max(candidates, key=lambda i: page.blocks[i].font_size or 0)
    if (page.blocks[title].font_size or 0) < body * TITLE_SIZE_RATIO:
        return {}
    labels = {}
    for i in candidates:
        if i == title:
            labels[i] = "title"
        elif DATE.search(text_of(page.blocks[i])):
            labels[i] = "date"
        elif i > title:
            labels[i] = "author"
    return labels


def repeated_headers(pages: list[PageBlocks]) -> set[str]:
    """ページの上の方（HEADER_ZONE）に、2ページ以上で同じ文字で出てくるもの（ヘッダー・柱）"""
    seen: dict[str, set[int]] = {}
    for page in pages:
        for b in page.blocks:
            if b.kind == "text" and b.n_rows == 1 and b.bbox[3] <= page.height * HEADER_ZONE:
                seen.setdefault(text_of(b), set()).add(page.page)
    return {text for text, on in seen.items() if len(on) >= 2}


def is_edge_block(page: PageBlocks, b: Block, headers: set[str]) -> bool:
    """ページ番号・ヘッダー: ページの上端・下端にある、1行の短いもの。
    ヘッダーは、ほかのページにも同じ文字がある（1ページだけの文書では、右寄せの短いもの）"""
    if b.kind != "text" or b.n_rows > 1 or len(text_of(b)) > 60:
        return False
    if b.bbox[1] >= page.height * (1 - EDGE * 1.5):
        return bool(PAGE_NUMBER.match(text_of(b))) or len(text_of(b)) <= 20
    if b.bbox[3] <= page.height * EDGE:
        return True
    if b.bbox[3] <= page.height * HEADER_ZONE:
        return text_of(b) in headers or b.bbox[0] >= page.width * 0.5
    return False


def body_font(pages: list[PageBlocks]) -> str:
    """文書の本文の書体（文字数で重みをつけた最頻値）"""
    counts: dict[str, int] = {}
    for page in pages:
        for b in page.blocks:
            if b.kind == "text" and b.font:
                counts[b.font] = counts.get(b.font, 0) + len(text_of(b))
    return max(counts, key=lambda f: counts[f]) if counts else ""


def classify_block(b: Block, body: float, font: str) -> str:
    text = text_of(b)
    if b.mono_ratio >= CODE_RATIO:
        return "code"
    if b.math_ratio >= MATH_RATIO:
        return "math"
    if b.list_head:
        return "list"
    short = len(text) <= HEADING_MAX_CHARS and b.n_rows <= 2
    big = (b.font_size or 0) >= body * HEADING_SIZE_RATIO
    # 番号から始まる1行で、太字か、本文と違う書体（英語の見出しは太字ではなくサンセリフ体になる）
    numbered = b.n_rows == 1 and bool(SECTION_NUMBER.match(text)) and (bool(b.bold) or b.font != font)
    if short and (big or numbered):
        return "heading"
    return "paragraph"


def heading_levels(items: list[tuple[Block, str]]) -> list[int | None]:
    """見出しのレベル。番号（1.2.3）があればその深さ、無ければ文字の大きい順（1〜3）"""
    sizes = sorted({b.font_size or 0 for b, label in items if label == "heading"}, reverse=True)
    levels: list[int | None] = []
    for b, label in items:
        if label != "heading":
            levels.append(None)
            continue
        m = SECTION_NUMBER.match(text_of(b))
        if m:
            levels.append(min(3, m.group(1).count(".") + 1))
        else:
            levels.append(min(3, sizes.index(b.font_size or 0) + 1))
    return levels


def label_pages(pages: list[PageBlocks]) -> list[list[tuple[Block, str]]]:
    """ページごとに (ブロック, ラベル) の並び（読み順）を返す"""
    body = body_size(pages)
    font = body_font(pages)
    headers = repeated_headers(pages)
    result = []
    for page in pages:
        captions = find_captions(page, body)
        titles = title_labels(page, body, headers)
        items = []
        for i, b in enumerate(page.blocks):
            if b.kind in ("table", "figure"):
                label = b.kind
            elif i in captions:
                label = "caption"
            elif i in titles:
                label = titles[i]
            elif is_edge_block(page, b, headers):
                label = "other"
            else:
                label = classify_block(b, body, font)
            items.append((b, label))
        result.append(items)
    return result


def to_labeled_ir(pages: list[PageBlocks], labeled: list[list[tuple[Block, str]]], source: str, structurer: str) -> IRDocument:
    """(ブロック, ラベル) の並びをラベル付きIRにする。見出しのレベルもここで決める"""
    flat = [item for items in labeled for item in items]
    levels = iter(heading_levels(flat))
    ir_pages = []
    for page, items in zip(pages, labeled):
        elements = [to_element(b, page.width, page.height, label=label, level=next(levels)) for b, label in items]
        ir_pages.append(Page(page=page.page, width=page.width, height=page.height, unit="pt", elements=elements))
    return IRDocument(source=source, extractor="hybrid_extractor", structurer=structurer, pages=ir_pages)


def predict(pdf_path: str) -> IRDocument:
    """PDF → ラベル付きIR（ルールだけ）。評価の登録簿（evaluation/models.json）から呼ぶ"""
    pages = extract_blocks(pdf_path)
    return to_labeled_ir(pages, label_pages(pages), os.path.basename(pdf_path), NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → ラベル付きIR（hybrid_extractor のルール分類器）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("out", help="出力するラベル付きIRのJSON")
    args = parser.parse_args()
    doc = predict(args.pdf)
    problems = validate(doc)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(doc, args.out)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()
