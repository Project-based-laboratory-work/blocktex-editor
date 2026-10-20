"""
PDF → ラベル付きIR（common/ir.py）→ エディタ用JSON を、ブロック分類モデルと数式OCRで作る。
rule_based/ir_adapter.py の「分類」の部分を、学習したモデルに置き換えたもの。

    python -m block_classifier.ir_adapter report.pdf -o out/
      → out/report.labeled.json  ラベル付きIR
        out/report.blocks.json   エディタ用JSON

処理の流れ（ページごと）:
  1. pdf_extract でテキストブロック・表・図を取り出す（ルールベースと同じ抽出）
  2. ブロック分類モデルで、各テキストブロックにラベルを付ける
  3. 細切れのブロックをまとめる
       - 別行立ての数式は `f(x) =` `∑` `i=1` のように細切れになるので、近くの math ブロックを1つにまとめる
       - 表のセルも行ごとに別ブロックになるので、近くの table ブロックを1つの表にまとめる
  4. まとめた数式の範囲をPDFから画像として切り抜き、数式OCR（math_ocr）で LaTeX にして tex に入れる
  5. 読み順に並べる。見出しのレベルは番号（「1.2 」→ 2）から決める
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import torch
from PIL import Image

from block_classifier.features import page_features
from block_classifier.predict import load
from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from math_ocr.dataset import preprocess
from math_ocr.predict import load_model as load_math_model
from rule_based.pdf_extract import extract
from rule_based.rule_based_classifier import ClassifiedBlock, _overlap_ratio, _sort_reading_order

NAME = "block_classifier"
# 見出しの番号。pdf_extract は1行の中の文字を左から順に並べ直さないので、番号が後ろに来ることがある
SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\s+|\s+(\d+(?:\.\d+)*)$")
MERGE_GAP = 6.0  # 表のセルは、これ以下の隙間（pt）で並んでいれば1つの表にまとめる
MATH_LINE_GAP = 2.0  # 数式の断片は、これ以下の隙間なら同じ行としてまとめる
SMALL_FONT_RATIO = 0.85  # これより小さい文字の断片（添字・∑の上下の範囲）は、隙間が MERGE_GAP 以下ならまとめる
SAME_ROW_GAP = 20.0  # 同じ高さに並ぶ数式の断片は、横の隙間がこれ以下ならまとめる
MATH_DPI = 200


@dataclass
class Item:
    """ページ内の要素1つ（まとめる前後で使う作業用の形。座標は pt）"""

    label: str
    bbox: tuple[float, float, float, float]
    text: str | None = None
    font_size: float | None = None
    bold: bool | None = None
    cells: list[list[str]] | None = None
    tex: str | None = None


def _union(a: tuple, b: tuple) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near(a: tuple, b: tuple, gap: float) -> bool:
    """2つの bbox の隙間が縦横とも gap 以下か（重なっていれば隙間は0とみなす）"""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return dx <= gap and dy <= gap


def _same_math(a: Item, b: Item) -> bool:
    """2つの数式の断片が、同じ1行の数式の一部か。

    別行立ての複数行の数式（align* の各行）は、行と行の隙間が2〜3ptしかない。一方、∑ の上下の範囲や
    添字も本体から3〜4pt離れることがある。そこで、隙間がごく小さいか、片方が小さい文字のときだけまとめる。
    """
    if _near(a.bbox, b.bbox, MATH_LINE_GAP):
        return True
    small, large = sorted([a.font_size or 0, b.font_size or 0])
    if large > 0 and small < large * SMALL_FONT_RATIO and _near(a.bbox, b.bbox, MERGE_GAP):
        return True
    # 同じ高さの帯に並ぶ断片（「χ² =」と ∑ と分数など）。複数行の数式の各行は、高さが重ならない
    vertical_overlap = min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1])
    horizontal_gap = max(a.bbox[0], b.bbox[0]) - min(a.bbox[2], b.bbox[2])
    return vertical_overlap > 0 and horizontal_gap <= SAME_ROW_GAP


def merge_fragments(items: list[Item], label: str) -> list[Item]:
    """同じラベルで近くにあるブロックを1つにまとめる（数式の細切れ・表のセル）"""
    if label == "math":
        belongs = _same_math
    else:
        def belongs(a: Item, b: Item) -> bool:
            return _near(a.bbox, b.bbox, MERGE_GAP)

    groups: list[list[Item]] = []
    for item in items:
        if item.label != label:
            continue
        near = [g for g in groups if any(belongs(item, other) for other in g)]
        merged = [item] + [other for g in near for other in g]
        groups = [g for g in groups if g not in near] + [merged]

    result = [item for item in items if item.label != label]
    for group in groups:
        group.sort(key=lambda it: (round(it.bbox[1]), it.bbox[0]))
        bbox = group[0].bbox
        for it in group[1:]:
            bbox = _union(bbox, it.bbox)
        text = "\n".join(it.text or "" for it in group)
        cells = [(it.text or "").split() for it in group] if label == "table" else None
        result.append(Item(label, bbox, text, max((it.font_size or 0) for it in group) or None, cells=cells))
    return result


def join_lines(items: list[Item]) -> list[Item]:
    """PDFで1行ずつ別の要素になった段落・箇条書きを、読み順に沿ってつなぐ。

    段落: 同じ段で上下に続く行をつなぐ。次の行が字下げされている、または前の行が途中で終わっている
          （段落の最後の行）ときは、新しい段落にする。
    箇条書き: 続く行を1つの要素にまとめる。項目の区切りは common/blocks.py が行頭の記号で判断する。
    """
    result: list[Item] = []
    for item in items:
        prev = result[-1] if result else None
        if prev is not None and item.label == prev.label and item.label in ("paragraph", "list"):
            size = item.font_size or 10.0
            same_column = abs(item.bbox[0] - prev.bbox[0]) < 3 * size
            gap = item.bbox[1] - prev.bbox[3]
            indented = item.bbox[0] > prev.bbox[0] + 0.5 * size  # 段落の先頭行は字下げされる
            prev_ends_early = prev.bbox[2] < item.bbox[2] - 2 * size
            if same_column and -0.5 * size < gap < 0.8 * size and (
                item.label == "list" or not (indented or prev_ends_early)
            ):
                prev.bbox = _union(prev.bbox, item.bbox)
                prev.text = f"{prev.text}\n{item.text}"
                continue
        result.append(Item(item.label, item.bbox, item.text, item.font_size, item.bold, item.cells, item.tex))
    return result


def split_heading(text: str) -> tuple[str, int | None]:
    """見出しの文字から番号を取り除き、番号の深さ（「1.2」→ 2）をレベルにする。

    番号はエディタがTeXにするとき \\section などで自動で振られるので、文字に残すと二重になる。
    """
    match = SECTION_NUMBER.search(text.strip())
    if not match:
        return text.strip(), None
    number = match.group(1) or match.group(2)
    return SECTION_NUMBER.sub("", text.strip(), count=1).strip(), min(number.count(".") + 1, 3)


@torch.no_grad()
def _read_math(page: pymupdf.Page, bbox: tuple, math_model, math_vocab) -> str:
    """PDFの数式の範囲を画像として切り抜き、数式OCRで LaTeX にする"""
    clip = pymupdf.Rect(bbox) + (-2, -2, 2, 2)  # 少し余白を付けて切り抜く
    pixmap = page.get_pixmap(clip=clip, dpi=MATH_DPI, colorspace=pymupdf.csGRAY)
    image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    ids = math_model.generate(preprocess(image).unsqueeze(0))[0]
    return math_vocab.decode(ids.tolist()[1:])


def classify_ir(pdf_path: str, checkpoint: Path, math_checkpoint: Path | None = None) -> IRDocument:
    model, labels = load(checkpoint)
    math_model, math_vocab = load_math_model(math_checkpoint) if math_checkpoint else (None, None)

    with pymupdf.open(pdf_path) as doc:
        pages = [Page(page=i + 1, width=p.rect.width, height=p.rect.height, unit="pt") for i, p in enumerate(doc)]

        for page_blocks in extract(pdf_path):
            page, pdf_page = pages[page_blocks.page], doc[page_blocks.page]
            items: list[Item] = []

            # 抽出で表・図として見つかった領域（中の文字はその表・図のものとして扱う）
            regions = [Item("table", t.bbox) for t in page_blocks.table_blocks]
            regions += [Item("figure", f.bbox) for f in page_blocks.figure_blocks]
            items += regions

            if page_blocks.text_blocks:
                features = page_features(page_blocks, page.height).unsqueeze(0)
                padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool)
                predicted = model(features, padding_mask)[0].argmax(-1).tolist()
                for block, label_id in zip(page_blocks.text_blocks, predicted):
                    if any(_overlap_ratio(block.bbox, r.bbox) >= 0.5 for r in regions):
                        continue
                    items.append(Item(labels[label_id], block.bbox, block.text, block.font_size, block.bold))

            items = merge_fragments(merge_fragments(items, "math"), "table")
            if math_model is not None:
                for item in items:
                    if item.label == "math":
                        item.tex = _read_math(pdf_page, item.bbox, math_model, math_vocab)

            # 読み順（2段組を考慮）はルールベースと同じ並べ方を使う
            order = _sort_reading_order(
                [ClassifiedBlock(page_blocks.page, item.label, item.bbox, str(i)) for i, item in enumerate(items)],
                page.width,
            )
            for item in join_lines([items[int(cb.text)] for cb in order]):
                is_text = item.label not in ("table", "figure")
                text, level = split_heading(item.text or "") if item.label == "heading" else (item.text, None)
                page.elements.append(Element(
                    kind="text" if is_text else item.label,
                    bbox=normalize_bbox(item.bbox, page.width, page.height),
                    text=text if is_text else None,
                    font_size=item.font_size,
                    bold=item.bold,
                    cells=item.cells,
                    label=item.label,
                    level=level,
                    tex=item.tex,
                ))  # fmt: skip

    return IRDocument(source=os.path.basename(pdf_path), extractor="rule_based", structurer=NAME, pages=pages)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → ラベル付きIR・エディタ用JSON（ブロック分類モデル＋数式OCR）")
    parser.add_argument("pdf")
    parser.add_argument("-o", "--out-dir", default=".")
    parser.add_argument("--checkpoint", type=Path, default=Path("block_classifier/checkpoints/transformer/best.pt"))
    parser.add_argument(
        "--math-checkpoint", type=Path, default=Path("math_ocr/checkpoints/math_ocr_v2/best.pt"),
        help="数式OCRのモデル。存在しなければ数式はPDFの文字のまま",
    )  # fmt: skip
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, os.path.splitext(os.path.basename(args.pdf))[0])
    math_checkpoint = args.math_checkpoint if args.math_checkpoint.exists() else None

    labeled = classify_ir(args.pdf, args.checkpoint, math_checkpoint)
    problems = validate(labeled)
    if problems:
        raise SystemExit("ラベル付きIRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(labeled, f"{stem}.labeled.json")

    document = ir_to_document(labeled)
    with open(f"{stem}.blocks.json", "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    print(f"{stem}.labeled.json\n{stem}.blocks.json（{len(document['blocks'])}ブロック）")


if __name__ == "__main__":
    main()
