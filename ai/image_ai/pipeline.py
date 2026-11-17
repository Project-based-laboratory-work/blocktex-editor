"""
画像（スクリーンショット）→ ラベル付きIR を、自作AIだけでつないで作る（画像読み込みのAI版の通し）。

    python -m image_ai.pipeline <画像> -o out.labeled.json
    python -m evaluation.run --model image_ai --dataset evaluation/data/synth_all   # チームの評価で採点（登録名 image_ai）

  1. ブロックの四角とラベル: 座標AI（image_layout）
  2. 文字: EasyOCR（チームと同じ日本語＋英語）でページ全体を1回読み、行をその中心が入るブロックに振り分ける
  3. ブロックごとの中身
       - table:  表AI（image_table）で行・列を区切り、セルの中身・罫線・揃えを付ける
       - math:   数式AI（math_ocr）でブロックを切り抜いて LaTeX に（tex）。複数行の数式は、行の間の空白で1行ずつに切って
                 読み、`\\` でつなぐ。画像読み込みの数式は align*（ai/CONSTRAINTS.md）なので、各行の最初の関係記号の前に & を置く
       - code:   行の左端のずれ ÷ 1文字の幅 → 行頭の空白（字下げ）
       - figure: 中身なし
       - それ以外: 行を上から順に "\\n" でつなぐ
  4. 読み順: 左右どちらの段にも複数のブロックがあれば2段組とみなし、両方の段にまたがるブロックで区切った区間ごとに
     左の段 → 右の段（image_rule_based と同じ考え方）。それ以外は上から順

どのブロックにも入らなかったOCRの行（座標AIが見落とした文字）は、縦に近い行どうしをまとめて paragraph にする。
ページの上端・下端の帯（ヘッダー・ページ番号の置き場所）にある、本文より大きくない1行だけのブロックは、
座標AIのラベルにかかわらず other にする（座標AIは数字だけのページ番号を math と間違えやすい）。
帯の幅は、学習側の image_dataset と評価用データの両方で、見出し・タイトルにかからない位置にした。
重みは各モデルの checkpoints/ の既定の場所から読む（引数で変えられる）。
"""

import argparse
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir

LAYOUT_CHECKPOINT = "image_layout/checkpoints/unet/best.pt"
TABLE_CHECKPOINT = "image_table/checkpoints/split/best.pt"
MATH_CHECKPOINT = "math_ocr/checkpoints/v4_im2latex/best.pt"
LANGS = ("ja", "en")  # easyocr_ocr の DEFAULT_LANGS と同じ
CROP_PAD = 4  # 表・数式を切り抜くときの余白（画素）
COLUMN_GUTTER = 0.02  # 段の境目（ページの真ん中）からこの割合までは、はみ出しても片方の段とみなす
TWO_COLUMN_MIN = 2  # 左右それぞれの段にこの数以上のブロックがあれば2段組とみなす
TOP_BAND, BOTTOM_BAND = 0.125, 0.90  # 中心がこれより上・下にある、本文より大きくない1行のブロックは other
MARGIN_LINE_HEIGHT = 1.3  # 帯の中でも、本文の行の高さのこの倍より高い（大きな文字の）ブロックは other にしない
KEEP_LABELS = {"table", "figure"}  # 帯の中でも other にしないラベル
MERGE_GAP = 0.8  # ブロックに入らなかった行どうしは、行の高さのこの倍までの縦のすき間ならつなぐ
MATH_LINE_GAP = 0.25  # 数式の行の間とみなす空白の高さ（文字の行の高さに対する割合）。分数の中の小さな空白では切らない
MATH_MIN_GAP = 6  # 文字の行の高さが分からないときの、行の間とみなす空白の高さ（画素）
RELATIONS = {"=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim", r"\simeq", r"\propto"}

Box = tuple[float, float, float, float]  # 画素


@dataclass
class Line:
    """OCRの1行"""

    box: Box
    text: str


@dataclass
class Block:
    label: str
    box: Box
    lines: list[Line]


@lru_cache(maxsize=None)
def _layout_model(path: str):
    from image_layout.evaluate import load_model

    return load_model(Path(path))


@lru_cache(maxsize=None)
def _table_model(path: str):
    from image_table.evaluate import load_model

    return load_model(Path(path))


@lru_cache(maxsize=None)
def _math_model(path: str):
    from math_ocr.predict import load_model

    return load_model(Path(path))


@lru_cache(maxsize=None)
def _reader():
    import easyocr

    return easyocr.Reader(list(LANGS), gpu=torch.cuda.is_available(), verbose=False)


def detect_blocks(image_path: str, layout_checkpoint: str) -> list[tuple[str, Box]]:
    """座標AI → [(ラベル, 画素の四角)]"""
    from image_layout.dataset import STRIDE, load_canvas
    from image_layout.postprocess import logits_to_classes, mask_to_blocks

    canvas, scale = load_canvas(image_path)
    with torch.no_grad():
        logits = _layout_model(layout_checkpoint)(torch.from_numpy(canvas)[None, None])[0].numpy()
    f = STRIDE / scale
    return [(label, (x0 * f, y0 * f, x1 * f, y1 * f)) for label, (x0, y0, x1, y1) in mask_to_blocks(logits_to_classes(logits))]


def read_lines(gray: np.ndarray) -> list[Line]:
    lines = []
    for poly, text, _ in _reader().readtext(gray):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        if text.strip():
            lines.append(Line((float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))), text))
    return lines


def assign_lines(boxes: list[tuple[str, Box]], lines: list[Line]) -> tuple[list[Block], list[Line]]:
    """OCRの行を、その中心が入るブロックに振り分ける。複数に入るときは一番小さいブロック。
    (ブロック, どのブロックにも入らなかった行) を返す"""
    blocks = [Block(label, box, []) for label, box in boxes]
    orphans = []
    for line in lines:
        cx, cy = (line.box[0] + line.box[2]) / 2, (line.box[1] + line.box[3]) / 2
        inside = [b for b in blocks if b.box[0] <= cx <= b.box[2] and b.box[1] <= cy <= b.box[3]]
        if inside:
            min(inside, key=lambda b: (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])).lines.append(line)
        else:
            orphans.append(line)
    return blocks, orphans


def orphan_blocks(orphans: list[Line], height: float) -> list[Block]:
    """ブロックに入らなかった行 → 上端・下端の帯の行は1行ずつ、それ以外は近い行をまとめて paragraph のブロックにする
    （帯の中のものは、あとで margin_others が other にする）"""

    def union(lines: list[Line]) -> Box:
        return (min(l.box[0] for l in lines), min(l.box[1] for l in lines), max(l.box[2] for l in lines), max(l.box[3] for l in lines))

    out, groups = [], []
    for row in rows_of(orphans):
        box = union(row)
        cy = (box[1] + box[3]) / 2
        if cy < TOP_BAND * height or cy > BOTTOM_BAND * height:
            out.append(Block("paragraph", box, row))
            continue
        if groups:
            last = union(groups[-1])
            line_height = box[3] - box[1]
            if box[1] - last[3] <= MERGE_GAP * line_height and box[0] < last[2] and last[0] < box[2]:
                groups[-1] += row
                continue
        groups.append(list(row))
    return out + [Block("paragraph", union(g), g) for g in groups]


def rows_of(lines: list[Line]) -> list[list[Line]]:
    """行を上から順に、同じ高さに並ぶもの（縦の範囲が半分以上重なる）ごとにまとめ、それぞれ左から並べる"""
    rows: list[list[Line]] = []
    for line in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        if rows:
            last = rows[-1][-1].box
            overlap = min(last[3], line.box[3]) - max(last[1], line.box[1])
            if overlap > 0.5 * min(last[3] - last[1], line.box[3] - line.box[1]):
                rows[-1].append(line)
                continue
        rows.append([line])
    return [sorted(row, key=lambda l: l.box[0]) for row in rows]


def margin_others(blocks: list[Block], height: float) -> None:
    """上端・下端の帯にある、本文より大きくない1行だけのブロックを other にする（ヘッダー・ページ番号）"""
    rows = [row for b in blocks for row in rows_of(b.lines)]
    heights = [max(l.box[3] for l in row) - min(l.box[1] for l in row) for row in rows]
    body = float(np.median(heights)) if heights else None
    for b in blocks:
        cy = (b.box[1] + b.box[3]) / 2
        if b.label in KEEP_LABELS or not (cy < TOP_BAND * height or cy > BOTTOM_BAND * height):
            continue
        if len(rows_of(b.lines)) <= 1 and (body is None or b.box[3] - b.box[1] <= MARGIN_LINE_HEIGHT * body):
            b.label = "other"


def block_text(lines: list[Line]) -> str:
    return "\n".join(" ".join(l.text for l in row) for row in rows_of(lines))


def code_text(lines: list[Line]) -> str:
    """コード: 行の左端が一番左の行からどれだけずれているかを、1文字の幅で割って行頭の空白にする"""
    rows = rows_of(lines)
    if not rows:
        return ""
    widths = [(l.box[2] - l.box[0]) / len(l.text) for row in rows for l in row if len(l.text) >= 3]
    char = float(np.median(widths)) if widths else 1.0
    left = min(row[0].box[0] for row in rows)
    return "\n".join(" " * round((row[0].box[0] - left) / char) + " ".join(l.text for l in row) for row in rows)


def reading_order(blocks: list[Block], width: float) -> list[Block]:
    center, gutter = width / 2, width * COLUMN_GUTTER

    def column(b: Block) -> int:
        if b.box[2] <= center + gutter:
            return 1
        if b.box[0] >= center - gutter:
            return 2
        return 0

    cols = [column(b) for b in blocks]
    if min(cols.count(1), cols.count(2)) < TWO_COLUMN_MIN:  # 1段組
        return sorted(blocks, key=lambda b: (b.box[1], b.box[0]))
    spanner_tops = sorted(b.box[1] for b, c in zip(blocks, cols) if c == 0)

    def key(pair: tuple[Block, int]) -> tuple:
        b, c = pair
        return (sum(1 for t in spanner_tops if t <= b.box[1]), c, b.box[1])

    return [b for b, _ in sorted(zip(blocks, cols), key=key)]


def math_lines(image: Image.Image, line_height: float | None) -> list[Image.Image]:
    """数式のブロックの画像 → 1行ずつの画像。行の間の空白（文字の行の高さの MATH_LINE_GAP 倍以上）で切る"""
    gray = np.asarray(image, dtype=np.float32)
    ink = (gray < 128) if np.median(gray) > 127 else (gray > 127)
    rows = ink.any(1)
    gap = max(MATH_MIN_GAP, MATH_LINE_GAP * line_height) if line_height else MATH_MIN_GAP
    bands, start, blank = [], None, 0
    for y, on in enumerate(rows):
        if on:
            if start is None:
                start = y
            elif blank >= gap:  # 前の行が終わって、十分な空白のあとに次の行が始まった
                bands.append((start, y - blank))
                start = y
            blank = 0
        elif start is not None:
            blank += 1
    if start is not None:
        bands.append((start, len(rows) - blank))
    if len(bands) <= 1:
        return [image]
    return [image.crop((0, max(0, a - 2), image.width, min(image.height, b + 2))) for a, b in bands]


def align_mark(tex: str) -> str:
    """最初の関係記号（{} の外にあるもの）の前に & を置く（align* の書き方）"""
    from math_ocr.tokenizer import detokenize, tokenize

    tokens, depth = tokenize(tex), 0
    for k, t in enumerate(tokens):
        depth += {"{": 1, "}": -1}.get(t, 0)
        if depth == 0 and t in RELATIONS:
            return detokenize(tokens[:k] + ["&"] + tokens[k:])
    return tex


def crop(image: Image.Image, box: Box) -> Image.Image:
    x0, y0, x1, y1 = box
    return image.crop((max(0, int(x0) - CROP_PAD), max(0, int(y0) - CROP_PAD),
                       min(image.width, int(np.ceil(x1)) + CROP_PAD), min(image.height, int(np.ceil(y1)) + CROP_PAD)))  # fmt: skip


def to_element(block: Block, image: Image.Image, table_checkpoint: str, math_checkpoint: str, use_ocr: bool) -> Element:
    bbox = normalize_bbox(block.box, image.width, image.height)
    if block.label == "figure":
        return Element(kind="figure", bbox=bbox, label="figure")
    if block.label == "table":
        from image_table.predict import table_element

        return table_element(crop(image, block.box), _table_model(table_checkpoint), bbox, _reader() if use_ocr else None)
    if block.label == "math":
        from math_ocr.dataset import preprocess

        model, vocab = _math_model(math_checkpoint)
        heights = [l.box[3] - l.box[1] for l in block.lines]
        parts = math_lines(crop(image, block.box), float(np.median(heights)) if heights else None)
        with torch.no_grad():
            ids = model.generate(torch.stack([preprocess(part) for part in parts]))
        tex = " \\\\\n".join(align_mark(vocab.decode(row.tolist()[1:])) for row in ids)
        return Element(kind="text", bbox=bbox, text=block_text(block.lines) or tex, label="math", tex=tex)
    text = code_text(block.lines) if block.label == "code" else block_text(block.lines)
    return Element(kind="text", bbox=bbox, text=text, label=block.label)


def predict(image_path: str, layout_checkpoint: str = LAYOUT_CHECKPOINT, table_checkpoint: str = TABLE_CHECKPOINT,
            math_checkpoint: str = MATH_CHECKPOINT, ocr: bool = True) -> IRDocument:  # fmt: skip
    """画像1枚 → ラベル付きIR（1ページ）。評価の登録簿（evaluation/models.json）の entry から呼ぶ"""
    with Image.open(image_path) as opened:
        image = opened.convert("L")
    blocks, orphans = assign_lines(detect_blocks(image_path, layout_checkpoint), read_lines(np.asarray(image)) if ocr else [])
    blocks += orphan_blocks(orphans, image.height)
    margin_others(blocks, image.height)
    elements = [to_element(b, image, table_checkpoint, math_checkpoint, ocr) for b in reading_order(blocks, image.width)]
    page = Page(page=1, width=image.width, height=image.height, unit="px", elements=elements)
    return IRDocument(source=os.path.basename(image_path), extractor="image_ai", structurer="image_ai", pages=[page])


def main() -> None:
    parser = argparse.ArgumentParser(description="画像を、自作AIだけでラベル付きIRにする")
    parser.add_argument("image")
    parser.add_argument("-o", "--out", default=None, help="出力先（省略時は <画像名>.labeled.json）")
    parser.add_argument("--no-ocr", action="store_true", help="文字を読まない（ブロックの位置とラベルだけ）")
    args = parser.parse_args()
    doc = predict(args.image, ocr=not args.no_ocr)
    out = args.out or os.path.splitext(args.image)[0] + ".labeled.json"
    save_ir(doc, out)
    print(f"{len(doc.pages[0].elements)} 個のブロック → {out}")


if __name__ == "__main__":
    main()
