"""
表の画像 → チーム共通のIRの表の要素（cells・aligns・row_rules・col_rules）。

    python -m image_table.predict <表の画像> [--checkpoint image_table/checkpoints/split/best.pt] [--no-ocr]

  1. 行・列の範囲: 表AI（model.py）で当てる
  2. セルの中身: 行の範囲 × 列の範囲の四角を切り抜き、EasyOCR（チームと同じ日本語＋英語）で読む
  3. 罫線: 行（列）の境目の空白の中に、表の幅（高さ）の半分以上続く線があるか
  4. 列の揃え: 列の中で、各セルの文字が左端・右端・真ん中のどれに寄っているか
     （列の幅いっぱいのセルは手がかりにならないので除く）
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common.ir import Element
from image_table.evaluate import load_model, predict

DEFAULT_CHECKPOINT = Path("image_table/checkpoints/split/best.pt")
LANGS = ("ja", "en")  # easyocr_ocr の DEFAULT_LANGS と同じ
LINE_SHARE = 0.5  # 表の幅（高さ）のこの割合以上続く線を罫線とみなす
CELL_PAD = 2  # セルを切り抜くときの余白（画素）
SAME = 1  # 左右の余りがこの画素数以下なら、列の幅いっぱいのセルとみなす


def ink_of(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if np.median(gray) < 128:
        gray = 255 - gray
    return gray < 160


def boundaries(spans: list[tuple[float, float]], length: int) -> list[tuple[int, int]]:
    """範囲の並び → 境目の空白（上端の外・範囲の間・下端の外）。範囲の数 + 1 個"""
    edges = [0.0] + [v for span in spans for v in span] + [float(length)]
    return [(int(edges[2 * k]), max(int(edges[2 * k]) + 1, int(np.ceil(edges[2 * k + 1])))) for k in range(len(spans) + 1)]


def rules(ink: np.ndarray, rows: list, cols: list) -> tuple[list[bool], list[bool]]:
    """(行の境目ごとの横線の有無, 列の境目ごとの縦線の有無)"""
    x0, x1 = int(cols[0][0]), int(np.ceil(cols[-1][1]))
    y0, y1 = int(rows[0][0]), int(np.ceil(rows[-1][1]))
    row_rules = [bool(ink[a:b, x0:x1].mean(1).max(initial=0) > LINE_SHARE) for a, b in boundaries(rows, ink.shape[0])]
    col_rules = [bool(ink[y0:y1, a:b].mean(0).max(initial=0) > LINE_SHARE) for a, b in boundaries(cols, ink.shape[1])]
    return row_rules, col_rules


def aligns(ink: np.ndarray, rows: list, cols: list) -> list[str]:
    """列ごとの揃え。基準は、その列で実際に文字がある左端・右端（予測した列の範囲は文字より少し広いことがあるので使わない）"""
    out = []
    for c0, c1 in cols:
        extents = []
        for r0, r1 in rows:
            xs = np.flatnonzero(ink[int(r0) : int(np.ceil(r1)), int(c0) : int(np.ceil(c1))].any(0))
            if len(xs):
                extents.append((xs[0], xs[-1]))
        if not extents:
            out.append("l")
            continue
        lo, hi = min(a for a, _ in extents), max(b for _, b in extents)
        # 列の幅いっぱいのセル（左右どちらにも余りが無い）は手がかりにならないので除く
        gaps = [(a - lo, hi - b) for a, b in extents if a - lo > SAME or hi - b > SAME]
        if not gaps:
            out.append("l")  # 全部のセルが同じ幅で、見た目では揃えが分からない
            continue
        lgap, rgap = float(np.median([g[0] for g in gaps])), float(np.median([g[1] for g in gaps]))
        out.append(min((lgap, "l"), (rgap, "r"), (abs(lgap - rgap), "c"))[1])
    return out


def read_cells(image: Image.Image, rows: list, cols: list, reader) -> list[list[str]]:
    gray = np.asarray(image.convert("L"))
    cells = []
    for r0, r1 in rows:
        row = []
        for c0, c1 in cols:
            crop = gray[max(0, int(r0) - CELL_PAD) : int(np.ceil(r1)) + CELL_PAD, max(0, int(c0) - CELL_PAD) : int(np.ceil(c1)) + CELL_PAD]
            # 上から下、同じ高さなら左から右の順につなげる
            found = sorted(reader.readtext(crop), key=lambda res: (round(res[0][0][1] / max(1, crop.shape[0] / 4)), res[0][0][0]))
            row.append(" ".join(text for _, text, _ in found))
        cells.append(row)
    return cells


def table_element(image: Image.Image, model, bbox=(0.0, 0.0, 1.0, 1.0), reader=None) -> Element:
    """表の画像 → IRの表の要素。reader（EasyOCRのReader）が None なら、セルの中身は空文字にする"""
    rows, cols = predict(model, image)
    return element_from_spans(image, rows, cols, bbox, reader)


def element_from_spans(image: Image.Image, rows: list, cols: list, bbox=(0.0, 0.0, 1.0, 1.0), reader=None) -> Element:
    """行・列の範囲が決まったあとの処理（セルの中身・罫線・揃え）。区切り方だけを入れ替えて比べるために分けてある"""
    if not rows or not cols:
        return Element(kind="table", bbox=bbox, label="table")
    ink = ink_of(image)
    row_rules, col_rules = rules(ink, rows, cols)
    cells = read_cells(image, rows, cols, reader) if reader is not None else [["" for _ in cols] for _ in rows]
    return Element(kind="table", bbox=bbox, cells=cells, aligns=aligns(ink, rows, cols),
                   row_rules=row_rules, col_rules=col_rules, label="table")  # fmt: skip


def get_reader():
    import easyocr
    import torch

    return easyocr.Reader(list(LANGS), gpu=torch.cuda.is_available(), verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="表の画像を、IRの表の要素（JSON）にする")
    parser.add_argument("image")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--no-ocr", action="store_true", help="セルの中身を読まない（区切り・罫線・揃えだけ）")
    args = parser.parse_args()
    with Image.open(args.image) as image:
        image.load()
    el = table_element(image, load_model(args.checkpoint), reader=None if args.no_ocr else get_reader())
    fields = ("cells", "aligns", "row_rules", "col_rules")
    print(json.dumps({k: getattr(el, k) for k in fields}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
