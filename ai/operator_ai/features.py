"""
オペレータAIの特徴量: 画像1枚から、設定を選ぶ手がかりになる数を測る。

**画像だけから測る**（正解IRの params は使わない）。本物のスクショには params が無いので、
ルール版（進め方4）も学習版（進め方5）も、振り分けにはここで測った値だけを使う。

使い方（ai/ で）:
    python -m operator_ai.features image_dataset/data/val
      → operator_ai/data/val/features.csv   1行 = 画像1枚（列は FEATURES）

    from operator_ai.features import measure
    measure("screenshot.png")   # → {"width": 1046, "page_dark": False, "char_height": 14.0, ...}
"""

import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

# 列の意味は operator_ai/README.md の「特徴量」
FEATURES = [
    "width", "height", "is_jpeg",
    "brightness", "page_dark", "page_bg",
    "band_ratio",
    "ink_ratio", "char_height", "char_height_rel", "n_chars",
    "center_gap",
]


def load_gray(image_path: str) -> np.ndarray:
    data = np.fromfile(image_path, dtype=np.uint8)  # 日本語のパスでも読めるように
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"画像を読めません: {image_path}")
    return gray


def page_box(gray: np.ndarray, page_bg: float) -> tuple[int, int, int, int]:
    """ビューアの帯・背景（ページと違う色で一様な行・列）を除いた、ページの範囲 (x0, y0, x1, y1)"""

    def keep(axis: int) -> np.ndarray:
        flat = gray.std(axis=axis) < 4
        other = np.abs(gray.mean(axis=axis) - page_bg) > 25
        return ~(flat & other)

    cols, rows = np.flatnonzero(keep(0)), np.flatnonzero(keep(1))
    h, w = gray.shape
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, w, h
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def measure(image_path: str) -> dict:
    gray = load_gray(image_path)
    h, w = gray.shape

    # 真ん中あたりの色を、ページの地の色とみなす（ダークモードなら暗い）
    center = gray[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
    page_bg = float(np.median(center))
    page_dark = page_bg < 128

    x0, y0, x1, y1 = page_box(gray, page_bg)
    page = gray[y0:y1, x0:x1]
    band_ratio = 1 - (x1 - x0) * (y1 - y0) / (w * h)

    # 文字＝白、地＝黒にそろえて二値化（大津）
    text = page if page_dark else 255 - page
    _, ink = cv2.threshold(text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_ratio = float((ink > 0).mean()) if ink.size else 0.0

    # 文字の高さ：連結成分のうち、文字くらいの大きさのものの高さの真ん中
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    heights = [
        int(stats[i, cv2.CC_STAT_HEIGHT]) for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= 4 and 3 <= stats[i, cv2.CC_STAT_HEIGHT] <= h / 10
        and stats[i, cv2.CC_STAT_WIDTH] <= w / 10
    ]
    char_height = float(statistics.median(heights)) if heights else 0.0

    # 段組み：ページの真ん中（40%〜60%）に、文字の高さくらいの幅のすき間が縦に通っているか。
    # 文字のある行のうち「すき間の左にも右にも文字があり、すき間には無い」行の割合の最大。
    # 1段組は、長い行はすき間を横切り、短い行（箇条書き・コード）は右に文字が無いので小さい。2段組は大きい
    center_gap = 0.0
    k = max(1, int(char_height * 0.6))
    joined = cv2.dilate(ink, np.ones((k, k), np.uint8))  # 文字の間を埋めて、1行を横に1本の帯にする
    text_rows = joined[(joined > 0).any(axis=1)] > 0
    pw = joined.shape[1]
    win = max(3, int(char_height))
    lo, hi = int(pw * 0.4), int(pw * 0.6) - win
    if len(text_rows) and hi > lo:
        cum = np.concatenate([np.zeros((len(text_rows), 1), int), text_rows.cumsum(axis=1)], axis=1)
        for x in range(lo, hi):
            empty = cum[:, x + win] - cum[:, x] == 0
            left = cum[:, x] > 0
            right = cum[:, -1] - cum[:, x + win] > 0
            center_gap = max(center_gap, float((empty & left & right).mean()))

    return {
        "width": w,
        "height": h,
        "is_jpeg": Path(image_path).suffix.lower() in (".jpg", ".jpeg"),
        "brightness": round(float(gray.mean()) / 255, 4),
        "page_dark": page_dark,
        "page_bg": round(page_bg / 255, 4),
        "band_ratio": round(band_ratio, 4),
        "ink_ratio": round(ink_ratio, 4),
        "char_height": char_height,
        "char_height_rel": round(char_height / h, 5),
        "n_chars": len(heights),
        "center_gap": round(center_gap, 4),
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="image_dataset の画像の特徴量を測って features.csv に書く")
    parser.add_argument("dataset", type=Path, help="image_dataset.build の出力先（images/ があるディレクトリ）")
    parser.add_argument("--limit", type=int, help="先頭からこの枚数だけ使う（試すとき）")
    parser.add_argument("--out", type=Path, help="出力先（既定は operator_ai/data/<データセット名>）")
    args = parser.parse_args(argv)

    images = sorted(p for p in (args.dataset / "images").iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    images = images[: args.limit]
    if not images:
        raise SystemExit(f"エラー: {args.dataset}/images/ に画像がありません。")
    out = args.out or Path("operator_ai/data") / args.dataset.name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "features.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["case", *FEATURES])
        writer.writeheader()
        for i, image in enumerate(images, 1):
            writer.writerow({"case": image.stem, **measure(str(image))})
            if i % 100 == 0:
                print(f"[{i}/{len(images)}]", flush=True)
    print(f"{csv_path} に {len(images)} 行を書きました")
    return csv_path


if __name__ == "__main__":
    main()
