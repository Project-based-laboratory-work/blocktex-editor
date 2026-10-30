"""
作った画像に正解ブロックを色分けして描き、正解がずれていないかを目で確かめる。

    python -m image_dataset.preview image_dataset/data/synth --n 10
      → image_dataset/data/synth/preview/doc_000000_p1_v0.png など（ラベルごとに色を変えた枠と、左上にラベル名）
"""

import argparse
from pathlib import Path

import cv2

from common.ir import denormalize_bbox, load_ir

# ラベルごとの枠の色（RGB）
COLORS = {
    "heading": (220, 40, 40),
    "paragraph": (40, 110, 220),
    "list": (30, 160, 80),
    "math": (200, 120, 0),
    "caption": (150, 60, 200),
    "table": (0, 160, 160),
    "figure": (120, 120, 0),
    "other": (130, 130, 130),
    "code": (230, 0, 160),
}


def draw(image_path: Path, truth_path: Path, out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    page = load_ir(str(truth_path)).pages[0]
    for el in page.elements:
        x0, y0, x1, y1 = (int(v) for v in denormalize_bbox(el.bbox, page.width, page.height))
        r, g, b = COLORS.get(el.label or "other", (0, 0, 0))
        cv2.rectangle(image, (x0, y0), (x1, y1), (b, g, r), 2)
        name = el.label + (str(el.level) if el.level else "")
        cv2.putText(image, name, (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (b, g, r), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="build の --out に指定したフォルダ")
    parser.add_argument("--n", type=int, default=10, help="描く枚数（先頭から）")
    args = parser.parse_args()

    out = args.data / "preview"
    out.mkdir(exist_ok=True)
    truths = sorted((args.data / "truth").glob("*.ir.json"))[: args.n]
    for truth in truths:
        stem = truth.name.removesuffix(".ir.json")
        image = next((args.data / "images").glob(f"{stem}.*"))
        draw(image, truth, out / f"{stem}.png")
    print(f"{len(truths)} 枚を {out} に描きました")


if __name__ == "__main__":
    main()
