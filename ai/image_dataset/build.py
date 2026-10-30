"""
画像の構造化器の学習データ（画像 + 正解のラベル付きIR）を大量に作る。

    python -m image_dataset.build --out image_dataset/data/train --docs 200 --seed 0
    python -m image_dataset.build --out image_dataset/data/val --docs 50 --seed 1

流れ（1文書ごと）:
  1. render.render_document: ランダムな文書のTeX → PDF → ページごとの正解ブロック（pt）
  2. ページごとに、解像度をランダムに選んで画像にする（正解の bbox も px に直す）
  3. augment.augment: style（clean / screenshot / photo）を選んで画像を崩す（bbox も一緒に動かす）
  4. 画像と、正解のラベル付きIR（common/ir.py。unit="px"、label_source="auto"）を保存する

出力:
    <out>/images/doc_000000_p1_v0.png   （JPEGの劣化を付けたものは .jpg）
    <out>/truth/doc_000000_p1_v0.ir.json 正解。画像1枚 = 1ページのIR。source は画像のファイル名
    <out>/tex/doc_000000.tex             元のTeX（確認・デバッグ用）
"""

import argparse
import random
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from image_dataset.augment import STYLES, augment
from image_dataset.render import TruthBlock, render_document, render_page

NAME = "image_dataset"
DEFAULT_STYLE_WEIGHTS = {"clean": 1, "screenshot": 2, "photo": 2}


def _pick_style(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def to_ir(image_name: str, width: int, height: int, blocks: list[TruthBlock], boxes: list, params: dict) -> IRDocument:
    """正解ブロック（bbox は px に直したもの）を、画像1枚 = 1ページのラベル付きIRにする"""
    elements = [
        Element(
            kind=block.kind,
            bbox=normalize_bbox(box, width, height),
            text=block.text,
            label=block.label,
            level=block.level,
            tex=block.tex,
            label_source="auto",
        )
        for block, box in zip(blocks, boxes)
    ]
    return IRDocument(source=image_name, extractor=NAME, pages=[Page(1, width, height, "px", elements)], params=params)


def build_one(args: tuple[int, int, Path, int, dict[str, float], tuple[float, float]]) -> Counter | None:
    index, seed, out, variants, style_weights, dpi_range = args
    rng = random.Random(seed * 1_000_003 + index)
    doc = render_document(rng)
    if doc is None:
        return None

    name = f"doc_{index:06d}"
    (out / "tex" / f"{name}.tex").write_text(doc.tex, encoding="utf-8")
    counts: Counter = Counter()
    for page in doc.pages:
        if not page.blocks:
            continue
        for variant in range(variants):
            dpi = rng.uniform(*dpi_range)
            image = render_page(doc.pdf, page.page, dpi)
            to_px = image.shape[1] / page.width
            boxes = [tuple(v * to_px for v in block.bbox) for block in page.blocks]
            style = _pick_style(rng, style_weights)
            result = augment(image, boxes, style, rng)
            blocks = [page.blocks[i] for i in result.keep]

            stem = f"{name}_p{page.page + 1}_v{variant}"
            suffix = ".jpg" if result.jpeg_quality is not None else ".png"
            options = [cv2.IMWRITE_JPEG_QUALITY, result.jpeg_quality] if result.jpeg_quality is not None else []
            cv2.imwrite(str(out / "images" / f"{stem}{suffix}"), cv2.cvtColor(result.image, cv2.COLOR_RGB2BGR), options)

            height, width = result.image.shape[:2]
            params = {"doc": name, "page": page.page + 1, "variant": variant, "seed": seed, "dpi": round(dpi, 1), **result.params}
            ir = to_ir(f"{stem}{suffix}", width, height, blocks, result.boxes, params)
            problems = validate(ir)
            if problems:
                raise ValueError(f"{stem}: 正解のIRに問題があります:\n" + "\n".join(problems))
            save_ir(ir, str(out / "truth" / f"{stem}.ir.json"))

            counts["images"] += 1
            counts[f"style:{style}"] += 1
            counts.update(f"label:{block.label}" for block in blocks)
    return counts


def parse_styles(text: str) -> dict[str, float]:
    """"clean=1,screenshot=2,photo=2" → {"clean": 1.0, ...}。重みを省くと1"""
    weights = {}
    for part in text.split(","):
        style, _, weight = part.partition("=")
        if style not in STYLES:
            raise argparse.ArgumentTypeError(f"不明なstyleです: {style!r}（使えるのは {STYLES}）")
        weights[style] = float(weight) if weight else 1.0
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("image_dataset/data/synth"))
    parser.add_argument("--docs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="学習用と検証用で変える（同じ文書が両方に入らないように）")
    parser.add_argument("--variants", type=int, default=1, help="1ページから作る画像の数（崩し方を変える）")
    parser.add_argument(
        "--styles",
        type=parse_styles,
        default=DEFAULT_STYLE_WEIGHTS,
        help="崩し方と重み。例: clean=1,screenshot=2,photo=2 / photo だけなら photo",
    )
    parser.add_argument("--dpi", type=float, nargs=2, default=(90, 200), metavar=("MIN", "MAX"), help="画像にするときの解像度の範囲")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for sub in ["images", "truth", "tex"]:
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed, args.out, args.variants, args.styles, tuple(args.dpi)) for i in range(args.docs)]
    with Pool(args.workers) as pool:
        results = pool.map(build_one, jobs)

    ok = [r for r in results if r is not None]
    total = sum(ok, Counter())
    print(f"{len(ok)}/{args.docs} 文書から {total['images']} 枚の画像を作成（コンパイル失敗 {args.docs - len(ok)}）")
    print("style ごとの枚数:", {k[6:]: v for k, v in sorted(total.items()) if k.startswith("style:")})
    print("ラベルごとのブロック数:", dict(sorted(((k[6:], v) for k, v in total.items() if k.startswith("label:")), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
