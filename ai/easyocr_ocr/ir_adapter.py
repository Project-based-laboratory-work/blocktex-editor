"""
EasyOCRで画像の文字を読み取り、共通のIR形式（common/ir.py）で出力するアダプタ（パイプライン IMG-A）。
要素はEasyOCRが返した行（または単語）単位のまま。ブロックへのまとめと読み順は行わない。

使い方（ai/ で）:
    python -m easyocr_ocr.ir_adapter ../ocr/data/easy1.jpg -o out/
      → out/easy1.base.ir.json  抽出結果（IR）
    python -m easyocr_ocr.ir_adapter ../ocr/data/easy1.jpg --preset up2 -o out/
      → out/easy1.up2.ir.json   プリセット（presets.py）を選んで実行
"""

import argparse
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from easyocr_ocr.preprocess import apply_preprocess
from easyocr_ocr.presets import PRESETS

NAME = "easyocr"
DEFAULT_LANGS = ("ja", "en")
DEFAULT_PRESET = "base"


@lru_cache(maxsize=None)
def get_reader(langs: tuple[str, ...]):
    """EasyOCRのReaderを作る。モデルの読み込みに数秒〜十数秒かかるので、言語の組ごとに1回だけ作る。"""
    import easyocr

    return easyocr.Reader(list(langs), gpu=False)


def to_ir(
    raw_results, source: str, width: int, height: int, params: dict, scale: float = 1.0, paragraph: bool = False
) -> IRDocument:
    """reader.readtext() の結果（4点の多角形・テキスト・信頼度の並び）をIRにする。

    width / height は元の画像の大きさ。前処理で拡大した場合、raw_results の座標は拡大後の画像のものなので、
    scale（拡大率）で割って元の画像の座標に戻す。
    paragraph=True の結果は箱が段落全体なので、箱の大きさから文字サイズを推定できない（font_size は書かない）。
    """
    elements = []
    for item in raw_results:
        # paragraph=True のときは信頼度が返らず、(box, text) の2つだけになる
        box, text = item[0], item[1]
        confidence = item[2] if len(item) > 2 else None

        # 4つの角の座標から左上と右下の点を取り出し、元の画像の座標に戻す
        xs = [float(x) / scale for x, _ in box]
        ys = [float(y) / scale for _, y in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if x1 <= x0 or y1 <= y0 or not text.strip():
            continue

        elements.append(
            Element(
                kind="text",
                bbox=normalize_bbox((x0, y0, x1, y1), width, height),
                text=text,
                # 横書きは高さ、縦書きは幅が文字の大きさに近い
                font_size=None if paragraph else round(min(x1 - x0, y1 - y0), 1),
                confidence=round(float(confidence), 4) if confidence is not None else None,
            )
        )

    page = Page(page=1, width=width, height=height, unit="px", elements=elements)
    return IRDocument(source=source, extractor=NAME, pages=[page], params=params)


# 解析の流れ：画像を開く → 前処理 → EasyOCRを実行 → IRに変換
def extract_ir(image_path: str, preset: str = DEFAULT_PRESET, langs: tuple[str, ...] = DEFAULT_LANGS) -> IRDocument:
    """画像1枚を、プリセットの設定でEasyOCRに読ませ、IRを返す。"""
    if preset not in PRESETS:
        raise ValueError(f"不明なプリセット {preset!r}（使えるのは {list(PRESETS)}）")
    settings = PRESETS[preset]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size

    processed, scale = apply_preprocess(rgb, settings["preprocess"])

    # 前処理した画像を渡すため、パスではなく配列（RGB）で渡す
    raw_results = get_reader(tuple(langs)).readtext(np.array(processed), **settings["readtext"])

    params = {"preset": preset, **settings, "lang": list(langs)}
    paragraph = bool(settings["readtext"].get("paragraph", False))
    return to_ir(raw_results, os.path.basename(image_path), width, height, params, scale, paragraph)


def main() -> None:
    parser = argparse.ArgumentParser(description="画像 → IR（EasyOCR）")
    parser.add_argument("image", help="入力画像")
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS), help="設定の組（presets.py）")
    parser.add_argument("--lang", nargs="+", default=list(DEFAULT_LANGS), help="EasyOCRの言語（例: --lang ja en）")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"エラー: {args.image} が見つかりません。")

    doc = extract_ir(args.image, args.preset, tuple(args.lang))
    problems = validate(doc)
    if problems:
        raise SystemExit("IR の形式に問題があります:\n" + "\n".join(problems))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{Path(args.image).stem}.{args.preset}.ir.json")
    save_ir(doc, out_path)
    print(f"{out_path}（要素 {len(doc.pages[0].elements)} 個）")


if __name__ == "__main__":
    main()
