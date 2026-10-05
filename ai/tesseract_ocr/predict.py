"""Tesseractで画像/PDFの文字を読み取り、文字の高さからサイズクラス（大/中/小）を判定する。

使い方（ai/ ディレクトリで）:
    python -m tesseract_ocr.predict rule_based/sample_data/sample.pdf
"""

import argparse
import os
from dataclasses import dataclass

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pytesseract import Output

LARGE_RATIO = 1.5  # ページ平均の何倍以上を「大（見出しクラス）」とするか
SMALL_RATIO = 0.7  # ページ平均の何倍未満を「小（補足・ルビなど）」とするか

LABELS = {"large": "🔴 大 (見出しクラス)", "small": "🔵 小 (補足・ルビなど)", "medium": "🟢 中 (本文クラス)"}


@dataclass
class Word:
    text: str
    height: int
    size_class: str  # "large" / "medium" / "small"


def load_images(path: str) -> list[Image.Image]:
    if os.path.splitext(path)[1].lower() == ".pdf":
        return convert_from_path(path)
    return [Image.open(path)]


def classify_size(height: float, avg_height: float) -> str:
    if height > avg_height * LARGE_RATIO:
        return "large"
    if height < avg_height * SMALL_RATIO:
        return "small"
    return "medium"


def predict(image: Image.Image, lang: str = "jpn+eng") -> list[Word]:
    """1ページ分の画像から、単語ごとのテキストとサイズクラスを返す。"""
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    found = [(t.strip(), h) for t, h in zip(data["text"], data["height"]) if t.strip()]
    if not found:
        return []
    avg_height = sum(h for _, h in found) / len(found)
    return [Word(t, h, classify_size(h, avg_height)) for t, h in found]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="画像またはPDFのパス")
    parser.add_argument("--lang", default="jpn+eng")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"エラー: {args.file} が見つかりません。")

    images = load_images(args.file)
    for i, img in enumerate(images):
        if len(images) > 1:
            print(f"\n--- {i + 1}ページ目 ---")
        words = predict(img, args.lang)
        if not words:
            print("文字が検出されませんでした。")
            continue
        for w in words:
            print(f"{LABELS[w.size_class]}\t高さ: {w.height}px\t文字: {w.text}")


if __name__ == "__main__":
    main()
