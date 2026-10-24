"""
OCRの前に画像にかける前処理の部品。プリセット（presets.py）から名前で選んで、順番にかける。

Tesseract側でも使えるように、EasyOCRには依存しない（入力も出力もPILの画像）。
2人で共有することが決まったら ai/image_common/ に移す。
"""

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageOps

# 背景の平均の明るさ（0〜255）がこれより暗ければ、ダークモードとみなして反転する
DARK_THRESHOLD = 128


def upscale(image: Image.Image, factor: float) -> Image.Image:
    """拡大する。小さい文字（スクショで10〜15px程度）を読みやすくするため。"""
    width, height = image.size
    return image.resize((round(width * factor), round(height * factor)), Image.LANCZOS)


def grayscale(image: Image.Image) -> Image.Image:
    """白黒（グレースケール）にする。色付きの背景や文字の影響を減らすため。"""
    return image.convert("L").convert("RGB")


def autocontrast(image: Image.Image) -> Image.Image:
    """明るさの幅を0〜255いっぱいに広げて、薄い文字をはっきりさせる。"""
    return ImageOps.autocontrast(image.convert("RGB"))


def invert_if_dark(image: Image.Image) -> Image.Image:
    """背景が暗ければ白黒反転する（ダークモードのスクショ対策）。明るい画像はそのまま。"""
    if np.asarray(image.convert("L")).mean() < DARK_THRESHOLD:
        return ImageOps.invert(image.convert("RGB"))
    return image


def otsu(image: Image.Image) -> Image.Image:
    """大津の方法で二値化する。EasyOCRでは悪化することもあるので、評価で確かめる。"""
    import cv2

    gray = np.asarray(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


# 適応的二値化の設定。BLOCK_SIZE は「周り」とみなす範囲（px、奇数）で、文字より十分大きくする。
# C は周りの明るさから引く値で、小さすぎると背景にノイズの点が出る
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10


def adaptive(image: Image.Image, method: str = "gaussian") -> Image.Image:
    """適応的二値化。場所ごとに周りの明るさから白黒の境目を決める。
    見出しの帯やコードブロックなど、一部だけ背景の色が違う画像で、大津の二値化より文字が残りやすい。
    method は "gaussian"（近くほど重く見る。ノイズに強い）か "mean"（範囲内の単純な平均）。"""
    import cv2

    methods = {"gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "mean": cv2.ADAPTIVE_THRESH_MEAN_C}
    gray = np.asarray(image.convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, methods[method], cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
    return Image.fromarray(binary).convert("RGB")


# 名前 → (処理, 拡大率)。拡大率は座標を元の画像に戻すときに使う
STEPS: dict[str, tuple[Callable[[Image.Image], Image.Image], float]] = {
    "upscale2": (lambda im: upscale(im, 2), 2.0),
    "upscale3": (lambda im: upscale(im, 3), 3.0),
    "grayscale": (grayscale, 1.0),
    "autocontrast": (autocontrast, 1.0),
    "invert_if_dark": (invert_if_dark, 1.0),
    "otsu": (otsu, 1.0),
    "adaptive": (lambda im: adaptive(im, "gaussian"), 1.0),
    "adaptive_mean": (lambda im: adaptive(im, "mean"), 1.0),
}


def apply_preprocess(image: Image.Image, steps: list[str]) -> tuple[Image.Image, float]:
    """steps の順に前処理をかける。(処理後の画像, 元の画像からの拡大率) を返す。"""
    scale = 1.0
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"不明な前処理 {name!r}（使えるのは {list(STEPS)}）")
        func, factor = STEPS[name]
        image = func(image)
        scale *= factor
    return image, scale
