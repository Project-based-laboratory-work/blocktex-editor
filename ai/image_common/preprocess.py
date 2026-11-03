"""
OCRの前に画像にかける前処理の部品。各OCRのプリセット（easyocr_ocr/presets.py, tesseract_ocr/presets.py）から名前で選んで、順番にかける。

EasyOCR・Tesseract の両方で使う。同じ前処理で比べられるように、ここ1か所にまとめる（入力も出力もPILの画像）。
"""

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageOps

# 背景の平均の明るさ（0〜255）がこれより暗ければ、ダークモードとみなして反転する
DARK_THRESHOLD = 128

# 小さい文字の認識度を上げるため拡大する
def upscale(image: Image.Image, factor: float) -> Image.Image:
    width, height = image.size
    return image.resize((round(width * factor), round(height * factor)), Image.LANCZOS)

# グレースケール化
def grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")

# 明るさの幅を0〜255いっぱいに広げることで、薄い文字の認識を上げる
def autocontrast(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(image.convert("RGB"))

# 背景が暗い時に白黒反転、ダークモード対策
def invert_if_dark(image: Image.Image) -> Image.Image:
    if np.asarray(image.convert("L")).mean() < DARK_THRESHOLD:
        return ImageOps.invert(image.convert("RGB"))
    return image

# 大津の2値化
def otsu(image: Image.Image) -> Image.Image:
    import cv2

    gray = np.asarray(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary).convert("RGB")


# 適応的二値化の設定、BLOCK_SIZE は「周り」とみなす範囲（px、奇数）で、文字より十分大きくする。
# C は周りの明るさから引く値で、小さすぎると背景にノイズの点が出る
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10

# 適応的2値化(method:gaussianとmean)
# 「周りより暗い画素を黒」にするので、白背景に黒文字が前提。暗い背景だと文字も背景も白になって消えるため、先に反転する
def adaptive(image: Image.Image, method: str = "gaussian", block_size: int = ADAPTIVE_BLOCK_SIZE) -> Image.Image:
    import cv2

    methods = {"gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C, "mean": cv2.ADAPTIVE_THRESH_MEAN_C}
    gray = np.asarray(invert_if_dark(image).convert("L"))
    binary = cv2.adaptiveThreshold(gray, 255, methods[method], cv2.THRESH_BINARY, block_size, ADAPTIVE_C)
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
    # 2倍に拡大した後用。周りの範囲も2倍にして、文字に対する範囲の割合を元の画像と同じにする（奇数なので61）
    "adaptive61": (lambda im: adaptive(im, "gaussian", 61), 1.0),
}

# 順に前処理を行う(処理後の画像、元の画像からの拡大率)を返す
def apply_preprocess(image: Image.Image, steps: list[str]) -> tuple[Image.Image, float]:
    scale = 1.0
    for name in steps:
        if name not in STEPS:
            raise ValueError(f"不明な前処理 {name!r}（使えるのは {list(STEPS)}）")
        func, factor = STEPS[name]
        image = func(image)
        scale *= factor
    return image, scale
