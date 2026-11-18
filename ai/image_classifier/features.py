"""
OCRの段落（IRの要素）1つを、モデルに入れる形にする。

  1. 数値の特徴: 位置・大きさ・文字の大きさ・文字列の形（PDF版のブロック分類と同じ考え方）
  2. その段落の部分の画像: 太字・等幅・数式の記号・表の線など、OCRの文字列や大きさには出ない手がかりを、
     CNNに直接見させる。チームのOCR側で「画像からは文字の太さ・フォントが取れない」と課題になっていた点

値は 0〜1 程度にそろえる（桁の大きい特徴ばかりが学習に効かないように）。
"""

import math
import re

import numpy as np
import torch
from PIL import Image

from common.ir import Element, Page

CROP_HEIGHT, CROP_WIDTH = 32, 256

SECTION_NUMBER = re.compile(r"^\d{1,2}(\s*[.,．]\s*\d{1,2}){0,2}\.?\s")  # 「1 」「1.2 」（OCRは「.」を「,」と読むことがある）
CAPTION_START = re.compile(r"^([図表]\s*\d|(Figure|Fig\.|Table)\s*\d)")
LIST_MARKER = re.compile(r"^([・•∙\-–—*]|\(\d+\)|\d+\.\s|[a-z]\)\s)")
JAPANESE = re.compile(r"[぀-ヿ一-鿿]")
MATH_SYMBOL = re.compile(r"[=+\-−×·÷≤≥≠≈∑∏∫√∞∂∇∈⊂∪∩→←⇒^_()\[\]{}|/<>!′αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]")
CODE_SYMBOL = re.compile(r"[;{}()\[\]=<>#\"'`]")


def _ratio(pattern: re.Pattern, text: str) -> float:
    chars = text.replace(" ", "").replace("\n", "")
    return len(pattern.findall(chars)) / max(len(chars), 1)


def font_size_of(el: Element, page: Page) -> float:
    """要素の文字の大きさ（px）。OCRが推定していなければ、箱の高さ ÷ 行数。"""
    if el.font_size is not None:
        return el.font_size
    lines = max(1, len([line for line in (el.text or "").split("\n") if line.strip()]))
    return (el.bbox[3] - el.bbox[1]) * page.height / lines


def body_font_size(page: Page) -> float:
    """本文の文字の大きさ = 文字数で重みを付けた中央値（OCRの大きさは少しずつばらつき、同じ値がほとんど出ないため）"""
    sizes = sorted((font_size_of(el, page), len((el.text or "").strip())) for el in page.elements)
    total = sum(n for _, n in sizes)
    count = 0
    for size, n in sizes:
        count += n
        if count * 2 >= total:
            return size
    return 12.0


def element_features(el: Element, page: Page, body_size: float, left_margin: float) -> list[float]:
    x0, y0, x1, y1 = el.bbox  # 0〜1 に正規化済み
    text = (el.text or "").strip()
    first_line = text.split("\n", 1)[0]
    size = font_size_of(el, page)
    return [
        # 位置と大きさ
        x0, y0, x1, y1, x1 - x0, y1 - y0, (x0 + x1) / 2,
        x0 - left_margin,  # 字下げ・中央寄せの手がかり
        # 文字の大きさ（本文との比と、ページの高さとの比）
        min(size / max(body_size, 1e-6), 4.0) / 4,
        min(size / page.height * 50, 2.0),
        # 文字列の形
        min(text.count("\n") + 1, 10) / 10,
        math.log1p(len(text)) / 6,
        float(bool(SECTION_NUMBER.match(first_line))),
        float(bool(CAPTION_START.match(first_line))),
        float(bool(LIST_MARKER.match(first_line))),
        float(text.endswith(("。", "．", "."))),
        _ratio(re.compile(r"\d"), text),
        _ratio(re.compile(r"[A-Za-z]"), text),
        _ratio(JAPANESE, text),
        _ratio(MATH_SYMBOL, text),
        _ratio(CODE_SYMBOL, text),
        # OCRの自信（読みにくい数式・記号では下がる）
        el.confidence if el.confidence is not None else 0.5,
    ]  # fmt: skip


NUM_FEATURES = 22


def load_ink(image_path: str) -> np.ndarray:
    """画像を「背景0・文字1」の配列にする。ダークモード（背景が暗い）の画像は反転してそろえる。"""
    with Image.open(image_path) as image:
        ink = 1.0 - np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if np.median(ink) > 0.5:  # 背景（いちばん多い色）が暗い
        ink = 1.0 - ink
    return ink


def crop(ink: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """要素の部分を切り出し、縦横比を保って CROP_HEIGHT × CROP_WIDTH に収める（左寄せ・上下中央）。uint8で返す。"""
    h, w = ink.shape
    x0, y0, x1, y1 = bbox
    px0, py0 = max(0, int(x0 * w) - 2), max(0, int(y0 * h) - 2)
    px1, py1 = min(w, int(math.ceil(x1 * w)) + 2), min(h, int(math.ceil(y1 * h)) + 2)
    canvas = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
    if px1 <= px0 or py1 <= py0:
        return canvas
    part = ink[py0:py1, px0:px1]
    scale = min(CROP_HEIGHT / part.shape[0], CROP_WIDTH / part.shape[1])
    new_w, new_h = max(1, round(part.shape[1] * scale)), max(1, round(part.shape[0] * scale))
    resized = np.asarray(Image.fromarray((part * 255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR))
    top = (CROP_HEIGHT - new_h) // 2
    canvas[top : top + new_h, :new_w] = resized
    return canvas


def page_inputs(page: Page, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """ページの全要素 → 特徴 (N, NUM_FEATURES) と画像 (N, CROP_HEIGHT, CROP_WIDTH)（uint8）"""
    elements = page.elements
    body_size = body_font_size(page)
    left_margin = min((el.bbox[0] for el in elements), default=0.0)
    features = [element_features(el, page, body_size, left_margin) for el in elements]
    ink = load_ink(image_path)
    crops = np.stack([crop(ink, el.bbox) for el in elements]) if elements else np.zeros((0, CROP_HEIGHT, CROP_WIDTH), np.uint8)
    return (
        torch.tensor(features, dtype=torch.float32).reshape(len(elements), NUM_FEATURES),
        torch.from_numpy(crops),
    )
