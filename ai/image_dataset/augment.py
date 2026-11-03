"""
ページの画像を「それらしく」崩す。正解ブロックの bbox も同じように動かす。

崩し方（style）は3通り:
  - clean:      何もしない（PDFをそのまま画像にしたもの）
  - screenshot: 画面のスクリーンショット風。ページの一部だけを切り出す、ダークモード、背景色、縮小、ときどきJPEG
  - photo:      紙を撮った写真風。紙の色、明るさのむら、傾き・ゆがみ（周りに机が写る）、ぼかし、ノイズ、JPEG

どれをどのくらい崩したかは params に記録する（正解のIRの params に入る）。
傾け・ゆがみでは、bbox の四隅を同じ変換で動かし、それを囲む軸に平行な四角を新しい bbox にする
（文字の周りに少し余白が増えるが、傾きが数度なので小さい）。
"""

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

Bbox = tuple[float, float, float, float]

STYLES = ["clean", "screenshot", "photo"]

# ダークモードの背景色と文字色（エディタやビューアでよくある色の範囲）
DARK_BACKGROUNDS = [(30, 30, 30), (40, 42, 54), (24, 26, 27), (45, 45, 48), (0, 0, 0)]
DARK_FOREGROUNDS = [(230, 230, 230), (212, 212, 212), (248, 248, 242), (200, 200, 200)]
# 明るい背景（セピアのビューア・薄い灰色・紙の色）
LIGHT_BACKGROUNDS = [(250, 244, 230), (244, 244, 244), (238, 236, 225), (255, 253, 245), (232, 236, 240)]
# 写真で紙の周りに写るもの（机など）
DESK_COLORS = [(120, 90, 60), (60, 60, 60), (200, 200, 195), (150, 120, 90), (30, 30, 35)]


@dataclass
class Augmented:
    image: np.ndarray  # 高さ × 幅 × RGB の uint8
    boxes: list[Bbox]  # 残したブロックの bbox（px）
    keep: list[int]  # 残したブロックの、元の並びでの番号（切り出しで外れたブロックは入らない）
    params: dict = field(default_factory=dict)  # どう崩したか
    jpeg_quality: int | None = None  # JPEGで保存するときの品質。None ならPNGで保存する


# ---- 部品（それぞれ単体でテストできるように、乱数は外から渡す） ----


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def gaps(boxes: np.ndarray, height: float) -> list[float]:
    """どのブロックにもかからない高さ（切り出しの上端・下端に使える位置）。ページの上端・下端と、ブロックの間の真ん中"""
    intervals = sorted((float(y0), float(y1)) for _, y0, _, y1 in boxes)
    cuts = [0.0]
    covered = 0.0
    for y0, y1 in intervals:
        if y0 > covered:
            cuts.append((covered + y0) / 2)
        covered = max(covered, y1)
    cuts.append(float(height))
    return sorted(set(cuts))


def crop(image: np.ndarray, boxes: np.ndarray, rng: random.Random, min_share: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """ページの一部（縦方向の範囲）を切り出す。上端・下端はブロックの間にするので、ブロックを途中で切らない。

    返すのは (画像, 残したブロックの bbox, 残したブロックの番号, params)
    """
    height, width = image.shape[:2]
    cuts = gaps(boxes, height)
    pairs = [(top, bottom) for top in cuts for bottom in cuts if bottom - top >= height * min_share]
    top, bottom = rng.choice(pairs) if pairs else (0.0, float(height))
    inside = np.where((boxes[:, 1] >= top) & (boxes[:, 3] <= bottom))[0]
    if len(inside) == 0:  # 何も残らない切り方はしない
        top, bottom, inside = 0.0, float(height), np.arange(len(boxes))
    # 横は、残すブロックにかからない範囲で左右の余白を削る
    left = rng.uniform(0, max(float(boxes[inside, 0].min()), 0))
    right = rng.uniform(min(float(boxes[inside, 2].max()), width), width)
    t, b, l, r = int(top), int(np.ceil(bottom)), int(left), int(np.ceil(right))
    shifted = boxes[inside] - np.array([l, t, l, t], dtype=float)
    return image[t:b, l:r].copy(), clip_boxes(shifted, r - l, b - t), inside, {"crop": [l, t, r, b]}


def dark_mode(image: np.ndarray, background: tuple[int, int, int], foreground: tuple[int, int, int]) -> np.ndarray:
    """白地に黒い文字を、暗い背景に明るい文字にする（明るさを反転して、背景色と文字色の間に割り当てる）"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ink = (1 - gray)[..., None]  # 文字のところが1
    bg, fg = np.array(background, np.float32), np.array(foreground, np.float32)
    return (bg + ink * (fg - bg)).clip(0, 255).astype(np.uint8)


def tint(image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """白いところを color にする（紙の色・ビューアの背景色）。黒い文字は黒いまま"""
    return (image.astype(np.float32) * (np.array(color, np.float32) / 255)).clip(0, 255).astype(np.uint8)


def scale(image: np.ndarray, boxes: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    return resized, boxes * np.array([size[0] / width, size[1] / height] * 2)


def warp(
    image: np.ndarray, boxes: np.ndarray, corners: np.ndarray, margin: int, border: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """画像の四隅（左上・右上・右下・左下）を corners（元の画像の座標）に動かし、周りに margin px の余白（border の色）を付ける"""
    height, width = image.shape[:2]
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    dst = corners.astype(np.float32) + margin
    matrix = cv2.getPerspectiveTransform(src, dst)
    size = (width + 2 * margin, height + 2 * margin)
    warped = cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_LINEAR, borderValue=border)
    # bbox の四隅を同じ変換で動かし、囲む四角にする
    x0, y0, x1, y1 = boxes.T
    points = np.stack([np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x1, y1], -1), np.stack([x0, y1], -1)], 1)
    moved = cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(-1, 4, 2)
    new_boxes = np.concatenate([moved.min(1), moved.max(1)], 1).astype(float)
    return warped, clip_boxes(new_boxes, *size)


def random_corners(width: int, height: int, rng: random.Random, max_angle: float, max_skew: float) -> tuple[np.ndarray, dict]:
    """少し回して、四隅を少しずつずらした（斜めから撮ったような）四隅の位置"""
    angle = np.deg2rad(rng.uniform(-max_angle, max_angle))
    center = np.array([width / 2, height / 2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    corners = (corners - center) @ rotation.T + center
    corners += np.array([[rng.uniform(-1, 1) * width, rng.uniform(-1, 1) * height] for _ in range(4)]) * max_skew
    return corners, {"angle": round(float(np.rad2deg(angle)), 2), "skew": max_skew}


def lighting(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    """明るさのむら（片側が暗い）。strength は一番暗いところの暗さ（0〜1）"""
    height, width = image.shape[:2]
    theta = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (xs / width - 0.5) * np.cos(theta) + (ys / height - 0.5) * np.sin(theta)  # -0.7〜0.7 くらい
    gain = 1 - strength * (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-6)
    return (image.astype(np.float32) * gain[..., None]).clip(0, 255).astype(np.uint8)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma) if sigma > 0 else image


def noise(image: np.ndarray, rng: random.Random, sigma: float) -> np.ndarray:
    gen = np.random.default_rng(rng.getrandbits(32))
    return (image.astype(np.float32) + gen.normal(0, sigma, image.shape)).clip(0, 255).astype(np.uint8)


# ---- style ごとの組み合わせ ----


def augment(image: np.ndarray, boxes: list[Bbox], style: str, rng: random.Random) -> Augmented:
    """style（STYLES のどれか）に従って画像を崩す"""
    if style not in STYLES:
        raise ValueError(f"不明なstyleです: {style!r}（使えるのは {STYLES}）")
    arr = np.array(boxes, dtype=float).reshape(-1, 4)
    keep = np.arange(len(boxes))
    params: dict = {"style": style}
    quality = None

    if style == "screenshot":
        if rng.random() < 0.5 and len(boxes) > 0:
            image, arr, keep, crop_params = crop(image, arr, rng, min_share=0.3)
            params.update(crop_params)
        r = rng.random()
        if r < 0.3:
            background, foreground = rng.choice(DARK_BACKGROUNDS), rng.choice(DARK_FOREGROUNDS)
            image = dark_mode(image, background, foreground)
            params["dark_mode"] = {"background": background, "foreground": foreground}
        elif r < 0.6:
            color = rng.choice(LIGHT_BACKGROUNDS)
            image = tint(image, color)
            params["background"] = color
        factor = rng.uniform(0.5, 1.0)
        image, arr = scale(image, arr, factor)
        params["scale"] = round(factor, 3)
        if rng.random() < 0.5:
            quality = rng.randint(50, 95)

    elif style == "photo":
        if rng.random() < 0.7:
            color = rng.choice(LIGHT_BACKGROUNDS)
            image = tint(image, color)
            params["paper"] = color
        height, width = image.shape[:2]
        corners, corner_params = random_corners(width, height, rng, max_angle=3.0, max_skew=0.03)
        margin = int(max(width, height) * rng.uniform(0.02, 0.08))
        desk = rng.choice(DESK_COLORS)
        image, arr = warp(image, arr, corners, margin, desk)
        params.update(corner_params, margin=margin, desk=desk)
        strength = rng.uniform(0.0, 0.35)
        image = lighting(image, rng, strength)
        sigma_blur, sigma_noise = rng.uniform(0.0, 1.2), rng.uniform(0.0, 6.0)
        image = noise(blur(image, sigma_blur), rng, sigma_noise)
        factor = rng.uniform(0.6, 1.0)
        image, arr = scale(image, arr, factor)
        params.update(lighting=round(strength, 3), blur=round(sigma_blur, 2), noise=round(sigma_noise, 2), scale=round(factor, 3))
        quality = rng.randint(60, 90)

    if quality is not None:  # JPEGの劣化は、保存するときにこの品質で保存して付ける（2回圧縮しないように）
        params["jpeg_quality"] = quality
    out_boxes = [tuple(round(float(v), 1) for v in box) for box in arr]
    return Augmented(image, out_boxes, [int(i) for i in keep], params, quality)
