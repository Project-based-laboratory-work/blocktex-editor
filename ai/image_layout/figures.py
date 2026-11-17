"""
学習用の水増し: 正解の図（figure）の四角の中に、ランダムなグラフや図を描き込む。

image_dataset の図は中身の無い枠（\\fbox）だけなので、そのまま学習すると「図＝空の枠」と覚えてしまい、
評価用データ（evaluation/synth_doc.py）やレポートの、棒グラフなど中身のある図を見つけられない（F1 0.417）。
学習のときだけ、図の枠の中身をランダムな図に描き替える。どのページの、どの図に描くかも毎回変わる。

描く図: 棒グラフ・折れ線・散布図・箱と矢印の図・写真のような濃淡。軸・目盛り・凡例の箱をときどき付ける。
色は画像の背景（地の明るさ）に合わせて、地と文字の間の濃さで描く（ダークモードの画像でも同じ見た目になる）。
"""

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FIGURE_RATE = 0.7  # 図1つあたり、描き替える割合（残りは枠のまま）


def _shade(rng: random.Random, bg: int, fg: int) -> int:
    """地（bg）と文字（fg）の間の濃さ"""
    t = rng.uniform(0.35, 1.0)
    return int(bg + (fg - bg) * t)


def _axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, rng: random.Random, color: int) -> tuple[int, int, int, int]:
    """軸と目盛りを描き、グラフを描く内側の四角を返す"""
    left, bottom = x0 + max(4, (x1 - x0) // 10), y1 - max(4, (y1 - y0) // 10)
    draw.line([(left, y0 + 2), (left, bottom), (x1 - 2, bottom)], fill=color, width=rng.choice([1, 1, 2]))
    for k in range(rng.randint(3, 6)):
        y = bottom - (bottom - y0) * (k + 1) // 7
        draw.line([(left - 3, y), (left, y)], fill=color)
        if rng.random() < 0.6:  # 目盛りの数字のかわりの短い線
            draw.line([(left - 3 - rng.randint(4, 10), y), (left - 5, y)], fill=color, width=2)
    return left + 2, y0 + 4, x1 - 4, bottom - 1


def _bars(draw, box, rng, bg, fg):
    x0, y0, x1, y1 = box
    n = rng.randint(3, 10)
    w = (x1 - x0) / n
    for k in range(n):
        h = rng.uniform(0.1, 0.95) * (y1 - y0)
        draw.rectangle([x0 + k * w + w * 0.15, y1 - h, x0 + (k + 1) * w - w * 0.15, y1], fill=_shade(rng, bg, fg),
                       outline=fg if rng.random() < 0.5 else None)  # fmt: skip


def _lines(draw, box, rng, bg, fg):
    x0, y0, x1, y1 = box
    for _ in range(rng.randint(1, 3)):
        n = rng.randint(4, 20)
        ys = np.cumsum(np.array([rng.uniform(-1, 1) for _ in range(n)]))
        ys = (ys - ys.min()) / max(np.ptp(ys), 1e-6)
        pts = [(x0 + (x1 - x0) * k / (n - 1), y1 - (y1 - y0) * (0.1 + 0.8 * v)) for k, v in enumerate(ys)]
        color = _shade(rng, bg, fg)
        draw.line(pts, fill=color, width=rng.choice([1, 2, 2, 3]))
        if rng.random() < 0.5:
            for x, y in pts:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)


def _scatter(draw, box, rng, bg, fg):
    x0, y0, x1, y1 = box
    r = rng.choice([1, 2, 3])
    for _ in range(rng.randint(15, 120)):
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_shade(rng, bg, fg))


def _diagram(draw, box, rng, bg, fg):
    """箱と矢印の図（処理の流れ図のようなもの）"""
    x0, y0, x1, y1 = box
    n = rng.randint(2, 5)
    horizontal = (x1 - x0) > (y1 - y0)
    cells = []
    for k in range(n):
        if horizontal:
            a, b = x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n
            cells.append((a + (b - a) * 0.12, y0 + (y1 - y0) * 0.3, b - (b - a) * 0.12, y1 - (y1 - y0) * 0.3))
        else:
            a, b = y0 + (y1 - y0) * k / n, y0 + (y1 - y0) * (k + 1) / n
            cells.append((x0 + (x1 - x0) * 0.25, a + (b - a) * 0.15, x1 - (x1 - x0) * 0.25, b - (b - a) * 0.15))
    for c in cells:
        draw.rectangle(c, outline=fg, width=rng.choice([1, 2]), fill=_shade(rng, bg, fg) if rng.random() < 0.3 else None)
        for _ in range(rng.randint(0, 2)):  # 箱の中の文字のかわりの短い線
            cy = rng.uniform(c[1] + 3, max(c[1] + 3, c[3] - 3))
            draw.line([(c[0] + 4, cy), (c[0] + 4 + (c[2] - c[0] - 8) * rng.uniform(0.3, 1.0), cy)], fill=fg, width=2)
    for a, b in zip(cells, cells[1:]):
        if horizontal:
            p, q = (a[2], (a[1] + a[3]) / 2), (b[0], (b[1] + b[3]) / 2)
        else:
            p, q = ((a[0] + a[2]) / 2, a[3]), ((b[0] + b[2]) / 2, b[1])
        draw.line([p, q], fill=fg, width=2)


def _photo(image: Image.Image, box, rng, bg, fg):
    """写真のような、ぼかした濃淡"""
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    rows, cols = rng.randint(3, 8), rng.randint(3, 8)
    small = np.array([[rng.uniform(0, 1) for _ in range(cols)] for _ in range(rows)])
    patch = Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    values = np.asarray(patch, dtype=np.float32) / 255
    image.paste(Image.fromarray((bg + (fg - bg) * (0.15 + 0.7 * values)).astype(np.uint8)), (x0, y0))


def draw_figure(image: Image.Image, box: tuple[float, float, float, float], rng: random.Random) -> None:
    """image（グレー、"L"）の box（画素）の中をランダムな図に描き替える"""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return
    pixels = np.asarray(image)
    bg = int(np.median(pixels))
    fg = 0 if bg > 127 else 255
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], fill=bg)  # 元の枠を消す
    if rng.random() < 0.4:
        draw.rectangle([x0, y0, x1, y1], outline=fg, width=rng.choice([1, 1, 2]))
    inner = (x0 + 3, y0 + 3, x1 - 3, y1 - 3)
    kind = rng.choice(["bars", "lines", "scatter", "diagram", "photo"])
    if kind == "photo":
        _photo(image, inner, rng, bg, fg)
        return
    if kind in ("bars", "lines", "scatter") and rng.random() < 0.8:
        inner = _axes(draw, *inner, rng, fg)
    if inner[2] - inner[0] < 8 or inner[3] - inner[1] < 8:
        return
    {"bars": _bars, "lines": _lines, "scatter": _scatter, "diagram": _diagram}[kind](draw, inner, rng, bg, fg)
    if kind != "diagram" and rng.random() < 0.3:  # 凡例の箱
        lx, ly = inner[2] - (inner[2] - inner[0]) // 4, inner[1] + 2
        draw.rectangle([lx, ly, inner[2] - 2, ly + (inner[3] - inner[1]) // 6], outline=fg)
