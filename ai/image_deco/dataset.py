"""
装飾AIの学習データの読み込み。generate.py が作った1行の画像と、装飾の範囲。

入力: 1行の画像を高さ HEIGHT にそろえ（縦横比は保つ）、「背景0・文字1」にしたもの。幅は行ごとに違うので、まとめるときに右を0で埋める
正解: 横の位置（1/STRIDE）ごとに [太字, 斜体, 下線] の 0/1

学習のときだけ、スクリーンショットに近づける崩しを入れる（解像度を落とす・JPEGで劣化させる・明暗の反転・上下の余白を変える）。
"""

import glob
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from image_deco.model import MARKS

HEIGHT = 40
STRIDE = 4
MAX_WIDTH = 2048  # これより長い行は、縦横比を崩して縮める（めったに無い）


MARGIN = 0.25  # 文字の上下に足す余白（文字の高さに対する割合）


def line_tensor(gray: Image.Image, margin: float = MARGIN) -> tuple[torch.Tensor, float]:
    """1行の画像 → (1, HEIGHT, W) の「背景0・文字1」と、横の拡大率。ダークモードは反転してそろえる。

    切り出し方（上下の余白）で文字の大きさが変わると、太さの見え方も変わって太字と間違える。
    そこで、文字のある行だけに縮めてから、文字の高さの margin 倍の余白を上下に足してそろえる"""
    ink = 1.0 - np.asarray(gray, dtype=np.float32) / 255.0
    if np.median(ink) > 0.5:
        ink = 1.0 - ink
    rows = np.flatnonzero(ink.max(axis=1) > 0.5)
    if len(rows):
        top, bottom = rows[0], rows[-1] + 1
        pad = max(1, round((bottom - top) * margin))
        ink = np.pad(ink[top:bottom], ((pad, pad), (0, 0)))
    h, w = ink.shape
    new_w = min(MAX_WIDTH, max(STRIDE, round(w * HEIGHT / h)))
    new_w -= new_w % STRIDE
    resized = Image.fromarray((ink * 255).astype(np.uint8)).resize((new_w, HEIGHT), Image.BILINEAR)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)[None], new_w / w


def degrade(gray: Image.Image, rng: random.Random) -> Image.Image:
    """スクリーンショットのような崩し"""
    if rng.random() < 0.5:  # 解像度を落として戻す（縮小表示）
        f = rng.uniform(0.5, 0.9)
        w, h = gray.size
        gray = gray.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.3:
        gray = gray.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.4:
        buf = io.BytesIO()
        gray.save(buf, "JPEG", quality=rng.randint(40, 90))
        gray = Image.open(buf).convert("L")
    if rng.random() < 0.2:  # ダークモード（地と文字の濃さもばらつかせる）
        bg, fg = rng.randint(20, 60), rng.randint(190, 240)
        a = np.asarray(gray, dtype=np.float32) / 255.0
        gray = Image.fromarray((fg + (bg - fg) * a).astype(np.uint8))
    return gray


def target(spans: list, scale: float, width: int) -> torch.Tensor:
    """装飾の範囲（画素） → (3, width / STRIDE) の 0/1"""
    out = torch.zeros(len(MARKS), width // STRIDE)
    for x0, x1, kind in spans:
        a, b = int(np.floor(x0 * scale / STRIDE)), int(np.ceil(x1 * scale / STRIDE))
        out[MARKS.index(kind), max(0, a) : max(0, b)] = 1.0
    return out


class DecoDataset(Dataset):
    def __init__(self, data_dir: Path, limit: int | None = None, augment: bool = False):
        self.data_dir = str(data_dir)
        self.label_paths = sorted(glob.glob(os.path.join(self.data_dir, "labels", "*.json")))[:limit]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.label_paths[index]
        label = json.loads(Path(path).read_text(encoding="utf-8"))
        name = os.path.splitext(os.path.basename(path))[0]
        with Image.open(os.path.join(self.data_dir, "images", f"{name}.png")) as image:
            gray = image.convert("L")
        if self.augment:
            gray = degrade(gray, random.Random())
        x, scale = line_tensor(gray, random.uniform(0.1, 0.4) if self.augment else MARGIN)
        return x, target(label["spans"], scale, x.shape[2])


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """幅をそろえる。(画像, 正解, 有効な位置の印)"""
    width = max(x.shape[2] for x, _ in batch)
    images = torch.zeros(len(batch), 1, HEIGHT, width)
    targets = torch.zeros(len(batch), len(MARKS), width // STRIDE)
    valid = torch.zeros(len(batch), width // STRIDE, dtype=torch.bool)
    for k, (x, t) in enumerate(batch):
        images[k, :, :, : x.shape[2]] = x
        targets[k, :, : t.shape[1]] = t
        valid[k, : t.shape[1]] = True
    return images, targets, valid
