"""
表AIの学習データの読み込み。generate.py が作った表の画像と、行・列の位置。

入力: 表の画像を、縦横比を保って CANVAS × CANVAS に収めたもの（左上に置き、余りは0で埋める）。「背景0・文字1」
正解: 縦の位置ごとに「行の外（境目・余白）なら1」、横の位置ごとに「列の外なら1」。長さは 1/STRIDE

行の中か外かは、その高さだけを見ても決まらない。折り返したセルの行と行の間も、本当の行の境目も、横一列が空白になる。
モデルは表全体を見て、「次の行で他の列にも文字が始まるなら行の境目」のように判断する必要がある。
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CANVAS = 768
STRIDE = 2  # 出力は入力の 1/2 の長さ
OUT = CANVAS // STRIDE


def load_canvas(image: Image.Image) -> tuple[np.ndarray, float]:
    """表の画像 → (CANVAS, CANVAS) の「背景0・文字1」の配列と、拡大率。ダークモードは反転してそろえる"""
    gray = image.convert("L")
    w, h = gray.size
    scale = min(CANVAS / h, CANVAS / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    ink = 1.0 - np.asarray(gray.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
    if np.median(ink) > 0.5:
        ink = 1.0 - ink
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    canvas[:new_h, :new_w] = ink
    return canvas, scale


def outside_mask(spans: list[list[float]], scale: float, length: int) -> np.ndarray:
    """範囲（画像の画素）のリスト → 出力の長さの配列。どの範囲にも入らない位置は1、入る位置は0"""
    out = np.ones(OUT, dtype=np.float32)
    f = scale / STRIDE
    for a, b in spans:
        lo, hi = int(np.floor(a * f)), int(np.ceil(b * f))
        out[max(0, lo) : min(OUT, hi)] = 0.0
    out[int(np.ceil(length * f)) :] = 1.0  # 画像の外（埋めた部分）も外
    return out


class TableDataset(Dataset):
    def __init__(self, data_dir: Path, limit: int | None = None):
        self.data_dir = str(data_dir)
        self.label_paths = sorted(glob.glob(os.path.join(self.data_dir, "labels", "*.json")))[:limit]

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.label_paths[index]
        label = json.loads(Path(path).read_text(encoding="utf-8"))
        name = os.path.splitext(os.path.basename(path))[0]
        with Image.open(os.path.join(self.data_dir, "images", f"{name}.png")) as image:
            canvas, scale = load_canvas(image)
        rows = outside_mask(label["rows"], scale, label["height"])
        cols = outside_mask(label["cols"], scale, label["width"])
        return torch.from_numpy(canvas)[None], torch.from_numpy(rows), torch.from_numpy(cols)
