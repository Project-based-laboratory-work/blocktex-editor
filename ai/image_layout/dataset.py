"""
画像のレイアウト解析（座標AI）の学習データ。image_dataset の画像と正解（ラベル付きIR）をそのまま使う。

入力: 画像を、縦横比を保って CANVAS_H × CANVAS_W に収めたもの（左上に置き、余りは0で埋める）。「背景0・文字1」
正解: 1/STRIDE の大きさの「塗り分け図」。各画素に、そこを覆うブロックのラベル番号（背景は0）を入れる

隣り合うブロックが1つにつながらないよう、正解の四角は SHRINK 画素ぶん内側に縮めて塗る。
予測したかたまりは、四角にしたあと同じだけ広げて元の大きさに戻す（postprocess.py）。
縮めて空いた外周（ブロックのふち）も、別の図として当てさせる（boundary_mask）。
"""

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common.ir import LABELS, load_ir
from image_layout.figures import FIGURE_RATE, draw_figure

CANVAS_H, CANVAS_W = 768, 576
STRIDE = 4  # 出力（塗り分け図）は入力の 1/4 の大きさ
OUT_H, OUT_W = CANVAS_H // STRIDE, CANVAS_W // STRIDE
SHRINK = 1  # 出力の画素で、四角を内側に縮める幅（1行の高さが約4画素なので、縮めすぎると1行のブロックが線になる）
CLASSES = ["background", *LABELS]  # 0 = 背景
CLASS_TO_ID = {label: i for i, label in enumerate(CLASSES)}


def load_canvas(image_path: str) -> tuple[np.ndarray, float]:
    """画像 → (CANVAS_H, CANVAS_W) の「背景0・文字1」の配列と、拡大率。ダークモードは反転してそろえる"""
    with Image.open(image_path) as image:
        return canvas_of(image.convert("L"))


def canvas_of(gray: Image.Image) -> tuple[np.ndarray, float]:
    """グレーの画像 → load_canvas と同じもの"""
    w, h = gray.size
    scale = min(CANVAS_H / h, CANVAS_W / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    ink = 1.0 - np.asarray(gray.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
    if np.median(ink) > 0.5:  # 背景が暗い
        ink = 1.0 - ink
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    canvas[:new_h, :new_w] = ink
    return canvas, scale


def _boxes(truth_path: str, scale: float) -> list[tuple[float, float, float, float, float, int]]:
    """正解のIR → [(面積, x0, y0, x1, y1, クラス番号)]。座標は出力の画素"""
    truth = load_ir(truth_path)
    page = truth.pages[0]
    boxes = []
    for el in page.elements:
        if not el.label:
            continue
        x0, y0, x1, y1 = el.bbox
        # 0〜1 → 元の画像の px → キャンバス → 出力の画素
        f = scale / STRIDE
        bx0, by0, bx1, by1 = x0 * page.width * f, y0 * page.height * f, x1 * page.width * f, y1 * page.height * f
        boxes.append(((bx1 - bx0) * (by1 - by0), bx0, by0, bx1, by1, CLASS_TO_ID[el.label]))
    return boxes


def _shrunk(bx0: float, by0: float, bx1: float, by1: float) -> tuple[int, int, int, int]:
    """内側に縮めた四角（両端を含む）。縮めると消えてしまう小さいブロックは、真ん中の1画素以上を残す"""
    sx0, sy0 = int(round(bx0)) + SHRINK, int(round(by0)) + SHRINK
    sx1, sy1 = int(round(bx1)) - SHRINK, int(round(by1)) - SHRINK
    if sx1 <= sx0:
        sx0 = sx1 = int((bx0 + bx1) / 2)
    if sy1 <= sy0:
        sy0 = sy1 = int((by0 + by1) / 2)
    return max(0, sx0), max(0, sy0), min(OUT_W - 1, sx1), min(OUT_H - 1, sy1)


def target_mask(truth_path: str, scale: float) -> np.ndarray:
    """正解のIR → (OUT_H, OUT_W) のラベル番号の図。小さいブロックが大きいブロックに隠れないよう、面積の大きい順に塗る"""
    mask = np.zeros((OUT_H, OUT_W), dtype=np.int64)
    for _, bx0, by0, bx1, by1, cls in sorted(_boxes(truth_path, scale), reverse=True):
        sx0, sy0, sx1, sy1 = _shrunk(bx0, by0, bx1, by1)
        mask[sy0 : sy1 + 1, sx0 : sx1 + 1] = cls
    return mask


def boundary_mask(truth_path: str, scale: float) -> np.ndarray:
    """
    正解のIR → (OUT_H, OUT_W) の「ブロックのふち」の図（ふち=1）。四角のうち、縮めて塗らなかった外周の部分。

    隣り合う段落の間は、ふちの1〜2画素しか空いていない。背景として当てさせるだけでは塗りつぶされやすいので、
    ふちを別に当てさせ、推論ではふちの画素を消してからかたまりに分ける（postprocess.py）。
    """
    ring = np.zeros((OUT_H, OUT_W), dtype=bool)
    inside = np.zeros((OUT_H, OUT_W), dtype=bool)
    for _, bx0, by0, bx1, by1, _ in _boxes(truth_path, scale):
        x0, y0 = max(0, int(round(bx0))), max(0, int(round(by0)))
        x1, y1 = min(OUT_W - 1, int(round(bx1))), min(OUT_H - 1, int(round(by1)))
        ring[y0 : y1 + 1, x0 : x1 + 1] = True
        sx0, sy0, sx1, sy1 = _shrunk(bx0, by0, bx1, by1)
        inside[sy0 : sy1 + 1, sx0 : sx1 + 1] = True
    return (ring & ~inside).astype(np.float32)  # 別のブロックの中身と重なるところは、中身を優先する


class LayoutDataset(Dataset):
    """augment_figures=True（学習用）なら、図の枠の中をランダムな図に描き替える（figures.py）"""

    def __init__(self, data_dir: Path, limit: int | None = None, augment_figures: bool = False):
        self.data_dir = str(data_dir)
        self.truth_paths = sorted(glob.glob(os.path.join(self.data_dir, "truth", "*.ir.json")))[:limit]
        self.augment_figures = augment_figures

    def __len__(self) -> int:
        return len(self.truth_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        truth_path = self.truth_paths[index]
        truth = load_ir(truth_path)
        with Image.open(os.path.join(self.data_dir, "images", truth.source)) as image:
            gray = image.convert("L")
        if self.augment_figures:
            rng = random.Random()
            page = truth.pages[0]
            for el in page.elements:
                if el.label == "figure" and rng.random() < FIGURE_RATE:
                    x0, y0, x1, y1 = el.bbox
                    draw_figure(gray, (x0 * page.width, y0 * page.height, x1 * page.width, y1 * page.height), rng)
        canvas, scale = canvas_of(gray)
        return (
            torch.from_numpy(canvas)[None],
            torch.from_numpy(target_mask(truth_path, scale)),
            torch.from_numpy(boundary_mask(truth_path, scale)),
        )
