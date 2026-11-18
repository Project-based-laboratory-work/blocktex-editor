"""
モデルの出力（位置ごとの「外らしさ」）→ 行・列の範囲 → マス目。

  1. 外らしさが 0.5 以下の位置が続くところを、1つの行（列）にする
  2. 短すぎるもの（MIN_LENGTH 未満）はノイズとして捨てる
  3. 出力の位置 → 元の画像の画素に戻す
"""

import numpy as np

from image_table.dataset import STRIDE

MIN_LENGTH = 2  # 出力の位置の数


def runs(outside: np.ndarray) -> list[tuple[int, int]]:
    """0/1 の並び → 0 が続く区間 [始まり, 終わり) のリスト"""
    inside = np.concatenate([[False], outside == 0, [False]])
    edges = np.flatnonzero(inside[1:] != inside[:-1])
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_LENGTH]


def spans(logits: np.ndarray, scale: float) -> list[tuple[float, float]]:
    """1次元の点数 → 元の画像の画素での範囲のリスト"""
    f = STRIDE / scale
    return [(a * f, b * f) for a, b in runs((logits > 0).astype(np.int64))]
