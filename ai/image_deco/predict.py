"""
1行の画像と、その行の文字（OCRの結果）から、装飾の付いた文字の範囲を決める。

  1. 装飾AIで、横の位置ごとに [太字, 斜体, 下線] の確率を出す
  2. 確率が 0.5 を超える位置が続くところを、装飾の範囲にする（短すぎるものはノイズとして捨てる）
  3. 範囲の左端・右端が行の幅のどこにあたるかの割合から、文字の位置に対応させる。
     1行の中の文字の幅はほぼそろっている（日本語は全角、英語は単語）とみなす。英語は単語の切れ目にそろえる
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_deco.dataset import STRIDE, line_tensor
from image_deco.model import MARKS, DecorationTagger, ModelConfig

MIN_COLUMNS = 3  # これより短い（出力の位置の数）装飾の範囲は捨てる


def load_model(checkpoint: Path) -> DecorationTagger:
    state = torch.load(checkpoint, map_location="cpu")
    model = DecorationTagger(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def line_spans(model: DecorationTagger, gray: Image.Image) -> list[tuple[float, float, str]]:
    """1行の画像 → [(左端の割合, 右端の割合, 装飾)]。割合は行の画像の幅に対する 0〜1"""
    x, _ = line_tensor(gray)
    probs = torch.sigmoid(model(x[None]))[0].numpy()  # (3, W/4)
    n = probs.shape[1]
    out = []
    for k, mark in enumerate(MARKS):
        on = np.concatenate([[False], probs[k] > 0.5, [False]])
        edges = np.flatnonzero(on[1:] != on[:-1])
        out += [(a / n, b / n, mark) for a, b in zip(edges[::2], edges[1::2]) if b - a >= MIN_COLUMNS]
    return out


def char_ranges(text: str, spans: list[tuple[float, float, str]]) -> list[tuple[int, int, str]]:
    """割合の範囲 → 文字の位置の範囲 [(開始, 終わり, 装飾)]。英語（空白で区切られた語）は語の切れ目まで広げる"""
    out = []
    n = len(text)
    for a, b, mark in spans:
        start, end = int(round(a * n)), int(round(b * n))
        if " " in text:  # 語の途中で切れたら、その語全体にする
            while 0 < start < n and text[start - 1] != " ":
                start -= 1
            while 0 < end < n and text[end] != " ":
                end += 1
            while start < end and text[start] == " ":
                start += 1
        if end > start:
            out.append((start, end, mark))
    return out


def pieces(text: str, ranges: list[tuple[int, int, str]]) -> list[tuple[str, frozenset[str]]]:
    """文字列を、装飾が同じ部分ごとに区切る → [(文字, 装飾の集合)]"""
    marks = [set() for _ in text]
    for start, end, mark in ranges:
        for i in range(start, min(end, len(text))):
            marks[i].add(mark)
    out: list[tuple[str, frozenset[str]]] = []
    for ch, m in zip(text, marks):
        m = frozenset(m)
        if out and out[-1][1] == m:
            out[-1] = (out[-1][0] + ch, m)
        else:
            out.append((ch, m))
    return out
