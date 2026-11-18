"""
塗り分け図（モデルの出力）→ ブロックの四角とラベル → チーム共通のIR。

  1. 画素ごとに一番点数の高いクラスを選ぶ。ふちの出力があれば、ふちと判定した画素は背景にする（隣のブロックと切り離す）。
     さらに、横幅 OPEN_WIDTH 未満の細い部分を消す（ふちの途切れたところから隣の段落へ漏れ出た「橋」を切る）
  2. 背景以外の画素がつながったかたまり（連結成分）を1つのブロックにする。クラスはかたまりの中で一番多いもの
  3. 学習で内側に縮めた分（dataset.SHRINK）だけ四角を広げ、元の画像の座標に戻す
"""

import cv2
import numpy as np

from common.ir import Element, IRDocument, Page, normalize_bbox
from image_layout.dataset import CLASSES, SHRINK, STRIDE

MIN_PIXELS = 3  # これより小さいかたまりはノイズとして捨てる（出力の画素数）
KIND = {"table": "table", "figure": "figure"}  # それ以外のラベルは kind="text"
# 後処理の設定。学習データ400枚で組み合わせを比べて決めた（検証データは使っていない）
BOUNDARY_THRESHOLD = 0.7  # ふちの確率がこれより大きい画素を背景にする
OPEN_WIDTH = 9  # 出力の画素。13 にすると、ページ番号などの短いブロックまで消えて悪くなる


def logits_to_classes(logits: np.ndarray) -> np.ndarray:
    """モデルの出力 (チャンネル, OUT_H, OUT_W) → クラス番号の図。チャンネルがクラス数より1つ多ければ、最後はふちの点数"""
    classes = logits[: len(CLASSES)].argmax(0)
    if logits.shape[0] > len(CLASSES):
        classes[logits[len(CLASSES)] > np.log(BOUNDARY_THRESHOLD / (1 - BOUNDARY_THRESHOLD))] = 0  # 点数 → 確率の逆
    # 「開く」処理: 横 OPEN_WIDTH 画素の窓が収まらない細い部分を消す。文字の行は横に長いので残る
    kernel = np.ones((1, OPEN_WIDTH), np.uint8)
    return classes * cv2.morphologyEx((classes > 0).astype(np.uint8), cv2.MORPH_OPEN, kernel)


def mask_to_blocks(classes: np.ndarray) -> list[tuple[str, tuple[float, float, float, float]]]:
    """(OUT_H, OUT_W) のクラス番号 → [(ラベル, 出力の画素での四角 x0, y0, x1, y1)]"""
    count, components, stats, _ = cv2.connectedComponentsWithStats((classes > 0).astype(np.uint8), connectivity=4)
    blocks = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < MIN_PIXELS:
            continue
        label = CLASSES[np.bincount(classes[components == i]).argmax()]
        blocks.append((label, (x - SHRINK, y - SHRINK, x + w + SHRINK, y + h + SHRINK)))
    return blocks


def to_ir(classes: np.ndarray, source: str, width: int, height: int, scale: float, texts=None) -> IRDocument:
    """ブロックをIRにする。texts（OCRの段落 [(bbox, text)]、0〜1）があれば、四角に中心が入る段落の文字をつなげて入れる"""
    page = Page(page=1, width=width, height=height, unit="px")
    f = STRIDE / scale  # 出力の画素 → 元の画像の px
    for label, (x0, y0, x1, y1) in mask_to_blocks(classes):
        box = (max(0.0, x0 * f), max(0.0, y0 * f), min(width, x1 * f), min(height, y1 * f))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        bbox = normalize_bbox(box, width, height)
        text = None
        if texts is not None:
            inside = [t for b, t in texts if bbox[0] <= (b[0] + b[2]) / 2 <= bbox[2] and bbox[1] <= (b[1] + b[3]) / 2 <= bbox[3]]
            text = "\n".join(inside) if inside else None
        kind = KIND.get(label, "text")
        # kind="text" の要素は text が必須（common/ir.py の形式チェック）。OCRを渡さないときは空文字にする
        page.elements.append(Element(kind=kind, bbox=bbox, text=(text or "") if kind == "text" else None, label=label))
    page.elements.sort(key=lambda el: (el.bbox[1], el.bbox[0]))
    return IRDocument(source=source, extractor="image_layout", structurer="image_layout", pages=[page])
