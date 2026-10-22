"""
予測したラベル付きIRと正解のラベル付きIRを比べる評価指標。
"""
from common.ir import Bbox


def area(bbox: Bbox) -> float:
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0) * max(y1 - y0, 0)


def iou(a: Bbox, b: Bbox) -> float:
    """2つのbboxの重なり具合（0〜1）。重なり部分の面積 ÷ 2つを合わせた部分の面積。"""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    inter_area = area(inter)
    union_area = area(a) + area(b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

IOU_THRESHOLD = 0.5  # これ以上重なっていれば同じブロックとみなす

def match(preds: list[Bbox], gts: list[Bbox], threshold: float = IOU_THRESHOLD) -> list[tuple[int, int]]:
    """IoUが大きい組から順に、予測と正解を1対1で対応させる。(予測の番号, 正解の番号) のリストを返す。"""
    candidates = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            score = iou(p, g)
            if score >= threshold:
                candidates.append((score, i, j))
    candidates.sort(reverse=True)

    used_preds = set()
    used_gts = set()
    pairs = []
    for score, i, j in candidates:
        if i in used_preds or j in used_gts:
            continue
        pairs.append((i, j))
        used_preds.add(i)
        used_gts.add(j)
    return pairs
