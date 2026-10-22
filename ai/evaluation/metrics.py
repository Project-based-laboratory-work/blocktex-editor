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
