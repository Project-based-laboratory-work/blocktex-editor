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

def safe_div(a: float, b: float) -> float:
    """a / b。b が 0 のときは 0.0（予測や正解が1つも無いページで止まらないように）。"""
    return a / b if b != 0 else 0

def detection_scores(n_pairs: int, n_preds: int, n_gts: int) -> dict[str, float]:
    """マッチングの結果から、検出の Precision / Recall / F1 を計算する。"""
    tp = n_pairs
    fp = n_preds - n_pairs
    fn = n_gts - n_pairs
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}

def label_accuracy(label_pairs: list[tuple[str, str]]) -> float:
    """ペアになった組のうち、ラベルが一致した割合。"""
    correct = sum(1 for gt, pred in label_pairs if gt == pred)
    return safe_div(correct, len(label_pairs))


def confusion_matrix(label_pairs: list[tuple[str, str]], labels: list[str]) -> list[list[int]]:
    """混同行列。matrix[正解の番号][予測の番号] に件数を数える。番号は labels の並び順。"""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for gt, pred in label_pairs:
        matrix[index[gt]][index[pred]] += 1
    return matrix
