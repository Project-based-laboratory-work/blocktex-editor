"""
予測したラベル付きIRと正解のラベル付きIRを比べる評価指標。
"""
from dataclasses import dataclass, field

from common.ir import Bbox, IRDocument, Page
from common.text import edit_distance

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

def normalize_text(text: str) -> str:
    """空白と改行を取り除く。PDFの折り返しや空白の入り方の違いを、文字の誤りとして数えないため。"""
    return "".join(text.split())


def char_errors(pred: str, gt: str) -> tuple[int, int]:
    """(誤りの数, 正解の文字数)。CER は、全部の組で足し合わせてから 誤りの数 ÷ 正解の文字数 で出す。"""
    p = normalize_text(pred)
    g = normalize_text(gt)
    return edit_distance(g, p), len(g)

def split_by_script(text: str) -> tuple[str, str]:
    """文字列を (ASCII以外の文字だけ, ASCIIの文字だけ) に分ける。並び順はそのまま。"""
    ja = "".join(c for c in text if not c.isascii())
    en = "".join(c for c in text if c.isascii())
    return ja, en


def char_errors_by_script(pred: str, gt: str) -> dict[str, tuple[int, int]]:
    """日本語の文字と英語の文字に分けて char_errors を計算する。「英語に強い」などを見るため。"""
    pred_ja, pred_en = split_by_script(pred)
    gt_ja, gt_en = split_by_script(gt)
    return {"ja": char_errors(pred_ja, gt_ja), "en": char_errors(pred_en, gt_en)}

@dataclass
class Counts:
    """足し合わせてから割る数。ページ・文書・データセット全体のどれでも、同じ形で足していける。"""
    n_pairs: int = 0
    n_preds: int = 0
    n_gts: int = 0
    iou_sum: float = 0.0  # ペアのIoUの合計。mIoU = iou_sum ÷ n_pairs
    label_pairs: list[tuple[str, str]] = field(default_factory=list)  # (正解, 予測)。ラベルを出すモデルのときだけ入る
    text_errors: int = 0
    text_chars: int = 0
    ja_errors: int = 0
    ja_chars: int = 0
    en_errors: int = 0
    en_chars: int = 0


def count_page(pred: Page, gt: Page, counts: Counts) -> None:
    """1ページ分の予測と正解を比べて、counts に足し込む。"""
    pairs = match([el.bbox for el in pred.elements], [el.bbox for el in gt.elements])
    counts.n_pairs += len(pairs)
    counts.n_preds += len(pred.elements)
    counts.n_gts += len(gt.elements)

    for i, j in pairs:
        p = pred.elements[i]
        g = gt.elements[j]
        counts.iou_sum += iou(p.bbox, g.bbox)

        if p.label is not None and g.label is not None:
            counts.label_pairs.append((g.label, p.label))
        if g.text is not None:
            pred_text = p.text or ""
            errors, n = char_errors(pred_text, g.text)
            counts.text_errors += errors
            counts.text_chars += n
            by_script = char_errors_by_script(pred_text, g.text)
            counts.ja_errors += by_script["ja"][0]
            counts.ja_chars += by_script["ja"][1]
            counts.en_errors += by_script["en"][0]
            counts.en_chars += by_script["en"][1]

def ratio_or_none(a: float, b: float) -> float | None:
    """a / b。b が 0 のときは None（比べるものが無かった＝対象外）。0.0 だと CER では「満点」に見えてしまうため。"""
    return a / b if b > 0 else None

def summarize(counts: Counts) -> dict[str, float | None]:
    """Counts から成績を計算する。数えるものが無かった指標は None。"""
    scores: dict[str, float | None] = detection_scores(counts.n_pairs, counts.n_preds, counts.n_gts)
    scores["miou"] = ratio_or_none(counts.iou_sum, counts.n_pairs)
    scores["label_acc"] = label_accuracy(counts.label_pairs) if counts.label_pairs else None
    scores["cer"] = ratio_or_none(counts.text_errors, counts.text_chars)
    scores["cer_ja"] = ratio_or_none(counts.ja_errors, counts.ja_chars)
    scores["cer_en"] = ratio_or_none(counts.en_errors, counts.en_chars)
    return scores


def count_document(pred: IRDocument, gt: IRDocument, counts: Counts) -> None:
    """文書1つ分を、正解のページごとに比べて counts に足し込む。予測に無いページは空のページとして比べる。"""
    pred_pages = {p.page:p for p in pred.pages}
    for gt_page in gt.pages:
        empty = Page(page=gt_page.page, width=gt_page.width, height=gt_page.height, unit=gt_page.unit)
        pred_page = pred_pages.get(gt_page.page, empty)
        count_page(pred_page, gt_page, counts)
