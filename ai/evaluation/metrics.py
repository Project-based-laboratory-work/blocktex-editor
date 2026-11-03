"""
予測したラベル付きIRと正解のラベル付きIRを比べる評価指標。
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from common.ir import Bbox, Element, IRDocument, Page, Run
from common.text import edit_distance

DECORATION_MARKS = ("bold", "italic", "underline")  # common/ir.py の Run の装飾
# 文中の装飾を採点するラベル。見出しやタイトルは書体ごと太字で印刷されるが、エディタでは装飾（runs）ではなく
# ブロックの種類で表すので、採点しない
DECORATED_LABELS = ("paragraph", "list", "caption")
# 文字の誤り率（CER）を数えないラベル。表の中身は cells で持つ抽出器が多いので、セルの正解率（table_cell_acc）で別に見る
CER_EXCLUDED_LABELS = ("table",)


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
    return a / b if b != 0 else 0.0


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
    level_correct: int = 0  # 正解も予測も見出しで、見出しレベルが一致した数
    level_total: int = 0    # 正解も予測も見出しで、両方にレベルがあった数
    deco_matched: int = 0   # 装飾（太字・斜体・下線）が正しく付いた文字の数
    deco_preds: int = 0     # 予測で装飾が付いた文字の数
    deco_gts: int = 0       # 正解で装飾が付いた文字の数
    # ブロックの種類（ラベル）ごとの数。正解のラベル（無ければ kind）ごと
    gt_labels: dict[str, int] = field(default_factory=dict)     # 正解の要素の数
    found_labels: dict[str, int] = field(default_factory=dict)  # 正解の要素のうち、予測と対応した数（ラベルは問わない）
    pred_labels: dict[str, int] = field(default_factory=dict)   # 予測の要素の数（予測のラベルごと。ラベルを出すモデルだけ）
    # 表（正解も予測も cells などを持つ組だけ）
    table_pairs: int = 0
    table_shape_ok: int = 0  # 行数・列数が一致した表の数
    cell_correct: int = 0    # 正解のセルのうち、同じ位置の予測のセルの文字が一致した数
    cell_total: int = 0
    align_correct: int = 0   # 列の揃え（l / c / r）が一致した数
    align_total: int = 0
    rule_correct: int = 0    # 罫線（縦・横の境目ごとの有無）が一致した数
    rule_total: int = 0
    # 数式（正解も予測も tex を持つ組だけ）。トークン単位の編集距離
    math_errors: int = 0
    math_tokens: int = 0
    # コード（正解が code の組）。行ごとの字下げ（行頭の空白の数）が一致した数
    indent_correct: int = 0
    indent_total: int = 0


def add_count(counter: dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = counter.get(key, 0) + n


def same_cell(pred: str, gt: str) -> bool:
    return normalize_text(pred) == normalize_text(gt)


def table_counts(pred: Element, gt: Element, counts: Counts) -> None:
    """表の中身（セル・列の揃え・罫線）を、同じ位置どうしで比べて counts に足す。予測に無い項目は数えない（対象外）"""
    if gt.cells is None or pred.cells is None:
        return
    counts.table_pairs += 1
    counts.table_shape_ok += int(len(pred.cells) == len(gt.cells) and all(len(p) == len(g) for p, g in zip(pred.cells, gt.cells)))
    for r, row in enumerate(gt.cells):
        for c, cell in enumerate(row):
            counts.cell_total += 1
            counts.cell_correct += int(r < len(pred.cells) and c < len(pred.cells[r]) and same_cell(pred.cells[r][c], cell))
    if gt.aligns is not None and pred.aligns is not None:
        counts.align_total += len(gt.aligns)
        counts.align_correct += sum(int(k < len(pred.aligns) and pred.aligns[k] == a) for k, a in enumerate(gt.aligns))
    for gt_rules, pred_rules in ((gt.col_rules, pred.col_rules), (gt.row_rules, pred.row_rules)):
        if gt_rules is not None and pred_rules is not None:
            counts.rule_total += len(gt_rules)
            counts.rule_correct += sum(int(k < len(pred_rules) and pred_rules[k] == v) for k, v in enumerate(gt_rules))


TEX_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")  # \frac、\{、1文字（math_ocr/tokenizer.py と同じ区切り方）


def tex_tokens(tex: str) -> list[str]:
    """LaTeXをトークンに分ける。& は揃えの位置なので数えない"""
    return [t for t in TEX_TOKEN.findall(tex) if t != "&"]


def indent_width(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def indent_counts(pred_text: str, gt_text: str) -> tuple[int, int]:
    """(字下げが一致した行の数, 正解の行の数)。行は上から順に対応させる（予測の行が足りなければ不一致）"""
    gt_lines = gt_text.split("\n")
    pred_lines = pred_text.split("\n")
    correct = sum(int(k < len(pred_lines) and indent_width(pred_lines[k]) == indent_width(line)) for k, line in enumerate(gt_lines))
    return correct, len(gt_lines)


def decorated_text(runs: list[Run] | None, mark: str) -> str:
    """runs のうち、mark（"bold" など）が付いた部分の文字をつなげる。空白・改行は数えない"""
    return normalize_text("".join(run.text for run in runs or [] if getattr(run, mark)))


def decoration_counts(pred_runs: list[Run] | None, gt_runs: list[Run] | None) -> tuple[int, int, int]:
    """(装飾が正しく付いた文字の数, 予測で装飾が付いた文字の数, 正解で装飾が付いた文字の数)。
    装飾ごとに、付いた文字の並びを比べ、一致した部分（SequenceMatcher の一致ブロック）の文字数を数える。
    runs が無い（None）のは、装飾が1つも無いものとして扱う"""
    matched = n_pred = n_gt = 0
    for mark in DECORATION_MARKS:
        p, g = decorated_text(pred_runs, mark), decorated_text(gt_runs, mark)
        matched += sum(block.size for block in SequenceMatcher(None, p, g, autojunk=False).get_matching_blocks())
        n_pred += len(p)
        n_gt += len(g)
    return matched, n_pred, n_gt


def count_page(pred: Page, gt: Page, counts: Counts) -> None:
    """1ページ分の予測と正解を比べて、counts に足し込む。"""
    pairs = match([el.bbox for el in pred.elements], [el.bbox for el in gt.elements])
    counts.n_pairs += len(pairs)
    counts.n_preds += len(pred.elements)
    counts.n_gts += len(gt.elements)
    for el in gt.elements:
        add_count(counts.gt_labels, el.label or el.kind)
    for el in pred.elements:
        if el.label is not None:
            add_count(counts.pred_labels, el.label)

    for i, j in pairs:
        p = pred.elements[i]
        g = gt.elements[j]
        counts.iou_sum += iou(p.bbox, g.bbox)
        add_count(counts.found_labels, g.label or g.kind)
        table_counts(p, g, counts)
        if g.tex is not None and p.tex is not None:
            gt_tokens = tex_tokens(g.tex)
            counts.math_errors += edit_distance(tex_tokens(p.tex), gt_tokens)
            counts.math_tokens += len(gt_tokens)
        if g.label == "code" and g.text is not None and p.text is not None:
            correct, total = indent_counts(p.text, g.text)
            counts.indent_correct += correct
            counts.indent_total += total

        if p.label is not None and g.label is not None:
            counts.label_pairs.append((g.label, p.label))
        if p.label == g.label == "heading" and p.level is not None and g.level is not None:
            counts.level_total += 1
            counts.level_correct += int(p.level == g.level)
        if g.text is not None and g.label in DECORATED_LABELS:
            matched, n_pred, n_gt = decoration_counts(p.runs, g.runs)
            counts.deco_matched += matched
            counts.deco_preds += n_pred
            counts.deco_gts += n_gt
        if g.text is not None and g.label not in CER_EXCLUDED_LABELS:
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
    scores["level_acc"] = ratio_or_none(counts.level_correct, counts.level_total)
    deco_precision = ratio_or_none(counts.deco_matched, counts.deco_preds)
    deco_recall = ratio_or_none(counts.deco_matched, counts.deco_gts)
    scores["deco_precision"] = deco_precision
    scores["deco_recall"] = deco_recall
    # 正解にも予測にも装飾が1つも無ければ、比べるものが無いので対象外
    if counts.deco_preds == 0 and counts.deco_gts == 0:
        scores["deco_f1"] = None
    else:
        scores["deco_f1"] = safe_div(2 * (deco_precision or 0.0) * (deco_recall or 0.0), (deco_precision or 0.0) + (deco_recall or 0.0))
    scores["table_shape_acc"] = ratio_or_none(counts.table_shape_ok, counts.table_pairs)
    scores["table_cell_acc"] = ratio_or_none(counts.cell_correct, counts.cell_total)
    scores["table_align_acc"] = ratio_or_none(counts.align_correct, counts.align_total)
    scores["table_rule_acc"] = ratio_or_none(counts.rule_correct, counts.rule_total)
    scores["math_ter"] = ratio_or_none(counts.math_errors, counts.math_tokens)
    scores["code_indent_acc"] = ratio_or_none(counts.indent_correct, counts.indent_total)
    return scores


def label_scores(counts: Counts) -> dict[str, dict[str, float | int | None]]:
    """ブロックの種類（正解のラベル）ごとの成績。
    found_rate: 正解のうち、ラベルを問わず見つけた割合（枠の検出）
    recall / precision / f1: ラベルまで合っていた組で数えたもの（ラベルを出さないモデルでは None）"""
    correct: dict[str, int] = {}
    for gt_label, pred_label in counts.label_pairs:
        if gt_label == pred_label:
            add_count(correct, gt_label)
    has_labels = bool(counts.pred_labels)
    result = {}
    for label in sorted(set(counts.gt_labels) | set(counts.pred_labels)):
        n_gt, n_pred, ok = counts.gt_labels.get(label, 0), counts.pred_labels.get(label, 0), correct.get(label, 0)
        recall = ratio_or_none(ok, n_gt) if has_labels else None
        precision = ratio_or_none(ok, n_pred) if has_labels else None
        f1 = None
        if recall is not None and precision is not None:
            f1 = safe_div(2 * recall * precision, recall + precision)
        elif recall is not None and has_labels:
            f1 = 0.0  # 予測が1つも無いラベル（一度も出さなかった）
        result[label] = {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "found_rate": ratio_or_none(counts.found_labels.get(label, 0), n_gt),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
    return result


def count_document(pred: IRDocument, gt: IRDocument, counts: Counts) -> None:
    """文書1つ分を、正解のページごとに比べて counts に足し込む。予測に無いページは空のページとして比べる。"""
    pred_pages = {p.page:p for p in pred.pages}
    for gt_page in gt.pages:
        empty = Page(page=gt_page.page, width=gt_page.width, height=gt_page.height, unit=gt_page.unit)
        pred_page = pred_pages.get(gt_page.page, empty)
        count_page(pred_page, gt_page, counts)
