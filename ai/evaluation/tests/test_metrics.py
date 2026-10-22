import pytest

from evaluation.metrics import (
    char_errors,
    char_errors_by_script,
    confusion_matrix,
    detection_scores,
    iou,
    label_accuracy,
    match,
    normalize_text,
    split_by_script,
    Counts,
    count_page
)
from common.ir import Element, Page

def test_iou_same_box_is_1():
    assert iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == 1.0


def test_iou_no_overlap_is_0():
    assert iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.8, 0.8)) == 0.0


def test_iou_partial_overlap():
    # a と b は右下に 0.2 ずつずれている
    assert iou((0.0, 0.0, 0.4, 0.4), (0.2, 0.2, 0.6, 0.6)) == pytest.approx(0.04 / 0.28)

def test_match_pairs_each_pred_with_nearest_gt():
    gts = [(0.1, 0.1, 0.9, 0.15), (0.1, 0.2, 0.9, 0.4)]  # 0: 見出し、1: 段落
    preds = [
        (0.1, 0.21, 0.9, 0.4),  # 0: 段落とほぼ同じ位置
        (0.1, 0.1, 0.9, 0.16),  # 1: 見出しとほぼ同じ位置
        (0.5, 0.8, 0.9, 0.9),   # 2: どの正解とも重ならない
    ]
    assert sorted(match(preds, gts)) == [(0, 1), (1, 0)]


def test_match_is_one_to_one():
    gts = [(0.0, 0.0, 0.4, 0.4), (0.05, 0.0, 0.45, 0.4)]  # 少しずれた正解が2つ
    preds = [(0.0, 0.0, 0.4, 0.4)]                         # 予測は1つだけ
    assert match(preds, gts) == [(0, 0)]

def test_detection_scores():
    # 2-b の1つ目のテストと同じ状況: 予測3つ、正解2つ、ペア2つ
    scores = detection_scores(n_pairs=2, n_preds=3, n_gts=2)
    assert scores["precision"] == pytest.approx(2 / 3)  # 予測3つのうち当たりは2つ
    assert scores["recall"] == pytest.approx(1.0)       # 正解2つを全部見つけた
    assert scores["f1"] == pytest.approx(0.8)           # 2 × (2/3) × 1 ÷ (2/3 + 1)


def test_detection_scores_with_no_preds():
    # 予測が1つも無くても止まらない
    assert detection_scores(n_pairs=0, n_preds=0, n_gts=3) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

LABEL_PAIRS = [  # (正解, 予測)
    ("heading", "heading"),
    ("heading", "paragraph"),   # 見出しを段落と間違えた
    ("paragraph", "paragraph"),
    ("paragraph", "paragraph"),
]


def test_label_accuracy():
    assert label_accuracy(LABEL_PAIRS) == pytest.approx(3 / 4)  # 4組のうち3組が一致
    assert label_accuracy([]) == 0.0                             # ペアが無くても止まらない


def test_confusion_matrix():
    matrix = confusion_matrix(LABEL_PAIRS, ["heading", "paragraph", "caption"])
    assert matrix == [
        [1, 1, 0],  # 正解 heading: heading と1回、paragraph と1回答えた
        [0, 2, 0],  # 正解 paragraph: 2回とも paragraph
        [0, 0, 0],  # caption は出てこないので全部0
    ]

def test_normalize_text_removes_spaces_and_newlines():
    assert normalize_text("これは\nテスト です") == "これはテストです"
    assert normalize_text("Hello\u3000World") == "HelloWorld"  # \u3000 は全角空白


def test_char_errors():
    assert char_errors("見出しA", "見出しB") == (1, 4)    # 1文字の置換
    assert char_errors("段落\nです", "段落です") == (0, 4)  # 改行の違いは数えない
    assert char_errors("", "abc") == (3, 3)               # 何も読めなければ全部誤り

def test_split_by_script():
    assert split_by_script("PDFの読み込み") == ("の読み込み", "PDF")


def test_char_errors_by_script():
    gt = "PDFの読み込み"  # 日本語5文字、英語3文字
    assert char_errors_by_script(gt, gt) == {"ja": (0, 5), "en": (0, 3)}
    assert char_errors_by_script("PBFの読み込み", gt) == {"ja": (0, 5), "en": (1, 3)}  # 英語だけ間違えた
    assert char_errors_by_script("PDFの読み込ろ", gt) == {"ja": (1, 5), "en": (0, 3)}  # 日本語だけ間違えた

A = (0.1, 0.1, 0.5, 0.15)
B = (0.1, 0.2, 0.9, 0.4)
C = (0.2, 0.5, 0.8, 0.8)
D = (0.1, 0.9, 0.3, 0.95)


def test_count_page():
    gt = Page(page=1, width=595, height=842, unit="pt", elements=[
        Element(kind="text", bbox=A, text="1. はじめに", label="heading"),
        Element(kind="text", bbox=B, text="本文です", label="paragraph"),
        Element(kind="figure", bbox=C, label="figure"),
    ])
    pred = Page(page=1, width=595, height=842, unit="pt", elements=[
        Element(kind="text", bbox=A, text="1. はじめに", label="heading"),
        Element(kind="text", bbox=B, text="本文でず", label="heading"),  # ラベルも1文字も間違えた
        Element(kind="text", bbox=D, text="3", label="other"),           # 正解に無い（ページ番号を拾った）
    ])
    counts = Counts()
    count_page(pred, gt, counts)

    assert (counts.n_pairs, counts.n_preds, counts.n_gts) == (2, 3, 3)  # 図は見逃し、ページ番号は誤検出
    assert counts.iou_sum == pytest.approx(2.0)                          # 2組とも位置はぴったり
    assert sorted(counts.label_pairs) == [("heading", "heading"), ("paragraph", "heading")]
    assert (counts.text_errors, counts.text_chars) == (1, 10)  # 「1.はじめに」6文字 +「本文です」4文字
    assert (counts.ja_errors, counts.ja_chars) == (1, 8)
    assert (counts.en_errors, counts.en_chars) == (0, 2)       # 「1.」の2文字


def test_count_page_adds_up():
    page = Page(page=1, width=595, height=842, unit="pt", elements=[Element(kind="figure", bbox=C)])
    counts = Counts()
    count_page(page, page, counts)
    count_page(page, page, counts)  # 2ページ目も同じ入れ物に足す
    assert (counts.n_pairs, counts.n_preds, counts.n_gts) == (2, 2, 2)
