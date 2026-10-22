import pytest

from evaluation.metrics import iou, match, detection_scores


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
