import numpy as np
import torch
from PIL import Image

from image_table.dataset import CANVAS, OUT, STRIDE, outside_mask
from image_table.evaluate import matched, rule_predict
from image_table.model import ModelConfig, TableSplitter
from image_table.postprocess import runs, spans
from image_table.predict import aligns, rules


def test_outside_mask_marks_gaps_and_padding():
    # 拡大率1の画像（高さ40画素）。行は 4〜10 と 14〜20 画素 → 出力の位置では 2〜5 と 7〜10
    out = outside_mask([[4, 10], [14, 20]], 1.0, 40)
    assert out[2:5].sum() == 0 and out[7:10].sum() == 0
    assert out[5:7].tolist() == [1, 1] and out[20 // STRIDE :].min() == 1


def test_runs_drop_short_pieces_and_spans_return_image_pixels():
    outside = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1])
    assert runs(outside) == [(1, 4), (7, 9)]  # 長さ1の区間（5）は捨てる
    logits = np.where(outside == 1, 3.0, -3.0)
    assert spans(logits, 0.5) == [(1 * STRIDE / 0.5, 4 * STRIDE / 0.5), (7 * STRIDE / 0.5, 9 * STRIDE / 0.5)]


def test_matched_needs_iou_half_and_one_to_one():
    gold = [(0, 10), (12, 20)]
    assert matched([(0, 9), (13, 20)], gold) == 2
    assert matched([(0, 22)], gold) == 0  # 2つの行を1つにまとめた予測は、どちらとも IoU 0.5 未満


def test_model_output_lengths():
    model = TableSplitter(ModelConfig()).eval()
    with torch.no_grad():
        rows, cols = model(torch.zeros(2, 1, CANVAS, CANVAS))
    assert rows.shape == (2, OUT) and cols.shape == (2, OUT)


def _table_image() -> np.ndarray:
    """2行×2列の表。上下に横線、左の列は左寄せ、右の列は右寄せ"""
    ink = np.zeros((60, 120), dtype=bool)
    ink[2, :] = ink[57, :] = True  # 上下の横線
    ink[10:20, 10:30] = ink[35:45, 10:40] = True  # 左の列（短いセルが左に寄る）
    ink[10:20, 90:110] = ink[35:45, 80:110] = True  # 右の列（短いセルが右に寄る）
    return ink


def test_rules_and_aligns_from_pixels():
    ink = _table_image()
    rows, cols = [(10, 20), (35, 45)], [(10, 50), (70, 110)]
    assert rules(ink, rows, cols) == ([True, False, True], [False, False, False])
    assert aligns(ink, rows, cols) == ["l", "r"]


def test_rule_baseline_ignores_rules_and_word_spaces():
    ink = _table_image()
    ink[10:20, 32:38] = True  # 左上のセルに2つ目の単語（間の空白は2画素で、文字の高さより狭い）
    image = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    rows, cols = rule_predict(image)
    assert len(rows) == 2 and len(cols) == 2
