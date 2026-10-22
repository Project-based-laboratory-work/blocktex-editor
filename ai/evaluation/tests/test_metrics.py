import pytest

from evaluation.metrics import iou


def test_iou_same_box_is_1():
    assert iou((0.1, 0.1, 0.5, 0.5), (0.1, 0.1, 0.5, 0.5)) == 1.0


def test_iou_no_overlap_is_0():
    assert iou((0.0, 0.0, 0.2, 0.2), (0.5, 0.5, 0.8, 0.8)) == 0.0


def test_iou_partial_overlap():
    # a と b は右下に 0.2 ずつずれている
    assert iou((0.0, 0.0, 0.4, 0.4), (0.2, 0.2, 0.6, 0.6)) == pytest.approx(0.04 / 0.28)