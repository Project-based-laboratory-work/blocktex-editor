import pytest

from evaluation.marks import read_marks


def test_read_marks_finds_all_marks(sample_pdf):
    marks = read_marks(sample_pdf)
    assert set(marks) == {"b0s", "b0e", "b1s", "b1e"}  # 見出し(b0)と段落(b1)の始まりと終わり


def test_read_marks_uses_ir_coordinates(sample_pdf):
    marks = read_marks(sample_pdf)
    assert marks["b0s"].page == 1                             # 1始まり
    assert marks["b0s"].y < marks["b1s"].y                    # 見出しは段落より上 → 上からの距離が小さい
    assert marks["b1s"].y == pytest.approx(172.16, abs=0.01)  # 段落1行目のベースライン（841.89 − 669.73）
    assert marks["b1e"].y == pytest.approx(189.62, abs=0.01)  # 段落の最後の行のベースライン
