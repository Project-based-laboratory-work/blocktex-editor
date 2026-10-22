import shutil
import subprocess
from pathlib import Path

import pytest

from evaluation.marks import read_marks

SAMPLE_TEX = Path(__file__).parent.parent / "sample_data" / "mark_test.tex"  # tests/ の1つ上の sample_data/


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> str:
    """mark_test.tex をPDFにする。PDFはgitに入れない決まり（.gitignore）なので、テストのたびに作る。"""
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無いので、目印のテストは飛ばす")
    work = tmp_path_factory.mktemp("marks")
    shutil.copy(SAMPLE_TEX, work)
    subprocess.run(["uplatex", "-interaction=nonstopmode", "-halt-on-error", "mark_test.tex"], cwd=work, check=True, capture_output=True)
    subprocess.run(["dvipdfmx", "-q", "-C", "0x0010", "mark_test.dvi"], cwd=work, check=True, capture_output=True)
    return str(work / "mark_test.pdf")


def test_read_marks_finds_all_marks(sample_pdf):
    marks = read_marks(sample_pdf)
    assert set(marks) == {"b0s", "b0e", "b1s", "b1e"}  # 見出し(b0)と段落(b1)の始まりと終わり


def test_read_marks_uses_ir_coordinates(sample_pdf):
    marks = read_marks(sample_pdf)
    assert marks["b0s"].page == 1                             # 1始まり
    assert marks["b0s"].y < marks["b1s"].y                    # 見出しは段落より上 → 上からの距離が小さい
    assert marks["b1s"].y == pytest.approx(172.16, abs=0.01)  # 段落1行目のベースライン（841.89 − 669.73）
    assert marks["b1e"].y == pytest.approx(189.62, abs=0.01)  # 段落の最後の行のベースライン
