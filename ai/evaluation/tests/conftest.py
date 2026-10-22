"""evaluation のテストで共通に使う準備（fixture）。pytest がこのファイルを自動で読み込む。"""
import shutil
import subprocess
from pathlib import Path

import pytest

SAMPLE_TEX = Path(__file__).parent.parent / "sample_data" / "mark_test.tex"  # tests/ の1つ上の sample_data/


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> str:
    """mark_test.tex をPDFにする。PDFはgitに入れない決まり（.gitignore）なので、テストのたびに作る。"""
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無いので、PDFを使うテストは飛ばす")
    work = tmp_path_factory.mktemp("marks")
    shutil.copy(SAMPLE_TEX, work)
    subprocess.run(["uplatex", "-interaction=nonstopmode", "-halt-on-error", "mark_test.tex"], cwd=work, check=True, capture_output=True)
    subprocess.run(["dvipdfmx", "-q", "-C", "0x0010", "mark_test.dvi"], cwd=work, check=True, capture_output=True)
    return str(work / "mark_test.pdf")
