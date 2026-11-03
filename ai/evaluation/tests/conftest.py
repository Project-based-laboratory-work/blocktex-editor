"""evaluation のテストで共通に使う準備（fixture）。pytest がこのファイルを自動で読み込む。"""
import shutil
from pathlib import Path

import pytest

from evaluation.ground_truth import compile_tex

SAMPLE_TEX = Path(__file__).parent.parent / "sample_data" / "mark_test.tex"  # tests/ の1つ上の sample_data/


@pytest.fixture(scope="session")
def has_tex() -> None:
    """uplatex / dvipdfmx が無い環境では、TeXをコンパイルするテストを飛ばす"""
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無いので、TeXをコンパイルするテストは飛ばす")


@pytest.fixture(scope="session")
def sample_pdf(has_tex, tmp_path_factory) -> str:
    """mark_test.tex をPDFにする。PDFはgitに入れない決まり（.gitignore）なので、テストのたびに作る。"""
    return str(compile_tex(SAMPLE_TEX.read_text(encoding="utf-8"), tmp_path_factory.mktemp("sample")))

@pytest.fixture(scope="session")
def small_dataset(has_tex, tmp_path_factory) -> Path:
    """日本語1件・英語1件の小さなデータセット（評価の実行のテスト用）"""
    from evaluation.make_dataset import main  # この fixture を使うテストのときだけ読み込む

    out = tmp_path_factory.mktemp("datasets") / "small"
    main(["--out", str(out), "--docs", "2"])
    return out
