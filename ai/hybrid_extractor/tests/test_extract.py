"""抽出器（extract.py）とルール分類器（rules.py）のテスト。

小さい関数は値を直接確かめ、最後にサンプルの TeX（sample_data/sample.tex）をPDFにして、全体の結果を確かめる
（uplatex が無い環境では、そのテストだけ飛ばす）。
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from common.ir import Run, validate
from hybrid_extractor.extract import (
    column_align,
    column_of,
    extract_blocks,
    is_two_column,
    join_words,
    reading_order,
    same_row,
    to_runs,
    union_bbox,
)
from hybrid_extractor.rules import predict

SAMPLE_TEX = Path(__file__).parent.parent / "sample_data" / "sample.tex"


# ---- 2-c-1 で書いた関数 ----


def test_same_row_number_and_title():
    # 見出しの番号「1」と「はじめに」は、縦の範囲がほとんど重なる
    one = {"bbox": (75.71, 289.99, 83.39, 303.94)}
    title = {"bbox": (98.79, 288.11, 156.89, 302.63)}
    assert same_row(one, title)
    assert same_row(title, one)  # 順番を入れ替えても同じ


def test_same_row_next_line():
    # 見出しとその下の段落は、縦に離れている
    title = {"bbox": (98.79, 288.11, 156.89, 302.63)}
    para = {"bbox": (85.80, 318.19, 519.56, 329.19)}
    assert not same_row(title, para)


def test_union_bbox():
    # 左・上は小さい方、右・下は大きい方
    assert union_bbox((75.71, 289.99, 83.39, 303.94), (98.79, 288.11, 156.89, 302.63)) == (75.71, 288.11, 156.89, 303.94)


# ---- 段組み・読み順 ----


def test_column_of():
    # A4 の幅 595pt。中央は 297.5
    assert column_of((70, 0, 280, 10), 595) == 1   # 左の段
    assert column_of((310, 0, 520, 10), 595) == 2  # 右の段
    assert column_of((70, 0, 520, 10), 595) == 0   # 段をまたぐ


def test_is_two_column():
    left = [(70, y, 280, y + 10) for y in range(100, 160, 15)]
    right = [(310, y, 520, y + 10) for y in range(100, 160, 15)]
    full = [(70, y, 520, y + 10) for y in range(100, 220, 15)]
    assert is_two_column(left + right, 595)
    assert not is_two_column(full, 595)


def test_reading_order_two_column():
    # タイトル（段をまたぐ）→ 左の段を上から → 右の段を上から
    boxes = [(310, 100, 520, 110), (70, 120, 280, 130), (150, 50, 450, 70), (70, 100, 280, 110)]
    assert reading_order(boxes, 595, two_column=True) == [2, 3, 1, 0]


# ---- 表 ----


def test_join_words():
    assert join_words("15", "時間") == "15時間"     # 日本語はそのまま
    assert join_words("Table", "1") == "Table 1"    # 英数字どうしは空白をはさむ
    assert join_words("", "項目") == "項目"


def test_column_align():
    # 左端がそろっている / 右端がそろっている
    assert column_align([(10, 30), (10, 50), (10, 40)], 5, 60) == "l"
    assert column_align([(30, 50), (10, 50), (20, 50)], 5, 55) == "r"
    # 全部同じ幅のときは、列の境目からの余白で決める（左右の余白が同じなら中央）
    assert column_align([(20, 40), (20, 40)], 10, 50) == "c"


# ---- 装飾 ----


def ch(c: str) -> dict:
    return {"c": c, "bbox": (0, 0, 1, 1)}


def test_to_runs_groups_same_decoration():
    chars = [ch(c) for c in "abcd"]
    decos = [(), ("bold",), ("bold",), ()]
    assert to_runs(chars, decos) == [Run("a"), Run("bc", bold=True), Run("d")]


def test_to_runs_none_when_all_same():
    # 装飾が無い、または全部同じ装飾（見出しが全部太字など）なら runs は書かない
    chars = [ch(c) for c in "ab"]
    assert to_runs(chars, [(), ()]) is None
    assert to_runs(chars, [("bold",), ("bold",)]) is None


# ---- サンプルの文書で全体を確かめる ----


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無い")
    work = tmp_path_factory.mktemp("sample")
    shutil.copy(SAMPLE_TEX, work / "sample.tex")
    for command in (["uplatex", "-interaction=nonstopmode", "sample.tex"],) * 2 + (["dvipdfmx", "sample.dvi"],):
        subprocess.run(command, cwd=work, check=True, capture_output=True, timeout=120)
    return work / "sample.pdf"


def test_sample_blocks(sample_pdf):
    pages = extract_blocks(str(sample_pdf))
    assert len(pages) == 3
    blocks = [b for p in pages for b in p.blocks]
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 2
    assert [b.kind for b in blocks].count("figure") == 1
    # 表1: 罫線のある表。列の揃えは TeX に書いたとおり l c r
    grid = tables[0].table
    assert grid.cells[0] == ["項目", "担当", "工数"]
    assert grid.aligns == ["l", "c", "r"]
    assert grid.col_rules == [True, True, True, True]
    assert grid.row_rules == [True, True, False, False, True]  # 上・見出しの下・下だけ
    # 表2: 折り返す列（p{5cm}）の幅。(5cm + 左右の余白 6pt×2) × 1.095（11pt の拡大）÷ A4 の幅
    assert tables[1].table.col_widths[0] is None
    assert tables[1].table.col_widths[1] == pytest.approx(0.2827, abs=0.003)
    # コードの字下げが残る
    code = next(b for b in blocks if b.mono_ratio > 0.5)
    assert code.text.splitlines()[1] == "    for page in doc:"


def test_sample_labels(sample_pdf):
    doc = predict(str(sample_pdf))
    assert validate(doc) == []
    labels = [el.label for page in doc.pages for el in page.elements]
    assert labels == [
        "title", "author", "date", "heading", "paragraph", "paragraph", "heading", "paragraph", "list", "paragraph",
        "list", "heading", "paragraph", "other",
        "heading", "paragraph", "math", "paragraph", "code", "heading", "paragraph", "caption", "table", "paragraph",
        "caption", "table", "paragraph", "other",
        "figure", "caption", "heading", "paragraph", "other",
    ]
    levels = [el.level for page in doc.pages for el in page.elements if el.label == "heading"]
    assert levels == [1, 2, 3, 1, 1, 1]
    # 段落の中の装飾
    deco = next(el for el in doc.pages[0].elements if el.text.startswith("段落の中には"))
    marked = {(r.text, r.bold, r.italic, r.underline) for r in deco.runs if r.bold or r.italic or r.underline}
    assert ("太字の語", True, None, None) in marked
    assert ("下線を引いた語", None, None, True) in marked
