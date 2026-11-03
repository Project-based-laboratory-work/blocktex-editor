import csv

import pytest
from PIL import Image

from common.ir import Element, IRDocument, Page
from easyocr_ocr import compare
from easyocr_ocr.compare import Sample, find_samples, page_text, reading_order, summarize


def el(text, x0, y0, x1, y1):
    return Element(kind="text", bbox=(x0, y0, x1, y1), text=text)


def texts(elements):
    return "".join(e.text for e in elements)


def test_horizontal_order_is_top_to_bottom_then_left_to_right():
    # 2行目の右・1行目の右・2行目の左・1行目の左 の順に渡す
    elements = [
        el("え", 0.5, 0.20, 0.9, 0.24),
        el("いう", 0.5, 0.10, 0.9, 0.14),
        el("あ", 0.1, 0.11, 0.4, 0.15),  # 少しずれていても同じ行
        el("お", 0.1, 0.20, 0.4, 0.24),
    ]
    # 1行目「あ」「いう」→ 2行目「お」「え」
    assert texts(reading_order(elements, vertical=False)) == "あいうおえ"


def test_vertical_order_is_right_column_first_then_top_to_bottom():
    # 右の列に「あいう」（上下2つの箱）、左の列に「えお」
    elements = [
        el("えお", 0.10, 0.1, 0.14, 0.6),
        el("う", 0.50, 0.5, 0.54, 0.7),
        el("あい", 0.51, 0.1, 0.55, 0.4),
    ]
    assert texts(reading_order(elements, vertical=True)) == "あいうえお"


def test_page_text_detects_vertical_from_box_shapes():
    vertical = [el("えお", 0.10, 0.1, 0.14, 0.6), el("あいう", 0.50, 0.1, 0.54, 0.7)]
    horizontal = [el("えお", 0.1, 0.5, 0.6, 0.54), el("あいう", 0.1, 0.1, 0.7, 0.14)]
    assert page_text(vertical) == "あいうえお"
    assert page_text(horizontal) == "あいうえお"


def test_find_samples_needs_matching_text(tmp_path):
    for name in ("a.jpg", "a.txt", "b.png", "c.txt"):
        (tmp_path / "jphorizon").mkdir(exist_ok=True)
        (tmp_path / "jphorizon" / name).write_bytes(b"")
    samples = find_samples(tmp_path)
    assert [(s.category, s.image.name) for s in samples] == [("jphorizon", "a.jpg")]
    assert find_samples(tmp_path, ["eng"]) == []


def test_score_counts_errors_like_evaluation(tmp_path):
    truth = tmp_path / "x.txt"
    truth.write_text("記憶もすべて\n色褪せたが", encoding="utf-8")
    sample = Sample("jphorizon", tmp_path / "x.jpg", truth)
    # 2文字違う（憶→億、褪→根）。改行と空白は数えない
    row = compare.score("base", sample, [el("記億もすべて 色根せたが", 0.1, 0.1, 0.9, 0.2)], 1.5)
    assert (row.errors, row.chars) == (2, 11)
    assert row.cer == pytest.approx(2 / 11)
    assert (row.ja_errors, row.ja_chars) == (2, 11) and row.en_chars == 0


def test_summarize_adds_up_errors_before_dividing():
    rows = [
        compare.Row("base", "eng", "a.jpg", 10, 100, 0, 0, 10, 100, 5, 2.0),
        compare.Row("base", "jp", "b.jpg", 0, 10, 0, 10, 0, 0, 5, 4.0),
        compare.Row("up2", "eng", "a.jpg", 5, 100, 0, 0, 5, 100, 5, 6.0),
        compare.Row("up2", "jp", "b.jpg", 0, 10, 0, 10, 0, 0, 5, None),
    ]
    summary = summarize(rows, ["eng", "jp"])
    assert [s["preset"] for s in summary] == ["up2", "base"]  # 全体のCERが小さい順
    base = summary[1]
    assert base["全体"] == pytest.approx(10 / 110)  # 画像ごとのCERの平均（0.05）ではない
    assert base["eng"] == pytest.approx(0.1) and base["jp"] == 0.0
    assert base["英語"] == pytest.approx(0.1) and base["日本語"] == 0.0
    assert base["秒/枚"] == 3.0
    assert summary[0]["秒/枚"] == 6.0  # 時間が分からない画像は平均に入れない


class CountingExtract:
    """extract_ir の代わり。呼ばれた回数を数え、決まった1要素を返す。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, image_path, preset):
        self.calls += 1
        page = Page(page=1, width=100, height=100, unit="px", elements=[el("abc", 0.1, 0.1, 0.5, 0.2)])
        return IRDocument(source="x.png", extractor="easyocr", pages=[page], params={"preset": preset})


def test_run_reuses_saved_results_and_writes_csv(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample" / "eng"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(sample_dir / "a.png")
    (sample_dir / "a.txt").write_text("abd", encoding="utf-8")
    fake = CountingExtract()
    monkeypatch.setattr(compare, "extract_ir", fake)
    samples = find_samples(tmp_path / "sample")
    out = tmp_path / "out"

    rows = compare.run(["base", "up2"], samples, out)
    assert fake.calls == 2
    assert [(r.preset, r.errors, r.chars) for r in rows] == [("base", 1, 3), ("up2", 1, 3)]
    assert (out / "ir" / "eng" / "a.base.ir.json").is_file()
    assert (out / "text" / "eng" / "a.up2.txt").read_text(encoding="utf-8") == "abc"

    # 2回目は保存済みを使うので、読み直さない。時間も前回の値が使われる
    rows_again = compare.run(["base", "up2"], samples, out)
    assert fake.calls == 2
    assert rows_again[0].seconds == rows[0].seconds
    compare.run(["base"], samples, out, force=True)
    assert fake.calls == 3

    compare.write_csv(out / "summary.csv", summarize(rows, ["eng"]))
    with (out / "summary.csv").open(encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    assert records[0]["preset"] == "base" and float(records[0]["全体"]) == pytest.approx(1 / 3)
    assert (out / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # Excel 用のBOM


def test_run_with_group_writes_grouped_text(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample" / "eng"
    sample_dir.mkdir(parents=True)
    Image.new("RGB", (10, 10), "white").save(sample_dir / "a.png")
    (sample_dir / "a.txt").write_text("abc", encoding="utf-8")
    monkeypatch.setattr(compare, "extract_ir", CountingExtract())
    out = tmp_path / "out"

    rows = compare.run(["base"], find_samples(tmp_path / "sample"), out, group=True)
    assert rows[0].errors == 0
    assert (out / "text" / "eng" / "a.base.group.txt").is_file()
    # 保存するIRはまとめる前のもの（--group の有無で読み直さずに比べられるように）
    assert "group" not in (out / "ir" / "eng" / "a.base.ir.json").read_text(encoding="utf-8")
