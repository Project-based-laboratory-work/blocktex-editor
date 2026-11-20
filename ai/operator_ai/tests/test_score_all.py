import csv
from pathlib import Path

import pytest

from common.ir import Element, IRDocument, Page, save_ir
from operator_ai import score_all
from operator_ai.score_all import Line, page_char_errors, parse_setting, reading_order, split_lines


def page(*elements: Element) -> Page:
    return Page(1, 1000, 1000, "px", list(elements))


def text(bbox, s, label=None) -> Element:
    return Element(kind="text", bbox=bbox, text=s, label=label)


HEADING = text((0.1, 0.10, 0.5, 0.14), "1 はじめに", "heading")
PARAGRAPH = text((0.1, 0.20, 0.9, 0.30), "This is a test.\nSecond line here.", "paragraph")
TABLE = Element(kind="table", bbox=(0.1, 0.50, 0.9, 0.70), text="a b\n1 2", label="table")


def test_perfect_reading_has_no_errors():
    truth = page(HEADING, PARAGRAPH)
    pred = page(text(HEADING.bbox, "1 はじめに"), text(PARAGRAPH.bbox, "This is a test.\nSecond line here."))
    c = page_char_errors(pred, truth)
    assert c["errors"] == 0
    assert c["chars"] == len("1はじめにThisisatest.Secondlinehere.")
    assert c["ja_chars"] == 4 and c["extra_chars"] == 0


def test_missing_block_counts_all_its_chars():
    c = page_char_errors(page(text(HEADING.bbox, "1 はじめに")), page(HEADING, PARAGRAPH))
    assert c["errors"] == len("Thisisatest.Secondlinehere.")


def test_text_outside_every_block_is_extra():
    pred = page(text(HEADING.bbox, "1 はじめに"), text((0.0, 0.95, 0.2, 0.99), "Page 3"))
    c = page_char_errors(pred, page(HEADING))
    assert c["extra_chars"] == 5 and c["errors"] == 5


def test_text_inside_table_is_ignored():
    pred = page(text(HEADING.bbox, "1 はじめに"), text((0.2, 0.55, 0.8, 0.65), "a b 1 2"))
    c = page_char_errors(pred, page(HEADING, TABLE))
    assert c["errors"] == 0 and c["chars"] == 5  # 表の文字は正解にも予測にも数えない


def test_paragraph_spanning_two_blocks_is_split_by_line():
    # Tesseract が見出しと本文を1つの段落にまとめても、行ごとに正しいブロックへ入る
    merged = text((0.1, 0.10, 0.9, 0.30), "1 はじめに\nThis is a test.\nSecond line here.")
    c = page_char_errors(page(merged), page(HEADING, PARAGRAPH))
    assert c["errors"] == 0


def test_block_order_does_not_matter():
    truth = page(HEADING, PARAGRAPH)
    pred = page(text(PARAGRAPH.bbox, "This is a test.\nSecond line here."), text(HEADING.bbox, "1 はじめに"))
    assert page_char_errors(pred, truth)["errors"] == 0


def test_reading_order_left_to_right_within_a_row():
    right = Line("world", (0.5, 0.10, 0.9, 0.12))
    left = Line("hello", (0.1, 0.101, 0.4, 0.121))
    below = Line("next", (0.1, 0.13, 0.4, 0.15))
    assert [ln.text for ln in reading_order([below, right, left])] == ["hello", "world", "next"]


def test_split_lines_skips_blank_lines():
    lines = split_lines([text((0, 0, 1, 0.3), "a\n\nb")])
    assert [ln.text for ln in lines] == ["a", "b"]


def test_parse_setting():
    assert parse_setting("tesseract:invert") == ("tesseract", "invert")
    with pytest.raises(ValueError):
        parse_setting("easyocr:no_such_preset")
    with pytest.raises(ValueError):
        parse_setting("invert")


def make_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "tex").mkdir()
    (root / "truth").mkdir()
    (root / "tex" / "doc_000000.tex").write_text("\\documentclass[twocolumn]{jsarticle}\n", encoding="utf-8")
    truth = IRDocument(
        source="doc_000000_p1_v0.png",
        extractor="image_dataset",
        pages=[page(HEADING, PARAGRAPH)],
        params={"doc": "doc_000000", "style": "screenshot", "dpi": 120.0, "dark_mode": {"background": [0, 0, 0]}},
    )
    save_ir(truth, str(root / "truth" / "doc_000000_p1_v0.ir.json"))
    return root


def test_main_writes_csv_and_reuses_ocr(tmp_path, monkeypatch):
    calls = []

    def fake_ocr(engine, preset, image_path):
        calls.append((engine, preset))
        doc = IRDocument(source="x.png", extractor=engine, pages=[page(text(HEADING.bbox, "1 はじめに"))])
        return score_all.OcrResult(doc, doc)

    monkeypatch.setattr(score_all, "run_ocr", fake_ocr)
    dataset = make_dataset(tmp_path / "ds")
    args = [str(dataset), "--settings", "tesseract:base", "tesseract:invert", "--out", str(tmp_path / "out")]
    csv_path = score_all.main(args)

    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == score_all.COLUMNS
    assert [r["setting"] for r in rows] == ["tesseract:base", "tesseract:invert"]
    row = rows[0]
    assert row["error"] == ""
    assert row["dark_mode"] == "True" and row["crop"] == "False" and row["columns"] == "2"
    assert float(row["cer_page"]) > 0  # 本文を読んでいないので取りこぼしがある
    assert row["sec_per_page"] != ""

    score_all.main(args)  # 2回目は保存した結果を使う
    assert len(calls) == 2


def test_main_records_ocr_errors(tmp_path, monkeypatch):
    def broken(engine, preset, image_path):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(score_all, "run_ocr", broken)
    dataset = make_dataset(tmp_path / "ds")
    csv_path = score_all.main([str(dataset), "--settings", "tesseract:base", "--out", str(tmp_path / "out")])
    with open(csv_path, encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["error"] == "RuntimeError: tesseract not found" and row["cer_page"] == ""
