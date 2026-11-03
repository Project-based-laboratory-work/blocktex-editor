"""pdf_extract の文字の装飾・表の構造（揃え・罫線・列幅）の判定。テスト用のPDFは reportlab で作る。"""

import pytest
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rule_based.pdf_extract import _column_align, _rule_flags, extract
from rule_based.rule_based_classifier import classify_page

pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))

BODY = ParagraphStyle("body", fontName="HeiseiMin-W3", fontSize=10, leading=15)
CAPTION = ParagraphStyle("caption", parent=BODY, alignment=TA_CENTER)
TITLE = ParagraphStyle("title", parent=BODY, fontSize=18, leading=24, alignment=TA_CENTER)
WRAP_WIDTH = 60 * mm


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "tables.pdf"
    booktabs = Table(
        [["項目", "区分", "金額"], ["交通費", "A", "1,200"], ["宿泊費", "BB", "15,000"], ["雑費", "C", "300"]],
        style=TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 1, colors.black),
            ]
        ),
    )
    no_rules = Table(
        [["甲", "100"], ["乙乙", "2000"], ["丙", "30"]],
        style=TableStyle([("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"), ("ALIGN", (1, 0), (1, -1), "RIGHT")]),
    )
    long_text = "赤くて丸い果物。長い説明を入れて、セルの中で折り返すかどうかを確かめる。"
    grid = Table(
        [["名前", "説明"], ["りんご", Paragraph(long_text, BODY)], ["みかん", "短い"]],
        colWidths=[25 * mm, WRAP_WIDTH],
        style=TableStyle([("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"), ("GRID", (0, 0), (-1, -1), 0.5, colors.black)]),
    )
    story = [
        Paragraph("表と装飾の確認", TITLE),
        Spacer(1, 4 * mm),  # \maketitle と同じく、タイトル・著者・日付の間を空ける
        Paragraph("山田 太郎", CAPTION),
        Spacer(1, 4 * mm),
        Paragraph("2026年11月2日", CAPTION),
        Spacer(1, 10 * mm),
        Paragraph(
            "これは<font name='HeiseiKakuGo-W5'>重要</font>な点で、<u>下線</u>もある。"
            "<font name='Helvetica'>English <font name='Helvetica-Bold'>bold</font> and "
            "<font name='Helvetica-Oblique'>italic</font>.</font>",
            BODY,
        ),
        Spacer(1, 10 * mm),
        Paragraph("表1 費用", CAPTION),
        booktabs,
        Spacer(1, 10 * mm),
        Paragraph("表2 罫線なし", CAPTION),
        no_rules,
        Spacer(1, 10 * mm),
        grid,
        Spacer(1, 10 * mm),
        Paragraph("表1の結果から、本文が表と間違われないことを確かめる。", BODY),
    ]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return extract(str(path))


def test_runs_have_bold_italic_and_underline(pages):
    paragraph = next(b for b in pages[0].text_blocks if b.text.startswith("これは"))
    marked = {run.text.strip(): (run.bold, run.italic, run.underline) for run in paragraph.runs if run.text.strip()}
    assert marked["重要"] == (True, None, None)  # 和文のゴシック体は太字
    assert marked["下線"] == (None, None, True)
    assert marked["bold"] == (True, None, None)
    assert marked["italic"] == (None, True, None)
    assert "".join(run.text for run in paragraph.runs) == paragraph.text


def test_block_without_marks_has_no_runs(pages):
    plain = next(b for b in pages[0].text_blocks if b.text.startswith("表1の結果"))
    assert plain.runs is None


def test_tables_without_vertical_rules_are_found_below_captions(pages):
    tables = sorted(pages[0].table_blocks, key=lambda t: t.bbox[1])
    assert len(tables) == 3  # 本文の「表1の結果から…」は表にしない
    booktabs, no_rules, grid = tables

    assert booktabs.cells == [["項目", "区分", "金額"], ["交通費", "A", "1,200"], ["宿泊費", "BB", "15,000"], ["雑費", "C", "300"]]
    assert booktabs.aligns == ["l", "c", "r"]
    assert booktabs.col_rules == [False] * 4
    assert booktabs.row_rules == [True, True, False, False, True]

    assert no_rules.cells == [["甲", "100"], ["乙乙", "2000"], ["丙", "30"]]
    assert no_rules.aligns == ["l", "r"]
    assert no_rules.row_rules == [False] * 4 and no_rules.col_rules == [False] * 3

    assert grid.col_rules == [True] * 3 and grid.row_rules == [True] * 4
    assert "\n" in grid.cells[1][1]
    assert grid.col_widths[0] is None and grid.col_widths[1] == pytest.approx(WRAP_WIDTH, abs=1.0)


def test_title_author_date_are_classified(pages):
    labels = {block.text: block.kind for block in classify_page(pages[0]) if block.text}
    assert labels["表と装飾の確認"] == "title"
    assert labels["山田 太郎"] == "author"
    assert labels["2026年11月2日"] == "date"
    assert labels["表1 費用"] == "caption"


@pytest.mark.parametrize(
    ("extents", "expected"),
    [
        ([(10, 30), (10, 50), (10, 40)], "l"),
        ([(20, 40), (10, 50), (15, 45)], "c"),
        ([(30, 50), (10, 50), (20, 50)], "r"),
        ([(10, 30), (40, 60)], "l"),  # どのセルも同じ幅なら決められないので左寄せ
        ([(10, 30)], "l"),
    ],
)
def test_column_align(extents, expected):
    assert _column_align(extents) == expected


def test_rule_flags_needs_long_enough_lines():
    rows = [(100.0, 110.0), (120.0, 130.0)]
    segments = [
        (95.0, 0.0, 100.0),  # 上端の外側の線（表の幅いっぱい）
        (115.0, 0.0, 30.0),  # 行の間の短い線（下線など）は罫線にしない
        (131.0, 0.0, 60.0),  # 下端の線（2本に分かれていても足して判定）
        (131.0, 60.0, 100.0),
    ]
    assert _rule_flags(rows, segments, (0.0, 100.0)) == [True, False, True]
