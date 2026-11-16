"""pdf_extract の文字の装飾・表の構造（揃え・罫線・列幅）の判定。テスト用のPDFは reportlab で作る。"""

import pytest
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas
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


def _draw_box(pdf: canvas.Canvas, top: float, rows: list[list[tuple[float, str]]]) -> None:
    """Word の網掛け段落と同じ形（外側の背景・行ごとの背景・細い塗りの枠線）の囲みを描く。"""
    x0, x1, height = 85.0, 553.0, 18.0
    page_height = A4[1]
    bottom = top + height * len(rows)
    pdf.setFillGray(0.975)
    pdf.rect(x0, page_height - bottom, x1 - x0, bottom - top, fill=1, stroke=0)
    for i in range(len(rows)):
        pdf.rect(x0 + 5, page_height - (top + (i + 1) * height), x1 - x0 - 10, height, fill=1, stroke=0)
    pdf.setFillGray(0.8)
    pdf.rect(x0, page_height - top, x1 - x0, 0.4, fill=1, stroke=0)
    pdf.rect(x0, page_height - bottom - 0.4, x1 - x0, 0.4, fill=1, stroke=0)
    pdf.rect(x0 - 0.4, page_height - bottom, 0.4, bottom - top, fill=1, stroke=0)
    pdf.rect(x1, page_height - bottom, 0.4, bottom - top, fill=1, stroke=0)
    pdf.setFillGray(0)
    pdf.setFont("HeiseiKakuGo-W5", 10)
    for i, row in enumerate(rows):
        for x, text in row:
            pdf.drawString(x, page_height - (top + i * height + 13), text)


@pytest.fixture(scope="module")
def boxes_page(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "boxes.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _draw_box(pdf, 100, [[(92, "名前:"), (200, "山田")], [(92, "学籍番号:"), (200, "12345")]])
    _draw_box(pdf, 200, [[(92, "int main(void) {")], [(92, '    printf("hello world");')], [(92, "}")]])
    _draw_box(pdf, 300, [[(92, "これは囲みの中の普通の文章で、表ではありません。")], [(92, "二行目も普通の文章が続いています。")]])
    _draw_box(pdf, 400, [[(92, "一行だけの注意書きです。")]])
    pdf.save()
    return extract(str(path))[0]


def test_box_with_columns_becomes_table(boxes_page):
    table, = boxes_page.table_blocks
    assert table.cells == [["名前:", "山田"], ["学籍番号:", "12345"]]
    assert table.col_rules == [True, False, True]  # 背景の端は列の境目にも罫線にもしない
    assert table.row_rules == [True, False, True]


def test_box_without_columns_stays_text(boxes_page):
    assert not boxes_page.figure_blocks  # 背景と枠線が図になると、中の文字が消える
    kinds = [block.kind for block in classify_page(boxes_page)]
    assert kinds.count("table") == 1 and "figure" not in kinds
    texts = {block.text for block in classify_page(boxes_page)}
    assert "int main(void) {" in texts and "一行だけの注意書きです。" in texts


def test_page_background_is_not_a_table(tmp_path):
    path = tmp_path / "background.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFillGray(0.97)
    pdf.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    pdf.setFillGray(0)
    pdf.setFont("HeiseiKakuGo-W5", 10)
    for i in range(5):
        pdf.drawString(60, 700 - i * 15, f"左の段の{i}行目")
        pdf.drawString(330, 700 - i * 15, f"右の段の{i}行目")
    pdf.save()
    assert extract(str(path))[0].table_blocks == []


def _ruled_table(rows: list[list], line_rows: list[int], col_widths: list[float] | None = None) -> Table:
    """外枠と縦線があり、line_rows の行の下にだけ横線がある表。"""
    style = [("FONTNAME", (0, 0), (-1, -1), "HeiseiMin-W3"), ("BOX", (0, 0), (-1, -1), 0.5, colors.black)]
    style += [("LINEAFTER", (0, 0), (-2, -1), 0.5, colors.black)]
    style += [("LINEBELOW", (0, r), (-1, r), 0.5, colors.black) for r in line_rows]
    return Table(rows, colWidths=col_widths, style=TableStyle(style))


def _tables_of(path, *tables: Table) -> list:
    story = []
    for table in tables:
        story += [table, Spacer(1, 15 * mm)]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return sorted(extract(str(path))[0].table_blocks, key=lambda t: t.bbox[1])


def test_rows_without_rules_are_split(tmp_path):
    header_only, no_inner = _tables_of(
        tmp_path / "rows.pdf",
        _ruled_table([["手法", "精度"], ["ルール", "0.81"], ["AI", "0.92"], ["混合", "0.95"]], [0]),
        _ruled_table([["名前", "山田"], ["番号", "12345"], ["所属", "情報工学"]], []),
    )
    assert header_only.cells == [["手法", "精度"], ["ルール", "0.81"], ["AI", "0.92"], ["混合", "0.95"]]
    assert header_only.row_rules == [True, True, False, False, True]  # 分けた行の間に罫線は足さない
    assert no_inner.cells == [["名前", "山田"], ["番号", "12345"], ["所属", "情報工学"]]


def test_wrapped_cells_are_not_split(tmp_path):
    text = "赤くて丸い果物。長い説明を入れて、セルの中で折り返すかどうかを確かめる。"
    one_column, two_columns = _tables_of(
        tmp_path / "wrap.pdf",
        _ruled_table([["名前", "説明"], ["りんご", Paragraph(text, BODY)], ["みかん", "短い"]], [0, 1], [25 * mm, WRAP_WIDTH]),
        _ruled_table([["説明", "備考"], [Paragraph(text, BODY), Paragraph(text, BODY)]], [0], [WRAP_WIDTH, WRAP_WIDTH]),
    )
    assert [len(row) for row in one_column.cells] == [2, 2, 2] and len(one_column.cells) == 3
    assert "\n" in one_column.cells[1][1]
    assert len(two_columns.cells) == 2  # 両方の列が同じ高さで折り返していても、行に分けない
