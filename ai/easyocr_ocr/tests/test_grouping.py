from common import ir
from common.ir import Element, IRDocument, Page
from easyocr_ocr.grouping import group_blocks, group_document, group_lines, group_page


def el(text, x0, y0, x1, y1, size=None):
    return Element(kind="text", bbox=(x0, y0, x1, y1), text=text, font_size=size, confidence=0.9)


def test_pieces_of_one_line_are_joined():
    # 実際の画像で起きた、1行が途中で2つに割れた箱（右の箱のほうが少し上にずれている）の座標
    pieces = [
        el("二つの箱に分かれて（読み取られた）後半です。", 0.345, 0.197, 0.765, 0.230),
        el("この行は途中で", 0.103, 0.207, 0.339, 0.232),
    ]
    lines = group_lines(pieces)
    assert [line.text for line in lines] == ["この行は途中で二つの箱に分かれて（読み取られた）後半です。"]
    assert lines[0].bbox == (0.103, 0.197, 0.765, 0.232)


def test_space_only_between_latin_words():
    lines = group_lines([el("Frederick", 0.1, 0.1, 0.3, 0.12), el("Douglass", 0.31, 0.1, 0.5, 0.12)])
    assert lines[0].text == "Frederick Douglass"


def test_close_lines_become_one_block():
    lines = [
        el("1行目", 0.1, 0.100, 0.9, 0.120, 20),
        el("2行目", 0.1, 0.125, 0.9, 0.145, 20),
        el("3行目", 0.1, 0.150, 0.8, 0.170, 20),
        # 大きく離れた次の段落
        el("次の段落", 0.1, 0.250, 0.9, 0.270, 20),
    ]
    blocks = group_blocks(lines)
    assert [b.text for b in blocks] == ["1行目\n2行目\n3行目", "次の段落"]
    assert blocks[0].bbox == (0.1, 0.100, 0.9, 0.170)


def test_heading_with_larger_text_stays_separate():
    lines = [el("見出し", 0.1, 0.10, 0.5, 0.14, 40), el("本文", 0.1, 0.145, 0.9, 0.165, 20)]
    assert [b.text for b in group_blocks(lines)] == ["見出し", "本文"]


def test_two_columns_are_not_mixed():
    lines = [
        el("左1", 0.05, 0.10, 0.45, 0.12, 20), el("右1", 0.55, 0.10, 0.95, 0.12, 20),
        el("左2", 0.05, 0.125, 0.45, 0.145, 20), el("右2", 0.55, 0.125, 0.95, 0.145, 20),
    ]  # fmt: skip
    blocks = group_blocks(group_lines(lines))
    assert sorted(b.text for b in blocks) == ["右1\n右2", "左1\n左2"]


def test_vertical_boxes_and_tables_are_left_as_they_are():
    vertical = el("縦書き", 0.80, 0.1, 0.82, 0.6)
    table = Element(kind="table", bbox=(0.1, 0.7, 0.9, 0.9), cells=[["a"]])
    page = Page(page=1, width=100, height=100, unit="px", elements=[el("横", 0.1, 0.1, 0.3, 0.12), vertical, table])
    grouped = group_page(page)
    assert vertical in grouped.elements and table in grouped.elements
    assert len(grouped.elements) == 3


def test_group_document_is_valid_and_recorded():
    page = Page(page=1, width=100, height=100, unit="px", elements=[
        el("1行目", 0.1, 0.100, 0.9, 0.120, 20), el("2行目", 0.1, 0.125, 0.9, 0.145, 20),
    ])  # fmt: skip
    doc = IRDocument(source="x.png", extractor="easyocr", pages=[page], params={"preset": "base"})
    grouped = group_document(doc)
    assert ir.validate(grouped) == []
    assert grouped.params == {"preset": "base", "group": "lines_to_blocks"}
    assert [e.text for e in grouped.pages[0].elements] == ["1行目\n2行目"]
    assert grouped.pages[0].elements[0].font_size == 20
