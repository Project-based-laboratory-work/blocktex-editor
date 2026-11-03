import random

import pytest

from common.ir import validate
from evaluation.ground_truth import (
    Glyph,
    block_bbox,
    block_lines,
    block_runs,
    block_text,
    build_ground_truth,
    compile_tex,
    read_glyphs,
    union_bbox,
)
from evaluation.marks import Mark, read_marks
from evaluation.synth_doc import FEATURES, SourceBlock, document_files, mark, preamble, random_document, title_blocks

# サンプルPDF（sample_data/mark_test.tex）から取った実際の値
GLYPHS = [
    Glyph(page=1, bbox=(75.71, 135.08, 83.39, 149.03), baseline=145.98),    # 見出しの番号「1」
    Glyph(page=1, bbox=(98.79, 133.2, 113.31, 147.72), baseline=145.98),    # 見出し「は」
    Glyph(page=1, bbox=(85.8, 163.28, 95.89, 173.37), baseline=172.16),     # 段落1行目「こ」
    Glyph(page=1, bbox=(75.71, 180.74, 85.8, 190.83), baseline=189.62),     # 段落の最後の行「が」
    Glyph(page=1, bbox=(294.91, 772.32, 300.36, 782.01), baseline=779.89),  # ページ番号「1」
]


def test_union_bbox():
    assert union_bbox([(1, 5, 3, 9), (0, 6, 4, 8)]) == (0, 5, 4, 9)


def test_block_bbox_heading():
    # 見出しの目印は b0s も b0e も y=145.98。目印より左にある番号「1」も入る
    assert block_bbox(GLYPHS, page=1, top=145.98, bottom=145.98) == (75.71, 133.2, 113.31, 149.03)


def test_block_bbox_paragraph_spans_lines():
    # 段落は1行目（172.16）から最後の行（189.62）まで。ページ番号は入らない
    assert block_bbox(GLYPHS, page=1, top=172.16, bottom=189.62) == (75.71, 163.28, 95.89, 190.83)


def test_block_bbox_allows_superscript():
    superscript = Glyph(page=1, bbox=(114.0, 130.0, 118.0, 138.0), baseline=142.0)  # ベースラインが4pt上
    assert block_bbox([*GLYPHS, superscript], page=1, top=145.98, bottom=145.98) == (75.71, 130.0, 118.0, 149.03)


def test_block_bbox_other_page_is_none():
    assert block_bbox(GLYPHS, page=2, top=145.98, bottom=145.98) is None


def test_block_boxes_from_real_pdf(sample_pdf):
    glyphs = read_glyphs(sample_pdf)
    marks = read_marks(sample_pdf)
    heading = block_bbox(glyphs, 1, marks["b0s"].y, marks["b0e"].y)
    paragraph = block_bbox(glyphs, 1, marks["b1s"].y, marks["b1e"].y)
    assert heading == pytest.approx((75.71, 133.2, 156.89, 149.03), abs=0.01)     # 「1 はじめに」
    assert paragraph == pytest.approx((75.71, 163.28, 519.56, 191.73), abs=0.01)  # 2行の段落。ページ番号は入らない


def test_read_glyphs_keeps_spaces(sample_pdf):
    glyphs = read_glyphs(sample_pdf)
    assert len(glyphs) == 83                         # 3-d の81文字 + 段落の中の空白2つ（「PDF の」「2 行」）
    assert any(g.c == " " for g in glyphs)
    assert all(g.page == 1 for g in glyphs)


def test_block_bbox_ignores_spaces():
    glyphs = [
        Glyph(page=1, bbox=(0, 0, 10, 10), baseline=8.0, c="A"),
        Glyph(page=1, bbox=(10, 0, 50, 10), baseline=8.0, c=" "),  # 行末の幅の広い空白
    ]
    assert block_bbox(glyphs, page=1, top=8.0, bottom=8.0) == (0, 0, 10, 10)  # 空白で右に広がらない


def test_block_text_breaks_lines_and_keeps_superscript():
    glyphs = [
        Glyph(page=1, bbox=(0, 0, 1, 1), baseline=10.0, c="A"),
        Glyph(page=1, bbox=(1, 0, 2, 1), baseline=10.0, c=" "),
        Glyph(page=1, bbox=(2, 0, 3, 1), baseline=10.0, c="b"),
        Glyph(page=1, bbox=(3, 0, 4, 1), baseline=10.0, c=" "),   # 行末の空白（strip で消える）
        Glyph(page=1, bbox=(0, 2, 1, 3), baseline=26.0, c="c"),   # 16pt 下 → 次の行
        Glyph(page=1, bbox=(1, 1, 2, 2), baseline=23.0, c="2"),   # 3pt 上の上付き文字 → 同じ行
    ]
    assert block_text(glyphs, page=1, top=10.0, bottom=26.0) == "A b\nc2"


def test_block_text_from_real_pdf(sample_pdf):
    glyphs = read_glyphs(sample_pdf)
    marks = read_marks(sample_pdf)
    assert block_text(glyphs, 1, marks["b0s"].y, marks["b0e"].y) == "1はじめに"  # 番号と題の間は空白の文字ではなく、すき間
    paragraph = block_text(glyphs, 1, marks["b1s"].y, marks["b1e"].y)
    assert paragraph.startswith("これは目印の動作を確かめるための段落です。PDF の中で")
    assert paragraph.count("\n") == 1  # 2行の段落なので、改行は1つ


def test_build_ground_truth_follows_blocks(has_tex, tmp_path):
    tex, blocks = random_document(random.Random(0), "en")
    doc = build_ground_truth(str(compile_tex(tex, tmp_path)), blocks, source="doc.pdf")
    assert validate(doc) == []  # common/ir.py の形式チェックに通る
    elements = [el for page in doc.pages for el in page.elements]
    assert [(el.label, el.level) for el in elements] == [(b.label, b.level) for b in blocks]  # 置いた順（読み順）
    assert all(el.label_source == "auto" for el in elements)
    assert all(el.text for el in elements)  # どの要素にも文字が入っている


def test_build_ground_truth_splits_block_across_pages(has_tex, tmp_path):
    long_text = " ".join(["This sentence makes the paragraph long enough to cross a page."] * 40)
    blocks = [
        SourceBlock(0, "heading", 1, rf"\section{{{mark('b0s')}Long{mark('b0e')}}}"),
        SourceBlock(1, "paragraph", None, r"\vspace*{15cm}" + "\n\n" + f"{mark('b1s')}{long_text}{mark('b1e')}"),
    ]
    tex = preamble("11pt") + "\n\n".join(b.tex for b in blocks) + "\n\\end{document}\n"
    doc = build_ground_truth(str(compile_tex(tex, tmp_path)), blocks, source="long.pdf")
    assert [len(p.elements) for p in doc.pages] == [2, 1]  # 1ページ目: 見出し + 段落の前半、2ページ目: 段落の後半
    first_half = doc.pages[0].elements[1]
    assert first_half.label == "paragraph"
    assert first_half.bbox[3] < 0.9  # ページの下の端（ページ番号の位置）まで伸びていない


def test_block_runs_marks_characters_between_decoration_marks():
    glyphs = [Glyph(page=1, bbox=(x, 0, x + 10, 10), baseline=8.0, c=c) for x, c in zip(range(0, 50, 10), "abcde")]
    lines = block_lines(glyphs, page=1, top=8.0, bottom=8.0)
    # 目印は「b」の左端（x=10）と「d」の左端（x=30）→「bc」が太字
    runs = block_runs(lines, [(Mark(page=1, x=10.0, y=8.0), Mark(page=1, x=30.0, y=8.0), "bold")])
    assert [(r.text, r.bold) for r in runs] == [("a", None), ("bc", True), ("de", None)]
    assert block_runs(lines, []) is None  # 装飾が無ければ runs を省く


def test_build_ground_truth_with_title_and_decoration(has_tex, tmp_path):
    blocks = title_blocks(random.Random(0), "en", 0)
    paragraph = SourceBlock(3, "paragraph", None, "", decorations=[("b3d0", "italic")])
    paragraph.tex = f"{mark('b3s')}Plain words and {mark('b3d0s')}\\textit{{slanted words}}{mark('b3d0e')} here.{mark('b3e')}"
    blocks.append(paragraph)
    tex = preamble("11pt") + "\n\n".join(b.tex for b in blocks if b.tex) + "\n\\end{document}\n"
    doc = build_ground_truth(str(compile_tex(tex, tmp_path)), blocks, source="doc.pdf")
    assert validate(doc) == []  # runs をつなげると text になる、なども確かめられる
    elements = doc.pages[0].elements
    assert [el.label for el in elements] == ["title", "author", "date", "paragraph"]
    italic = [run.text for run in elements[3].runs if run.italic]
    assert italic == ["slanted words"]


@pytest.mark.parametrize("lang, seed", [("ja", 1), ("en", 2)])
def test_build_ground_truth_with_every_feature(has_tex, tmp_path, lang, seed):
    features = set(FEATURES)
    tex, blocks = random_document(random.Random(seed), lang, features)
    pdf = compile_tex(tex, tmp_path, document_files(blocks))
    doc = build_ground_truth(str(pdf), blocks, source="doc.pdf", two_column=True)
    assert validate(doc) == []
    elements = [el for page in doc.pages for el in page.elements]
    by_label = {el.label: el for el in elements}
    for label in ("title", "table", "figure", "math", "code", "caption", "other"):
        assert label in by_label, label

    table = by_label["table"]
    source_table = next(b for b in blocks if b.label == "table")
    assert table.kind == "table" and table.cells == source_table.cells and table.row_rules == source_table.row_rules
    codes = [el for el in elements if el.label == "code"]
    if len(codes) == 1:  # 段・ページをまたがなければ、字下げごとソースのとおり（またぐとPDFの文字から取る）
        assert codes[0].text == next(b for b in blocks if b.label == "code").text
    assert all(el.text for el in codes)
    assert by_label["math"].tex is not None
    assert by_label["figure"].kind == "figure" and by_label["figure"].text is None
    # ヘッダー（柱）は各ページの先頭、ページ番号は各ページの最後。どちらも label="other"
    for page in doc.pages:
        assert page.elements[0].label == "other" and not page.elements[0].text.isdigit()
        assert page.elements[-1].label == "other" and page.elements[-1].text == str(page.page)
    # 2段組なので、段をまたぐもの（タイトル、table* / figure* とそのキャプション、ヘッダー・ページ番号）以外は、
    # 左右どちらかの段に収まる
    wide_labels = {b.label for b in blocks if b.full_width} | {"other"}
    for el in elements:
        if el.label not in wide_labels:
            assert el.bbox[2] <= 0.52 or el.bbox[0] >= 0.48, (el.label, el.bbox)
