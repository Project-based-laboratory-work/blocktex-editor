import random

import pytest

from evaluation.synth_doc import FEATURES, HEADING_COMMANDS, figure_blocks, preamble, random_document, table_blocks, title_blocks


def test_every_block_has_start_and_end_marks():
    tex, blocks = random_document(random.Random(0), "ja")
    for block in blocks:
        assert tex.count(rf"\mk{{b{block.id}s}}") == 1  # 始まりの目印がちょうど1つ
        assert tex.count(rf"\mk{{b{block.id}e}}") == 1  # 終わりの目印がちょうど1つ


def test_heading_command_matches_level():
    _, blocks = random_document(random.Random(0), "ja")
    headings = [b for b in blocks if b.label == "heading"]
    assert headings  # 見出しが1つ以上ある
    for block in headings:
        assert block.tex.startswith("\\" + HEADING_COMMANDS[block.level - 1] + "{")  # レベル2なら \subsection{


def test_english_document_is_ascii_only():
    for seed in range(5):
        tex, _ = random_document(random.Random(seed), "en")
        assert tex.isascii()  # 「英語の文書」に日本語が混ざっていない


def test_same_seed_gives_same_document():
    assert random_document(random.Random(42), "ja")[0] == random_document(random.Random(42), "ja")[0]


def test_decorations_have_marks_and_fit_the_language():
    for lang, allowed in [("ja", {"bold", "underline"}), ("en", {"bold", "italic", "underline"})]:
        decorations = []
        for seed in range(10):
            tex, blocks = random_document(random.Random(seed), lang)
            for block in blocks:
                for name, kind in block.decorations:
                    assert tex.count(rf"\mk{{{name}s}}") == 1 and tex.count(rf"\mk{{{name}e}}") == 1
                    decorations.append(kind)
        assert decorations                  # 10文書あれば、装飾は1つ以上ある
        assert set(decorations) <= allowed  # 日本語に斜体は付けない（jsarticle では見た目が変わらない）


def test_title_blocks():
    blocks = title_blocks(random.Random(0), "ja", 0)
    assert [b.label for b in blocks] == ["title", "author", "date"]
    assert r"\maketitle" in blocks[0].tex and blocks[1].tex == blocks[2].tex == ""  # TeXはタイトルがまとめて持つ


def test_features_put_each_block_into_the_document():
    tex, blocks = random_document(random.Random(0), "ja", {"table", "figure", "math", "code"})
    labels = [b.label for b in blocks]
    for label in ("table", "figure", "math", "code", "caption"):
        assert label in labels
    assert "title" not in labels  # 選ばなかったものは入らない
    assert any(b.files for b in blocks)  or "\\fbox" in tex  # 図は画像か、枠線だけの図


def test_unknown_feature_is_an_error():
    with pytest.raises(ValueError):
        random_document(random.Random(0), "ja", {"tabel"})


def test_table_ground_truth_matches_its_shape():
    table, caption = table_blocks(random.Random(3), "en", 5)
    rows, cols = len(table.cells), len(table.cells[0])
    assert len(table.aligns) == cols
    assert len(table.col_rules) == cols + 1 and len(table.row_rules) == rows + 1  # 境目の数は列数（行数）+1
    assert caption.label == "caption" and caption.id == 6


def test_preamble_options():
    assert ",twocolumn" in preamble(two_column=True)
    assert r"\pagestyle{empty}" in preamble()
    assert r"\mk{pn\thepage s}" in preamble(page_number=True)  # ページ番号の前後に目印
    assert set(FEATURES) >= {"table", "figure", "math", "code", "two_column", "page_number"}


def test_wrapped_column_has_width():
    tables = [table_blocks(random.Random(seed), "ja", 0)[0] for seed in range(30)]
    wrapped = [t for t in tables if t.col_widths is not None]
    assert wrapped  # 30個あれば、折り返す列のある表は1つ以上ある
    for t in wrapped:
        widths = [w for w in t.col_widths if w is not None]
        # (5cm ＋ 左右の余白 6pt×2) ÷ A4の幅（595.28pt）。10pt の文書なので拡大は無し
        assert len(t.col_widths) == len(t.cells[0]) and widths == [0.2582]
        assert r"p{5.0cm}" in t.tex


def test_wide_floats_only_in_two_column():
    wide = [b for seed in range(30) for b in table_blocks(random.Random(seed), "en", 0, two_column=True) + figure_blocks(random.Random(seed), "en", 2, two_column=True)]
    assert any(b.full_width for b in wide) and any(r"\begin{table*}" in b.tex or r"\begin{figure*}" in b.tex for b in wide)
    assert not any(b.full_width for seed in range(30) for b in table_blocks(random.Random(seed), "en", 0))  # 1段組では使わない


def test_header_in_preamble():
    head = preamble(header="Evaluation document")
    assert r"\mk{hd\thepage s}Evaluation document\mk{hd\thepage e}" in head
    assert r"\pagestyle{empty}" not in head
