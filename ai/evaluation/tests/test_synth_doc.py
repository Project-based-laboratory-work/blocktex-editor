import random

from evaluation.synth_doc import HEADING_COMMANDS, random_document


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
