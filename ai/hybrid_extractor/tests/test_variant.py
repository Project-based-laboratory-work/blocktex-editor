"""見た目の制限（doc_gen.Variant）と中身の材料のパック（content.py）のテスト。コンパイルはしない"""
import argparse
import random

import pytest

from hybrid_extractor.content import ALT, DEFAULT, aozora_sentences, build_formulas, gutenberg_sentences
from hybrid_extractor.doc_gen import Variant, add_variant_args, pick, random_document, use_variant, variant_from_args


def test_default_variant_keeps_the_same_documents():
    # 何も指定しなければ、今までと同じ文書（Variant() を渡しても渡さなくても同じ）
    for seed in range(5):
        a = random_document(random.Random(seed), "ja")[0]
        b = random_document(random.Random(seed), "ja", Variant())[0]
        assert a == b


def test_only_and_exclude_restrict_the_style():
    times = ["times", "newtxtext", "mathptmx"]
    for seed in range(20):
        _, _, only = random_document(random.Random(seed), "en", Variant(only={"text_font": times}))
        _, _, excluded = random_document(random.Random(seed), "en", Variant(exclude={"text_font": times}))
        assert only.text_font in times
        assert excluded.text_font not in times


def test_pick_raises_when_nothing_left():
    with use_variant(Variant(only={"cls": ["nothing"]})):
        with pytest.raises(ValueError):
            pick(random.Random(0), "cls", ["jsarticle"])


def test_variant_from_args():
    parser = argparse.ArgumentParser()
    add_variant_args(parser)
    args = parser.parse_args(["--only", "cls=bxjsarticle", "--exclude", "text_font=times,mathptmx", "--content", "alt"])
    v = variant_from_args(args)
    assert v.only == {"cls": ["bxjsarticle"]}
    assert v.exclude == {"text_font": ["times", "mathptmx"]}
    assert v.content == "alt"
    with pytest.raises(SystemExit):
        variant_from_args(parser.parse_args(["--only", "size=10pt"]))


def test_alt_content_uses_other_words():
    tex = "".join(random_document(random.Random(s), "ja", Variant(content="alt"))[0] for s in range(10))
    assert any(h in tex for h in ALT.words["ja"]["headings"])
    assert not any(h in tex for h in DEFAULT.words["ja"]["headings"] if h not in ALT.words["ja"]["headings"]
                   and len(h) >= 3)


def test_build_formulas_filters_and_splits():
    lines = [r"x _ { i } + y _ { i } .",                       # 短い → 文中（末尾の . は外す）
             r"\Gamma ( z + 1 ) = \int _ { 0 } ^ { \infty } d x \, e ^ { - x } x ^ { z } = z \Gamma ( z ) + c",
             r"\mbox { if } x > 0",                               # 対応していない命令 → 使わない
             r"a & = b"]                                          # & は使わない
    inline, display = build_formulas(lines)
    assert inline == ["x _ { i } + y _ { i }"]
    assert len(display) == 1 and display[0].startswith(r"\Gamma")


def test_aozora_and_gutenberg_sentences():
    page = ('<div class="main_text">寺田<ruby><rb>寅彦</rb><rp>（</rp><rt>とらひこ</rt><rp>）</rp></ruby>は'
            '物理学者であり随筆家でもあった人である。短い。<br />［＃改ページ］</div>'
            '<div class="bibliographical_information"></div>').encode("cp932")
    assert aozora_sentences(page) == ["寺田寅彦は物理学者であり随筆家でもあった人である。"]
    text = ("header\n*** START OF THE BOOK ***\nThe flame of a candle is a beautiful thing to study closely. "
            "It burns.\n*** END OF THE BOOK ***")
    assert gutenberg_sentences(text) == ["The flame of a candle is a beautiful thing to study closely."]
