"""書体を自由にしたとき（PDF読み込みの許可リストの段階1）のための判定のテスト。

書体の名前（CMTT・CMMI など）に頼らず、文字の送り幅・斜体の並び・ベースラインの間隔で決める関数を確かめる。
最後に、書体を変えた小さな文書を実際に PDF にして確かめる（uplatex が無い環境では飛ばす）。
"""
import shutil
import subprocess
from pathlib import Path

import pymupdf
import pytest

from hybrid_extractor.extract import (
    EQUATION_NUMBER,
    extract_blocks,
    fixed_pitch_fonts,
    is_equation_number,
    is_line_number,
    mark_char_kinds,
    page_lines,
    row_baseline,
)


def ch(c, x0, width, font="F", size=10.0, flags=0, baseline=10.0):
    return {"c": c, "bbox": (x0, baseline - 8, x0 + width, baseline + 2), "font": font, "size": size, "flags": flags,
            "baseline": baseline}


def prop(c):
    """ふつうの（等幅でない）書体の文字の幅。i・l は細く、m・w は太い"""
    return {"i": 2.8, "l": 2.8, "t": 3.3, "m": 7.8, "w": 7.2, "E": 6.1, "=": 7.8}.get(c, 4.6)


def chars_of(text, width_of, font, size=10.0, flags=0, x0=0.0, baseline=10.0):
    out = []
    for c in text:
        w = width_of(c)
        out.append(ch(c, x0, w, font, size, flags, baseline))
        x0 += w
    return out


def test_fixed_pitch_fonts_by_advance_width():
    # 等幅: どの文字も 0.6em。ふつうの書体: i は細く m は太い。数字だけの書体は決めない（数字はふつうの書体でも幅がそろう）
    mono = chars_of("while queue:", lambda c: 6.0, "NimbusMonL-Regu")
    prop = chars_of("milling", lambda c: {"i": 2.8, "l": 2.8, "m": 8.3}.get(c, 5.0), "NimbusRomNo9L-Regu")
    digits = chars_of("2024", lambda c: 5.0, "Digits")
    result = fixed_pitch_fonts(mono + prop + digits)
    assert result == {"NimbusMonL-Regu": True, "NimbusRomNo9L-Regu": False}


def test_mark_char_kinds_italic_letters_and_words():
    # mathptmx: 数式の x・y は本文の斜体と同じ書体。短い並びは数式、長い並び（斜体の単語）は数式ではない
    italic = "NimbusRomNo9L-ReguItal"
    line = {"chars": [*chars_of("x", prop, italic, flags=2), *chars_of(" = ", prop, "NimbusRomNo9L-Regu"),
                      *chars_of("ab", prop, italic, flags=2, x0=30),
                      *chars_of(" word", prop, "NimbusRomNo9L-Regu", x0=40),
                      *chars_of("italic", prop, italic, flags=2, x0=70)]}
    mark_char_kinds([line])
    marks = [(c["c"], c["math_char"], c["math_like_char"]) for c in line["chars"] if c["c"] != " "]
    assert marks[0] == ("x", True, True)
    assert marks[1] == ("=", False, True)     # 記号は行の数え方（math_like）だけ
    assert marks[2:4] == [("a", True, True), ("b", True, True)]
    assert marks[4] == ("w", False, False)
    assert not any(c["math_char"] for c in line["chars"][-6:])   # 「italic」は単語


def test_mark_char_kinds_mono_is_not_math():
    # コードの = や < は数式の記号に数えない
    line = {"chars": chars_of("k < n; k++", lambda c: 6.0, "LMMono10-Regular")}
    mark_char_kinds([line])
    assert all(c["mono_char"] and not c["math_like_char"] for c in line["chars"])


def test_row_baseline_ignores_scripts():
    row = {"chars": [ch("x", 0, 5, baseline=100.0), ch("2", 5, 3, size=7.0, baseline=96.0), ch("y", 8, 5, baseline=100.0)],
           "bbox": (0, 90, 13, 102)}
    assert row_baseline(row) == 100.0


def test_line_number_and_equation_number():
    code = {"text": "    total = 0", "chars": chars_of("total = 0", lambda c: 5.25, "LMMono10-Regular", x0=60),
            "bbox": (60, 2, 107, 12), "size": 10.0}
    mark_char_kinds([code])
    number = {"text": "2", "chars": [ch("2", 10, 3.5, "CMR7", size=7.0)], "bbox": (10, 4, 13.5, 12), "size": 7.0}
    assert is_line_number(number, code)
    assert not is_line_number(code, number)
    math = {"text": "E = mc", "chars": chars_of("E=mc", prop, "CMMI10", x0=200), "bbox": (200, 2, 224, 12),
            "size": 10.0}
    mark_char_kinds([math])
    eq = {"text": "(1)", "chars": chars_of("(1)", prop, "CMR10", x0=500), "bbox": (500, 2, 512, 12), "size": 10.0}
    assert EQUATION_NUMBER.match("(2.3)")
    assert is_equation_number(eq, math)
    assert not is_equation_number(eq, code)   # コードの行の右の (1) はつながない


FONT_DOC = r"""\documentclass[uplatex,dvipdfmx]{jsarticle}
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{listings}
\begin{document}
本文の段落。Body text with \textit{italic words}.

\begin{lstlisting}[basicstyle=\ttfamily,columns=fixed,numbers=left,numberstyle=\tiny]
def mean(xs):
    total = 0
    for x in xs:
        total += x
    return total
\end{lstlisting}
\end{document}
"""


@pytest.fixture(scope="module")
def font_pdf(tmp_path_factory) -> Path:
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無い")
    work = tmp_path_factory.mktemp("fonts")
    (work / "doc.tex").write_text(FONT_DOC, encoding="utf-8")
    subprocess.run(["uplatex", "-interaction=nonstopmode", "doc.tex"], cwd=work, check=True, capture_output=True)
    subprocess.run(["dvipdfmx", "-q", "doc.dvi"], cwd=work, check=True, capture_output=True)
    return work / "doc.pdf"


def test_times_code_is_one_block_with_indent(font_pdf):
    # times の等幅（NimbusMonL）は名前に Mono が無く、文字の枠が低いので行のすき間が大きく見える。
    # 行番号も字下げの深い行ではコードから離れる。それでもコードは1つのブロックになり、字下げが戻る
    lines = page_lines(pymupdf.open(font_pdf)[0])
    code_chars = [c for line in lines for c in line["chars"] if c["c"] in "defmantol"]
    assert code_chars and all(c["mono_char"] for c in code_chars if "Mon" in c["font"])
    blocks = [b for page in extract_blocks(str(font_pdf)) for b in page.blocks]
    code = [b for b in blocks if b.mono_ratio >= 0.6]
    assert len(code) == 1
