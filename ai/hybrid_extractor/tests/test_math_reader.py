"""数式を読む係（math_reader.py）のテスト。作り物の文字で部品を確かめ、サンプルの PDF で全体を確かめる"""
import shutil
import subprocess
from pathlib import Path

import pytest

from common.ir import Run
from hybrid_extractor.extract import runs_with_math
from hybrid_extractor.math_reader import (
    FallbackReader,
    PdfTextReader,
    RuleReader,
    balance_braces,
    inline_math_spans,
    make_reader,
)

SAMPLE_TEX = Path(__file__).parent.parent / "sample_data" / "sample.tex"


class NoDrawings:
    """分数の線の無いページ（RuleReader は page.get_drawings() だけを使う）"""

    def get_drawings(self):
        return []


def g(c: str, x: float, size: float = 10.0, baseline: float = 100.0, font: str = "CMMI10") -> dict:
    """1文字。幅は文字サイズの半分、高さは文字サイズ"""
    return {"c": c, "bbox": (x, baseline - size * 0.75, x + size * 0.5, baseline + size * 0.25), "font": font,
            "size": size, "flags": 0, "baseline": baseline}


def read(glyphs: list[dict]) -> str | None:
    return RuleReader().read(NoDrawings(), glyphs)


def test_superscript_and_subscript():
    # m c² … 小さくて上にずれた 2 は上付き。a_i … 小さくて下にずれた i は下付き
    assert read([g("m", 0), g("c", 5), g("2", 10, size=7, baseline=96)]) == "mc^{2}"
    assert read([g("a", 0), g("i", 5, size=7, baseline=101.6)]) == "a_{i}"


def test_consecutive_scripts_are_grouped():
    assert read([g("x", 0), g("1", 5, size=7, baseline=96), g("0", 8.5, size=7, baseline=96)]) == "x^{10}"


def test_sum_with_limits():
    # ∑ の真上の n が上限、真下の i=1 が下限
    sigma = {"c": "∑", "bbox": (10, 85, 25, 105), "font": "CMEX10", "size": 10.0, "flags": 0, "baseline": 95}
    upper = g("n", 14, size=7, baseline=82)
    lower = [g("i", 11, size=7, baseline=112), g("=", 14, size=7, baseline=112, font="CMR7"),
             g("1", 18, size=7, baseline=112, font="CMR7")]
    glyphs = [g("y", 0), g("=", 5, font="CMR10"), sigma, upper, *lower, g("x", 28)]
    assert read(glyphs) == r"y=\sum_{i=1}^{n}x"


def test_greek_and_functions():
    glyphs = [g("l", 0, font="CMR10"), g("o", 5, font="CMR10"), g("g", 10, font="CMR10"), g("α", 16)]
    assert read(glyphs) == r"\log \alpha"


def test_equation_number_is_dropped():
    glyphs = [g("E", 0), g("=", 6, font="CMR10"), g("m", 12),
              g("(", 200, font="CMR10"), g("1", 205, font="CMR10"), g(")", 210, font="CMR10")]
    assert read(glyphs) == "E=m"


def test_sqrt_without_bar_is_left_to_ocr():
    # 根号の横線が見つからなければ読まない（rule+ocr なら OCR に回る）
    assert read([g("√", 0, font="CMSY10"), g("x", 6)]) is None


def test_stacked_scripts_are_sub_then_sup():
    # e の右に上付き 2 と下付き 0 が重なっている → TeX の書き方どおり e_{0}^{2}
    assert read([g("e", 0), g("2", 5, size=7, baseline=96), g("0", 5.2, size=7, baseline=101.6)]) == "e_{0}^{2}"


def test_inline_sum_scripts_are_on_the_right():
    # 文中の ∑ は、上限・下限が右横に添字として付く
    sigma = {"c": "∑", "bbox": (0, 91, 10, 103), "font": "CMSY10", "size": 10.0, "flags": 0, "baseline": 100}
    glyphs = [sigma, g("n", 10.5, size=7, baseline=95), g("i", 10.5, size=7, baseline=102),
              g("=", 13, size=7, baseline=102, font="CMR7"), g("1", 16, size=7, baseline=102, font="CMR7"), g("x", 21)]
    assert read(glyphs) == r"\sum_{i=1}^{n}x"


def test_fallback_reader():
    class Fixed:
        name = "fixed"

        def read(self, page, glyphs):
            return "OCR"

    reader = FallbackReader(RuleReader(), Fixed())
    assert reader.read(NoDrawings(), [g("x", 0)]) == "x"                         # ルールで読めた
    assert reader.read(NoDrawings(), [g("√", 0, font="CMSY10")]) == "OCR"         # 読めなければ OCR


def test_pdf_text_reader():
    assert PdfTextReader().read(NoDrawings(), [g("c", 5), g("m", 0), g("2", 10, size=7)]) == "m c 2"


def test_make_reader_needs_checkpoint_for_ocr(tmp_path):
    assert make_reader("none") is None
    assert make_reader("rule").name == "rule"
    with pytest.raises(FileNotFoundError):
        make_reader("rule+ocr", tmp_path / "missing.pt")


def test_balance_braces():
    assert balance_braces(r"\frac{a}{b") == r"\frac{a}{b}"
    assert balance_braces("a}b{c") == "ab{c}"


# ---- 文中の数式 ----


def text_chars(text: str, font: str = "Mincho") -> list[dict]:
    return [g(c, i * 10.0, font=font) for i, c in enumerate(text)]


def test_inline_math_spans():
    # 「式 E = mc² です」: E・m・c が数式のフォント。間の「 = 」と後ろの小さい 2 も数式に入る
    chars = text_chars("式 ") + [g("E", 30), g(" ", 40, font="CMR10"), g("=", 50, font="CMR10"), g(" ", 60, font="CMR10"),
                                 g("m", 70), g("c", 80), g("2", 90, size=7, baseline=96, font="CMR7")] + text_chars("です")
    assert inline_math_spans(chars) == [(2, 9)]


def test_inline_math_spans_include_function_name():
    # 「sin x」の sin は立体（数式のフォントではない）だが、数式に入れる
    chars = text_chars("式 ") + [g(c, 20 + k * 5, font="CMR10") for k, c in enumerate("sin")] + [g(" ", 35, font="CMR10"), g("x", 40)]
    assert inline_math_spans(chars) == [(2, 7)]


def test_inline_math_spans_stop_at_words():
    # 「x と y」: 間に日本語があれば別の数式
    chars = [g("x", 0)] + text_chars("と") + [g("y", 20)]
    assert inline_math_spans(chars) == [(0, 1), (2, 3)]


def test_runs_with_math():
    chars = text_chars("ab") + [g("x", 20)] + text_chars("c")
    decos = [(), ("bold",), (), ()]
    runs = runs_with_math(chars, decos, {(2, 3): "x"})
    assert runs == [Run("a"), Run("b", bold=True), Run("x", tex="x"), Run("c")]
    # 読めなかった数式は tex を書かない
    assert runs_with_math(chars, decos, {(2, 3): None})[2] == Run("x")


# ---- サンプルの PDF で全体を確かめる ----


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    if shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None:
        pytest.skip("uplatex / dvipdfmx が無い")
    work = tmp_path_factory.mktemp("sample")
    shutil.copy(SAMPLE_TEX, work / "sample.tex")
    for command in (["uplatex", "-interaction=nonstopmode", "sample.tex"],) * 2 + (["dvipdfmx", "sample.dvi"],):
        subprocess.run(command, cwd=work, check=True, capture_output=True, timeout=120)
    return work / "sample.pdf"


def test_sample_math(sample_pdf):
    from hybrid_extractor.operator_rules import predict

    doc = predict(str(sample_pdf), math="rule")
    elements = [el for page in doc.pages for el in page.elements]
    # 別行立ての数式（sample.tex の align*）。分数・∑ の上下・積分の上下・2行
    math = next(el for el in elements if el.label == "math")
    assert math.tex == r"f(x)=\sum_{i=1}^{n}a_{i}x^{i} \\ \int_{0}^{1}x^{2}dx=\frac{1}{3}"
    # 文中の数式（$E = mc^2$ と $a_i + b_i$）
    para = next(el for el in elements if el.text.startswith("文中に"))
    assert [r.tex for r in para.runs if r.tex] == ["E=mc^{2}", "a_{i}+b_{i}"]
    # math="none" なら読まない
    plain = predict(str(sample_pdf), math="none")
    assert all(el.tex is None for page in plain.pages for el in page.elements)


# ---- 対応させたい記号の一覧（math_ocr v2 が読める記号）でのまとめた確認 ----


def test_rule_reader_on_symbol_list(tmp_path):
    """比較実験（inline_math_eval）と同じ作り方の文書2つで、ルールが文中の数式をほとんど正しく読めること。
    一覧の項目（アクセント・\\binom・\\mathbb・\\dots・\\lim など）が読めなくなったら、ここで気づける"""
    if shutil.which("uplatex") is None:
        pytest.skip("uplatex が無い")
    from hybrid_extractor.inline_math_eval import evaluate, make_case

    (tmp_path / "pdfs").mkdir()
    cases = [make_case((i, 7, ["ja", "en"][i % 2], str(tmp_path))) for i in range(2)]
    report = evaluate(cases, tmp_path, {"rule": make_reader("rule")})
    assert report["read"]["rule"]["exact"] >= 0.9
    assert report["detect"]["recall"] >= 0.8
