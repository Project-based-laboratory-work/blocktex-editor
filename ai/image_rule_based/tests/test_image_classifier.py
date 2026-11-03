from common.ir import Element, Page
from image_rule_based.image_classifier import body_font_size, classify_page, font_size_of


def el(text, x0, y0, x1, y1, size=None, kind="text"):
    return Element(kind=kind, bbox=(x0, y0, x1, y1), text=text, font_size=size)


def page_of(*elements, width=1000, height=1400):
    return Page(page=1, width=width, height=height, unit="px", elements=list(elements))


BODY = "本文の段落です。" * 10  # 80文字。見出しの上限（50文字）より長い


def test_body_size_is_weighted_median():
    # OCRの推定値は少しずつばらつく。文字数の多い本文の大きさが選ばれる
    assert body_font_size([(18.3, 100), (18.7, 120), (19.1, 90), (30.0, 5)]) == 18.7
    assert body_font_size([]) is None
    assert body_font_size([(20.0, 0)]) is None


def test_font_size_falls_back_to_line_height():
    assert font_size_of("1行目\n2行目\n3行目", (0, 100, 500, 160), None) == 20.0
    assert font_size_of("1行目", (0, 100, 500, 160), 18.0) == 18.0


def test_heading_paragraph_and_caption():
    blocks = classify_page(page_of(
        el("1 はじめに", 0.1, 0.10, 0.4, 0.13, 27.0),
        el(BODY, 0.1, 0.15, 0.9, 0.30, 18.3),
        el(BODY, 0.1, 0.32, 0.9, 0.47, 18.9),
        el("図1 システムの構成", 0.3, 0.50, 0.7, 0.52, 17.0),
        el("Figure 2. Results", 0.3, 0.55, 0.7, 0.57, 17.0),
    ))  # fmt: skip
    assert [b.kind for b in blocks] == ["heading", "paragraph", "paragraph", "caption", "caption"]
    assert blocks[0].page == 0 and blocks[0].bbox == (100.0, 140.0, 400.0, 182.0)


def test_large_but_long_text_is_paragraph():
    blocks = classify_page(page_of(el(BODY, 0.1, 0.1, 0.9, 0.2, 30.0), el(BODY, 0.1, 0.3, 0.9, 0.4, 18.0)))
    assert [b.kind for b in blocks] == ["paragraph", "paragraph"]


def test_small_difference_is_not_heading():
    # 本文の1.3倍に届かない（PDF版と同じしきい値）
    blocks = classify_page(page_of(el("やや大きい", 0.1, 0.1, 0.4, 0.12, 22.0), el(BODY, 0.1, 0.2, 0.9, 0.3, 18.0)))
    assert blocks[0].kind == "paragraph"


def test_bold_short_text_is_heading():
    heading = Element(kind="text", bbox=(0.1, 0.1, 0.4, 0.12), text="概要", font_size=18.0, bold=True)
    assert classify_page(page_of(heading, el(BODY, 0.1, 0.2, 0.9, 0.3, 18.0)))[0].kind == "heading"


def test_numbered_line_is_heading_even_if_not_large():
    # OCRの大きさでは本文と同じでも、番号で始まる短い1行は見出し
    blocks = classify_page(page_of(
        el("4.1 CPUの評価", 0.1, 0.10, 0.4, 0.12, 18.0),
        el("1,1 文書の比較", 0.1, 0.14, 0.4, 0.16, 18.0),  # OCRが「.」を「,」と読み違えたもの
        el("2 Related Work", 0.1, 0.18, 0.4, 0.20, 18.0),
        el(BODY, 0.1, 0.22, 0.9, 0.35, 18.0),
    ))  # fmt: skip
    assert [b.kind for b in blocks] == ["heading", "heading", "heading", "paragraph"]


def test_numbered_sentence_or_number_only_is_not_heading():
    blocks = classify_page(page_of(
        el("1. データを読み込む。", 0.1, 0.10, 0.5, 0.12, 18.0),  # 文の終わり方 → 番号付きの箇条書き
        el("2024", 0.1, 0.14, 0.2, 0.16, 18.0),  # 数字だけ
        el("3 実験\n本文が続く", 0.1, 0.18, 0.5, 0.22, 18.0),  # 2行
        el(BODY, 0.1, 0.25, 0.9, 0.35, 18.0),
    ))  # fmt: skip
    assert [b.kind for b in blocks] == ["paragraph"] * 4


def test_text_inside_table_is_absorbed():
    table = Element(kind="table", bbox=(0.1, 0.5, 0.9, 0.7), cells=[["a", "b"]])
    blocks = classify_page(page_of(el(BODY, 0.1, 0.1, 0.9, 0.3, 18.0), table, el("a b", 0.2, 0.55, 0.5, 0.6, 18.0)))
    assert [b.kind for b in blocks] == ["paragraph", "table"]


def test_two_columns_are_read_left_then_right():
    elements = [
        el("題名", 0.2, 0.05, 0.8, 0.08, 30.0),  # 両方の段にまたがる
        el("右の上" + BODY, 0.52, 0.10, 0.95, 0.40, 18.0),
        el("左の下" + BODY, 0.05, 0.45, 0.48, 0.80, 18.0),
        el("左の上" + BODY, 0.05, 0.10, 0.48, 0.40, 18.0),
        el("右の下" + BODY, 0.52, 0.45, 0.95, 0.80, 18.0),
    ]
    blocks = classify_page(page_of(*elements))
    assert [b.text[:3] for b in blocks] == ["題名", "左の上", "左の下", "右の上", "右の下"]


def test_result_does_not_depend_on_resolution():
    # 同じレイアウトを小さい画像と大きい画像で。長さは割合で決めるので、ラベルも読み順も変わらない
    def run(width, scale):
        elements = [
            el("見出し", 0.2, 0.05, 0.8, 0.08, 30.0 * scale),
            el("右" + BODY, 0.505, 0.10, 0.95, 0.40, 18.0 * scale),  # 中央線から少しはみ出している
            el("左" + BODY, 0.05, 0.10, 0.495, 0.40, 18.0 * scale),
        ]
        return [(b.kind, b.text[:1]) for b in classify_page(page_of(*elements, width=width, height=width * 1.4))]

    assert run(600, 1.0) == run(3000, 5.0) == [("heading", "見"), ("paragraph", "左"), ("paragraph", "右")]
