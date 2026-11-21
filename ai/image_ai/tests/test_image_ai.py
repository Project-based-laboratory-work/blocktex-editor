import numpy as np
from PIL import Image

from image_ai.pipeline import (Block, Line, align_mark, assign_lines, block_text, code_text, margin_others, math_lines,
                               orphan_blocks, reading_order)  # fmt: skip


def test_lines_go_to_the_smallest_block_containing_their_center():
    boxes = [("paragraph", (0, 0, 100, 100)), ("caption", (10, 10, 60, 30))]
    blocks, orphans = assign_lines(boxes, [Line((12, 12, 50, 28), "表1 結果"), Line((10, 50, 90, 60), "本文"), Line((200, 0, 220, 10), "外")])
    assert [l.text for l in blocks[0].lines] == ["本文"] and [l.text for l in blocks[1].lines] == ["表1 結果"]
    assert [l.text for l in orphans] == ["外"]


def test_orphan_lines_become_paragraphs_and_margin_lines_become_page_numbers():
    height = 1000
    lines = [Line((480, 960, 520, 975), "3"), Line((100, 400, 900, 420), "見落とした段落の1行目"),
             Line((100, 425, 600, 445), "2行目"), Line((100, 700, 900, 720), "離れた行")]  # fmt: skip
    blocks = orphan_blocks(lines, height)
    margin_others(blocks, height)
    assert [(b.label, block_text(b.lines)) for b in blocks] == [
        ("other", "3"), ("paragraph", "見落とした段落の1行目\n2行目"), ("paragraph", "離れた行")]


def test_margin_others_relabels_small_single_lines_but_keeps_big_titles_and_tables():
    height = 1000
    body = [Block("paragraph", (100, 300 + 30 * k, 900, 320 + 30 * k), [Line((100, 300 + 30 * k, 900, 320 + 30 * k), "本文")]) for k in range(5)]
    number = Block("math", (480, 950, 520, 970), [Line((480, 950, 520, 970), "12")])
    header = Block("date", (700, 100, 900, 118), [Line((700, 100, 900, 118), "評価用の文書")])
    title = Block("title", (200, 60, 800, 110), [Line((200, 60, 800, 110), "大きなタイトル")])
    table = Block("table", (100, 920, 900, 990), [])
    margin_others(body + [number, header, title, table], height)
    assert (number.label, header.label, title.label, table.label) == ("other", "other", "title", "table")
    assert all(b.label == "paragraph" for b in body)


def test_block_text_joins_rows_top_to_bottom_and_words_left_to_right():
    lines = [Line((50, 20, 90, 30), "world"), Line((0, 0, 40, 10), "first"), Line((0, 21, 45, 31), "hello")]
    assert block_text(lines) == "first\nhello world"


def test_code_indent_from_left_offset():
    # 1文字の幅は 10 画素。2行目は 40 画素右にずれている → 空白4つ
    lines = [Line((0, 0, 120, 10), "for i in ran"), Line((40, 15, 140, 25), "total += i")]
    assert code_text(lines) == "for i in ran\n    total += i"


def _b(label, box):
    return Block(label, box, [])


def test_two_column_reading_order_uses_spanning_blocks_as_sections():
    width = 100
    title = _b("title", (10, 0, 90, 5))
    l1, l2, r1, r2 = _b("p", (5, 10, 45, 20)), _b("p", (5, 30, 45, 40)), _b("p", (55, 10, 95, 20)), _b("p", (55, 30, 95, 40))
    wide = _b("table", (10, 50, 90, 60))
    l3, r3 = _b("p", (5, 70, 45, 80)), _b("p", (55, 65, 95, 75))
    order = reading_order([r3, wide, r1, l2, title, l3, r2, l1], width)
    assert order == [title, l1, l2, r1, r2, wide, l3, r3]


def test_one_column_page_keeps_top_to_bottom_even_with_short_left_blocks():
    width = 100
    heading, body, date = _b("heading", (5, 0, 30, 5)), _b("p", (5, 10, 95, 30)), _b("date", (70, 35, 95, 40))
    assert reading_order([date, body, heading], width) == [heading, body, date]


def test_math_lines_split_at_wide_gaps_but_not_inside_a_fraction():
    ink = np.zeros((100, 50), dtype=bool)
    ink[5:15, 5:45] = True  # 1行目の分子
    ink[17:18, 5:45] = True  # 分数の横線（2画素の空白）
    ink[20:30, 5:45] = True  # 分母
    ink[60:75, 5:45] = True  # 2行目（30画素の空白）
    image = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    parts = math_lines(image, line_height=40)
    assert len(parts) == 2 and parts[0].height < 40


def test_align_mark_goes_before_the_first_top_level_relation():
    assert align_mark(r"y=\frac{a}{b}") == r"y&=\frac{a}{b}"
    assert align_mark(r"f_{x=1}\leq2") == r"f_{x=1}&\leq2"
    assert align_mark("abc") == "abc"
