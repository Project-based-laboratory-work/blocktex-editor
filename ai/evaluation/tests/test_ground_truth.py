from evaluation.ground_truth import Glyph, block_bbox, union_bbox

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
