"""
TeXで作ったPDFから、正解のラベル付きIRを作る。

目印（marks.py）から各ブロックの縦の範囲（1行目〜最後の行のベースライン）が分かるので、
その範囲にある文字の枠を全部包む枠を、そのブロックの正解の枠にする。
"""
from dataclasses import dataclass

# 上付き・下付きの文字は行のベースラインから少しずれるので、その分だけ範囲を広げる（pt）。
# 行と行の間は16pt前後あるので、5ptなら隣の行の文字は入らない
BASELINE_TOLERANCE = 5.0

PtBox = tuple[float, float, float, float]  # (x0, y0, x1, y1)。pt、上からの距離（正規化する前）


@dataclass
class Glyph:
    """PDFに印刷された1文字"""
    page: int        # 1始まり
    bbox: PtBox
    baseline: float  # 文字が乗っている線の、上からの距離（pt）


def union_bbox(boxes: list[PtBox]) -> PtBox:
    """全部の枠を包む、一番小さい枠"""
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def block_bbox(glyphs: list[Glyph], page: int, top: float, bottom: float) -> PtBox | None:
    """page の、ベースラインが top〜bottom（前後に BASELINE_TOLERANCE の余裕）にある文字を包む枠。文字が無ければ None"""
    boxes = [g.bbox for g in glyphs if top - BASELINE_TOLERANCE <= g.baseline <= bottom + BASELINE_TOLERANCE and g.page == page]
    if not boxes:
        return None
    return union_bbox(boxes)
