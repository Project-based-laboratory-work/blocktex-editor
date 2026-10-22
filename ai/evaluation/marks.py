"""
TeXに埋め込んだ目印（PDFの名前付きリンク先）を読む。

目印は \\mk{名前} で埋め込み、dvipdfmx -C 0x0010 で消さずにPDFに残す。
名前は b{ブロック番号}s（始まり）と b{ブロック番号}e（終わり）。
"""
from dataclasses import dataclass

import pymupdf


@dataclass
class Mark:
    page: int  # 1始まり（IRと同じ）
    x: float   # 左端からの距離（pt）
    y: float   # 上端からの距離（pt。IRと同じ向き）。文字のベースライン（文字が乗っている線）の高さ


def read_marks(pdf_path: str) -> dict[str, Mark]:
    """PDFの目印を {名前: Mark} で返す。PDFの座標（左下が原点、ページは0始まり）をIRの向きに直す。"""
    marks = {}
    with pymupdf.open(pdf_path) as doc:
        for name, dest in doc.resolve_names().items():
            page_index = dest["page"]
            x, y = dest["to"]
            height = doc[page_index].rect.height
            marks[name] = Mark(page=page_index + 1, x=x, y= height - y)
    return marks
