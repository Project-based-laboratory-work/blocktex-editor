"""
読み込んだPDF・画像から、図（label="figure"）の範囲を切り取って画像にする。

エディタの画像ブロックは、画像ファイルを持てる（src にファイル名、中身はエディタが手元に持つ）。
ラベル付きIRには図の位置（bbox、0〜1）しか無いので、元のPDF・画像をその位置で切り取り、PNG にして添付する。
モデルによらない（図の bbox を出すモデルなら、どれでも画像が付く）。

    figures = crop_figures("input.pdf", labeled)          # [Figure(name, png), ...] 図の並び順（= 画像ブロックの並び順）
    attach_figures(document, figures)                      # エディタ用JSONの画像ブロックの src に名前を入れる

- PDF: その範囲だけを画像にする（page.get_pixmap(clip=...)）。埋め込んだ写真も、線で描いた図も同じように切り取れる
- 画像（png / jpg）: 元の画像をその範囲で切り取る
"""
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.ir import IRDocument, denormalize_bbox

FIGURE_DPI = 200       # PDFの図を画像にする解像度（A4 の幅いっぱいの図で 1650 ピクセルくらい）
MARGIN = 0.004         # 切り取る範囲の上下左右の余白（ページの幅・高さに対する割合）。枠の線が切れないように
MIN_SIZE = 0.01        # これより小さい図（幅か高さがページの1%未満）は切り取らない
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class Figure:
    name: str    # 画像ブロックの src にするファイル名（figure-p1-1.png）。エディタの PDF プレビューでもこの名前で送る
    png: bytes


def figure_elements(doc: IRDocument) -> list[tuple[int, Any]]:
    """図の要素を、ページ・要素の順（= common.blocks.ir_to_document が画像ブロックを作る順）に並べる"""
    return [(page.page, el) for page in doc.pages for el in page.elements if el.label == "figure"]


def padded(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """余白を足した範囲（0〜1）。小さすぎる図は None"""
    x0, y0, x1, y1 = bbox
    if x1 - x0 < MIN_SIZE or y1 - y0 < MIN_SIZE:
        return None
    return (max(0.0, x0 - MARGIN), max(0.0, y0 - MARGIN), min(1.0, x1 + MARGIN), min(1.0, y1 + MARGIN))


def crop_figures(input_path: str, doc: IRDocument) -> list[Figure | None]:
    """図ごとに、切り取った PNG。切り取れなかった図（小さすぎる・ページが無い）は None。並びは figure_elements と同じ"""
    items = figure_elements(doc)
    if not items:
        return []
    if Path(input_path).suffix.lower() in IMAGE_SUFFIXES:
        return _crop_image(input_path, items)
    return _crop_pdf(input_path, items)


def _crop_pdf(pdf_path: str, items: list[tuple[int, Any]]) -> list[Figure | None]:
    import pymupdf

    figures: list[Figure | None] = []
    counts: dict[int, int] = {}
    with pymupdf.open(pdf_path) as pdf:
        for page_no, el in items:
            box = padded(el.bbox)
            if box is None or not 1 <= page_no <= pdf.page_count:
                figures.append(None)
                continue
            page = pdf[page_no - 1]
            clip = pymupdf.Rect(*denormalize_bbox(box, page.rect.width, page.rect.height))
            counts[page_no] = counts.get(page_no, 0) + 1
            png = page.get_pixmap(clip=clip, dpi=FIGURE_DPI, alpha=False).tobytes("png")
            figures.append(Figure(f"figure-p{page_no}-{counts[page_no]}.png", png))
    return figures


def _crop_image(image_path: str, items: list[tuple[int, Any]]) -> list[Figure | None]:
    from PIL import Image

    figures: list[Figure | None] = []
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        for k, (page_no, el) in enumerate(items, start=1):
            box = padded(el.bbox)
            if box is None or page_no != 1:   # 画像は1枚 = 1ページ
                figures.append(None)
                continue
            x0, y0, x1, y1 = denormalize_bbox(box, image.width, image.height)
            out = io.BytesIO()
            image.crop((round(x0), round(y0), round(x1), round(y1))).save(out, format="PNG")
            figures.append(Figure(f"figure-p1-{k}.png", out.getvalue()))
    return figures


def attach_figures(document: dict[str, Any], figures: list[Figure | None]) -> None:
    """エディタ用JSON（ir_to_document の結果）の画像ブロックに、切り取った図の名前を src として入れる（書き換える）。
    画像ブロックは図の要素と同じ順に作られるので、順番で対応させる。切り取れなかった図の src は空のまま（エディタで選び直す）"""
    images = [block for block in document["blocks"] if block["type"] == "image"]
    for block, figure in zip(images, figures):
        if figure is not None:
            block["src"] = figure.name
