"""図の範囲を切り取って画像ブロックに付ける（common/figures.py）のテスト"""
import io

import pymupdf
from PIL import Image

from common.blocks import ir_to_document
from common.figures import attach_figures, crop_figures, figure_elements
from common.ir import Element, IRDocument, Page


def labeled(unit: str, width: float, height: float, figures: list[tuple[float, float, float, float]]) -> IRDocument:
    page = Page(page=1, width=width, height=height, unit=unit, elements=[
        Element(kind="text", bbox=(0.1, 0.05, 0.9, 0.08), text="本文の段落です。", label="paragraph"),
        *[Element(kind="figure", bbox=b, label="figure") for b in figures],
        Element(kind="text", bbox=(0.1, 0.9, 0.9, 0.93), text="図1 結果", label="caption"),
    ])
    return IRDocument(source="input", extractor="test", pages=[page], structurer="test")


def png_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def test_crop_pdf_figure_keeps_the_drawing(tmp_path):
    # A4 のページの、上から 1/2〜3/4・左右 1/4〜3/4 に赤い四角（線で描いた図）
    pdf = pymupdf.open()
    page = pdf.new_page(width=595, height=842)
    page.draw_rect(pymupdf.Rect(149, 421, 446, 631), color=(1, 0, 0), fill=(1, 0, 0))
    path = tmp_path / "input.pdf"
    pdf.save(path)

    doc = labeled("pt", 595, 842, [(0.25, 0.5, 0.75, 0.75)])
    figures = crop_figures(str(path), doc)
    assert [f.name for f in figures] == ["figure-p1-1.png"]
    w, h = png_size(figures[0].png)
    # 200dpi で 297pt × 210pt（＋ 余白）→ 825 × 583 ピクセルくらい
    assert 820 <= w <= 850 and 580 <= h <= 610
    # 真ん中は赤（図の中身が写っている）
    image = Image.open(io.BytesIO(figures[0].png)).convert("RGB")
    assert image.getpixel((w // 2, h // 2)) == (255, 0, 0)


def test_crop_image_and_skip_tiny_figures(tmp_path):
    image = Image.new("RGB", (1000, 800), "white")
    path = tmp_path / "page.png"
    image.save(path)
    doc = labeled("px", 1000, 800, [(0.2, 0.2, 0.6, 0.5), (0.5, 0.6, 0.505, 0.61)])   # 2つ目は小さすぎる
    figures = crop_figures(str(path), doc)
    assert figures[1] is None
    # 400 × 240 ピクセル ＋ 上下左右に 0.4% の余白
    assert png_size(figures[0].png) == (408, 246)


def test_attach_figures_sets_src_in_order():
    doc = labeled("pt", 595, 842, [(0.2, 0.2, 0.6, 0.4), (0.2, 0.5, 0.6, 0.7)])
    document = ir_to_document(doc)
    images = [b for b in document["blocks"] if b["type"] == "image"]
    assert len(images) == len(figure_elements(doc)) == 2
    assert all(b["src"] == "" for b in images)   # 切り取る前は空（エディタで選び直す）

    from common.figures import Figure
    attach_figures(document, [Figure("figure-p1-1.png", b""), None])
    assert [b["src"] for b in images] == ["figure-p1-1.png", ""]   # 切り取れなかった図は空のまま
