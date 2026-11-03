from PIL import Image

from evaluation.images import render_pages


def test_render_pages_makes_one_png_per_page(sample_pdf, tmp_path):
    paths = render_pages(sample_pdf, tmp_path, dpi=72)
    assert [p.name for p in paths] == ["page-1.png"]
    with Image.open(paths[0]) as image:
        width, height = image.size
        assert abs(width - 595) <= 1 and abs(height - 842) <= 1  # 72dpi なら、A4（pt）とほぼ同じ大きさのピクセルになる
