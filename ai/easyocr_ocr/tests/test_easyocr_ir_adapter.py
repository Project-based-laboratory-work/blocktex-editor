import pytest
from PIL import Image

from common import ir
from easyocr_ocr import ir_adapter
from easyocr_ocr.ir_adapter import extract_ir, to_ir
from image_common.preprocess import STEPS
from easyocr_ocr.presets import PRESETS

WIDTH, HEIGHT = 4284, 5712


def test_to_ir_is_valid():
    raw = [
        ([[531, 352], [972, 352], [972, 512], [531, 512]], "【見出し】", 0.9864),
        ([[810, 485], [3135, 485], [3135, 787], [810, 787]], "《本文の一行目のテキスト》", 0.8516),
    ]
    doc = to_ir(raw, "sample.jpg", WIDTH, HEIGHT, {"preprocess": "none"})

    assert ir.validate(doc) == []
    assert doc.extractor == "easyocr" and doc.structurer is None
    page = doc.pages[0]
    assert page.page == 1 and page.unit == "px" and (page.width, page.height) == (WIDTH, HEIGHT)

    first = page.elements[0]
    assert first.kind == "text" and first.text == "【見出し】"
    assert first.bbox == (0.12395, 0.06162, 0.22689, 0.08964)
    assert first.font_size == 160.0
    assert all(el.label is None for el in page.elements)


def test_vertical_box_uses_width_as_font_size():
    # 縦書き1列（幅40px・高さ400px）
    raw = [([[100, 100], [140, 100], [140, 500], [100, 500]], "縦書き", 0.9)]
    doc = to_ir(raw, "v.png", 1000, 1000, {})
    assert doc.pages[0].elements[0].font_size == 40.0


def test_skips_empty_boxes_and_blank_text():
    raw = [
        ([[10, 10], [10, 10], [10, 20], [10, 20]], "幅ゼロ", 0.9),
        ([[10, 10], [50, 10], [50, 20], [10, 20]], "  ", 0.9),
        ([[10, 10], [50, 10], [50, 20], [10, 20]], "残る", 0.9),
    ]
    doc = to_ir(raw, "x.png", 100, 100, {})
    assert [el.text for el in doc.pages[0].elements] == ["残る"]


def test_accepts_numpy_values():
    np = pytest.importorskip("numpy")
    raw = [(np.array([[0, 0], [50, 0], [50, 10], [0, 10]], dtype=np.int32), "abc", np.float64(0.5))]
    doc = to_ir(raw, "x.png", 100, 100, {})
    assert ir.validate(doc) == []
    assert isinstance(doc.pages[0].elements[0].confidence, float)


def test_upscaled_coordinates_go_back_to_original():
    # 2倍に拡大した画像での座標 → 元の画像（100×100）の座標に戻る
    raw = [([[20, 40], [100, 40], [100, 60], [20, 60]], "abc", 0.9)]
    doc = to_ir(raw, "x.png", 100, 100, {}, scale=2.0)
    el = doc.pages[0].elements[0]
    assert el.bbox == (0.1, 0.2, 0.5, 0.3)
    assert el.font_size == 10.0


def test_paragraph_results_have_no_confidence_or_font_size():
    # 段落全体の箱（高さ60px）。短い辺は段落の高さで、文字の大きさではない
    raw = [([[0, 0], [90, 0], [90, 60], [0, 60]], "段落の1行目\n2行目")]
    doc = to_ir(raw, "x.png", 100, 100, {}, paragraph=True)
    assert ir.validate(doc) == []
    element = ir.to_dict(doc)["pages"][0]["elements"][0]
    assert "confidence" not in element and "font_size" not in element


def test_presets_use_known_steps():
    for name, settings in PRESETS.items():
        assert set(settings) == {"preprocess", "readtext"}, name
        assert all(step in STEPS for step in settings["preprocess"]), name


class FakeReader:
    """EasyOCRの代わり。受け取った画像の大きさと引数を覚えておき、画像の左上1/4に箱を1つ返す。"""

    def readtext(self, image, **kwargs):
        self.shape = image.shape
        self.kwargs = kwargs
        h, w = image.shape[:2]
        return [([[0, 0], [w // 2, 0], [w // 2, h // 4], [0, h // 4]], "テスト", 0.5)]


@pytest.mark.parametrize("preset", list(PRESETS))
def test_extract_ir_with_each_preset(preset, tmp_path, monkeypatch):
    if any(step.startswith(("otsu", "adaptive")) for step in PRESETS[preset]["preprocess"]):
        pytest.importorskip("cv2")  # 二値化は OpenCV を使う
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    reader = FakeReader()
    monkeypatch.setattr(ir_adapter, "get_reader", lambda langs: reader)

    doc = extract_ir(str(image_path), preset)

    assert ir.validate(doc) == []
    assert doc.params["preset"] == preset and doc.params["lang"] == ["ja", "en"]
    assert reader.kwargs == PRESETS[preset]["readtext"]
    page = doc.pages[0]
    # ページの大きさは元の画像のまま。拡大しても bbox は同じ割合になる
    assert (page.width, page.height) == (200, 100)
    assert page.elements[0].bbox == (0.0, 0.0, 0.5, 0.25)


def test_unknown_preset_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        extract_ir(str(tmp_path / "x.png"), "nope")
