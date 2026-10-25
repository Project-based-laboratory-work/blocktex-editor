import pytest
from PIL import Image

from common import ir
from easyocr_ocr.preprocess import STEPS
from tesseract_ocr import ir_adapter
from tesseract_ocr.evaluate import default_setting
from tesseract_ocr.ir_adapter import OcrRun, extract_ir, is_vertical, join_words, to_elements
from tesseract_ocr.presets import PRESETS


def make_data(words: list[tuple[str, int, int, int, int, int, int, int, float]]) -> dict[str, list]:
    """image_to_data の DICT と同じ形を作る。words は (文字, block, par, line, left, top, width, height, conf)。"""
    keys = ["text", "block_num", "par_num", "line_num", "left", "top", "width", "height", "conf"]
    return {k: [w[n] for w in words] for n, k in enumerate(keys)}


def test_horizontal_paragraph_is_valid():
    data = make_data([
        ("Hello", 1, 1, 1, 10, 10, 50, 20, 96.0),
        ("world", 1, 1, 1, 70, 10, 50, 20, 90.0),
        ("次の", 1, 1, 2, 10, 40, 30, 20, 80.0),
        ("行", 1, 1, 2, 40, 40, 20, 20, 80.0),
    ])
    elements = to_elements(OcrRun("horizontal", "jpn+eng", 3, data), 200, 100)

    assert len(elements) == 1
    el = elements[0]
    assert el.kind == "text" and el.text == "Hello world\n次の行"
    assert el.bbox == (0.05, 0.1, 0.6, 0.6)
    assert el.font_size == 20.0
    assert el.confidence == 0.865


def test_vertical_lines_read_right_to_left_top_to_bottom():
    # 縦書き2列（右の列が1行目）。Tesseract が左の列・下の文字から返しても読み順に並べ直す
    data = make_data([
        ("く", 1, 1, 1, 10, 50, 30, 30, 90.0),
        ("か", 1, 1, 1, 10, 10, 30, 30, 90.0),
        ("い", 1, 1, 2, 60, 50, 30, 30, 90.0),
        ("あ", 1, 1, 2, 60, 10, 30, 30, 90.0),
    ])
    el = to_elements(OcrRun("vertical", "jpn_vert+eng", 5, data), 100, 100)[0]
    assert el.text == "あい\nかく"
    assert el.font_size == 30.0  # 縦書きは列の幅


def test_upscaled_coordinates_go_back_to_original():
    data = make_data([("abc", 1, 1, 1, 20, 40, 80, 20, 90.0)])
    el = to_elements(OcrRun("horizontal", "eng", 3, data), 100, 100, scale=2.0)[0]
    assert el.bbox == (0.1, 0.2, 0.5, 0.3)
    assert el.font_size == 10.0


def test_skips_blank_words_and_splits_paragraphs():
    data = make_data([
        ("", 1, 0, 0, 0, 0, 100, 100, -1.0),  # ブロックの枠
        ("  ", 1, 1, 1, 0, 0, 10, 10, 95.0),
        ("one", 1, 1, 1, 10, 10, 30, 10, 95.0),
        ("two", 2, 1, 1, 10, 50, 30, 10, 95.0),
    ])
    elements = to_elements(OcrRun("horizontal", "eng", 3, data), 100, 100)
    assert [el.text for el in elements] == ["one", "two"]


def test_join_words_spaces_only_between_ascii():
    assert join_words(["Hello", "world"]) == "Hello world"
    assert join_words(["本", "文", "です"]) == "本文です"
    assert join_words(["Tesseract", "で", "OCR"]) == "TesseractでOCR"


def test_mixed_decides_by_line_shape():
    assert is_vertical("mixed", [(0, 0, 20, 200), (30, 0, 50, 180)])
    assert not is_vertical("mixed", [(0, 0, 200, 20)])
    assert is_vertical("vertical", [(0, 0, 200, 20)])


def test_presets_use_known_steps():
    for name, settings in PRESETS.items():
        assert set(settings) == {"preprocess", "tesseract"}, name
        assert all(step in STEPS for step in settings["preprocess"]), name


@pytest.mark.parametrize("dataset, expected", [
    ("eng", ("horizontal", "eng")),
    ("jphorizon", ("horizontal", None)),
    ("jpvertical", ("vertical", None)),
])
def test_default_setting_from_folder_name(dataset, expected):
    assert default_setting(dataset) == expected


def fake_tesseract(calls: list):
    """Tesseract の代わり。画像の左上1/4に単語を1つ返す。縦書きの実行だけ信頼度を高くする。"""

    def run(image, direction, lang, psm, oem):
        calls.append((image.size, direction, lang, psm, oem))
        w, h = image.size
        conf = 95.0 if direction == "vertical" else 40.0
        return OcrRun(direction, lang, psm, make_data([("テスト", 1, 1, 1, 0, 0, w // 2, h // 4, conf)]))

    return run


@pytest.mark.parametrize("preset", list(PRESETS))
def test_extract_ir_with_each_preset(preset, tmp_path, monkeypatch):
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (200, 100), "white").save(image_path)
    calls: list = []
    monkeypatch.setattr(ir_adapter, "run_tesseract", fake_tesseract(calls))
    monkeypatch.setattr(ir_adapter.pytesseract, "get_tesseract_version", lambda: "5.5.0")

    doc = extract_ir(str(image_path), preset, "auto")

    assert ir.validate(doc) == []
    assert doc.extractor == "tesseract" and doc.params["preset"] == preset
    # auto は横・縦の両方で読み、点数の高い縦書きを選ぶ
    assert [c[1] for c in calls] == ["horizontal", "vertical"]
    assert doc.params["chosen_direction"] == "vertical" and doc.params["lang"] == "jpn_vert+eng"
    if "psm" in PRESETS[preset]["tesseract"]:
        assert {c[3] for c in calls} == {PRESETS[preset]["tesseract"]["psm"]}
    page = doc.pages[0]
    # ページの大きさは元の画像のまま。拡大しても bbox は同じ割合になる
    assert (page.width, page.height) == (200, 100)
    assert page.elements[0].bbox == (0.0, 0.0, 0.5, 0.25)


def test_lang_option_overrides_default(tmp_path, monkeypatch):
    image_path = tmp_path / "shot.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    calls: list = []
    monkeypatch.setattr(ir_adapter, "run_tesseract", fake_tesseract(calls))
    monkeypatch.setattr(ir_adapter.pytesseract, "get_tesseract_version", lambda: "5.5.0")

    doc = extract_ir(str(image_path), "base", "horizontal", "eng")
    assert calls == [((40, 40), "horizontal", "eng", 3, 1)]
    assert "direction_scores" not in doc.params


def test_unknown_preset_or_direction_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        extract_ir(str(tmp_path / "x.png"), "nope")
    with pytest.raises(ValueError):
        extract_ir(str(tmp_path / "x.png"), "base", "diagonal")
