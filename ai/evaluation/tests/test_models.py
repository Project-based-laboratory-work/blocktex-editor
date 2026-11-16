import pytest

from common.ir import IRDocument, Page, validate
from evaluation.metrics import iou
import evaluation.models
from evaluation.models import ModelSpec, load_entry, load_registry, predict, predict_image


def test_registry_has_rule_based():
    spec = load_registry()["rule_based"]
    assert spec.entry == "rule_based.ir_adapter:classify_ir"
    assert spec.input == "pdf"
    assert "label" in spec.outputs


def test_load_entry_finds_function():
    assert load_entry("evaluation.metrics:iou") is iou  # 普通に import したものと同じ関数


def test_predict_with_rule_based(sample_pdf):
    doc = predict(load_registry()["rule_based"], sample_pdf)
    assert validate(doc) == []  # common/ir.py の形式に合っている
    assert doc.structurer == "rule_based"
    assert any(el.label == "heading" for page in doc.pages for el in page.elements)  # 「1 はじめに」は見出しと判定される


def test_predict_rejects_unknown_input():
    spec = ModelSpec(id="x", name="x", entry="evaluation.metrics:iou", input="audio", outputs=["bbox"])
    with pytest.raises(ValueError):  # この中でエラーが起きれば成功、起きなければ失敗
        predict(spec, "dummy.pdf")


def test_every_registered_entry_can_be_found():
    # OCRのライブラリは関数の中で読み込むので、入っていなくても関数は見つかる
    for spec in load_registry().values():
        assert callable(load_entry(spec.entry))
        assert spec.input in ("pdf", "image")


def test_predict_image_model_gets_each_page_as_image(sample_pdf, monkeypatch):
    seen = []

    def fake_ocr(image_path: str, dpi_was_removed: bool = True) -> IRDocument:
        seen.append(image_path)
        return IRDocument(source="x.png", extractor="fake", pages=[Page(page=1, width=827, height=1170, unit="px")])

    monkeypatch.setattr(evaluation.models, "load_entry", lambda entry: fake_ocr)  # 登録簿の関数の代わりに fake_ocr を呼ばせる
    spec = ModelSpec(id="fake", name="fake", entry="x:y", input="image", outputs=["bbox"], params={"dpi": 100})
    doc = predict(spec, sample_pdf)
    assert len(seen) == 1 and seen[0].endswith("page-1.png")  # 1ページのPDFなので1枚。dpi は関数に渡さない
    assert [p.page for p in doc.pages] == [1]
    assert doc.pages[0].unit == "px"


def test_predict_image_passes_the_image_as_is(tmp_path, monkeypatch):
    seen = []

    def fake_ocr(image_path: str) -> IRDocument:
        seen.append(image_path)
        return IRDocument(source="x.png", extractor="fake", pages=[Page(page=1, width=827, height=1170, unit="px")])

    monkeypatch.setattr(evaluation.models, "load_entry", lambda entry: fake_ocr)
    spec = ModelSpec(id="fake", name="fake", entry="x:y", input="image", outputs=["bbox"], params={"dpi": 100})
    image = str(tmp_path / "slide.png")
    doc = predict_image(spec, image)
    assert seen == [image]  # 画像はそのまま1枚渡し、dpi は関数に渡さない
    assert doc.source == "slide.png"
    assert [p.page for p in doc.pages] == [1]


def test_predict_image_rejects_pdf_only_model():
    with pytest.raises(ValueError):  # PDFしか読めないモデルに画像を渡すとエラー
        predict_image(load_registry()["rule_based"], "slide.png")
