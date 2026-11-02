import json
import sys

from common import ir
from common.blocks import ir_to_document
from common.ir import Element, IRDocument, Page
from image_rule_based import ir_adapter
from image_rule_based.ir_adapter import classify_ir

BODY = "This is a paragraph of body text. " * 4


def sample_doc() -> IRDocument:
    elements = [
        Element(kind="text", bbox=(0.1, 0.20, 0.9, 0.35), text=BODY, font_size=18.4, confidence=0.9),
        Element(kind="text", bbox=(0.4, 0.10, 0.6, 0.13), text="CHAPTER I", font_size=40.0, confidence=0.95),
        Element(kind="table", bbox=(0.1, 0.50, 0.9, 0.70), cells=[["a", "b"], ["1", "2"]]),
        Element(kind="text", bbox=(0.1, 0.72, 0.6, 0.74), text="Table 1 Results", font_size=17.0),
    ]
    page = Page(page=1, width=1240, height=1755, unit="px", elements=elements)
    return IRDocument(source="eng_001.jpg", extractor="easyocr", pages=[page], params={"preset": "base", "group": "lines_to_blocks"})


def test_classify_ir_is_valid_and_in_reading_order():
    labeled = classify_ir(sample_doc())
    assert ir.validate(labeled) == []
    assert labeled.structurer == "image_rule_based" and labeled.extractor == "easyocr"
    assert labeled.source == "eng_001.jpg" and labeled.params["preset"] == "base"

    elements = labeled.pages[0].elements
    assert [e.label for e in elements] == ["heading", "paragraph", "table", "caption"]
    assert [e.kind for e in elements] == ["text", "text", "table", "text"]
    # 元の要素の値を引き継ぐ
    assert elements[0].bbox == (0.4, 0.1, 0.6, 0.13)
    assert elements[0].font_size == 40.0 and elements[0].confidence == 0.95
    assert elements[2].cells == [["a", "b"], ["1", "2"]]


def test_labeled_ir_becomes_editor_blocks():
    document = ir_to_document(classify_ir(sample_doc()))
    assert [b["type"] for b in document["blocks"]] == ["heading", "paragraph", "table"]
    assert document["blocks"][0]["text"] == "CHAPTER I"
    assert document["blocks"][2]["caption"] == "Table 1 Results"


def test_main_writes_files(tmp_path, monkeypatch):
    src = tmp_path / "eng_001.base.group.ir.json"
    ir.save_ir(sample_doc(), str(src))
    monkeypatch.setattr(sys, "argv", ["ir_adapter", str(src), "-o", str(tmp_path / "out")])
    ir_adapter.main()

    labeled = ir.load_ir(str(tmp_path / "out" / "eng_001.base.group.labeled.json"))
    assert ir.validate(labeled) == []
    blocks = json.loads((tmp_path / "out" / "eng_001.base.group.blocks.json").read_text(encoding="utf-8"))
    assert blocks["title"] == "eng_001.jpg" and len(blocks["blocks"]) == 3


def test_main_stdout(tmp_path, monkeypatch, capsys):
    src = tmp_path / "x.ir.json"
    ir.save_ir(sample_doc(), str(src))
    monkeypatch.setattr(sys, "argv", ["ir_adapter", str(src), "--stdout", "--title", "テスト"])
    ir_adapter.main()

    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"labeled", "document"}
    assert out["document"]["title"] == "テスト"
    assert out["labeled"]["structurer"] == "image_rule_based"
