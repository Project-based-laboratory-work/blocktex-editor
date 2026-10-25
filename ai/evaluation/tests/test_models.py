import pytest

from common.ir import validate
from evaluation.metrics import iou
from evaluation.models import ModelSpec, load_entry, load_registry, predict


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
    spec = ModelSpec(id="x", name="x", entry="evaluation.metrics:iou", input="image", outputs=["bbox"])
    with pytest.raises(ValueError):  # この中でエラーが起きれば成功、起きなければ失敗
        predict(spec, "dummy.pdf")
