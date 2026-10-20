from typing import get_args

import pytest

from common import ir
from rule_based import labels
from rule_based.ir_adapter import classify_ir, extract_ir, labels_to_ir
from rule_based.labels import LabeledBlock, save_labels
from rule_based.sample_data import generate_sample_pdf


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    generate_sample_pdf.OUTPUT_PATH = str(path)
    generate_sample_pdf.build()
    return str(path)


def test_label_definitions_match_common():
    assert labels.FINE_LABELS == ir.LABELS
    assert list(get_args(labels.Source)) == ir.LABEL_SOURCES


def test_extract_ir_is_valid(sample_pdf):
    doc = extract_ir(sample_pdf)
    assert ir.validate(doc) == []
    assert doc.source == "sample.pdf" and doc.structurer is None
    page = doc.pages[0]
    assert page.page == 1 and page.unit == "pt" and round(page.width) == 595
    kinds = [el.kind for el in page.elements]
    assert kinds.count("table") == 1 and kinds.count("figure") == 1
    table = next(el for el in page.elements if el.kind == "table")
    assert table.cells == [["項目", "値"], ["工数", "15時間"], ["担当", "加茂大地"]]
    assert all(el.label is None for el in page.elements)


def test_classify_ir_is_valid_and_labeled(sample_pdf):
    doc = classify_ir(sample_pdf)
    assert ir.validate(doc) == []
    elements = doc.pages[0].elements
    assert all(el.label is not None for el in elements)
    headings = [el for el in elements if el.label == "heading"]
    assert headings[0].text == "第1章 サンプル文書" and headings[0].font_size is not None


def test_labels_to_ir(sample_pdf, tmp_path):
    blocks = [
        LabeledBlock(page=0, bbox=(72.0, 100.0, 520.0, 130.0), label="heading", text="1. はじめに", source="manual"),
        LabeledBlock(page=0, bbox=(100.0, 320.0, 400.0, 500.0), label="table", text=None, source="auto"),
    ]
    label_json = tmp_path / "labels.json"
    save_labels("sample.pdf", blocks, str(label_json))

    doc = labels_to_ir(str(label_json), sample_pdf)
    assert ir.validate(doc) == []
    heading, table = doc.pages[0].elements
    assert heading.label == "heading" and heading.label_source == "manual" and heading.kind == "text"
    assert table.kind == "table" and table.text is None
    assert heading.bbox == ir.normalize_bbox(blocks[0].bbox, doc.pages[0].width, doc.pages[0].height)
