import json

import pytest

from common.ir import (
    IR_VERSION,
    Element,
    IRDocument,
    Page,
    Run,
    from_dict,
    load_ir,
    normalize_bbox,
    save_ir,
    to_dict,
    validate,
)


def sample_doc() -> IRDocument:
    page = Page(
        page=1,
        width=595.0,
        height=842.0,
        unit="pt",
        elements=[
            Element(kind="text", bbox=(0.1, 0.1, 0.5, 0.13), text="1. はじめに", font_size=18.0, bold=True),
            Element(kind="table", bbox=(0.1, 0.2, 0.9, 0.4), cells=[["項目", "値"], ["工数", "15時間"]]),
            Element(
                kind="text",
                bbox=(0.1, 0.5, 0.9, 0.52),
                text="これは重要な点です。",
                runs=[Run("これは"), Run("重要", bold=True, underline=True), Run("な点です。")],
            ),
            Element(kind="table", bbox=(0.1, 0.6, 0.9, 0.7), cells=[["項目", "値"], ["工数", "15"]], aligns=["l", "r"]),
        ],
    )
    return IRDocument(source="sample.pdf", extractor="test", pages=[page], params={"scale": 1.5})


def test_normalize_bbox_divides_by_page_size_and_clamps():
    assert normalize_bbox((59.5, 84.2, 297.5, 421.0), 595.0, 842.0) == (0.1, 0.1, 0.5, 0.5)
    assert normalize_bbox((-3.0, 10.0, 600.0, 20.0), 595.0, 842.0)[0::2] == (0.0, 1.0)  # はみ出しは0〜1に収める


def test_save_and_load_round_trip(tmp_path):
    doc = sample_doc()
    path = tmp_path / "ir.json"
    save_ir(doc, str(path))
    assert load_ir(str(path), check=True) == doc


def test_none_fields_are_omitted_from_json():
    data = to_dict(sample_doc())
    table = data["pages"][0]["elements"][1]
    assert set(table) == {"kind", "bbox", "cells"}  # 取れなかった値はキーごと無い
    assert "structurer" not in data
    runs = data["pages"][0]["elements"][2]["runs"]
    assert runs == [{"text": "これは"}, {"text": "重要", "bold": True, "underline": True}, {"text": "な点です。"}]
    assert from_dict(json.loads(json.dumps(data))) == sample_doc()


def test_version_1_is_read_as_current_version():
    data = to_dict(sample_doc())
    data["ir_version"] = 1
    doc = from_dict(data)
    assert doc.ir_version == IR_VERSION
    assert validate(doc) == []


def test_validate_accepts_correct_document():
    assert validate(sample_doc()) == []


def test_validate_reports_broken_document():
    doc = sample_doc()
    doc.pages[0].page = 0
    doc.pages[0].elements = [
        Element(kind="txt", bbox=(72.0, 90.0, 200.0, 112.0), text="正規化忘れ"),
        Element(kind="text", bbox=(0.5, 0.1, 0.2, 0.3)),
        Element(kind="text", bbox=(0.1, 0.1, 0.2, 0.2), text="x", label="headding", level=1),
        Element(kind="text", bbox=(0.1, 0.1, 0.2, 0.2), text="x", label="heading", level=4),
    ]
    problems = "\n".join(validate(doc))
    for expected in ["1始まり", "不明なkind", "0〜1の範囲外", "左右が逆", "text がない", "不明なlabel", "level は label='heading'", "level は1〜3"]:
        assert expected in problems


def test_validate_reports_broken_runs_and_aligns():
    doc = sample_doc()
    doc.pages[0].elements = [
        Element(kind="text", bbox=(0.1, 0.1, 0.2, 0.2), text="重要な点", runs=[Run("重要", bold=True), Run("")]),
        Element(kind="text", bbox=(0.1, 0.1, 0.2, 0.2), text="x", runs=[Run("x", italic="yes")]),
        Element(kind="figure", bbox=(0.1, 0.1, 0.2, 0.2), runs=[Run("x")]),
        Element(kind="table", bbox=(0.1, 0.1, 0.2, 0.2), cells=[["a", "b", "c"]], aligns=["l", "center"]),
        Element(kind="text", bbox=(0.1, 0.1, 0.2, 0.2), text="x", aligns=["l"]),
    ]
    problems = "\n".join(validate(doc))
    for expected in [
        "runs[1] の text が空",
        "要素の text と違う",
        "italic は true / false",
        "runs は kind='text'",
        "不明なalign 'center'",
        "cells の列数 3",
        "aligns は kind='table'",
    ]:
        assert expected in problems


def test_new_labels_are_valid():
    doc = sample_doc()
    doc.pages[0].elements = [
        Element(kind="text", bbox=(0.3, 0.1, 0.7, 0.13), text="日本語と言語類型", label=label)
        for label in ("title", "author", "date")
    ]
    assert validate(doc) == []


def test_load_with_check_raises_on_problem(tmp_path):
    doc = sample_doc()
    doc.ir_version = 999
    path = tmp_path / "ir.json"
    save_ir(doc, str(path))
    with pytest.raises(ValueError, match="ir_version"):
        load_ir(str(path), check=True)
