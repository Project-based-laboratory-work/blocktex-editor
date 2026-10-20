import re
from pathlib import Path

from common.blocks import EDITOR_SCHEMA_VERSION, ir_to_document
from common.ir import Element, IRDocument, Page

MODEL_TS = Path(__file__).resolve().parents[3] / "web" / "src" / "model.ts"


def text(label: str, body: str, y: float = 0.1, **kwargs) -> Element:
    return Element(kind="text", bbox=(0.1, y, 0.9, y + 0.02), text=body, label=label, **kwargs)


def convert(*elements: Element, pages: list[list[Element]] | None = None) -> list[dict]:
    pages = pages or [list(elements)]
    ir_pages = [Page(page=i + 1, width=595.0, height=842.0, unit="pt", elements=els) for i, els in enumerate(pages)]
    return ir_to_document(IRDocument(source="t.pdf", extractor="test", structurer="test", pages=ir_pages))["blocks"]


def test_schema_version_matches_editor():
    match = re.search(r"const SCHEMA_VERSION = (\d+);", MODEL_TS.read_text(encoding="utf-8"))
    assert match, "web/src/model.ts に SCHEMA_VERSION が見つからない"
    assert int(match.group(1)) == EDITOR_SCHEMA_VERSION, "model.ts の版数が上がった。common/blocks.py の変換を見直すこと"


def test_document_has_editor_fields():
    doc = ir_to_document(IRDocument(source="t.pdf", extractor="test", pages=[]))
    assert doc["schemaVersion"] == EDITOR_SCHEMA_VERSION and doc["title"] == "t.pdf" and doc["blocks"] == []
    assert isinstance(doc["id"], str)


def test_heading_levels_follow_font_size():
    blocks = convert(
        text("heading", "1章", font_size=20.0),
        text("heading", "1.1 節", font_size=14.0),
        text("heading", "1.1.1 項", font_size=12.0),
        text("heading", "小見出し", font_size=11.0),
        text("heading", "明示", font_size=20.0, level=2),
    )
    assert [b["level"] for b in blocks] == [1, 2, 3, 3, 2]
    assert blocks[0] == {"id": blocks[0]["id"], "type": "heading", "level": 1, "text": "1章"}


def test_paragraph_joins_wrapped_lines():
    blocks = convert(text("paragraph", "日本語の文が\n折り返されている。"), text("paragraph", "English words\nare wrapped."))
    assert blocks[0]["content"] == [{"type": "text", "text": "日本語の文が折り返されている。"}]
    assert blocks[1]["content"][0]["text"] == "English words are wrapped."


def test_unlabeled_elements_use_kind():
    blocks = convert(Element(kind="text", bbox=(0.1, 0.1, 0.9, 0.2), text="ラベルなし"), Element(kind="figure", bbox=(0.1, 0.3, 0.5, 0.5)))
    assert [b["type"] for b in blocks] == ["paragraph", "image"]


def test_consecutive_list_elements_are_merged():
    blocks = convert(text("list", "・りんご\n・みかんと\nぶどう"), text("list", "・バナナ"))
    assert len(blocks) == 1
    assert blocks[0]["marker"] == "dot"
    assert [item["content"][0]["text"] for item in blocks[0]["items"]] == ["りんご", "みかんとぶどう", "バナナ"]


def test_list_marker_kinds():
    assert convert(text("list", "1. 一つ目\n2. 二つ目"))[0]["marker"] == "number"
    assert convert(text("list", "(1) 一つ目"))[0]["marker"] == "paren"
    assert convert(text("list", "• one"))[0]["marker"] == "bullet"
    assert convert(text("list", "- one"))[0]["marker"] == "dash"


def test_caption_is_attached_to_figure_and_table():
    figure = Element(kind="figure", bbox=(0.2, 0.3, 0.6, 0.5), label="figure")
    table = Element(kind="table", bbox=(0.1, 0.7, 0.9, 0.8), label="table", cells=[["a", "b"], ["c"]])
    blocks = convert(figure, text("caption", "図1: 図", 0.52), text("caption", "表1: 表", 0.65), table)
    assert [b["type"] for b in blocks] == ["image", "table"]
    assert blocks[0]["caption"] == "図1: 図"
    assert blocks[1]["caption"] == "表1: 表"
    assert blocks[1]["columns"] == [{"align": "l"}, {"align": "l"}]
    assert blocks[1]["rows"] == [{"cells": ["a", "b"]}, {"cells": ["c", ""]}]  # 列数を揃える


def test_caption_without_target_stays_paragraph():
    figure = Element(kind="figure", bbox=(0.2, 0.3, 0.6, 0.5), label="figure")
    # 図が次のページにあるときは付けない
    blocks = convert(pages=[[text("caption", "図1: 離れた図")], [figure]])
    assert [b["type"] for b in blocks] == ["paragraph", "image"]


def test_figure_width_is_relative_to_text_area():
    figure = Element(kind="figure", bbox=(0.3, 0.3, 0.5, 0.5), label="figure")
    blocks = convert(Element(kind="text", bbox=(0.1, 0.1, 0.9, 0.2), text="本文", label="paragraph"), figure)
    assert blocks[1]["src"] == ""
    assert blocks[1]["widthRatio"] == 0.25  # 本文の幅 0.8 に対して 0.2


def test_math_uses_tex_if_available_and_other_is_dropped():
    blocks = convert(text("math", "x2", tex="x^2"), text("math", "a + b"), text("other", "- 1 -"))
    assert [b["tex"] for b in blocks] == ["x^2", "a + b"]
    assert all(b["type"] == "math" and b["displayMode"] is True for b in blocks)
