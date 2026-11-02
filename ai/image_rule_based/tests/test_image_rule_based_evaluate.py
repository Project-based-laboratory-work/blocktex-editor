import collections

from common.ir import Element, IRDocument, Page
from image_rule_based.evaluate import count_headings


def doc(*labels) -> IRDocument:
    elements = [Element(kind="text", bbox=(0.1, 0.1 * i, 0.9, 0.1 * i + 0.05), text="x", label=label) for i, label in enumerate(labels, 1)]
    return IRDocument(source="a.png", extractor="x", pages=[Page(page=1, width=100, height=100, unit="px", elements=elements)])


def test_count_headings():
    counts = collections.Counter()
    count_headings(doc("heading", "heading", "paragraph", "paragraph"), doc("heading", "paragraph", "heading", "paragraph"), counts)
    assert counts == {"tp": 1, "fp": 1, "fn": 1}
