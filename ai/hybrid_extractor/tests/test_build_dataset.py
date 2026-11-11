"""学習データづくり（build_dataset.py・doc_gen.py）のテスト"""
import random
import shutil

import pytest

from common.ir import Element, IRDocument, Page
from hybrid_extractor.build_dataset import make_doc, match_labels, overlap_ratio
from hybrid_extractor.doc_gen import random_document


def test_overlap_ratio():
    # 内側の箱の面積のうち、外側と重なる割合
    assert overlap_ratio((0, 0, 2, 2), (0, 0, 4, 4)) == 1.0
    assert overlap_ratio((0, 0, 2, 2), (1, 0, 4, 4)) == 0.5
    assert overlap_ratio((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_match_labels():
    gt = IRDocument(source="x", extractor="ground_truth", pages=[Page(1, 100, 100, "pt", [
        Element(kind="text", bbox=(0.1, 0.1, 0.9, 0.2), text="a", label="heading", level=2),
        Element(kind="text", bbox=(0.1, 0.3, 0.9, 0.6), text="b", label="paragraph"),
    ])])
    assert match_labels(gt, 1, (0.1, 0.1, 0.5, 0.2)) == ("heading", 2, 1.0)
    assert match_labels(gt, 1, (0.1, 0.35, 0.9, 0.45)) == ("paragraph", None, 1.0)
    label, level, ratio = match_labels(gt, 1, (0.1, 0.7, 0.9, 0.8))  # どの正解とも重ならない
    assert label is None and ratio == 0.0


def test_random_document_is_reproducible():
    # 同じ種なら同じ文書
    a, _, _ = random_document(random.Random(5), "ja")
    b, _, _ = random_document(random.Random(5), "ja")
    assert a == b


def test_make_doc(tmp_path):
    if shutil.which("uplatex") is None:
        pytest.skip("uplatex が無い")
    (tmp_path / "docs").mkdir()
    (tmp_path / "pdfs").mkdir()
    info = make_doc((0, 3, "en", str(tmp_path)))
    assert info is not None
    assert (tmp_path / "docs" / "00000.json").exists()
    assert (tmp_path / "pdfs" / "00000.pdf").exists()
    labeled = sum(n for label, n in info["counts"].items() if label != "(none)")
    assert labeled >= sum(info["counts"].values()) * 0.8  # ほとんどのブロックに正解が付く
    # 区切りの専門家のデータ: 隣り合う2行の組は、行の数 - 1 個
    import json
    doc = json.loads((tmp_path / "docs" / "00000.json").read_text(encoding="utf-8"))
    page = doc["pages"][0]
    assert len(page["pairs"]) == len(page["rows"]) - 1
    assert {"same", "rule_same", "prev_group_kind"} <= set(page["pairs"][0])
