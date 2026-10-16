import json
import shutil

import pytest
import torch

from block_classifier.build_dataset import Segment, build_one, label_of
from block_classifier.dataset import IGNORE, collate
from block_classifier.features import NUM_FEATURES, _font_features, block_features
from block_classifier.model import ModelConfig, build_model
from rule_based.labels import FINE_LABELS
from rule_based.pdf_extract import TextBlock


def text_block(bbox, text="本文", font="Ryumin-Light", size=10.0, page=0) -> TextBlock:
    return TextBlock(page=page, text=text, bbox=bbox, font_size=size, font_name=font, bold=False)


def test_font_features_are_relative_to_body():
    # 本文がゴシック体の文書では、ゴシック体のブロックは「本文と同じ」と表される
    assert _font_features("IPAexGothic", "IPAexGothic")[:2] == [1.0, 1.0]
    # 本文が明朝体なら、ゴシック体は「本文より太い種類」
    assert _font_features("GothicBBB-Medium", "Ryumin-Light")[4] == 1.0
    assert _font_features("CMTT10", "Ryumin-Light")[2] == 1.0  # 等幅
    assert _font_features("CMMI10", "Ryumin-Light")[3] == 1.0  # 数式用


def test_block_features_length():
    features = block_features(text_block((72, 100, 300, 112)), 595, 842, 10.0, "Ryumin-Light", 72)
    assert len(features) == NUM_FEATURES


def test_label_of_picks_the_enclosing_block():
    segments = [
        Segment("heading", page=0, column=0, top=100, bottom=100),
        Segment("paragraph", page=0, column=0, top=130, bottom=170),
    ]
    assert label_of(text_block((72, 90, 300, 104)), segments, 595, False) == "heading"
    assert label_of(text_block((72, 140, 300, 152)), segments, 595, False) == "paragraph"
    assert label_of(text_block((290, 780, 300, 790)), segments, 595, False) == "other"  # ページ番号など


def test_label_of_spanning_segment_does_not_swallow_column():
    """2段組のタイトル（段をまたぐ1行）が、左の段全体を飲み込まないこと"""
    segments = [
        Segment("other", page=0, column=-1, top=80, bottom=80),
        Segment("paragraph", page=0, column=0, top=200, bottom=300),
    ]
    assert label_of(text_block((60, 250, 280, 262)), segments, 595, True) == "paragraph"
    assert label_of(text_block((150, 70, 450, 84)), segments, 595, True) == "other"


def test_collate_pads_and_masks():
    features, labels, mask = collate([
        (torch.ones(2, NUM_FEATURES), torch.tensor([0, 1])),
        (torch.ones(3, NUM_FEATURES), torch.tensor([2, 3, 4])),
    ])  # fmt: skip
    assert features.shape == (2, 3, NUM_FEATURES)
    assert labels[0].tolist() == [0, 1, IGNORE]
    assert mask.tolist() == [[False, False, True], [False, False, False]]


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_padding_does_not_change_predictions(kind):
    """埋め草のブロックを足しても、本物のブロックの出力が変わらないこと（padding mask が効いている）"""
    torch.manual_seed(0)
    model = build_model(ModelConfig(num_features=NUM_FEATURES, num_classes=len(FINE_LABELS), kind=kind)).eval()
    features = torch.rand(1, 4, NUM_FEATURES)
    padded = torch.cat([features, torch.rand(1, 3, NUM_FEATURES)], dim=1)
    mask = torch.tensor([[False] * 4 + [True] * 3])
    with torch.no_grad():
        plain = model(features, torch.zeros(1, 4, dtype=torch.bool))
        with_pad = model(padded, mask)[:, :4]
    assert plain.shape == (1, 4, len(FINE_LABELS))
    assert torch.allclose(plain, with_pad, atol=1e-5)


@pytest.mark.skipif(shutil.which("uplatex") is None, reason="uplatex が無い環境ではスキップ")
def test_build_one_document(tmp_path):
    for sub in ["pdfs", "labels"]:
        (tmp_path / sub).mkdir()
    counts = build_one((0, 123, tmp_path))
    assert counts is not None, "コンパイルに失敗した"
    blocks = json.loads((tmp_path / "labels" / "doc_000000.json").read_text(encoding="utf-8"))["blocks"]
    assert all(block["label"] in FINE_LABELS for block in blocks)
    assert counts.get("heading", 0) > 0 and counts.get("paragraph", 0) > 0
