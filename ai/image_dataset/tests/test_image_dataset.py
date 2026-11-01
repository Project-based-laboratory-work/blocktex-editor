import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from common.ir import load_ir, validate
from image_dataset.augment import STYLES, augment, crop, gaps, scale, warp
from image_dataset.build import DEFAULT_STYLE_WEIGHTS, build_one, parse_styles

# 縦に並んだ3つのブロック（幅200 × 高さ300 の画像）
BOXES = np.array([[20, 20, 180, 60], [20, 100, 180, 150], [20, 200, 120, 280]], dtype=float)
IMAGE = np.full((300, 200, 3), 255, dtype=np.uint8)


def test_gaps_are_between_blocks():
    assert gaps(BOXES, 300) == [0.0, 10.0, 80.0, 175.0, 300.0]


def test_crop_never_cuts_a_block():
    for seed in range(30):
        image, boxes, keep, params = crop(IMAGE, BOXES, random.Random(seed), min_share=0.3)
        height, width = image.shape[:2]
        assert len(keep) > 0
        l, t, _, _ = params["crop"]
        # 残したブロックは元の位置から切り出しの左上だけずれていて、画像の中に収まっている
        np.testing.assert_allclose(boxes, BOXES[keep] - [l, t, l, t])
        assert (boxes[:, [0, 1]] >= 0).all() and (boxes[:, 2] <= width).all() and (boxes[:, 3] <= height).all()


def test_scale_moves_boxes():
    image, boxes = scale(IMAGE, BOXES, 0.5)
    assert image.shape[:2] == (150, 100)
    np.testing.assert_allclose(boxes, BOXES * 0.5)


def test_warp_without_moving_corners_only_adds_margin():
    corners = np.array([[0, 0], [200, 0], [200, 300], [0, 300]], dtype=float)
    image, boxes = warp(IMAGE, BOXES, corners, margin=10, border=(0, 0, 0))
    assert image.shape[:2] == (320, 220)
    np.testing.assert_allclose(boxes, BOXES + 10, atol=1e-3)


@pytest.mark.parametrize("style", STYLES)
def test_augment_keeps_boxes_inside_image(style):
    for seed in range(5):
        result = augment(IMAGE, [tuple(b) for b in BOXES], style, random.Random(seed))
        height, width = result.image.shape[:2]
        assert len(result.boxes) == len(result.keep) > 0
        for x0, y0, x1, y1 in result.boxes:
            assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert result.params["style"] == style


def test_augment_is_reproducible():
    a = augment(IMAGE, [tuple(b) for b in BOXES], "photo", random.Random(3))
    b = augment(IMAGE, [tuple(b) for b in BOXES], "photo", random.Random(3))
    assert a.params == b.params and a.boxes == b.boxes
    np.testing.assert_array_equal(a.image, b.image)


def test_parse_styles():
    assert parse_styles("clean=1,photo=3") == {"clean": 1.0, "photo": 3.0}
    assert parse_styles("screenshot") == {"screenshot": 1.0}


@pytest.mark.skipif(shutil.which("uplatex") is None or shutil.which("dvipdfmx") is None, reason="uplatex と dvipdfmx が必要")
def test_build_one_writes_valid_truth(tmp_path: Path):
    for sub in ["images", "truth", "tex"]:
        (tmp_path / sub).mkdir()
    counts = build_one((0, 0, tmp_path, 1, DEFAULT_STYLE_WEIGHTS, (72, 100)))
    assert counts is not None and counts["images"] > 0

    truths = sorted((tmp_path / "truth").glob("*.ir.json"))
    assert len(truths) == counts["images"]
    labels = set()
    for truth in truths:
        doc = load_ir(str(truth))
        assert validate(doc) == []
        page = doc.pages[0]
        assert page.unit == "px"
        image = cv2.imread(str(tmp_path / "images" / doc.source))
        assert image.shape[:2] == (page.height, page.width)  # 正解の座標は保存した画像の大きさに合っている
        for el in page.elements:
            assert el.label_source == "auto"
            if el.label == "heading":
                assert el.level in (1, 2, 3)
        labels |= {el.label for el in page.elements}
    assert {"heading", "paragraph"} <= labels
