import json

import numpy as np
import torch

from common.ir import Element, IRDocument, Page, save_ir
from image_layout.dataset import CLASS_TO_ID, CLASSES, OUT_H, OUT_W, SHRINK, STRIDE, boundary_mask, target_mask
from image_layout.model import LayoutUNet, ModelConfig
from image_layout.postprocess import logits_to_classes, mask_to_blocks, to_ir


def _truth(tmp_path, elements, width=576, height=768):
    path = tmp_path / "t.ir.json"
    doc = IRDocument(source="t.png", extractor="image_dataset",
                     pages=[Page(page=1, width=width, height=height, unit="px", elements=elements)])  # fmt: skip
    save_ir(doc, str(path))
    return str(path)


def test_target_mask_paints_shrunk_boxes(tmp_path):
    # キャンバスと同じ大きさの画像（拡大率1）。四角は出力の画素で x 10〜50, y 20〜30
    bbox = (10 * STRIDE / 576, 20 * STRIDE / 768, 50 * STRIDE / 576, 30 * STRIDE / 768)
    path = _truth(tmp_path, [Element(kind="text", bbox=bbox, text="見出し", label="heading")])
    mask = target_mask(path, 1.0)
    assert mask.shape == (OUT_H, OUT_W)
    ys, xs = np.nonzero(mask == CLASS_TO_ID["heading"])
    assert (xs.min(), ys.min()) == (10 + SHRINK, 20 + SHRINK) and (xs.max(), ys.max()) == (50 - SHRINK, 30 - SHRINK)


def test_boundary_is_the_ring_left_by_shrinking(tmp_path):
    # 上下に接した2つの段落（境目は y=30）。ふちは各四角の外周1画素で、2つの中身の間の1画素がふちになる
    top = (10 * STRIDE / 576, 20 * STRIDE / 768, 50 * STRIDE / 576, 30 * STRIDE / 768)
    bottom = (10 * STRIDE / 576, 30 * STRIDE / 768, 50 * STRIDE / 576, 40 * STRIDE / 768)
    path = _truth(tmp_path, [Element(kind="text", bbox=b, text="本文", label="paragraph") for b in (top, bottom)])
    ring, mask = boundary_mask(path, 1.0), target_mask(path, 1.0)
    assert not (ring.astype(bool) & (mask > 0)).any()
    assert ring[29:32, 30].tolist() == [0, 1, 0] and mask[29:32, 30].tolist() == [CLASS_TO_ID["paragraph"], 0, CLASS_TO_ID["paragraph"]]


def test_boundary_channel_separates_touching_blocks():
    logits = np.zeros((len(CLASSES) + 1, OUT_H, OUT_W), dtype=np.float32)
    logits[CLASS_TO_ID["paragraph"], 10:30, 10:60] = 5.0  # 2つの段落が1つにつながって塗られている
    logits[len(CLASSES), 19:21, 10:60] = 3.0  # その間をふちと判定した
    assert [label for label, _ in mask_to_blocks(logits_to_classes(logits))] == ["paragraph", "paragraph"]


def test_tiny_block_keeps_at_least_one_pixel(tmp_path):
    bbox = (10 * STRIDE / 576, 20 * STRIDE / 768, 11 * STRIDE / 576, 21 * STRIDE / 768)
    path = _truth(tmp_path, [Element(kind="text", bbox=bbox, text="x", label="math")])
    assert (target_mask(path, 1.0) == CLASS_TO_ID["math"]).sum() >= 1


def test_mask_to_blocks_restores_the_shrunk_margin():
    classes = np.zeros((OUT_H, OUT_W), dtype=np.int64)
    classes[21:30, 11:50] = CLASS_TO_ID["table"]
    [(label, box)] = mask_to_blocks(classes)
    assert label == "table" and box == (11 - SHRINK, 21 - SHRINK, 50 + SHRINK, 30 + SHRINK)


def test_to_ir_fills_text_from_ocr_and_marks_tables():
    classes = np.zeros((OUT_H, OUT_W), dtype=np.int64)
    classes[10:20, 10:60] = CLASS_TO_ID["paragraph"]
    classes[40:60, 10:60] = CLASS_TO_ID["table"]
    texts = [((15 * STRIDE / 576, 12 * STRIDE / 768, 50 * STRIDE / 576, 18 * STRIDE / 768), "本文")]
    doc = to_ir(classes, "t.png", 576, 768, 1.0, texts)
    kinds = [(el.kind, el.label, el.text) for el in doc.pages[0].elements]
    assert kinds == [("text", "paragraph", "本文"), ("table", "table", None)]
    json.dumps(doc.pages[0].elements[0].bbox)  # 保存できる値になっている


def test_model_output_is_quarter_resolution():
    for boundary in (False, True):
        model = LayoutUNet(ModelConfig(num_classes=len(CLASS_TO_ID), boundary=boundary)).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, OUT_H * STRIDE, OUT_W * STRIDE))
        assert out.shape == (1, len(CLASS_TO_ID) + boundary, OUT_H, OUT_W)


def test_draw_figure_handles_every_kind_and_both_themes():
    import random

    from PIL import Image

    from image_layout.figures import draw_figure

    for seed in range(60):  # 図の種類・軸・凡例の組み合わせを一通り通す
        for background in (255, 30):
            image = Image.new("L", (300, 200), background)
            draw_figure(image, (20, 20, 280, 180), random.Random(seed))
            assert np.asarray(image)[20:180, 20:280].std() > 0  # 何か描かれている
