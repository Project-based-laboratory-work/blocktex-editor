import torch

from common.ir import Element, Page
from image_classifier.dataset import IGNORE, LABEL_TO_ID, collate, target_labels
from image_classifier.features import CROP_HEIGHT, CROP_WIDTH, NUM_FEATURES, body_font_size, crop, element_features
from image_classifier.model import ModelConfig, PageClassifier


def _el(bbox, label=None, text="本文の段落です。", font_size=12.0):
    return Element(kind="text", bbox=bbox, text=text, font_size=font_size, label=label)


def test_target_label_comes_from_the_block_covering_most_of_the_ocr_box():
    truth = [_el((0.1, 0.1, 0.9, 0.2), "heading"), _el((0.1, 0.3, 0.9, 0.6), "paragraph")]
    ocr = [(0.1, 0.1, 0.8, 0.19), (0.1, 0.35, 0.9, 0.5), (0.1, 0.12, 0.9, 0.4)]
    # 3つ目は2つのブロックにまたがり、どちらにも半分以上入っていないので外す
    assert target_labels(ocr, truth) == [LABEL_TO_ID["heading"], LABEL_TO_ID["paragraph"], IGNORE]


def test_features_have_the_declared_length_and_body_size_is_weighted_by_characters():
    page = Page(page=1, width=1000, height=1400, unit="px",
                elements=[_el((0.1, 0.1, 0.5, 0.12), font_size=30, text="見出し"), _el((0.1, 0.2, 0.9, 0.4), text="本文" * 50)])  # fmt: skip
    assert body_font_size(page) == 12.0
    assert len(element_features(page.elements[0], page, 12.0, 0.1)) == NUM_FEATURES


def test_crop_keeps_the_fixed_size():
    ink = torch.zeros(200, 300).numpy()
    ink[50:60, 20:280] = 1.0
    out = crop(ink, (0.05, 0.2, 0.95, 0.35))
    assert out.shape == (CROP_HEIGHT, CROP_WIDTH) and out.max() > 0


def test_collate_pads_and_masks():
    a = (torch.zeros(2, NUM_FEATURES), torch.zeros(2, CROP_HEIGHT, CROP_WIDTH, dtype=torch.uint8), torch.tensor([1, 2]))
    b = (torch.zeros(4, NUM_FEATURES), torch.zeros(4, CROP_HEIGHT, CROP_WIDTH, dtype=torch.uint8), torch.tensor([0, 1, 2, 3]))
    features, crops, labels, padding_mask = collate([a, b])
    assert features.shape == (2, 4, NUM_FEATURES) and crops.shape == (2, 4, CROP_HEIGHT, CROP_WIDTH)
    assert labels[0].tolist() == [1, 2, IGNORE, IGNORE]
    assert padding_mask.tolist() == [[False, False, True, True], [False] * 4]


def test_model_output_shape_with_and_without_image():
    features = torch.randn(2, 5, NUM_FEATURES)
    crops = torch.rand(2, 5, CROP_HEIGHT, CROP_WIDTH)
    padding_mask = torch.tensor([[False] * 5, [False, False, False, True, True]])
    for use_image in (True, False):
        model = PageClassifier(ModelConfig(num_features=NUM_FEATURES, num_classes=12, use_image=use_image)).eval()
        assert model(features, crops, padding_mask).shape == (2, 5, 12)
