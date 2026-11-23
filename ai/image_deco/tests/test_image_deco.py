import torch
from PIL import Image

from image_deco.dataset import HEIGHT, STRIDE, collate, line_tensor, target
from image_deco.model import MARKS, DecorationTagger, ModelConfig
from image_deco.predict import char_ranges, pieces


def test_target_marks_the_decorated_columns():
    t = target([[8, 24, "bold"], [40, 48, "underline"]], scale=1.0, width=64)
    assert t.shape == (len(MARKS), 64 // STRIDE)
    assert t[MARKS.index("bold")].nonzero().flatten().tolist() == [2, 3, 4, 5]
    assert t[MARKS.index("underline")].nonzero().flatten().tolist() == [10, 11]
    assert t[MARKS.index("italic")].sum() == 0


def test_line_tensor_keeps_aspect_and_height():
    x, scale = line_tensor(Image.new("L", (400, 20), 255))
    assert x.shape[1] == HEIGHT and x.shape[2] % STRIDE == 0 and abs(scale - 2.0) < 0.05


def test_collate_pads_and_marks_valid_positions():
    a = (torch.zeros(1, HEIGHT, 16), torch.zeros(3, 4))
    b = (torch.zeros(1, HEIGHT, 32), torch.ones(3, 8))
    images, targets, valid = collate([a, b])
    assert images.shape == (2, 1, HEIGHT, 32) and valid.tolist() == [[True] * 4 + [False] * 4, [True] * 8]


def test_model_output_has_one_score_per_mark_and_position():
    out = DecorationTagger(ModelConfig()).eval()(torch.zeros(2, 1, HEIGHT, 64))
    assert out.shape == (2, len(MARKS), 64 // STRIDE)


def test_char_ranges_map_fractions_and_snap_english_to_words():
    assert char_ranges("本文の重要な点", [(3 / 7, 5 / 7, "bold")]) == [(3, 5, "bold")]
    # 「important」の途中から「note」の途中まで → 2語全体
    text = "this important note here"
    assert char_ranges(text, [(0.3, 0.7, "underline")]) == [(5, 19, "underline")]


def test_pieces_split_by_decoration():
    assert pieces("abcde", [(1, 3, "bold"), (2, 4, "underline")]) == [
        ("a", frozenset()), ("b", frozenset({"bold"})), ("c", frozenset({"bold", "underline"})),
        ("d", frozenset({"underline"})), ("e", frozenset())]  # fmt: skip
