import random

import torch
from torch import nn

from math_ocr.dataset import IMAGE_HEIGHT, IMAGE_WIDTH, Collator, preprocess
from math_ocr.metrics import edit_distance, evaluate
from math_ocr.model import MathOCRModel, ModelConfig
from math_ocr.synth_data import random_formula, render
from math_ocr.tokenizer import Vocab, detokenize, tokenize
from math_ocr.train import compute_loss


def small_model(vocab_size: int = 20) -> MathOCRModel:
    config = ModelConfig(
        vocab_size=vocab_size, pad_id=0, bos_id=1, eos_id=2,
        d_model=32, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64, dropout=0.0,
    )  # fmt: skip
    return MathOCRModel(config)


def test_tokenize_splits_commands_and_ignores_spaces():
    assert tokenize(r"\frac{a}{2} + \alpha_{1}") == [r"\frac", "{", "a", "}", "{", "2", "}", "+", r"\alpha", "_", "{", "1", "}"]
    assert tokenize(r"\{ x \}") == [r"\{", "x", r"\}"]


def test_detokenize_keeps_command_boundary():
    assert detokenize([r"\sin", "x"]) == r"\sin x"  # \sinx にならない
    assert detokenize([r"\alpha", "+", "1"]) == r"\alpha+1"


def test_vocab_round_trip():
    vocab = Vocab.build([r"\frac{a}{b}", r"x^{2}"])
    ids = vocab.encode(r"\frac{a}{b}")
    assert ids[0] == vocab.bos_id and ids[-1] == vocab.eos_id
    assert vocab.decode(ids[1:]) == r"\frac{a}{b}"
    assert vocab.encode("z")[1] == vocab.stoi["<unk>"]  # 語彙に無いトークン


def test_generated_formulas_are_renderable():
    rng = random.Random(0)
    for _ in range(50):
        render(random_formula(rng))  # mathtext が描画できない書き方を生成していれば ValueError になる


def test_preprocess_shape_and_range():
    x = preprocess(render(r"\frac{a}{b}"))
    assert x.shape == (1, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert 0.0 <= x.min() and x.max() <= 1.0 and x.max() > 0.5  # 文字（=1付近）がある


def test_collator_pads_to_longest():
    images, seqs = Collator(pad_id=0)([
        (torch.zeros(1, 4, 4), torch.tensor([1, 5, 2])),
        (torch.zeros(1, 4, 4), torch.tensor([1, 5, 6, 7, 2])),
    ])  # fmt: skip
    assert images.shape == (2, 1, 4, 4)
    assert seqs.tolist() == [[1, 5, 2, 0, 0], [1, 5, 6, 7, 2]]


def test_model_output_shape():
    model = small_model()
    logits = model(torch.rand(2, 1, IMAGE_HEIGHT, IMAGE_WIDTH), torch.randint(3, 20, (2, 7)))
    assert logits.shape == (2, 7, 20)


def test_causal_mask_hides_future_tokens():
    """位置 t の出力は t より後ろのトークンに影響されない（カンニングしていない）ことを確認する"""
    model = small_model().eval()
    images = torch.rand(1, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    a = torch.tensor([[1, 5, 6, 7, 8]])
    b = torch.tensor([[1, 5, 6, 9, 10]])  # 位置3以降だけ変える
    with torch.no_grad():
        la, lb = model(images, a), model(images, b)
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)
    assert not torch.allclose(la[:, 3:], lb[:, 3:])


def test_can_overfit_tiny_batch():
    """固定の4サンプルを暗記できる＝勾配が流れ、1つずらしとマスクが正しい"""
    torch.manual_seed(0)
    model = small_model()
    images = torch.rand(4, 1, IMAGE_HEIGHT, IMAGE_WIDTH)
    seqs = torch.tensor([[1, 3, 4, 5, 2], [1, 6, 7, 2, 0], [1, 8, 2, 0, 0], [1, 9, 10, 11, 2]])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(150):
        loss = compute_loss(model, criterion, images, seqs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    generated = model.generate(images, max_len=6)
    for pred, ref in zip(generated.tolist(), seqs.tolist()):
        ref_body = ref[: ref.index(2) + 1]
        assert pred[: len(ref_body)] == ref_body


def test_metrics():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    result = evaluate([["a", "b"], ["x"]], [["a", "b"], ["y", "z"]])
    assert result["exact_match"] == 0.5
    assert result["token_error_rate"] == 2 / 4
