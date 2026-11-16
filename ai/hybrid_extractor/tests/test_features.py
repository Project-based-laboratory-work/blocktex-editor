"""特徴（features.py）と AI の専門家の部品のテスト。torch が無い環境では、torch を使うテストだけ飛ばす"""
import numpy as np
import pytest

from hybrid_extractor.features import (
    BLOCK_FEATURE_NAMES,
    CHAR_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    block_char_size,
    block_features,
    char_features,
    doc_info,
    pair_features,
)

CTX = {"width": 595.0, "height": 842.0, "body": 10.0, "two_column": False,
       "margins": {"0": 0.13}, "rights": {"0": 0.87}}


def row(x0, y0, x1, y1, text="本文の行", kind="text", size=10.0, font="Mincho", list_head=False):
    return {"bbox": [x0, y0, x1, y1], "text": text, "size": size, "font": font, "kind": kind,
            "list_head": list_head, "indented": False, "column": 0, "n_chars": len(text)}


def test_pair_features_length_and_meaning():
    first = row(0.13, 0.10, 0.87, 0.112)
    prev = row(0.13, 0.12, 0.87, 0.132, text="文の途中で折り返す")
    nxt = row(0.13, 0.14, 0.5, 0.152, text="続きの行。")
    f = pair_features(first, prev, nxt, CTX)
    assert len(f) == len(PAIR_FEATURE_NAMES)
    named = dict(zip(PAIR_FEATURE_NAMES, f))
    # すき間 = (0.14 - 0.132) × 842 ÷ 10 ≈ 0.67（本文の文字 0.67 個分）
    assert named["gap"] == pytest.approx(0.6736, abs=1e-3)
    assert named["prev_end_sentence"] == 0.0  # 前の行は「す」で終わる（文の途中）
    assert named["group_text"] == 1.0


def test_block_features_length_and_neighbors():
    blocks = [
        {"kind": "figure", "bbox": [0.3, 0.2, 0.7, 0.4]},
        {"kind": "text", "bbox": [0.4, 0.41, 0.6, 0.43], "text": "図1 枠線だけの図", "font_size": 9.0,
         "bold": False, "font": "Mincho", "mono_ratio": 0.0, "math_ratio": 0.0, "n_rows": 1, "first_indent": 0.0,
         "list_head": False},
    ]
    page = {"page": 1, "width": 595.0, "height": 842.0, "blocks": blocks}
    info = doc_info([page])
    f = block_features(blocks, 1, page, info)
    assert len(f) == len(BLOCK_FEATURE_NAMES)
    named = dict(zip(BLOCK_FEATURE_NAMES, f))
    assert named["caption_head"] == 1.0
    assert named["prev_is_figure"] == 1.0  # キャプションは図のすぐ隣にある


def test_char_features_length_and_script_shift():
    # x² の 2 は小さく、ベースラインが上にずれる（負）
    chars = [
        {"c": "x", "font": "CMMI10", "size": 10.0, "bbox": [0.10, 0.100, 0.11, 0.112], "baseline": 0.110, "flags": 2},
        {"c": "2", "font": "CMR7", "size": 7.0, "bbox": [0.11, 0.096, 0.115, 0.105], "baseline": 0.104, "flags": 0},
        {"c": " ", "font": "", "size": 0.0, "bbox": None, "baseline": None, "flags": 0},
    ]
    f = char_features(chars, block_char_size(chars), 595.0, 842.0)
    assert [len(v) for v in f] == [len(CHAR_FEATURE_NAMES)] * 3
    named = [dict(zip(CHAR_FEATURE_NAMES, v)) for v in f]
    assert named[0]["math_font"] == 1.0 and named[1]["small"] == 1.0
    assert named[1]["base_shift"] < 0
    assert named[2]["space"] == 1.0


def test_spans_from_probs():
    pytest.importorskip("torch")
    from hybrid_extractor.ai_experts import spans_from_probs

    chars = [{"c": c, "bbox": None if c in " \n" else (0, 0, 1, 1), "size": 10.0} for c in "a x = y\nz b"]
    probs = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=float)
    # 前後の空白を外し、改行で切る: "x = y" と "z"
    assert spans_from_probs(chars, probs) == [(2, 7), (8, 9)]
    # 数式の中の空白の確率が低くても、間が空白・演算子だけならつなぐ: "x = y" は1つ
    probs = np.array([0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=float)
    assert spans_from_probs(chars, probs) == [(2, 7)]


def test_mlp_learns_a_simple_rule():
    pytest.importorskip("torch")
    from hybrid_extractor.nn import binary_scores, predict_binary, train_binary

    # 「1つ目の特徴が正なら はい」を学べること（学習のループが正しく動くかの確認）
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    model, std, _ = train_binary(x[:400], y[:400], x[400:], y[400:], epochs=40, log=False)
    assert binary_scores(predict_binary(model, std, x[400:]), y[400:])["acc"] >= 0.95
