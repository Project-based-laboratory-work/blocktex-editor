"""学習した重み（checkpoints/、train.py で作る）を読み込んで、AIの専門家にする。

    boundary_expert() → 区切りの専門家（extract.Continues の形。group_rows に渡す）
    label_experts()   → ラベルの専門家の一覧（experts.Expert の形。オペレータの AI モードが聞く）
    inline_expert()   → 文中の数式の専門家（experts.InlineMathExpert の形）

どれも、学習データと同じ features.py の関数で特徴を作ってから、モデルに通す。
torch が要るので、このファイルは AI を使うときだけ読み込む（ルールだけなら要らない）。
"""

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from hybrid_extractor.experts import Expert, InlineMathExpert
from hybrid_extractor.extract import Block, Continues, PageContext
from hybrid_extractor.math_reader import include_function_name
from hybrid_extractor.features import (
    block_char_size,
    char_features,
    ctx_record,
    pair_features,
    row_record,
    to_char_records,
)
from hybrid_extractor.nn import MLP, CharTagger, Sequences, load_checkpoint, predict_tagger

CHECKPOINTS = Path(__file__).parent / "checkpoints"
MAX_LEN = 512       # train.py と同じ（文中の数式の専門家が一度に見る文字の数）
THRESHOLD = 0.5
# 文中の数式の専門家が「数式の文字」と言うしきい値。検証用データ（inline_val）で、数式のまとまりの F1 が一番良い値を選んだ
# （0.5: 0.920 / 0.7: 0.936 / 0.9: 0.944。ルール版は 0.932）。学習で「はい」の見逃しを重く数えた（pos_weight）ので、
# 0.5 では手を挙げすぎる（英文の中の1文字を数式と言う）
INLINE_THRESHOLD = 0.9


@lru_cache(maxsize=None)
def load_mlp(path: Path) -> tuple[MLP, object, dict]:
    ckpt = load_checkpoint(path)
    model = MLP(ckpt["config"]["n_in"], ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["std"], ckpt["config"]


def mlp_prob(model: MLP, std, features: list[float]) -> float:
    with torch.no_grad():
        x = torch.tensor(std(np.array([features], dtype=np.float32)))
        return float(torch.sigmoid(model(x))[0])


# ---- 区切りの専門家 ----


def boundary_expert(path: Path = CHECKPOINTS / "boundary.pt") -> Continues:
    """「今のブロック（group）」に、次の行（row）が続く自信を返す関数。
    ブロックの1行目・最後の行・次の行の記録と、ページの記録から特徴を作る（train.boundary_examples と同じ）"""
    model, std, _ = load_mlp(path)
    records: dict[int, dict] = {}   # 同じ行の記録は作り直さない（行の id → 記録）
    contexts: dict[int, dict] = {}

    def continues(group: list[dict], gkind: str, row: dict, ctx: PageContext) -> float:
        if id(ctx) not in contexts:
            contexts.clear()
            records.clear()
            contexts[id(ctx)] = ctx_record(ctx)

        def rec(r: dict) -> dict:
            if id(r) not in records:
                records[id(r)] = row_record(r, ctx)
            return records[id(r)]

        return mlp_prob(model, std, pair_features(rec(group[0]), rec(group[-1]), rec(row), contexts[id(ctx)]))

    return continues


# ---- ラベルの専門家 ----


def label_experts(directory: Path = CHECKPOINTS / "labels") -> list[Expert]:
    """ラベルごとの専門家。特徴はオペレータが前もって文書全体で作り、DocInfo.features に入れておく
    （前後のブロックや、文書の本文の文字サイズを見るため）"""
    experts = []
    for path in sorted(directory.glob("*.pt")):
        model, std, config = load_mlp(path)

        def judge(block: Block, info, model=model, std=std) -> float:
            return mlp_prob(model, std, info.features[id(block)])

        experts.append(Expert(f"{config['label']}_ai", config["label"], judge))
    return experts


# ---- 文中の数式の専門家 ----


@lru_cache(maxsize=None)
def load_tagger(path: Path) -> tuple[CharTagger, object, dict]:
    ckpt = load_checkpoint(path)
    c = ckpt["config"]
    model = CharTagger(c["vocab_size"], c["n_feat"], d_model=c["d_model"], layers=c["layers"], heads=c["heads"],
                       max_len=c["max_len"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["std"], ckpt["extra"]["vocab"]


# 2つの数式のかけらの間にあっても、1つの数式としてつないでよい文字（空白・演算子・かっこ・区切り）
JOIN = re.compile(r"[\s=+\-−×·÷±<>≤≥≠≈≡∼∝/|,.:;()\[\]{}′']")


def spans_from_probs(chars: list[dict], probs: np.ndarray, threshold: float = THRESHOLD,
                     max_gap: int = 1) -> list[tuple[int, int]]:
    """1文字ずつの「数式か」の確率から、数式の範囲を作る。

    1. 確率が threshold 以上の文字が続くところを、数式のかけらにする（改行では切る）
    2. 隣り合う2つのかけらの間が、空白・演算子・かっこ（JOIN）だけで、空白以外が max_gap 文字以下なら、つなぐ。
       数式の中の空白（r = 1 の前後の空白）は確率が低くなりやすく、そのままだと数式が割れるため
    3. 前の関数名（sin・log・lim。ルール版と同じ include_function_name）を入れる
    4. 前後の空白を外し、見える文字が無ければ捨てる"""
    runs: list[list[int]] = []
    k = 0
    while k < len(chars):
        if probs[k] < threshold or chars[k]["c"] == "\n":
            k += 1
            continue
        start = k
        while k < len(chars) and probs[k] >= threshold and chars[k]["c"] != "\n":
            k += 1
        runs.append([start, k])
    merged: list[list[int]] = []
    for run in runs:
        if merged:
            between = chars[merged[-1][1]:run[0]]
            letters = [ch for ch in between if not ch["c"].isspace()]
            same_line = all(ch["c"] != "\n" for ch in between)  # 改行をまたいではつながない
            if same_line and len(letters) <= max_gap and all(JOIN.fullmatch(ch["c"]) for ch in between):
                merged[-1][1] = run[1]
                continue
        merged.append(run)
    spans = []
    for start, end in merged:
        while start < end and chars[start]["c"].isspace():
            start += 1
        while end > start and chars[end - 1]["c"].isspace():
            end -= 1
        if not any(ch["bbox"] is not None and not ch["c"].isspace() for ch in chars[start:end]):
            continue
        size = max(ch["size"] for ch in chars[start:end] if ch["bbox"] is not None)
        spans.append((include_function_name(chars, start, size), end))
    return spans


def inline_expert(path: Path = CHECKPOINTS / "inline_math.pt") -> InlineMathExpert:
    model, std, vocab = load_tagger(path)

    def find(chars: list[dict], width: float, height: float) -> list[tuple[int, int]]:
        if not chars:
            return []
        recs = to_char_records(chars, width, height)
        feats = np.array(char_features(recs, block_char_size(recs), width, height), dtype=np.float32)
        ids = np.array([vocab.get(ch["c"], 1) for ch in chars])
        seqs = Sequences([ids[k:k + MAX_LEN] for k in range(0, len(chars), MAX_LEN)],
                         [feats[k:k + MAX_LEN] for k in range(0, len(chars), MAX_LEN)],
                         [np.zeros(len(ids[k:k + MAX_LEN])) for k in range(0, len(chars), MAX_LEN)])
        probs = np.concatenate(predict_tagger(model, std, seqs))
        return spans_from_probs(chars, probs, INLINE_THRESHOLD)

    return InlineMathExpert("inline_math_ai", find)
