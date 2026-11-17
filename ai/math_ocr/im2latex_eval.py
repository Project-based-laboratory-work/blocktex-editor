"""
公開データ IM2LATEX-100K（arXivの論文から集めた、人が書いた数式）の test で数式OCRを評価する。

  1. 数式の書き方をそろえる（見た目が変わらない違いを消す）
       - 見た目に影響しない {} のまとまり（引数でない {}）を外す: { \\frac { a } { b } } → \\frac { a } { b }
       - 空白の指定（~ \\; \\quad \\hspace{..} など）を消す
       - x ^ { \\prime } → x '
       - そのあと real_eval.normalize（別名の統一、1文字の添字に {} を付ける）
  2. 対応範囲（学習した記号だけで書けて50トークン以下）の数式を選ぶ
  3. データに付いている画像（作者が描画したもの）を読ませて、正解と比べる

    python -m math_ocr.im2latex_eval <test.parquet> <out_dir> <ckpt...>

Deng et al., "Image-to-Markup Generation with Coarse-to-Fine Attention", ICML 2017（CC0 で配布）
"""

import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from math_ocr import real_eval as R
from math_ocr.dataset import preprocess
from math_ocr.predict import load_model
from math_ocr.tokenizer import detokenize, tokenize

# 引数を {} で取るコマンドと、その数
ARGS = {
    "\\frac": 2, "\\binom": 2, "\\stackrel": 2, "\\sqrt": 1, "\\mathbb": 1, "\\mathbf": 1, "\\mathrm": 1,
    "\\mathcal": 1, "\\operatorname": 1, "\\bar": 1, "\\hat": 1, "\\vec": 1, "\\tilde": 1, "\\dot": 1, "\\ddot": 1,
    "\\overline": 1, "\\underline": 1, "\\widetilde": 1, "\\widehat": 1, "\\text": 1, "\\hspace": 1,
}  # fmt: skip
SPACES = {"~", "\\;", "\\,", "\\:", "\\!", "\\ ", "\\quad", "\\qquad"}


def strip_groups(tokens: list[str]) -> list[str]:
    """引数になっていない { } を外す（中身は残す）"""
    n = len(tokens)

    def parse(i: int) -> tuple[list[str], int]:
        out, pending, bracket = [], 0, 0
        while i < n:
            t = tokens[i]
            if t == "}":
                return out, i
            if t == "{":
                inner, j = parse(i + 1)
                i = j + 1
                if pending > 0:
                    out += ["{", *inner, "}"]
                    pending -= 1
                else:
                    out += inner
                continue
            out.append(t)
            i += 1
            if t == "[":
                bracket += 1
            elif t == "]":
                bracket = max(0, bracket - 1)
            elif bracket:
                continue
            elif t in ("^", "_"):
                pending = 1
            elif t in ARGS:
                pending = ARGS[t]
            elif pending > 0:
                pending -= 1  # x ^ 2 のように {} なしで1トークンを引数に取った
        return out, i

    return parse(0)[0]


def canonical(latex: str) -> list[str]:
    tokens = tokenize(latex)
    out, i = [], 0
    while i < len(tokens):  # \hspace { ... } は引数ごと消す
        if tokens[i] == "\\hspace" and i + 1 < len(tokens) and tokens[i + 1] == "{":
            depth, i = 0, i + 1
            while i < len(tokens):
                depth += {"{": 1, "}": -1}.get(tokens[i], 0)
                i += 1
                if depth == 0:
                    break
            continue
        if tokens[i] not in SPACES:
            out.append(tokens[i])
        i += 1
    out = strip_groups(out)
    joined = " ".join(out).replace("^ { \\prime }", "'").replace("^ \\prime", "'").replace("\\prime", "'")
    return R.normalize(joined)


def main() -> None:
    parquet, out_dir, checkpoints = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    rows = pq.read_table(parquet, columns=["filename", "formula", "image"]).to_pylist()
    vocab_tokens = set(load_model(Path(checkpoints[0]))[1].itos)

    scoped = []
    for row in rows:
        gt = canonical(row["formula"])
        if R.in_scope(gt, vocab_tokens):
            scoped.append((row["filename"], gt, row["image"]["bytes"]))
    lengths = sorted(len(gt) for _, gt, _ in scoped)
    print(f"test {len(rows)} 個のうち、対応範囲 {len(scoped)} 個（トークン数の中央値 {lengths[len(lengths) // 2]}）")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scoped.tsv", "w", encoding="utf-8") as f:
        for name, gt, _ in scoped:
            f.write(f"{name}\t{detokenize(gt)}\n")

    images = [preprocess(Image.open(io.BytesIO(data))) for _, _, data in scoped]
    print("model\t全体\t1〜9\t10〜19\t20〜50")
    for checkpoint in checkpoints:
        model, vocab = load_model(Path(checkpoint))
        hits = []
        with torch.no_grad():
            for i in range(0, len(scoped), 64):
                batch = torch.stack(images[i : i + 64])
                for (_, gt, _), ids in zip(scoped[i : i + 64], model.generate(batch)):
                    hits.append((len(gt), canonical(vocab.decode(ids.tolist()[1:])) == gt))
        groups = [hits, [h for h in hits if h[0] < 10], [h for h in hits if 10 <= h[0] < 20], [h for h in hits if h[0] >= 20]]
        cells = [f"{sum(ok for _, ok in g) / max(len(g), 1):.1%} ({len(g)})" for g in groups]
        print(Path(checkpoint).parent.name + "\t" + "\t".join(cells), flush=True)


if __name__ == "__main__":
    main()
