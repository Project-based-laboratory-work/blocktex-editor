"""学習データだけを変えた3モデルを、同じ条件で評価する。

  1. v2の検証データ（matplotlib / pdflatex 各1000件）。学習データ（両方）と同じ数式は除く
  2. 自分のレポートの数式（real_eval と同じ手順）

使い方（ai/ で）: python -m math_ocr.compare_eval <ckpt1> <ckpt2> ... -- <tex...>
"""

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from math_ocr import real_eval as R
from math_ocr.dataset import preprocess
from math_ocr.predict import load_model
from math_ocr.tokenizer import tokenize

DATA = Path("math_ocr/data")


def rows(name: str, split: str) -> list[list[str]]:
    return [line.rstrip("\n").split("\t", 1) for line in open(DATA / name / f"{split}.tsv", encoding="utf-8")]


def main() -> None:
    rest = sys.argv[1:]
    split = rest.index("--")
    checkpoints, tex_files = rest[:split], rest[split + 1 :]
    seen = {latex for name in ["v2_mathtext", "v2_latex"] for _, latex in rows(name, "train")}

    # 実際の文書の数式は、モデルによらず同じものを使う
    formulas, keys = [], set()
    for path in tex_files:
        for f in R.extract_formulas(Path(path)):
            key = (f.display, " ".join(R.normalize(f.latex)))
            if key not in keys:
                keys.add(key)
                formulas.append(f)
    with tempfile.TemporaryDirectory() as tmp:
        images = [(f, R.render(f, Path(tmp))) for f in formulas]
    images = [(f, im) for f, im in images if im is not None]

    print("model\tmathtext_val\tlatex_val\treal_all\treal_scope\treal_scope_10plus")
    for checkpoint in checkpoints:
        model, vocab = load_model(Path(checkpoint))
        vocab_tokens = set(vocab.itos)
        scores = []
        with torch.no_grad():
            for name in ["v2_mathtext", "v2_latex"]:
                data = [r for r in rows(name, "val") if r[1] not in seen]
                correct = 0
                for i in range(0, len(data), 64):
                    batch = data[i : i + 64]
                    x = torch.stack([preprocess(Image.open(DATA / name / "images" / f)) for f, _ in batch])
                    for (_, latex), ids in zip(batch, model.generate(x)):
                        correct += vocab.ids_to_tokens(ids.tolist()[1:]) == tokenize(latex)
                scores.append(f"{correct / len(data):.1%} ({correct}/{len(data)})")

            results = []
            for f, im in images:
                ids = model.generate(preprocess(im).unsqueeze(0))[0]
                gt, pred = R.normalize(f.latex), R.normalize(vocab.decode(ids.tolist()[1:]))
                results.append((gt, gt == pred, R.in_scope(gt, vocab_tokens)))
        scope = [r for r in results if r[2]]
        long = [r for r in scope if len(r[0]) >= 10]
        for group in (results, scope, long):
            ok = sum(r[1] for r in group)
            scores.append(f"{ok / len(group):.1%} ({ok}/{len(group)})")
        print(Path(checkpoint).parent.name + "\t" + "\t".join(scores), flush=True)


if __name__ == "__main__":
    main()
