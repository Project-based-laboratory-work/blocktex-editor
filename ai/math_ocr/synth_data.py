"""
学習用の「数式画像とLaTeXの組」を自動生成する。

ランダムに数式を組み立て、matplotlib の mathtext で画像に描画する。正解のLaTeXを自分で作ってから
描画するので、人手のラベル付けが要らない（スライドの「組版の逆をたどる」の学習データ版）。

mathtext は LaTeX のサブセットしか描画できない（\\begin{align} などの環境は非対応）。まずはこれで
モデルの仕組みを確認し、実際のuplatexで描画したデータ（docker/ のTeX環境を使う）は後で追加する。

同じ数式でも書き方が何通りもあると（x_i と x_{i} など）、モデルはどちらを出せばよいか分からず
学習が難しくなる。そのため生成する数式は「添字は常に {} で囲む」などの書き方に統一している。

    python -m math_ocr.synth_data --out math_ocr/data/math_synth --train 20000 --val 1000
"""

import argparse
import io
import random
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import mathtext  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from PIL import Image  # noqa: E402

from math_ocr.tokenizer import tokenize  # noqa: E402

LOWER = list("abcdefghijkmnpqrstuvwxyz")  # l と o は 1 と 0 に見えやすいので除外
UPPER = list("ABCDEFGHKLMNPQRSTUVWXYZ")
GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\theta", r"\lambda",
    r"\mu", r"\pi", r"\sigma", r"\phi", r"\omega", r"\Delta", r"\Omega",
]  # fmt: skip
FUNCS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp"]
BIN_OPS = ["+", "-", r"\times", r"\cdot", r"\pm"]
RELATIONS = ["=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx"]
INDEX_VARS = list("ijkn")

FONTSETS = ["cm", "stix", "dejavuserif"]  # 書体を変えて、特定のフォントだけに過適合しないようにする

# 以下の生成関数は乱数生成器 rng を引数で受け取る。グローバルな random を使わないことで、
# 並列生成してもサンプル番号ごとに同じ結果を再現できる。
# 部品同士は空白で連結する（"\times" と "x" を直結すると "\timesx" という別コマンドになるため）。


def _symbol(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return rng.choice(LOWER)
    if r < 0.8:
        return rng.choice(GREEK)
    return rng.choice(UPPER)


def _number(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return str(rng.randint(0, 9))
    if r < 0.9:
        return str(rng.randint(10, 99))
    return f"{rng.randint(0, 9)}.{rng.randint(0, 99)}"


def _script(rng: random.Random) -> str:
    """添字・指数に入る短い式"""
    r = rng.random()
    if r < 0.4:
        return _number(rng)
    if r < 0.8:
        return rng.choice(LOWER + INDEX_VARS)
    return f"{rng.choice(INDEX_VARS)} {rng.choice(['+', '-'])} 1"


def _atom(rng: random.Random, allow_number: bool = True) -> str:
    """変数や数に、添字・指数が付いたもの（x, 2, x_{i}, \\alpha^{2} など）"""
    base = _number(rng) if allow_number and rng.random() < 0.3 else _symbol(rng)
    r = rng.random()
    if r < 0.2:
        return f"{base}_{{{_script(rng)}}}"
    if r < 0.4:
        return f"{base}^{{{_script(rng)}}}"
    if r < 0.5:
        return f"{base}_{{{_script(rng)}}}^{{{_script(rng)}}}"
    return base


def _term(rng: random.Random, depth: int) -> str:
    """項。depth が残っていれば分数・根号・括弧・関数・総和/積分で入れ子にする"""
    if depth <= 0:
        return _atom(rng)
    r = rng.random()
    if r < 0.4:
        return _atom(rng)
    if r < 0.5:
        return f"{_number(rng)} {_atom(rng, allow_number=False)}"  # 係数付き（2 x）
    if r < 0.62:
        return rf"\frac{{{_expr(rng, depth - 1)}}}{{{_expr(rng, depth - 1)}}}"
    if r < 0.7:
        return rf"\sqrt{{{_expr(rng, depth - 1)}}}"
    if r < 0.8:
        power = f"^{{{_script(rng)}}}" if rng.random() < 0.4 else ""
        return f"( {_expr(rng, depth - 1)} ){power}"
    if r < 0.9:
        func = rng.choice(FUNCS)
        if rng.random() < 0.5:
            return f"{func} {_atom(rng)}"
        return f"{func} ( {_expr(rng, depth - 1)} )"
    if rng.random() < 0.5:
        i = rng.choice(INDEX_VARS)
        upper = rng.choice(["n", "N", r"\infty", str(rng.randint(2, 9))])
        return rf"\sum_{{{i} = {rng.randint(0, 1)}}}^{{{upper}}} {_term(rng, depth - 1)}"
    var = rng.choice(["x", "t", "u"])
    return rf"\int_{{{_number(rng)}}}^{{{_symbol(rng)}}} {_term(rng, depth - 1)} d {var}"


def _expr(rng: random.Random, depth: int) -> str:
    """式 = 項 (演算子 項)*"""
    parts = [_term(rng, depth)]
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        parts += [rng.choice(BIN_OPS), _term(rng, depth)]
    return " ".join(parts)


def random_formula(rng: random.Random, max_tokens: int = 40) -> str:
    while True:
        formula = _expr(rng, depth=rng.choice([1, 1, 2]))
        if rng.random() < 0.6:
            formula = f"{formula} {rng.choice(RELATIONS)} {_expr(rng, depth=rng.choice([0, 1]))}"
        if len(tokenize(formula)) <= max_tokens:
            return formula


def render(latex: str, fontset: str = "cm", dpi: int = 150) -> Image.Image:
    """数式を白背景・黒文字のグレースケール画像にする。描画できない数式は ValueError を投げる。"""
    buf = io.BytesIO()
    prop = FontProperties(size=12, math_fontfamily=fontset)
    mathtext.math_to_image(f"${latex}$", buf, prop=prop, dpi=dpi, format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGBA")
    # math_to_image は背景を透明にして出力するので、白い背景の上に合成する
    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("L")


def _make_sample(args: tuple[int, int, Path, str]) -> tuple[str, str] | None:
    index, seed, image_dir, split = args
    rng = random.Random(seed * 1_000_003 + index)
    latex = random_formula(rng)
    try:
        image = render(latex, fontset=rng.choice(FONTSETS), dpi=rng.randint(100, 200))
    except ValueError:
        return None
    filename = f"{split}_{index:06d}.png"
    image.save(image_dir / filename)
    return filename, latex


def generate(out_dir: Path, split: str, count: int, seed: int, workers: int) -> None:
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, seed, image_dir, split) for i in range(count)]
    with Pool(workers) as pool:
        results = pool.map(_make_sample, jobs, chunksize=64)

    ok = [r for r in results if r is not None]
    with open(out_dir / f"{split}.tsv", "w", encoding="utf-8") as f:
        for filename, latex in ok:
            f.write(f"{filename}\t{latex}\n")
    print(f"{split}: {len(ok)}/{count} 件を生成（描画失敗 {count - len(ok)} 件）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("math_ocr/data/math_synth"))
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # train と val でシードを変え、同じ数式が両方に入りにくくする
    generate(args.out, "train", args.train, seed=0, workers=args.workers)
    generate(args.out, "val", args.val, seed=1, workers=args.workers)


if __name__ == "__main__":
    main()
