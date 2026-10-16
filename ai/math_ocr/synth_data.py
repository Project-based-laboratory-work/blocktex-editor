"""
学習用の「数式画像とLaTeXの組」を自動生成する。

ランダムに数式を組み立て、matplotlib の mathtext で画像に描画する。正解のLaTeXを自分で作ってから
描画するので、人手のラベル付けが要らない（スライドの「組版の逆をたどる」の学習データ版）。

mathtext は LaTeX のサブセットしか描画できない（\\begin{align} などの環境は非対応）。まずはこれで
モデルの仕組みを確認し、実際のuplatexで描画したデータ（docker/ のTeX環境を使う）は後で追加する。

同じ数式でも書き方が何通りもあると（x_i と x_{i} など）、モデルはどちらを出せばよいか分からず
学習が難しくなる。そのため生成する数式は「添字は常に {} で囲む」などの書き方に統一している。

    python -m math_ocr.synth_data --out data/math_synth --train 20000 --val 1000
    python -m math_ocr.synth_data --out data/formulas --formulas-only --seed 1   # 数式だけ（render_latex.py 用）
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

LOWER = list("abcdefghijklmnopqrstuvwxyz")
UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
# \varepsilon は入れない: フォントによって \epsilon が ε の形で描かれ、画像からは区別できないため、
# どちらの形も \epsilon と読ませる。同じ理由で、見た目が同じになる別名（\le と \leq、\rightarrow と \to、
# \ldots と \dots など）は片方だけを使う。
GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta", r"\eta", r"\theta", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\phi", r"\varphi", r"\chi",
    r"\psi", r"\omega", r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Pi", r"\Sigma", r"\Phi", r"\Psi",
    r"\Omega",
]  # fmt: skip
ACCENTS = [r"\bar", r"\hat", r"\vec", r"\tilde", r"\dot"]
NUMBER_SETS = ["R", "N", "Z", "Q", "C"]  # \mathbb{R} など
FUNCS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp"]
FUNC_NAMES = list("fghFGPu")
INDEX_VARS = list("ijkn")

# (候補, 重み)。+ や = のようによく使う記号が多く出るように重みを付ける
BIN_OPS = (["+", "-", r"\times", r"\cdot", r"\pm", "/", r"\div", r"\cup", r"\cap", r"\circ"],
           [30, 25, 10, 10, 5, 6, 3, 4, 4, 3])  # fmt: skip
RELATIONS = (["=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\in", r"\to", r"\equiv", r"\sim",
              r"\subset", r"\propto"],
             [40, 6, 6, 8, 8, 6, 6, 6, 4, 3, 3, 2, 2])  # fmt: skip

FONTSETS = ["cm", "stix", "dejavuserif"]  # 書体を変えて、特定のフォントだけに過適合しないようにする

# 以下の生成関数は乱数生成器 rng を引数で受け取る。グローバルな random を使わないことで、
# 並列生成してもサンプル番号ごとに同じ結果を再現できる。
# 部品同士は空白で連結する（"\times" と "x" を直結すると "\timesx" という別コマンドになるため）。


def _pick(rng: random.Random, weighted: tuple[list[str], list[int]]) -> str:
    return rng.choices(weighted[0], weights=weighted[1])[0]


def _symbol(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.5:
        return rng.choice(LOWER)
    if r < 0.7:
        return rng.choice(GREEK)
    if r < 0.86:
        return rng.choice(UPPER)
    if r < 0.92:
        return f"{rng.choice(ACCENTS)}{{{rng.choice(LOWER)}}}"
    if r < 0.95:
        return rf"\mathbf{{{rng.choice(LOWER + UPPER)}}}"
    if r < 0.98:
        return rf"\mathbb{{{rng.choice(NUMBER_SETS)}}}"
    return rng.choice([r"\infty", r"\ell"])


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
    if r < 0.75:
        return rng.choice(LOWER + INDEX_VARS)
    if r < 0.9:
        return f"{rng.choice(INDEX_VARS)} {rng.choice(['+', '-'])} 1"
    return f"{rng.choice(INDEX_VARS)} , {rng.choice(INDEX_VARS)}"  # a_{i , j}


def _atom(rng: random.Random, allow_number: bool = True) -> str:
    """変数や数に、添字・指数などが付いたもの（x, 2, x_{i}, \\alpha^{2}, f', n! など）"""
    base = _number(rng) if allow_number and rng.random() < 0.3 else _symbol(rng)
    r = rng.random()
    if r < 0.2:
        return f"{base}_{{{_script(rng)}}}"
    if r < 0.38:
        return f"{base}^{{{_script(rng)}}}"
    if r < 0.47:
        return f"{base}_{{{_script(rng)}}}^{{{_script(rng)}}}"
    if r < 0.5:
        return f"{base} '"
    if r < 0.53:
        return f"{base} !"
    return base


def _args(rng: random.Random) -> str:
    """カンマ区切りの引数（x , y）"""
    return " , ".join(_atom(rng) for _ in range(rng.choice([1, 1, 2, 2, 3])))


def _parenthesized(rng: random.Random, depth: int) -> str:
    inner = _expr(rng, depth - 1)
    left, right = rng.choice([("(", ")"), ("(", ")"), ("(", ")"), ("[", "]"), ("|", "|")])
    # 中に分数があると括弧の高さが足りなくなるので、実際の文書では \left( \right) で伸ばすことが多い
    if r"\frac" in inner and rng.random() < 0.7:
        left, right = rf"\left{left}", rf"\right{right}"
    power = f"^{{{_script(rng)}}}" if rng.random() < 0.3 else ""
    return f"{left} {inner} {right}{power}"


def _big_operator(rng: random.Random, depth: int) -> str:
    """総和・総乗・積分・極限・最大最小"""
    body = _term(rng, depth - 1)
    r = rng.random()
    if r < 0.4:
        op = rng.choice([r"\sum", r"\sum", r"\prod"])
        i = rng.choice(INDEX_VARS)
        upper = rng.choice(["n", "N", r"\infty", str(rng.randint(2, 9))])
        return rf"{op}_{{{i} = {rng.randint(0, 1)}}}^{{{upper}}} {body}"
    if r < 0.7:
        var = rng.choice(["x", "t", "u"])
        if rng.random() < 0.25:
            return rf"\int {body} d {var}"  # 不定積分
        return rf"\int_{{{_number(rng)}}}^{{{_symbol(rng)}}} {body} d {var}"
    if r < 0.88:
        var = rng.choice(["x", "n", "t", "h"])
        limit = rng.choice(["0", r"\infty", "a", "1"])
        return rf"\lim_{{{var} \to {limit}}} {body}"
    return rf"{rng.choice([r'\max', r'\min'])}_{{{rng.choice(LOWER)}}} {body}"


def _derivative(rng: random.Random) -> str:
    d = rng.choice(["d", r"\partial"])
    return rf"\frac{{{d} {rng.choice(FUNC_NAMES + ['y'])}}}{{{d} {rng.choice(['x', 't', 'y'])}}}"


def _term(rng: random.Random, depth: int) -> str:
    """項。depth が残っていれば分数・根号・括弧・関数・総和/積分などで入れ子にする"""
    if depth <= 0:
        return _atom(rng)
    r = rng.random()
    if r < 0.34:
        return _atom(rng)
    if r < 0.43:
        return f"{_number(rng)} {_atom(rng, allow_number=False)}"  # 係数付き（2 x）
    if r < 0.54:
        return rf"\frac{{{_expr(rng, depth - 1)}}}{{{_expr(rng, depth - 1)}}}"
    if r < 0.61:
        index = f"[{rng.randint(3, 9)}]" if rng.random() < 0.15 else ""  # \sqrt[3]{x}
        return rf"\sqrt{index}{{{_expr(rng, depth - 1)}}}"
    if r < 0.71:
        return _parenthesized(rng, depth)
    if r < 0.79:
        func = rng.choice(FUNCS)
        if func == r"\log" and rng.random() < 0.3:
            func = rf"\log_{{{rng.choice(['2', '10', 'a'])}}}"
        if rng.random() < 0.5:
            return f"{func} {_atom(rng)}"
        return f"{func} ( {_expr(rng, depth - 1)} )"
    if r < 0.86:
        return f"{rng.choice(FUNC_NAMES)} ( {_args(rng)} )"  # f ( x , y )
    if r < 0.9:
        return _derivative(rng)
    if r < 0.92:
        return rf"\binom{{{_script(rng)}}}{{{_script(rng)}}}"
    if r < 0.94:
        return rf"\nabla {_symbol(rng)}"
    return _big_operator(rng, depth)


def _expr(rng: random.Random, depth: int) -> str:
    """式 = 項 (演算子 項)*"""
    parts = [_term(rng, depth)]
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        parts += [_pick(rng, BIN_OPS), _term(rng, depth)]
    return " ".join(parts)


def _sequence(rng: random.Random) -> str:
    """a_{1} , a_{2} , \\dots , a_{n} や a_{1} + a_{2} + \\cdots + a_{n}"""
    base, last = rng.choice(LOWER), rng.choice(["n", "N", "k", "m"])
    items = [f"{base}_{{1}}", f"{base}_{{2}}", None, f"{base}_{{{last}}}"]
    # 点々は、カンマの間では下寄り（\dots）、演算子の間では中央（\cdots）と書き分ける
    separator, dots = rng.choice([(",", r"\dots"), (",", r"\dots"), ("+", r"\cdots"), (r"\cdot", r"\cdots")])
    return f" {separator} ".join(item or dots for item in items)


def _set(rng: random.Random) -> str:
    elements = " , ".join(_atom(rng) for _ in range(rng.randint(2, 4)))
    return rf"\{{ {elements} \}}"


def _statement(rng: random.Random) -> str:
    """数式全体の形を選ぶ"""
    r = rng.random()
    if r < 0.7:
        formula = _expr(rng, depth=rng.choice([1, 1, 2]))
        for _ in range(rng.choices([0, 1, 2], weights=[35, 57, 8])[0]):
            formula = f"{formula} {_pick(rng, RELATIONS)} {_expr(rng, depth=rng.choice([0, 1]))}"
        return formula
    if r < 0.77:
        return f"{rng.choice(FUNC_NAMES)} ( {_args(rng)} ) = {_expr(rng, depth=rng.choice([1, 2]))}"
    if r < 0.84:
        sequence = _sequence(rng)
        return sequence if rng.random() < 0.5 else f"{_atom(rng)} = {sequence}"
    if r < 0.9:
        return rng.choice([_set(rng), rf"{_symbol(rng)} \in {_set(rng)}", f"{rng.choice(UPPER)} = {_set(rng)}"])
    if r < 0.95:
        left, right = rng.choice([("[", "]"), ("(", ")"), ("[", ")")])  # 区間
        return rf"{_symbol(rng)} \in {left} {_atom(rng)} , {_atom(rng)} {right}"
    return rf"{rng.choice(FUNC_NAMES)} : {_symbol(rng)} \to {_symbol(rng)}"  # 写像 f : X \to Y


def random_formula(rng: random.Random, max_tokens: int = 50) -> str:
    while True:
        formula = _statement(rng)
        if rng.random() < 0.06:  # 文中の数式は末尾に句読点が付くことがある
            formula = f"{formula} {rng.choice([',', '.'])}"
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


def _make_sample(args: tuple[int, int, Path | None, str]) -> tuple[str, str] | None:
    index, seed, image_dir, split = args
    rng = random.Random(seed * 1_000_003 + index)
    latex = random_formula(rng)
    if image_dir is None:  # 数式だけ欲しい場合（render_latex.py で別途描画する）
        return "-", latex
    try:
        image = render(latex, fontset=rng.choice(FONTSETS), dpi=rng.randint(100, 200))
    except ValueError:
        return None
    filename = f"{split}_{index:06d}.png"
    image.save(image_dir / filename)
    return filename, latex


def generate(out_dir: Path, split: str, count: int, seed: int, workers: int, formulas_only: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = None if formulas_only else out_dir / "images"
    if image_dir:
        image_dir.mkdir(exist_ok=True)
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
    parser.add_argument("--out", type=Path, default=Path("data/math_synth"))
    parser.add_argument("--train", type=int, default=20000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="変えると別の数式の集合になる")
    parser.add_argument("--formulas-only", action="store_true", help="画像を描かず、数式の一覧（tsv）だけ出力する")
    args = parser.parse_args()

    # train と val でシードを変え、同じ数式が両方に入りにくくする
    generate(args.out, "train", args.train, args.seed * 2, args.workers, args.formulas_only)
    generate(args.out, "val", args.val, args.seed * 2 + 1, args.workers, args.formulas_only)


if __name__ == "__main__":
    main()
