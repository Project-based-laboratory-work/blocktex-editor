"""
実際の文書（自分のレポートのTeX）に書かれた数式で、数式OCRの精度を測る。

    python -m math_ocr.real_eval math_ocr/checkpoints/math_ocr_v2/best.pt report1.tex report2.tex ...

流れ:
  1. TeXから数式を取り出す（別行立て \\[ \\] / equation など、文中 $ $ / \\( \\)）。コードの中は除く
  2. 元と同じ形（別行立てか文中か）でLaTeXで組み、画像にする
  3. 数式OCRで読み、正解（元のTeX）と比べる

同じ数式でも書き方は何通りもある（x_i と x_{i}、\\le と \\leq など）。そのまま比べると、
見た目が同じでも「不正解」になってしまうので、比べる前に書き方をそろえる（normalize）。

結果は「全部の数式」と「対応範囲の数式」（語彙にある記号だけで書けて、50トークン以下）に分けて出す。
都合の悪い数式を外さずに全部で測り、狙った範囲での実力も別に見せるため。
"""

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from math_ocr.dataset import preprocess
from math_ocr.metrics import edit_distance
from math_ocr.predict import load_model
from math_ocr.render_latex import _crop_to_ink
from math_ocr.tokenizer import tokenize

MAX_TOKENS = 50  # 学習した数式の長さの上限（synth_data.random_formula の max_tokens）

# 見た目が同じになる別名 → 学習データで使っている書き方
ALIASES = {
    r"\le": r"\leq", r"\ge": r"\geq", r"\ne": r"\neq", r"\rightarrow": r"\to", r"\ldots": r"\dots",
    r"\varepsilon": r"\epsilon", r"\lbrace": r"\{", r"\rbrace": r"\}",
}  # fmt: skip
# 見た目にほぼ影響しない（空白・大きさの指定）ので、比べるときは無視する
IGNORED = {r"\,", r"\;", r"\:", r"\!", r"\ ", r"\quad", r"\qquad", r"\displaystyle", r"\textstyle"}

PAGE = r"""\documentclass[uplatex]{jsarticle}
\usepackage{amsmath,amssymb}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


@dataclass
class Formula:
    source: str  # どのファイルから取ったか
    display: bool  # 別行立てか（False なら文中）
    latex: str


def extract_formulas(tex_path: Path) -> list[Formula]:
    tex = tex_path.read_text(encoding="utf-8", errors="ignore")
    tex = re.sub(r"\\begin\{(Verbatim|verbatim|lstlisting|minted)\}.*?\\end\{\1\}", "", tex, flags=re.S)
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    body = tex[tex.index(r"\begin{document}") :] if r"\begin{document}" in tex else tex
    display = re.findall(r"\\\[(.+?)\\\]", body, flags=re.S)
    display += re.findall(r"\\begin\{(?:equation|align|gather|eqnarray)\*?\}(.+?)\\end\{", body, flags=re.S)
    inline = re.findall(r"(?<![\\$])\$([^$]+?)\$", body) + re.findall(r"\\\((.+?)\\\)", body)
    name = tex_path.parent.name + "/" + tex_path.name
    formulas = [Formula(name, True, f.strip()) for f in display] + [Formula(name, False, f.strip()) for f in inline]
    return [f for f in formulas if f.latex]


def normalize(latex: str) -> list[str]:
    """比べる前に書き方をそろえる。添字・指数の1文字の引数は {} で囲む（x_i → x_{i}）"""
    tokens = [ALIASES.get(t, t) for t in tokenize(latex) if t not in IGNORED and t != "&"]
    result: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        result.append(token)
        if token in ("^", "_") and i + 1 < len(tokens) and tokens[i + 1] != "{":
            result += ["{", tokens[i + 1], "}"]
            i += 2
            continue
        i += 1
    return result


def in_scope(tokens: list[str], vocab_tokens: set[str]) -> bool:
    """線引きした対応範囲か: 学習した記号だけで書けて、50トークン以下"""
    return len(tokens) <= MAX_TOKENS and all(t in vocab_tokens for t in tokens)


def render(formula: Formula, work_dir: Path, dpi: int = 150) -> Image.Image | None:
    """元と同じ形（別行立て/文中）でLaTeXで組み、余白を切った画像にする。組めなければ None"""
    body = rf"\[ {formula.latex} \]" if formula.display else rf"\noindent ${formula.latex}$"
    (work_dir / "f.tex").write_text(PAGE % body, encoding="utf-8")
    try:
        for command in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], ["dvipdfmx", "-q", "f.dvi"],
                        ["pdftoppm", "-r", str(dpi), "-gray", "-png", "-singlefile", "f.pdf", "f"]):  # fmt: skip
            subprocess.run(command, cwd=work_dir, check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return _crop_to_ink(Image.open(work_dir / "f.png"))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tex", type=Path, nargs="+")
    parser.add_argument("--show", type=int, default=15, help="間違えた例を何件表示するか")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint)
    vocab_tokens = set(vocab.itos)

    # 同じ数式（同じ形）は1回だけ数える
    formulas, seen = [], set()
    for path in args.tex:
        for formula in extract_formulas(path):
            key = (formula.display, " ".join(normalize(formula.latex)))
            if key not in seen:
                seen.add(key)
                formulas.append(formula)

    results = []  # (formula, 正解トークン, 予測トークン)
    with tempfile.TemporaryDirectory() as tmp:
        for formula in formulas:
            image = render(formula, Path(tmp))
            if image is None:
                print(f"組めなかった: {formula.latex!r}（{formula.source}）")
                continue
            ids = model.generate(preprocess(image).unsqueeze(0))[0]
            results.append((formula, normalize(formula.latex), normalize(vocab.decode(ids.tolist()[1:]))))

    def summary(name: str, rows: list) -> None:
        if not rows:
            print(f"{name}: 0 件")
            return
        exact = sum(gt == pred for _, gt, pred in rows)
        errors = sum(edit_distance(pred, gt) for _, gt, pred in rows)
        tokens = sum(len(gt) for _, gt, _ in rows)
        print(f"{name}: {len(rows)} 件  完全一致率 {exact / len(rows):.1%}  トークン誤り率 {errors / max(tokens, 1):.3f}")

    scoped = [r for r in results if in_scope(r[1], vocab_tokens)]
    print(f"\n=== 実際の文書の数式（{len(args.tex)} ファイル、重複を除いて {len(formulas)} 個）===")
    summary("全部の数式    ", results)
    summary("対応範囲の数式", scoped)
    summary("  うち別行立て", [r for r in scoped if r[0].display])
    summary("  うち文中    ", [r for r in scoped if not r[0].display])

    print("\n間違えた例（対応範囲）:")
    for formula, gt, pred in [r for r in scoped if r[1] != r[2]][: args.show]:
        print(f"  正解: {''.join(gt)}\n  予測: {''.join(pred)}   （{formula.source}）")
    out_of_scope = [r for r in results if not in_scope(r[1], vocab_tokens)]
    unknown = sorted({t for _, gt, _ in out_of_scope for t in gt if t not in vocab_tokens})
    print(f"\n対応範囲の外: {len(out_of_scope)} 件。学習していない記号: {unknown}")


if __name__ == "__main__":
    main()
