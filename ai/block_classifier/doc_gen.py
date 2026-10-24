"""
学習データ用に、ランダムな文書を作ってTeXにする。

各ブロックの始まりと終わりに「目印」（PDFの名前付きリンク先, named destination）を埋め込む。
dvipdfmx は目印を置いた位置をPDFの座標でそのまま記録するので、PDFから取り出した塊が
どのブロックの範囲に入っているかを見れば、正解ラベルが分かる（auto_label.py で行う）。

文字の内容で対応付ける方法は、別行立ての数式が `f(x) =` `∑` `i=1` のように細切れになり、
元のTeXと文字が一致しないため使えない。色で見分ける方法は使わない（チームの方針）。

TeXの書き方は、エディタのTeX変換（web/src/tex.ts）に合わせている（jsarticle、align*、
itemize/enumerate のマーカーなど）。
"""

import random
from dataclasses import dataclass

from math_ocr.synth_data import random_formula

# ---- ブロック ----


@dataclass
class SourceBlock:
    """文書に置いたブロック1つ。id は目印の名前（b{id}s / b{id}e）に使う"""

    id: int
    label: str  # rule_based/labels.py の FINE_LABELS のどれか
    tex: str  # 目印を埋め込んだTeX


def _mark(name: str) -> str:
    """dvipdfmx に、この位置を名前付きリンク先としてPDFに記録させる"""
    return rf"\mk{{{name}}}"


# ---- 文章の材料（内容は分類にほぼ影響しないので、それらしい日本語になれば良い） ----

NOUNS = [
    "数式", "文書", "画像", "モデル", "手法", "データ", "精度", "構造", "ブロック", "段落", "見出し", "学習",
    "評価", "結果", "入力", "出力", "特徴", "座標", "ページ", "変換", "システム", "アルゴリズム", "誤差",
    "パラメータ", "実験", "推論", "計算量", "符号", "記号", "表", "図", "レイアウト", "認識", "抽出",
]  # fmt: skip
VERBS = ["検討する", "提案する", "比較する", "評価した", "示す", "述べる", "改善した", "実装した", "確認した", "分析する"]
CONNECTORS = ["また、", "さらに、", "一方で、", "そこで、", "その結果、", "本研究では、", "次に、", "ここで、", ""]
HEADINGS = [
    "はじめに", "関連研究", "提案手法", "実験", "実験方法", "結果", "考察", "まとめ", "今後の課題",
    "背景", "目的", "データセット", "評価方法", "実装",
]  # fmt: skip
ENGLISH = ["PDF", "LaTeX", "Transformer", "CNN", "OCR", "JSON", "GPU", "CPU", "PyTorch", "XY-cut"]


def _noun(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.15:
        return rng.choice(ENGLISH)
    if r < 0.3:
        return f"{rng.choice(NOUNS)}の{rng.choice(NOUNS)}"
    return rng.choice(NOUNS)


def _sentence(rng: random.Random, inline_math: bool = True) -> str:
    """日本語の文を1つ。太字・下線・文中数式をときどき混ぜる"""
    parts = [rng.choice(CONNECTORS), _noun(rng), rng.choice(["を", "について", "に対して", "と"])]
    if rng.random() < 0.5:
        parts += [_noun(rng), "の", _noun(rng), "を"]
    noun = _noun(rng)
    r = rng.random()
    if r < 0.08:
        noun = rf"\textbf{{{noun}}}"
    elif r < 0.14:
        noun = rf"\underline{{{noun}}}"
    parts += [noun, "として", rng.choice(VERBS)]
    sentence = "".join(parts)
    if inline_math and rng.random() < 0.2:
        sentence += f"。ここで${random_formula(rng, max_tokens=12)}$とする"
    return sentence + "。"


def _paragraph_text(rng: random.Random) -> str:
    return "".join(_sentence(rng) for _ in range(rng.choices([1, 2, 3, 4, 6], weights=[2, 4, 4, 3, 1])[0]))


def _heading_text(rng: random.Random) -> str:
    return rng.choice(HEADINGS) if rng.random() < 0.6 else f"{_noun(rng)}の{rng.choice(['評価', '検討', '設計', '比較'])}"


# ---- ブロックごとのTeX ----


def _heading(rng: random.Random, i: int) -> SourceBlock:
    command = rng.choices(["section", "subsection", "subsubsection"], weights=[5, 4, 1])[0]
    return SourceBlock(i, "heading", rf"\{command}{{{_mark(f'b{i}s')}{_heading_text(rng)}{_mark(f'b{i}e')}}}")


def _paragraph(rng: random.Random, i: int) -> SourceBlock:
    return SourceBlock(i, "paragraph", f"{_mark(f'b{i}s')}{_paragraph_text(rng)}{_mark(f'b{i}e')}")


LIST_SETUPS = [
    ("itemize", r"\renewcommand{\labelitemi}{・}"),
    ("itemize", ""),
    ("itemize", r"\renewcommand{\labelitemi}{--}"),
    ("enumerate", ""),
    ("enumerate", r"\renewcommand{\labelenumi}{(\arabic{enumi})}"),
]


def _list(rng: random.Random, i: int) -> SourceBlock:
    env, setup = rng.choice(LIST_SETUPS)
    items = [_sentence(rng, inline_math=False) if rng.random() < 0.6 else _noun(rng) for _ in range(rng.randint(2, 5))]
    items[0] = _mark(f"b{i}s") + items[0]
    items[-1] = items[-1] + _mark(f"b{i}e")
    lines = [rf"\begin{{{env}}}", *([f"  {setup}"] if setup else []), *(rf"  \item {item}" for item in items)]
    return SourceBlock(i, "list", "\n".join([*lines, rf"\end{{{env}}}"]))


def _math(rng: random.Random, i: int) -> SourceBlock:
    lines = [random_formula(rng, max_tokens=30) for _ in range(rng.choices([1, 2, 3], weights=[6, 3, 1])[0])]
    lines[0] = _mark(f"b{i}s") + lines[0]
    lines[-1] = lines[-1] + _mark(f"b{i}e")
    return SourceBlock(i, "math", "\\begin{align*}\n" + " \\\\\n".join(lines) + "\n\\end{align*}")


def _caption(i: int, rng: random.Random) -> str:
    return rf"\caption{{{_mark(f'b{i}s')}{_noun(rng)}の{rng.choice(['結果', '比較', '概要', '例', '構成'])}{_mark(f'b{i}e')}}}"


def _table(rng: random.Random, i: int) -> list[SourceBlock]:
    cols = rng.randint(2, 5)
    rows = [" & ".join(_noun(rng) for _ in range(cols))]
    rows += [" & ".join(f"{rng.uniform(0, 100):.1f}" for _ in range(cols)) for _ in range(rng.randint(2, 6))]
    body = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  {_caption(i + 1, rng)}",
        # 目印は表の上と下の別の行に置く。表と同じ行に置くと、表全体が1行の文字のように扱われ、
        # 目印が表の縦の真ん中の高さに記録されてしまう
        f"  {_mark(f'b{i}s')}\\par",
        rf"  \begin{{tabular}}{{{'l' * cols}}}",
        r"    \toprule",
        f"    {rows[0]} \\\\",
        r"    \midrule",
        *(f"    {row} \\\\" for row in rows[1:]),
        r"    \bottomrule",
        r"  \end{tabular}",
        f"  \\par{_mark(f'b{i}e')}",
        r"\end{table}",
    ]
    tex = "\n".join(body)
    # 表とキャプションは同じ環境の中にあるので、TeXは表側にまとめて持たせ、キャプションは目印だけ使う
    return [SourceBlock(i, "table", tex), SourceBlock(i + 1, "caption", "")]


def _figure(rng: random.Random, i: int) -> list[SourceBlock]:
    width, height = rng.uniform(4, 12), rng.uniform(2, 6)
    body = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        f"  {_mark(f'b{i}s')}\\par",
        rf"  \fbox{{\rule{{0pt}}{{{height:.1f}cm}}\hspace{{{width:.1f}cm}}}}",
        f"  \\par{_mark(f'b{i}e')}",
        f"  {_caption(i + 1, rng)}",
        r"\end{figure}",
    ]
    return [SourceBlock(i, "figure", "\n".join(body)), SourceBlock(i + 1, "caption", "")]


CODE_LINES = [
    "int main(void) {", "    return 0;", "}", "for (i = 0; i < n; i++) {", "    sum += a[i];", "x = x + 1;",
    "def f(x):", "    return x * 2", "if x > 0:", "    print(x)", "import numpy as np", "y = np.zeros(10)",
    "movq %rsp, %rbp", "pushq %rbx", "addq $8, %rax", "jmp L1", "L1:", "cmpq $0, %rcx",
    "$ ./a.out", "$ gcc -O2 main.c", "while (p != NULL) {", "    p = p->next;", "char *s = argv[1];",
    "return self.value", "class Node:", "for line in f:", "printf(\"%d\\n\", x);",
]  # fmt: skip


def _code(rng: random.Random, i: int) -> SourceBlock:
    """プログラムのコード。中身はTeXとして解釈されないので、目印はコードの上と下の別の行に置く"""
    lines = [rng.choice(CODE_LINES) for _ in range(rng.randint(2, 12))]
    if rng.random() < 0.5:
        env = "verbatim"
        begin = r"\begin{verbatim}"
    else:
        env = "lstlisting"
        options = [r"basicstyle=\ttfamily\small"]
        if rng.random() < 0.5:
            options.append("numbers=left")
        if rng.random() < 0.4:
            options.append("frame=single")
        begin = rf"\begin{{lstlisting}}[{','.join(options)}]"
    body = [f"{_mark(f'b{i}s')}\\par", begin, *lines, rf"\end{{{env}}}", f"\\par{_mark(f'b{i}e')}"]
    return SourceBlock(i, "code", "\n".join(body))


# ---- 文書全体 ----

PREAMBLE = r"""%(documentclass)s
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{listings}
%(fonts)s
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
"""

# PDFに埋め込む日本語フォント（dvipdfmx のフォントマップ）。None は既定（Ryumin / GothicBBB の名前で出る）
FONT_MAPS = [None, "uptex-ipaex.map", "uptex-haranoaji.map", "uptex-ipa.map"]


def _preamble(rng: random.Random) -> str:
    """文書クラスとフォントの設定をランダムに選ぶ。

    実際の文書では、本文がゴシック体だったり、見出しが明朝体だったりする。学習データの書体が
    1通りだと、モデルが「ゴシック体なら見出し」のような近道を覚えてしまうので、散らしておく。
    """
    size = rng.choice(["10pt", "11pt", "12pt"])
    columns = ",twocolumn" if rng.random() < 0.25 else ""
    if rng.random() < 0.3:
        documentclass = rf"\documentclass[uplatex,dvipdfmx,ja=standard,a4paper,{size}{columns}]{{bxjsarticle}}"
    else:
        documentclass = rf"\documentclass[uplatex,a4paper,{size}{columns}]{{jsarticle}}"
    fonts = []
    if rng.random() < 0.2:
        fonts.append(r"\renewcommand{\kanjifamilydefault}{\gtdefault}")  # 本文もゴシック体
    if rng.random() < 0.2:
        fonts.append(r"\renewcommand{\headfont}{\mcfamily\bfseries}")  # 見出しを明朝体に
    fonts.append(rng.choices(["", r"\usepackage{lmodern}", r"\usepackage{mathptmx}"], weights=[6, 2, 2])[0])
    return PREAMBLE % {"documentclass": documentclass, "fonts": "\n".join(fonts)}


def random_document(rng: random.Random) -> tuple[str, list[SourceBlock]]:
    """ランダムな文書のTeXと、置いたブロックの一覧を返す"""
    blocks: list[SourceBlock] = []
    next_id = 0

    def add(new: SourceBlock | list[SourceBlock]) -> None:
        nonlocal next_id
        new = new if isinstance(new, list) else [new]
        blocks.extend(new)
        next_id += len(new)

    title_lines = []
    if rng.random() < 0.6:
        title_lines = [
            rf"\title{{{_mark(f'b{next_id}s')}{_noun(rng)}を用いた{_noun(rng)}の{rng.choice(VERBS[:3])[:-2]}{_mark(f'b{next_id}e')}}}",
            r"\author{" + rng.choice(["山田 太郎", "佐藤 花子", "情報 太郎"]) + "}",
            r"\date{2026年" + f"{rng.randint(1, 12)}月{rng.randint(1, 28)}日" + "}",
        ]
        add(SourceBlock(next_id, "other", r"\maketitle"))  # タイトル（ラベル定義に title が無いので other）

    for _ in range(rng.randint(3, 8)):  # 節の数
        add(_heading(rng, next_id))
        for _ in range(rng.randint(1, 5)):
            kind = rng.choices(["paragraph", "list", "math", "table", "figure", "code"], weights=[10, 2, 3, 1, 1, 2])[0]
            builder = {
                "paragraph": _paragraph, "list": _list, "math": _math, "table": _table, "figure": _figure, "code": _code,
            }[kind]  # fmt: skip
            add(builder(rng, next_id))

    body = "\n\n".join(block.tex for block in blocks if block.tex)
    tex = _preamble(rng) + "\n".join(title_lines) + "\n\\begin{document}\n\n" + body + "\n\n\\end{document}\n"
    return tex, blocks
