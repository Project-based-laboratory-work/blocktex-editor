"""学習データ用に、ランダムな文書（TeX）を作る。正解は目印（\\special{pdf:dest}）で作るので、色は使わない。

評価用の文書（evaluation/synth_doc.py）とは別の作り方にしている。同じ作り方の文書で学習すると、
評価で実力より高い点が出るため（block_classifier と同じ考え方）。言葉も作りも変え、
ルールが苦手になるように、見た目をわざとばらつかせている:

    - 文書クラス（jsarticle / bxjsarticle / article）、文字サイズ、2段組、本文の書体（明朝 / ゴシック、Times / Helvetica）
    - 見出し: 番号あり / 番号なし（\\section*）/ 本文と同じ大きさの太字の1行 / 中央寄せ
    - 段落: 字下げなし（\\noindent）、太字で始まる段落、文中の数式
    - 箇条書き: itemize / enumerate / description、記号の違う項目（★ や ― など）
    - 表: 罫線の種類、キャプションが表の上 / 下
    - 図: 画像 / 枠線だけ、キャプションが図の下 / 上
    - 数式: equation（式番号つき）/ align* / gather* / \\[ \\]
    - コード: verbatim / lstlisting（枠・行番号つき）

正解を目印から作る部分（evaluation/ground_truth.py の build_ground_truth）と、ブロックの入れ物（SourceBlock）・
ページ番号とヘッダーの目印（page_style）は、評価と同じ仕組みを使う（作り方の決まりなので、モデルが有利にはならない）。
"""

import io
import random
from contextlib import contextmanager
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from evaluation.synth_doc import SourceBlock, mark, page_style
from hybrid_extractor.content import DEFAULT as DEFAULT_CONTENT
from hybrid_extractor.content import Content, get_content

# 中身の材料（単語・見出し・数式の部品・コード）は content.py のパック。下の名前は今までの名前（default のパック）
WORDS = DEFAULT_CONTENT.words
CELL_WORDS = DEFAULT_CONTENT.cell_words
MATH_PARTS = DEFAULT_CONTENT.math_parts
CODE_SNIPPETS = DEFAULT_CONTENT.code_snippets
LIST_LABELS = [r"\textbullet", "★", "―", "・", r"$\triangleright$", "(\\alph*)", "\\roman*."]


# 書体（PDF読み込みの許可リストの段階1。ai/CONSTRAINTS.md）。どれも、書体の名前に頼るルールが崩れないかを見るため。
# 数式の書体は CM 系（既定・lmodern・mathpazo・mathptmx）だけ（数式OCR の学習書体と同じ）。
# (プリアンブル, 選ぶ重み)。mathptmx・mathpazo は数式の文字も本文の斜体と同じ書体になる
TEXT_FONTS = {
    "default": ("", 3),
    "lmodern": (r"\usepackage{lmodern}", 2),
    "newtxtext": (r"\usepackage{newtxtext}", 1),
    "times": (r"\usepackage{times}", 1),
    "mathptmx": (r"\usepackage{mathptmx}", 1),
    "mathpazo": (r"\usepackage{mathpazo}", 1),
    "helvet": (r"\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}", 1),
    "cm_sans": (r"\renewcommand{\familydefault}{\sfdefault}", 1),
}
MONO_FONTS = {"default": "", "inconsolata": r"\usepackage{inconsolata}"}
JA_FONT_MAPS = {"default": None, "ipaex": "uptex-ipaex.map", "ipa": "uptex-ipa.map", "haranoaji": "uptex-haranoaji.map"}


@dataclass
class Variant:
    """作る文書の「見た目の制限」と「中身の材料」。何も指定しなければ今までと同じ文書（同じ種なら同じ文書）になる。

    見たことのない見た目・中身で AI を評価するため（AI.md の 7.7）:
      学習データ: exclude={"text_font": ["times", "newtxtext", "mathptmx"]}（Times の仲間を見せない）
      評価データ: only={"text_font": ["times", "newtxtext", "mathptmx"]}（Times の仲間だけ）、content="real"（本物の文と数式）
    制限できる属性は VARIANT_ATTRS"""
    only: dict[str, list[str]] = field(default_factory=dict)      # 属性 → 使ってよい値
    exclude: dict[str, list[str]] = field(default_factory=dict)   # 属性 → 使わない値
    content: str = "default"                                       # 中身の材料のパック（content.py）


VARIANT_ATTRS = ("cls", "text_font", "mono_font", "ja_font", "list_label")
_ACTIVE: list = [Variant(), DEFAULT_CONTENT]   # 今作っている文書の Variant と材料（use_variant の中で入れ替える）


@contextmanager
def use_variant(variant: Variant | None):
    """この中で作る文書に、variant の制限と材料を使う"""
    old = list(_ACTIVE)
    variant = variant or Variant()
    _ACTIVE[:] = [variant, get_content(variant.content)]
    try:
        yield
    finally:
        _ACTIVE[:] = old


def add_variant_args(parser) -> None:
    """データを作るコマンドに、Variant の指定（--only / --exclude / --content）を足す"""
    parser.add_argument("--only", action="append", default=[], metavar="属性=値,値",
                        help=f"その属性はこの値だけにする（例: text_font=times,newtxtext,mathptmx）。属性は {', '.join(VARIANT_ATTRS)}")
    parser.add_argument("--exclude", action="append", default=[], metavar="属性=値,値",
                        help="その属性はこの値を使わない（学習データから見た目を抜くとき）")
    parser.add_argument("--content", default="default", help="中身の材料のパック（default / alt / real。content.py）")


def variant_from_args(args) -> Variant:
    def parse(items: list[str]) -> dict[str, list[str]]:
        out = {}
        for item in items:
            attr, _, values = item.partition("=")
            if attr not in VARIANT_ATTRS:
                raise SystemExit(f"制限できない属性: {attr}（{', '.join(VARIANT_ATTRS)}）")
            out[attr] = values.split(",")
        return out

    return Variant(only=parse(args.only), exclude=parse(args.exclude), content=args.content)


def C() -> Content:
    """今使っている中身の材料"""
    return _ACTIVE[1]


def pick(rng: random.Random, attr: str, options: list, weights: list | None = None):
    """options から1つ選ぶ。Variant の only / exclude で、選べる値を絞る（絞らなければ rng.choice と同じ乱数の使い方）"""
    variant = _ACTIVE[0]
    keep = [k for k, o in enumerate(options)
            if (attr not in variant.only or o in variant.only[attr])
            and (attr not in variant.exclude or o not in variant.exclude[attr])]
    if not keep:
        raise ValueError(f"{attr} に選べる値が無い（only {variant.only.get(attr)} / exclude {variant.exclude.get(attr)}）")
    if len(keep) == len(options):
        return rng.choices(options, weights=weights)[0] if weights else rng.choice(options)
    filtered = [options[k] for k in keep]
    return rng.choices(filtered, weights=[weights[k] for k in keep])[0] if weights else rng.choice(filtered)


@dataclass
class Style:
    """1つの文書の見た目の設定"""
    lang: str
    cls: str           # 文書クラス
    size: str          # 文字サイズ
    two_column: bool
    gothic_body: bool  # 本文をゴシック体にする（日本語だけ）
    text_font: str     # 欧文の本文の書体（TEXT_FONTS のキー）。数式の書体も変わるものがある（mathptmx・mathpazo）
    t1: bool           # \usepackage[T1]{fontenc}（欧文の書体の名前が CMR → SFRM・CMTT → SFTT などに変わる）
    mono_font: str     # 等幅の書体（MONO_FONTS のキー）
    ja_font: str       # 和文の書体（JA_FONT_MAPS のキー。dvipdfmx のマップを \special{pdf:mapfile} で切り替える）
    numbered: bool     # 見出しに番号を付ける（False なら \section*）
    centered_headings: bool
    page_number: bool
    header: bool


def random_style(rng: random.Random, lang: str) -> Style:
    cls = pick(rng, "cls", ["jsarticle", "jsarticle", "bxjsarticle"] + (["article", "article"] if lang == "en" else []))
    return Style(
        lang=lang,
        cls=cls,
        size=rng.choice(["10pt", "11pt", "12pt"]),
        two_column=rng.random() < 0.3,
        gothic_body=lang == "ja" and rng.random() < 0.2,
        text_font=pick(rng, "text_font", list(TEXT_FONTS), weights=[w for _, w in TEXT_FONTS.values()]),
        t1=rng.random() < 0.4,
        mono_font=pick(rng, "mono_font", list(MONO_FONTS)),
        ja_font=pick(rng, "ja_font", list(JA_FONT_MAPS)),
        numbered=rng.random() < 0.65,
        centered_headings=cls == "article" and rng.random() < 0.4,
        page_number=rng.random() < 0.6,
        header=rng.random() < 0.3,
    )


def preamble(style: Style) -> str:
    # dvipdfmx は文書クラスの全体の指定にする。graphicx だけに付けると、newtxtext を読み込んだ文書で画像の命令が
    # 別のドライバ用（em: graph）になり、dvipdfmx が画像を黙って捨てる（図が消えてキャプションだけ残る）
    options = ["dvipdfmx", "a4paper", style.size] + (["twocolumn"] if style.two_column else [])
    if style.cls in ("jsarticle", "bxjsarticle"):
        options.insert(0, "uplatex")
    lines = [
        rf"\documentclass[{','.join(options)}]{{{style.cls}}}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{booktabs}",
        r"\usepackage{listings}",
        r"\usepackage{enumitem}",
        r"\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}",
    ]
    if style.lang == "ja" and style.cls == "article":
        raise ValueError("article クラスは英語の文書だけ")
    if style.t1:
        lines.append(r"\usepackage[T1]{fontenc}")
    for package in (TEXT_FONTS[style.text_font][0], MONO_FONTS[style.mono_font]):
        if package:
            lines.append(package)
    if JA_FONT_MAPS[style.ja_font]:
        lines.append(rf"\AtBeginDvi{{\special{{pdf:mapfile {JA_FONT_MAPS[style.ja_font]}}}}}")
    if style.gothic_body:
        lines.append(r"\renewcommand{\kanjifamilydefault}{\gtdefault}")
    if style.centered_headings:
        lines.append(r"\usepackage{sectsty}\allsectionsfont{\centering}")  # sectsty は jsarticle 系では使えない
    header = {"ja": "学習用の文書", "en": "Training document"}[style.lang] if style.header else None
    lines.append(page_style(style.page_number, header))
    lines.append(r"\begin{document}")
    return "\n".join(lines) + "\n"


# ---- 文と段落 ----


def sentence(rng: random.Random, lang: str, math: bool = False) -> str:
    w = C().words[lang]
    if C().sentences:
        text = rng.choice(C().sentences[lang])   # 本物の文（real のパック）
    elif C().templates:
        text = fill_template(rng, rng.choice(C().templates[lang]), w, lang)   # いくつかの文の形（wide のパック）
    elif lang == "ja":
        text = rng.choice(w["nouns"]) + "と" + rng.choice(w["nouns"]) + rng.choice(w["verbs"]) + rng.choice(w["ends"])
    else:
        text = f"{rng.choice(w['nouns'])} {rng.choice(w['verbs'])} {rng.choice(w['nouns'])}{rng.choice(w['ends'])}"
        text = text[0].upper() + text[1:]
    if math:
        part = f"${rng.choice(C().inline_formulas)}$" if C().inline_formulas else rng.choice(C().inline_parts)
        text = (f"{part}について、" if lang == "ja" else f"For {part}, ") + (text if lang == "ja" else text[0].lower() + text[1:])
    return text


def fill_template(rng: random.Random, template: str, w: dict, lang: str) -> str:
    """文の形の {n}（名詞）・{N}（文頭の名詞）・{v}（動詞）・{e}（文の終わり）を、前から順に埋める"""
    out = template
    for key, pool in (("{N}", "nouns"), ("{n}", "nouns"), ("{v}", "verbs"), ("{e}", "ends")):
        while key in out:
            word = rng.choice(w[pool])
            if key == "{N}":
                word = word[0].upper() + word[1:]
            out = out.replace(key, word, 1)
    return out


def sentences(rng: random.Random, lang: str, n: int, math_rate: float = 0.0) -> str:
    return C().words[lang]["sep"].join(sentence(rng, lang, rng.random() < math_rate) for _ in range(n))


def decorate(rng: random.Random, text: str, lang: str) -> str:
    """ときどき語を太字・下線・斜体にする（正解の runs は作らない。分類の学習には使わないため）"""
    if rng.random() > 0.3:
        return text
    command = rng.choice([r"\textbf", r"\underline"] + ([r"\textit"] if lang == "en" else []))
    word = rng.choice(C().words[lang]["nouns"])
    return text.replace(word, f"{command}{{{word}}}", 1) if word in text else text


def paragraph(rng: random.Random, lang: str, i: int) -> SourceBlock:
    text = decorate(rng, sentences(rng, lang, rng.randint(1, 6), math_rate=0.2), lang)
    prefix = ""
    r = rng.random()
    if r < 0.15:
        prefix = r"\noindent "  # 字下げの無い段落
    elif r < 0.3:
        lead = rng.choice(C().words[lang]["leads"])
        text = rf"\textbf{{{lead}}}{C().words[lang]['sep'] or '　'}{text}"  # 太字で始まる段落（見出しと間違えやすい）
    elif r < 0.42:
        # 表・図を参照する文で始まる段落（「表1 で始まればキャプション」という近道を覚えさせないため）
        text = reference_lead(rng, lang) + (C().words[lang]["sep"]) + text
    return SourceBlock(i, "paragraph", None, f"{prefix}{mark(f'b{i}s')}{text}{mark(f'b{i}e')}")


def reference_lead(rng: random.Random, lang: str) -> str:
    """表・図を参照する文（キャプションと同じく「表1」「Figure 2」で始まる）"""
    n = rng.randint(1, 5)
    if lang == "ja":
        return rng.choice([f"表{n}に示すように、{rng.choice(C().words[lang]['nouns'])}を比べた。",
                           f"図{n}は、{rng.choice(C().words[lang]['nouns'])}の様子である。",
                           f"表{n} の結果から、{rng.choice(C().words[lang]['nouns'])}が分かる。"])
    return rng.choice([f"Table {n} shows {rng.choice(C().words[lang]['nouns'])}.",
                       f"Figure {n} compares {rng.choice(C().words[lang]['nouns'])} and {rng.choice(C().words[lang]['nouns'])}.",
                       f"Fig. {n} illustrates {rng.choice(C().words[lang]['nouns'])}."])


def heading(rng: random.Random, style: Style, i: int) -> SourceBlock:
    level = rng.choice([1, 1, 2, 2, 3])
    title = rng.choice(C().words[style.lang]["headings"])
    s, e = mark(f"b{i}s"), mark(f"b{i}e")
    if level == 3 and rng.random() < 0.4:
        # 本文と同じ大きさの、太字の1行だけの見出し（番号も無い）
        return SourceBlock(i, "heading", 3, rf"\par\medskip\noindent{s}\textbf{{{title}}}{e}\par")
    command = ["section", "subsection", "subsubsection"][level - 1] + ("" if style.numbered else "*")
    return SourceBlock(i, "heading", level, rf"\{command}{{{s}{title}{e}}}")


def list_block(rng: random.Random, lang: str, i: int) -> SourceBlock:
    env = rng.choice(["itemize", "itemize", "enumerate", "description"])
    items = [decorate(rng, sentences(rng, lang, rng.randint(1, 2)), lang) for _ in range(rng.randint(2, 5))]
    if env == "description":
        terms = rng.sample(C().words[lang]["leads"] + C().words[lang]["headings"], len(items))
        lines = [rf"\item[{t}] {item}" for t, item in zip(terms, items)]
        opts = ""
    else:
        lines = [rf"\item {item}" for item in items]
        label = pick(rng, "list_label", LIST_LABELS)
        if env == "itemize" and "*" not in label and rng.random() < 0.5:
            opts = f"[label={label}]"
        elif env == "enumerate" and "*" in label and rng.random() < 0.5:
            opts = f"[label={label}]"
        else:
            opts = ""
    # 目印は最初の項目の文字の前と、最後の項目の文字の後
    first = lines[0]
    head_end = first.index("]") + 1 if env == "description" else len(r"\item ")
    lines[0] = first[:head_end] + mark(f"b{i}s") + first[head_end:]
    lines[-1] += mark(f"b{i}e")
    tex = "\n".join([rf"\begin{{{env}}}{opts}", *("  " + line for line in lines), rf"\end{{{env}}}"])
    return SourceBlock(i, "list", None, tex)


def title_blocks(rng: random.Random, lang: str, i: int) -> list[SourceBlock]:
    """タイトル・著者・（ときどき）日付。日付を出さない文書もある（\\date{}）"""
    w = C().words[lang]
    title = (rng.choice(w["nouns"]) + "と" + rng.choice(w["nouns"]) + "の" + rng.choice(["調査", "記録", "検討"])) if lang == "ja" \
        else f"On {rng.choice(w['nouns'])} and {rng.choice(w['nouns'])}".title()
    with_date = rng.random() < 0.7
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    date = f"2025年{month}月{day}日" if lang == "ja" else rng.choice([f"2025/{month}/{day}", f"{day} March 2025"])
    tex = [rf"\title{{{mark(f'b{i}s')}{title}{mark(f'b{i}e')}}}",
           rf"\author{{{mark(f'b{i + 1}s')}{rng.choice(w['authors'])}{mark(f'b{i + 1}e')}}}",
           rf"\date{{{mark(f'b{i + 2}s')}{date}{mark(f'b{i + 2}e')}}}" if with_date else r"\date{}",
           r"\maketitle"]
    blocks = [SourceBlock(i, "title", None, "\n".join(tex), full_width=True),
              SourceBlock(i + 1, "author", None, "", full_width=True)]
    if with_date:
        blocks.append(SourceBlock(i + 2, "date", None, "", full_width=True))
    return blocks


# ---- 表・図・数式・コード ----


def caption(rng: random.Random, lang: str, i: int) -> str:
    text = rng.choice(C().words[lang]["nouns"]) + rng.choice(C().words[lang]["captions"])
    return rf"\caption{{{mark(f'b{i}s')}{text[0].upper() + text[1:]}{mark(f'b{i}e')}}}"


def table_blocks(rng: random.Random, lang: str, i: int) -> list[SourceBlock]:
    """表とキャプション。キャプションは表の上か下。番号: 表 i、キャプション i+1"""
    cols, rows = rng.randint(2, 5), rng.randint(2, 6)
    style = rng.choice(["grid", "booktabs", "hline", "none"])
    header = rng.sample(C().cell_words[lang], cols)
    body = [[f"{rng.uniform(0, 500):.{rng.choice([0, 1, 2])}f}" for _ in range(cols)] for _ in range(rows)]
    rule = "|" if style == "grid" else ""
    spec = rule + rule.join(rng.choice("lcr") for _ in range(cols)) + rule
    top, mid, bottom = {"grid": (r"\hline",) * 3, "hline": (r"\hline",) * 3,
                        "booktabs": (r"\toprule", r"\midrule", r"\bottomrule"), "none": ("", "", "")}[style]
    lines = [top, " & ".join(header) + r" \\", mid]
    for k, row in enumerate(body):
        lines.append(" & ".join(row) + r" \\")
        if style == "grid" and k < rows - 1:
            lines.append(r"\hline")
    lines.append(bottom)
    table_tex = [f"  {mark(f'b{i}s')}\\par", rf"  \begin{{tabular}}{{{spec}}}", *(f"    {x}" for x in lines if x),
                 r"  \end{tabular}", f"  \\par{mark(f'b{i}e')}"]
    cap = f"  {caption(rng, lang, i + 1)}"
    above = rng.random() < 0.6
    body_tex = [cap, *table_tex] if above else [*table_tex, cap]
    tex = "\n".join([r"\begin{table}[htbp]", r"  \centering", *body_tex, r"\end{table}"])
    table = SourceBlock(i, "table", None, tex, kind="table", own_line_marks=True,
                        text="\n".join(" ".join(r) for r in [header, *body]))
    cap_block = SourceBlock(i + 1, "caption", None, "")
    return [cap_block, table] if above else [table, cap_block]


def figure_png(rng: random.Random) -> bytes:
    """図の画像（折れ線か円）"""
    w, h = 500, rng.randint(250, 400)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    if rng.random() < 0.5:
        pts = [(30 + k * (w - 60) // 8, rng.randint(30, h - 30)) for k in range(9)]
        d.line(pts, fill=(30, 60, 160), width=4)
        d.line([(30, h - 20), (w - 20, h - 20)], fill="black", width=2)
    else:
        for _ in range(rng.randint(3, 6)):
            x, y, r = rng.randint(60, w - 60), rng.randint(60, h - 60), rng.randint(15, 50)
            d.ellipse([x - r, y - r, x + r, y + r], outline=(160, 40, 40), width=3)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def figure_blocks(rng: random.Random, lang: str, i: int) -> list[SourceBlock]:
    """図とキャプション。キャプションはたいてい図の下。番号: 図 i、キャプション i+1"""
    files = {}
    if rng.random() < 0.7:
        name = f"pic{i}.png"
        files[name] = figure_png(rng)
        body = rf"\includegraphics[width={rng.uniform(0.35, 0.75):.2f}\linewidth]{{{name}}}"
    else:
        body = rf"\fbox{{\rule{{0pt}}{{{rng.uniform(1.5, 3.5):.1f}cm}}\hspace{{{rng.uniform(0.3, 0.6):.2f}\linewidth}}}}"
    fig_tex = [f"  {mark(f'b{i}s')}\\par", f"  {body}", f"  \\par{mark(f'b{i}e')}"]
    cap = f"  {caption(rng, lang, i + 1)}"
    below = rng.random() < 0.8
    tex = "\n".join([r"\begin{figure}[htbp]", r"  \centering", *(fig_tex + [cap] if below else [cap] + fig_tex),
                     r"\end{figure}"])
    fig = SourceBlock(i, "figure", None, tex, kind="figure", own_line_marks=True, files=files)
    cap_block = SourceBlock(i + 1, "caption", None, "")
    return [fig, cap_block] if below else [cap_block, fig]


def math_block(rng: random.Random, i: int) -> SourceBlock:
    """別行立ての数式。式番号つき（equation）、複数行（align* / gather*）、\\[ \\]"""
    env = rng.choice(["equation", "align*", "gather*", "display"])
    n = 1 if env in ("equation", "display") else rng.randint(2, 3)
    exprs = []
    for _ in range(n):
        if C().display_formulas:
            exprs.append(rng.choice(C().display_formulas))   # 本物の数式（real のパック）。align* でも & は付けない
            continue
        parts = rng.sample(C().math_parts, rng.randint(1, 3))
        lhs = rng.choice(["E", "g(x)", "P", "L_{0}"])
        amp = "&" if env == "align*" else ""
        exprs.append(f"{lhs} {amp}= " + " + ".join(parts))
    exprs[0] = mark(f"b{i}s") + exprs[0]
    exprs[-1] += mark(f"b{i}e")
    body = " \\\\\n".join(exprs)
    tex = rf"\[{body}\]" if env == "display" else f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"
    return SourceBlock(i, "math", None, tex)


def code_block(rng: random.Random, i: int) -> SourceBlock:
    lines = rng.choice(C().code_snippets)
    if rng.random() < 0.5:
        begin, end = r"\begin{verbatim}", r"\end{verbatim}"
    else:
        opts = [r"basicstyle=\ttfamily" + rng.choice(["", r"\small", r"\footnotesize"]), "columns=fixed"]
        if rng.random() < 0.3:
            opts.append("frame=single")
        if rng.random() < 0.3:
            opts.append(r"numbers=left,numberstyle=\tiny")
        begin, end = rf"\begin{{lstlisting}}[{','.join(opts)}]", r"\end{lstlisting}"
    tex = "\n".join([f"{mark(f'b{i}s')}\\par", begin, *lines, end, f"\\par{mark(f'b{i}e')}"])
    return SourceBlock(i, "code", None, tex, own_line_marks=True, text="\n".join(lines))


# ---- 文書 ----


def random_document(rng: random.Random, lang: str, variant: Variant | None = None) -> tuple[str, list[SourceBlock], Style]:
    """ランダムな文書の TeX、置いたブロックの一覧、見た目の設定を返す。variant で見た目を絞り、中身の材料を変えられる"""
    with use_variant(variant):
        return _random_document(rng, lang)


def _random_document(rng: random.Random, lang: str) -> tuple[str, list[SourceBlock], Style]:
    style = random_style(rng, lang)
    blocks: list[SourceBlock] = []
    if rng.random() < 0.6:
        blocks += title_blocks(rng, lang, 0)
    for _ in range(rng.randint(2, 5)):
        blocks.append(heading(rng, style, len(blocks)))
        for _ in range(rng.randint(1, 4)):
            r = rng.random()
            if r < 0.5:
                blocks.append(paragraph(rng, lang, len(blocks)))
            elif r < 0.65:
                blocks.append(list_block(rng, lang, len(blocks)))
            elif r < 0.73:
                blocks += table_blocks(rng, lang, len(blocks))
            elif r < 0.8:
                blocks += figure_blocks(rng, lang, len(blocks))
            elif r < 0.9:
                blocks.append(math_block(rng, len(blocks)))
            else:
                blocks.append(code_block(rng, len(blocks)))
    body = "\n\n".join(b.tex for b in blocks if b.tex)
    return preamble(style) + body + "\n\\end{document}\n", blocks, style


def document_files(blocks: list[SourceBlock]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for b in blocks:
        files.update(b.files)
    return files
