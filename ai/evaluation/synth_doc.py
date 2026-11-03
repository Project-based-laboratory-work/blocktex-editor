"""
評価用のランダムな文書（TeX）を作る。各ブロックの始まりと終わりに目印 \\mk{b{番号}s} / \\mk{b{番号}e} を埋め込む。
文中の装飾（太字・斜体・下線）も、装飾した語の前後に目印 \\mk{b{番号}d{何個目}s} / \\mk{b{番号}d{何個目}e} を置く。
ページ番号には \\mk{pn{ページ}s} / \\mk{pn{ページ}e}、ヘッダー（柱）には \\mk{hd{ページ}s} / \\mk{hd{ページ}e} を置く。

文書に入れるもの（FEATURES）を選べるので、表だけ・2段組だけ、のように機能ごとに評価できる。

block_classifier/doc_gen.py（学習データ用）とは別に作る。同じ作り方の文書で評価すると、
block_classifier だけが「学習で見たことのある文書」で評価されて、実力より高く出てしまうため。
"""
import io
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

# 文書に入れられるもの。見出し・段落・箇条書きはいつも入る
FEATURES = ["title", "decoration", "table", "figure", "math", "code", "two_column", "page_number", "header"]

PREAMBLE = r"""\documentclass[uplatex,a4paper,%(size)s%(columns)s]{jsarticle}
\usepackage[dvipdfmx]{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{listings}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
%(pagestyle)s
\begin{document}
"""

# ページ番号もヘッダーも出さない（ページをまたぐブロックの範囲に入らないように）
NO_PAGE_STYLE = r"\pagestyle{empty}"
# ページ番号・ヘッダーを出し、その前後に目印を置く。jsarticle の \maketitle は1ページ目を plainfoot / plainhead にするので、
# それも同じにする（1ページ目だけ目印が無い、とならないように）
PAGE_STYLE = r"""\makeatletter
\def\ps@evalpage{\let\@mkboth\@gobbletwo%%
  \def\@oddhead{%(head)s}\let\@evenhead\@oddhead%%
  \def\@oddfoot{%(foot)s}\let\@evenfoot\@oddfoot}
\let\ps@plain\ps@evalpage
\let\ps@plainfoot\ps@evalpage
\let\ps@plainhead\ps@evalpage
\makeatother
\pagestyle{evalpage}"""
PAGE_NUMBER_FOOT = r"\hfil\mk{pn\thepage s}\thepage\mk{pn\thepage e}\hfil"
HEADER_TEXT = {"ja": "評価用の文書", "en": "Evaluation document"}  # ヘッダー（柱）の文字。右寄せにする

# 言語ごとの材料。英語の文書がASCIIだけになるように、英語側には日本語を入れない
WORDS = {
    "ja": {
        "nouns": ["数式", "文書", "画像", "モデル", "手法", "データ", "精度", "構造", "段落", "見出し", "評価", "結果", "座標", "変換"],
        "verbs": ["を検討する", "を提案する", "を比較した", "を評価した", "を示す", "を改善した", "を実装した", "を確認した"],
        "headings": ["はじめに", "関連研究", "提案手法", "実験", "結果", "考察", "まとめ", "今後の課題"],
        "authors": ["山田 太郎", "佐藤 花子", "情報 太郎"],
        "captions": ["の結果", "の比較", "の概要", "の例", "の構成"],
        "sep": "",    # 単語・文の区切り（日本語は詰める）
        "end": "。",
    },
    "en": {
        "nouns": ["the model", "the document", "the layout", "the method", "the dataset", "the accuracy", "the result", "each block"],
        "verbs": ["improves", "describes", "compares", "evaluates", "shows", "extracts", "detects", "predicts"],
        "headings": ["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion", "Conclusion"],
        "authors": ["Taro Yamada", "Hanako Sato", "Alice Smith"],
        "captions": [" results", " comparison", " overview", " example", " structure"],
        "sep": " ",
        "end": ".",
    },
}
# 表のセルに入れる短い語（2段組でもはみ出さないように短くする）
TABLE_WORDS = {
    "ja": ["精度", "時間", "手法", "条件", "件数", "誤差", "平均", "最大"],
    "en": ["Acc", "Time", "Method", "Case", "Count", "Error", "Mean", "Max"],
}
HEADING_COMMANDS = ["section", "subsection", "subsubsection"]  # 見出しレベル 1, 2, 3

# 文中の装飾。jsarticle の日本語は斜体の書体が無く、\textit を付けても見た目が変わらないので、日本語には付けない
DECORATIONS = {"ja": ["bold", "underline"], "en": ["bold", "italic", "underline"]}
DECORATION_TEX = {"bold": r"\textbf", "italic": r"\textit", "underline": r"\underline"}
DECORATION_RATE = 0.04  # 1文の中で、語を1つ装飾する割合（文書の半分くらいに装飾が入る）
TITLE_RATE = 0.6        # features を指定しないとき、文書の先頭にタイトル・著者・日付（\maketitle）を置く割合

# 表の罫線の種類。エディタ（web/src/tex.ts）で作れる形に合わせる
TABLE_RULE_STYLES = ["booktabs", "grid", "hline", "none"]

# 数式の部品（ASCIIだけ）。2〜4個を + - = でつなぐ
MATH_TERMS = [
    r"x^{2}", r"\frac{a}{b}", r"\sum_{i=1}^{n} x_{i}", r"\sqrt{y}", r"\alpha", r"\beta x", r"f(x)", r"e^{-t}",
    r"\int_{0}^{1} g(t) \, dt", r"\log n", r"2 \pi r", r"\lambda_{k}",
]  # fmt: skip

# コードの例（字下げを含む）。中身は verbatim / lstlisting なので TeX として解釈されない
CODE_SNIPPETS = [
    ["def area(w, h):", "    if w < 0:", "        return 0", "    return w * h"],
    ["for i in range(n):", "    total += a[i]", "print(total)"],
    ["int main(void) {", "    int x = 0;", "    while (x < 10) {", "        x++;", "    }", "    return x;", "}"],
    ["class Node:", "    def __init__(self, v):", "        self.value = v", "        self.next = None"],
    ["if (p != NULL) {", "    p = p->next;", "}"],
]


def page_style(page_number: bool, header: str | None) -> str:
    """ページ番号・ヘッダーの設定。どちらも無ければ何も出さない"""
    if not page_number and header is None:
        return NO_PAGE_STYLE
    head = rf"\hfil\mk{{hd\thepage s}}{header}\mk{{hd\thepage e}}" if header is not None else ""
    return PAGE_STYLE % {"head": head, "foot": PAGE_NUMBER_FOOT if page_number else ""}


def preamble(size: str = "11pt", two_column: bool = False, page_number: bool = False, header: str | None = None) -> str:
    """文書の頭（\\begin{document} まで）。header はヘッダー（柱）に出す文字"""
    return PREAMBLE % {
        "size": size,
        "columns": ",twocolumn" if two_column else "",
        "pagestyle": page_style(page_number, header),
    }


@dataclass
class SourceBlock:
    """文書に置いたブロック1つ。正解IRの label / level / runs などはここから取る"""
    id: int             # 目印の名前 b{id}s / b{id}e に使う
    label: str          # common/ir.py の Label のどれか
    level: int | None   # 見出しのときだけ 1〜3
    tex: str            # 空のブロック（author / date / caption）は、TeXを同じ環境のブロックがまとめて持つ
    # 文中の装飾。(目印の名前の頭, 装飾)。目印は <頭>s と <頭>e で、その間の文字に装飾が付いている
    decorations: list[tuple[str, str]] = field(default_factory=list)
    kind: str = "text"            # 正解IRの kind（"text" / "table" / "figure"）
    full_width: bool = False      # 2段組でも段をまたいで1つ（\maketitle のタイトルなど）
    own_line_marks: bool = False  # 目印をブロックの上と下の別の行に置いた（表・図・コード）。範囲を広げない
    text: str | None = None       # 正解の文字列を、PDFからではなくここから取る（コードの字下げを残すため）
    tex_value: str | None = None  # 数式の正解のLaTeX
    cells: list[list[str]] | None = None  # 表の正解（行 × 列）
    aligns: list[str] | None = None
    col_rules: list[bool] | None = None
    row_rules: list[bool] | None = None
    col_widths: list[float | None] | None = None
    files: dict[str, bytes] = field(default_factory=dict)  # コンパイルに要るファイル（図の画像）


def mark(name: str) -> str:
    return rf"\mk{{{name}}}"


def sentence(rng: random.Random, lang: str, block: SourceBlock | None = None) -> str:
    """文を1つ。block を渡すと、ときどき目的語を装飾して、その目印を block.decorations に記録する"""
    words = WORDS[lang]
    parts = [rng.choice(words["nouns"]), rng.choice(words["verbs"]), rng.choice(words["nouns"])]
    if block is not None and rng.random() < DECORATION_RATE:
        decoration = rng.choice(DECORATIONS[lang])
        name = f"b{block.id}d{len(block.decorations)}"
        block.decorations.append((name, decoration))
        # 目印は装飾の命令の外に置く（命令の中だと、下線などの描画に目印が混ざらないように）
        parts[2] = f"{mark(name + 's')}{DECORATION_TEX[decoration]}{{{parts[2]}}}{mark(name + 'e')}"
    text = words["sep"].join(parts) + words["end"]
    return text[0].upper() + text[1:]  # 英語は文頭を大文字に（日本語は変わらない）


def heading(rng: random.Random, lang: str, i: int) -> SourceBlock:
    level = rng.choice([1, 2, 3])
    command = HEADING_COMMANDS[level - 1]
    title = rng.choice(WORDS[lang]["headings"])
    return SourceBlock(i, "heading", level, rf"\{command}{{{mark(f'b{i}s')}{title}{mark(f'b{i}e')}}}")


def paragraph(rng: random.Random, lang: str, i: int, decorate: bool = True) -> SourceBlock:
    block = SourceBlock(i, "paragraph", None, "")
    text = WORDS[lang]["sep"].join(sentence(rng, lang, block if decorate else None) for _ in range(rng.randint(2, 6)))
    start, end = mark(f"b{i}s"), mark(f"b{i}e")
    block.tex = f"{start}{text}{end}"
    return block


def list_block(rng: random.Random, lang: str, i: int, decorate: bool = True) -> SourceBlock:
    block = SourceBlock(i, "list", None, "")
    items = [sentence(rng, lang, block if decorate else None) for _ in range(rng.randint(2, 4))]
    items[0] = mark(f"b{i}s") + items[0]
    items[-1] += mark(f"b{i}e")
    lines = [r"\begin{itemize}", *(rf"  \item {item}" for item in items), r"\end{itemize}"]
    block.tex = "\n".join(lines)
    return block


def title_blocks(rng: random.Random, lang: str, i: int) -> list[SourceBlock]:
    """タイトル・著者・日付の3ブロック（番号は i, i+1, i+2）。\\maketitle で1か所に出るので、TeXはタイトルにまとめる"""
    title, author, date = i, i + 1, i + 2
    words = WORDS[lang]
    title_text = words["sep"].join([rng.choice(words["nouns"]), rng.choice(words["verbs"]), rng.choice(words["nouns"])])
    title_text = title_text[0].upper() + title_text[1:]
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    date_text = f"2026年{month}月{day}日" if lang == "ja" else f"2026-{month:02d}-{day:02d}"
    tex = "\n".join([
        rf"\title{{{mark(f'b{title}s')}{title_text}{mark(f'b{title}e')}}}",
        rf"\author{{{mark(f'b{author}s')}{rng.choice(words['authors'])}{mark(f'b{author}e')}}}",
        rf"\date{{{mark(f'b{date}s')}{date_text}{mark(f'b{date}e')}}}",
        r"\maketitle",
    ])
    return [
        SourceBlock(title, "title", None, tex, full_width=True),
        SourceBlock(author, "author", None, "", full_width=True),
        SourceBlock(date, "date", None, "", full_width=True),
    ]


def caption_tex(rng: random.Random, lang: str, i: int) -> str:
    """キャプション。「表1:」「図1:」の番号は目印の外に出るが、同じ行なので正解の文字に入る"""
    words = WORDS[lang]
    text = rng.choice(words["nouns"]) + rng.choice(words["captions"])
    return rf"\caption{{{mark(f'b{i}s')}{text[0].upper() + text[1:]}{mark(f'b{i}e')}}}"


# 折り返す列（p{幅}）の幅。IR の col_widths は、エディタ（web/src/tex.ts）と同じく列の左右の余白（\tabcolsep）を含めた
# 列の幅を、ページの幅（A4 = 595.28pt）で割った値
WRAP_WIDTH_CM = {False: 5.0, True: 3.0}  # 1段組 / 2段組の段の中
CM = 72 / 2.54                            # 1cm を PDF の pt（1/72インチ）にした値
TABCOLSEP = 6 * 72 / 72.27                # jsarticle の \tabcolsep（6pt）を PDF の pt にした値
A4_WIDTH_PT = 595.28
# jsarticle は 11pt・12pt のとき文書全体を拡大する（\mag）。cm や pt で書いた長さも、PDFの上ではこの倍率になる
MAGNIFICATION = {"10pt": 1.0, "11pt": 1.095, "12pt": 1.2}


def wrap_column_width(two_column: bool, size: str) -> float:
    """折り返す列の、IR の col_widths の値"""
    width_pt = (WRAP_WIDTH_CM[two_column] * CM + 2 * TABCOLSEP) * MAGNIFICATION[size]
    return round(width_pt / A4_WIDTH_PT, 4)


def table_blocks(rng: random.Random, lang: str, i: int, two_column: bool = False, size: str = "10pt") -> list[SourceBlock]:
    """表（番号 i）とキャプション（番号 i+1）。列の揃え・罫線の種類をランダムに変え、正解の cells・揃え・罫線・列幅も持つ。
    ときどき1列を幅指定（p{幅}）にして、セルの中で折り返させる。2段組では、ときどき段をまたぐ表（table*）にする"""
    wide = two_column and rng.random() < 0.4
    cols = rng.randint(2, 4)
    n_body = rng.randint(2, 4)
    style = rng.choice(TABLE_RULE_STYLES)
    aligns = [rng.choice("lcr") for _ in range(cols)]
    wrap_col = rng.randrange(cols) if rng.random() < 0.3 else None
    narrow = two_column and not wide  # 2段組の段の中に置く表
    wrap_cm = WRAP_WIDTH_CM[narrow]
    header = rng.sample(TABLE_WORDS[lang], cols)
    body = [
        [
            WORDS[lang]["sep"].join(sentence(rng, lang) for _ in range(2)) if c == wrap_col else f"{rng.uniform(0, 100):.1f}"
            for c in range(cols)
        ]
        for _ in range(n_body)
    ]
    cells = [header, *body]
    if wrap_col is not None:
        aligns[wrap_col] = "l"  # 折り返す列は左寄せ（エディタの p{} 列と同じ）

    specs = [rf">{{\raggedright\arraybackslash}}p{{{wrap_cm}cm}}" if c == wrap_col else a for c, a in enumerate(aligns)]
    rule = "|" if style == "grid" else ""
    spec = rule + rule.join(specs) + rule
    top, middle, bottom = {
        "booktabs": (r"\toprule", r"\midrule", r"\bottomrule"),
        "grid": (r"\hline", r"\hline", r"\hline"),
        "hline": (r"\hline", r"\hline", r"\hline"),
        "none": (None, None, None),
    }[style]
    lines = [*([top] if top else []), " & ".join(header) + r" \\", *([middle] if middle else [])]
    for k, row in enumerate(body):
        lines.append(" & ".join(row) + r" \\")
        if style == "grid" and k < n_body - 1:
            lines.append(r"\hline")  # grid は行の間にもすべて線を引く
    lines += [bottom] if bottom else []

    # 横線: 上端・見出しの下・行の間・下端（行数+1個）。縦線: 左端・列の間・右端（列数+1個）
    has_rules = style != "none"
    row_rules = [has_rules, has_rules, *([style == "grid"] * (n_body - 1)), has_rules]
    col_rules = [style == "grid"] * (cols + 1)
    # 列幅は、折り返す列だけページの幅で割った値を入れ、ほかは None（IR の決まり）
    col_widths = [wrap_column_width(narrow, size) if c == wrap_col else None for c in range(cols)] if wrap_col is not None else None

    env = "table*" if wide else "table"
    tex = "\n".join([
        rf"\begin{{{env}}}[htbp]",
        r"  \centering",
        f"  {caption_tex(rng, lang, i + 1)}",
        # 目印は表の上と下の別の行に置く。表と同じ行に置くと、目印が表の縦の真ん中の高さに記録されてしまう
        f"  {mark(f'b{i}s')}\\par",
        rf"  \begin{{tabular}}{{{spec}}}",
        *(f"    {line}" for line in lines),
        r"  \end{tabular}",
        f"  \\par{mark(f'b{i}e')}",
        rf"\end{{{env}}}",
    ])
    table = SourceBlock(
        i, "table", None, tex, kind="table", own_line_marks=True, full_width=wide,
        text="\n".join(" ".join(row) for row in cells), cells=cells, aligns=aligns, col_rules=col_rules, row_rules=row_rules,
        col_widths=col_widths,
    )
    return [table, SourceBlock(i + 1, "caption", None, "", full_width=wide)]


def figure_image(rng: random.Random) -> bytes:
    """図の画像（棒グラフ風のPNG）"""
    width, height = 600, rng.randint(280, 420)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    bars = rng.randint(3, 7)
    for k in range(bars):
        x0 = 40 + k * (width - 80) // bars
        bar_height = rng.randint(height // 5, height - 60)
        color = tuple(rng.randint(40, 200) for _ in range(3))
        draw.rectangle([x0 + 8, height - 30 - bar_height, x0 + (width - 80) // bars - 8, height - 30], fill=color)
    draw.line([30, height - 30, width - 30, height - 30], fill="black", width=3)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def figure_blocks(rng: random.Random, lang: str, i: int, two_column: bool = False) -> list[SourceBlock]:
    """図（番号 i）とキャプション（番号 i+1）。たいていは画像、ときどき枠線だけの図（ベクターの描画）。
    2段組では、ときどき段をまたぐ図（figure*）にする"""
    wide = two_column and rng.random() < 0.4
    env = "figure*" if wide else "figure"
    files = {}
    if rng.random() < 0.7:
        name = f"fig{i}.png"
        files[name] = figure_image(rng)
        body = rf"\includegraphics[width={rng.uniform(0.4, 0.8):.2f}\linewidth]{{{name}}}"
    else:
        body = rf"\fbox{{\rule{{0pt}}{{{rng.uniform(2, 4):.1f}cm}}\hspace{{{rng.uniform(0.3, 0.7):.2f}\linewidth}}}}"
    tex = "\n".join([
        rf"\begin{{{env}}}[htbp]",
        r"  \centering",
        f"  {mark(f'b{i}s')}\\par",
        f"  {body}",
        f"  \\par{mark(f'b{i}e')}",
        f"  {caption_tex(rng, lang, i + 1)}",
        rf"\end{{{env}}}",
    ])
    figure = SourceBlock(i, "figure", None, tex, kind="figure", own_line_marks=True, full_width=wide, files=files)
    return [figure, SourceBlock(i + 1, "caption", None, "", full_width=wide)]


def math_block(rng: random.Random, lang: str, i: int) -> SourceBlock:
    """別行立ての数式（align*）。1〜3行"""
    lines = []
    for _ in range(rng.choice([1, 1, 2, 3])):
        terms = rng.sample(MATH_TERMS, rng.randint(2, 4))
        expr = terms[0]
        for term in terms[1:]:
            expr += f" {rng.choice(['+', '-'])} {term}"
        lines.append(f"{rng.choice(['y', 'z', 'F(x)', 'S'])} &= {expr}")
    marked = list(lines)
    marked[0] = mark(f"b{i}s") + marked[0]
    marked[-1] = marked[-1] + mark(f"b{i}e")
    tex = "\\begin{align*}\n" + " \\\\\n".join(marked) + "\n\\end{align*}"
    return SourceBlock(i, "math", None, tex, tex_value=" \\\\\n".join(lines))


def code_block(rng: random.Random, lang: str, i: int) -> SourceBlock:
    """プログラムのコード（字下げあり）。中身はTeXとして解釈されないので、目印はコードの上と下の別の行に置く。
    正解の文字はPDFからではなくソースから取る（行頭の空白＝字下げを正しく持つため）"""
    lines = rng.choice(CODE_SNIPPETS)
    if rng.random() < 0.5:
        begin, end = r"\begin{verbatim}", r"\end{verbatim}"
    else:
        frame = ",frame=single" if rng.random() < 0.4 else ""
        begin, end = rf"\begin{{lstlisting}}[basicstyle=\ttfamily\small,columns=fixed{frame}]", r"\end{lstlisting}"
    tex = "\n".join([f"{mark(f'b{i}s')}\\par", begin, *lines, end, f"\\par{mark(f'b{i}e')}"])
    return SourceBlock(i, "code", None, tex, own_line_marks=True, text="\n".join(lines))


def random_document(rng: random.Random, lang: str, features: set[str] | None = None) -> tuple[str, list[SourceBlock]]:
    """ランダムな文書のTeXと、置いたブロックの一覧を返す。

    features（FEATURES のうち入れるもの）を渡すと、そのとおりに入れる（表なら1つ以上の節に表を置く）。
    渡さないときは、ときどきタイトル・著者・日付を置き、文中の装飾を混ぜる（表・図などは入れない）。
    """
    if features is None:
        features = {"decoration"} | ({"title"} if rng.random() < TITLE_RATE else set())
    unknown = features - set(FEATURES)
    if unknown:
        raise ValueError(f"不明な features {sorted(unknown)}（使えるのは {FEATURES}）")
    decorate = "decoration" in features

    size = rng.choice(["10pt", "11pt", "12pt"])  # 表の列幅の正解が文字サイズで変わるので、先に決める
    blocks: list[SourceBlock] = []
    if "title" in features:
        blocks += title_blocks(rng, lang, 0)

    # 表・図・数式・コードは、どれかの節の最後に1つずつ置く（入れると決めたものは必ず入る）
    extras = [name for name in ("table", "figure", "math", "code") if name in features]
    n_sections = max(rng.randint(2, 4), len(extras))
    placed = dict(zip(rng.sample(range(n_sections), len(extras)), extras))
    for section in range(n_sections):
        blocks.append(heading(rng, lang, len(blocks)))  # 番号は「今までに置いたブロックの数」= 0, 1, 2, ...
        for _ in range(rng.randint(1, 3)):
            builder = rng.choice([paragraph, paragraph, list_block])  # 段落を出やすく
            blocks.append(builder(rng, lang, len(blocks), decorate))
        extra = placed.get(section)
        if extra == "table":
            blocks += table_blocks(rng, lang, len(blocks), "two_column" in features, size)
        elif extra == "figure":
            blocks += figure_blocks(rng, lang, len(blocks), "two_column" in features)
        elif extra == "math":
            blocks.append(math_block(rng, lang, len(blocks)))
        elif extra == "code":
            blocks.append(code_block(rng, lang, len(blocks)))

    head = preamble(
        size,
        "two_column" in features,
        "page_number" in features,
        HEADER_TEXT[lang] if "header" in features else None,
    )
    body = "\n\n".join(b.tex for b in blocks if b.tex)
    return head + body + "\n\\end{document}\n", blocks


def document_files(blocks: list[SourceBlock]) -> dict[str, bytes]:
    """コンパイルに要るファイル（図の画像）をまとめる"""
    files: dict[str, bytes] = {}
    for block in blocks:
        files.update(block.files)
    return files
