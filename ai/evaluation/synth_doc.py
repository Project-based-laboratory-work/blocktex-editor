"""
評価用のランダムな文書（TeX）を作る。各ブロックの始まりと終わりに目印 \\mk{b{番号}s} / \\mk{b{番号}e} を埋め込む。
文中の装飾（太字・斜体・下線）も、装飾した語の前後に目印 \\mk{b{番号}d{何個目}s} / \\mk{b{番号}d{何個目}e} を置く。

block_classifier/doc_gen.py（学習データ用）とは別に作る。同じ作り方の文書で評価すると、
block_classifier だけが「学習で見たことのある文書」で評価されて、実力より高く出てしまうため。
"""
import random
from dataclasses import dataclass, field

PREAMBLE = r"""\documentclass[uplatex,a4paper,%(size)s]{jsarticle}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
\pagestyle{empty}
\begin{document}
"""

# 言語ごとの材料。英語の文書がASCIIだけになるように、英語側には日本語を入れない
WORDS = {
    "ja": {
        "nouns": ["数式", "文書", "画像", "モデル", "手法", "データ", "精度", "構造", "段落", "見出し", "評価", "結果", "座標", "変換"],
        "verbs": ["を検討する", "を提案する", "を比較した", "を評価した", "を示す", "を改善した", "を実装した", "を確認した"],
        "headings": ["はじめに", "関連研究", "提案手法", "実験", "結果", "考察", "まとめ", "今後の課題"],
        "authors": ["山田 太郎", "佐藤 花子", "情報 太郎"],
        "sep": "",    # 単語・文の区切り（日本語は詰める）
        "end": "。",
    },
    "en": {
        "nouns": ["the model", "the document", "the layout", "the method", "the dataset", "the accuracy", "the result", "each block"],
        "verbs": ["improves", "describes", "compares", "evaluates", "shows", "extracts", "detects", "predicts"],
        "headings": ["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion", "Conclusion"],
        "authors": ["Taro Yamada", "Hanako Sato", "Alice Smith"],
        "sep": " ",
        "end": ".",
    },
}
HEADING_COMMANDS = ["section", "subsection", "subsubsection"]  # 見出しレベル 1, 2, 3

# 文中の装飾。jsarticle の日本語は斜体の書体が無く、\textit を付けても見た目が変わらないので、日本語には付けない
DECORATIONS = {"ja": ["bold", "underline"], "en": ["bold", "italic", "underline"]}
DECORATION_TEX = {"bold": r"\textbf", "italic": r"\textit", "underline": r"\underline"}
DECORATION_RATE = 0.04  # 1文の中で、語を1つ装飾する割合（文書の半分くらいに装飾が入る）
TITLE_RATE = 0.6        # 文書の先頭にタイトル・著者・日付（\maketitle）を置く割合


@dataclass
class SourceBlock:
    """文書に置いたブロック1つ。正解IRの label / level / runs はここから取る"""
    id: int             # 目印の名前 b{id}s / b{id}e に使う
    label: str          # "heading" / "paragraph" / "list" / "title" / "author" / "date"
    level: int | None   # 見出しのときだけ 1〜3
    tex: str            # 空のブロック（author / date）は、TeXを title のブロックがまとめて持つ
    # 文中の装飾。(目印の名前の頭, 装飾)。目印は <頭>s と <頭>e で、その間の文字に装飾が付いている
    decorations: list[tuple[str, str]] = field(default_factory=list)


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


def paragraph(rng: random.Random, lang: str, i: int) -> SourceBlock:
    block = SourceBlock(i, "paragraph", None, "")
    text = WORDS[lang]["sep"].join(sentence(rng, lang, block) for _ in range(rng.randint(2, 6)))
    start, end = mark(f"b{i}s"), mark(f"b{i}e")
    block.tex = f"{start}{text}{end}"
    return block


def list_block(rng: random.Random, lang: str, i: int) -> SourceBlock:
    block = SourceBlock(i, "list", None, "")
    items = [sentence(rng, lang, block) for _ in range(rng.randint(2, 4))]
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
    return [SourceBlock(title, "title", None, tex), SourceBlock(author, "author", None, ""), SourceBlock(date, "date", None, "")]


def random_document(rng: random.Random, lang: str) -> tuple[str, list[SourceBlock]]:
    """ランダムな文書のTeXと、置いたブロックの一覧を返す。
    ときどき先頭にタイトル・著者・日付を置き、節（見出し + 1〜3個のブロック）を2〜4個並べる"""
    blocks: list[SourceBlock] = []
    if rng.random() < TITLE_RATE:
        blocks += title_blocks(rng, lang, 0)
    for _ in range(rng.randint(2, 4)):
        blocks.append(heading(rng, lang, len(blocks)))  # 番号は「今までに置いたブロックの数」= 0, 1, 2, ...
        for _ in range(rng.randint(1, 3)):
            builder = rng.choice([paragraph, paragraph, list_block])  # 段落を出やすく
            blocks.append(builder(rng, lang, len(blocks)))
    size = rng.choice(["10pt", "11pt", "12pt"])
    body = "\n\n".join(b.tex for b in blocks if b.tex)
    return PREAMBLE % {"size": size} + body + "\n\\end{document}\n", blocks
