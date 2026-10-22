"""
評価用のランダムな文書（TeX）を作る。各ブロックの始まりと終わりに目印 \\mk{b{番号}s} / \\mk{b{番号}e} を埋め込む。

block_classifier/doc_gen.py（学習データ用）とは別に作る。同じ作り方の文書で評価すると、
block_classifier だけが「学習で見たことのある文書」で評価されて、実力より高く出てしまうため。
"""
import random
from dataclasses import dataclass

PREAMBLE = r"""\documentclass[uplatex,a4paper,%(size)s]{jsarticle}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
\begin{document}
"""

# 言語ごとの材料。英語の文書がASCIIだけになるように、英語側には日本語を入れない
WORDS = {
    "ja": {
        "nouns": ["数式", "文書", "画像", "モデル", "手法", "データ", "精度", "構造", "段落", "見出し", "評価", "結果", "座標", "変換"],
        "verbs": ["を検討する", "を提案する", "を比較した", "を評価した", "を示す", "を改善した", "を実装した", "を確認した"],
        "headings": ["はじめに", "関連研究", "提案手法", "実験", "結果", "考察", "まとめ", "今後の課題"],
        "sep": "",    # 単語・文の区切り（日本語は詰める）
        "end": "。",
    },
    "en": {
        "nouns": ["the model", "the document", "the layout", "the method", "the dataset", "the accuracy", "the result", "each block"],
        "verbs": ["improves", "describes", "compares", "evaluates", "shows", "extracts", "detects", "predicts"],
        "headings": ["Introduction", "Related Work", "Method", "Experiments", "Results", "Discussion", "Conclusion"],
        "sep": " ",
        "end": ".",
    },
}
HEADING_COMMANDS = ["section", "subsection", "subsubsection"]  # 見出しレベル 1, 2, 3


@dataclass
class SourceBlock:
    """文書に置いたブロック1つ。正解IRの label / level はここから取る"""
    id: int             # 目印の名前 b{id}s / b{id}e に使う
    label: str          # "heading" / "paragraph" / "list"
    level: int | None   # 見出しのときだけ 1〜3
    tex: str


def mark(name: str) -> str:
    return rf"\mk{{{name}}}"


def sentence(rng: random.Random, lang: str) -> str:
    words = WORDS[lang]
    text = words["sep"].join([rng.choice(words["nouns"]), rng.choice(words["verbs"]), rng.choice(words["nouns"])])
    text += words["end"]
    return text[0].upper() + text[1:]  # 英語は文頭を大文字に（日本語は変わらない）


def heading(rng: random.Random, lang: str, i: int) -> SourceBlock:
    level = rng.choice([1, 2, 3])
    command = HEADING_COMMANDS[level - 1]
    title = rng.choice(WORDS[lang]["headings"])
    return SourceBlock(i, "heading", level, rf"\{command}{{{mark(f'b{i}s')}{title}{mark(f'b{i}e')}}}")


def paragraph(rng: random.Random, lang: str, i: int) -> SourceBlock:
    text = WORDS[lang]["sep"].join(sentence(rng, lang) for _ in range(rng.randint(2, 6)))
    start, end = mark(f"b{i}s"), mark(f"b{i}e")
    return SourceBlock(i, "paragraph", None, f"{start}{text}{end}")


def list_block(rng: random.Random, lang: str, i: int) -> SourceBlock:
    items = [sentence(rng, lang) for _ in range(rng.randint(2, 4))]
    items[0] = mark(f"b{i}s") + items[0]
    items[-1] += mark(f"b{i}e")
    lines = [r"\begin{itemize}", *(rf"  \item {item}" for item in items), r"\end{itemize}"]
    return SourceBlock(i, "list", None, "\n".join(lines))


def random_document(rng: random.Random, lang: str) -> tuple[str, list[SourceBlock]]:
    """ランダムな文書のTeXと、置いたブロックの一覧を返す。節（見出し + 1〜3個のブロック）を2〜4個並べる"""
    blocks: list[SourceBlock] = []
    for _ in range(rng.randint(2, 4)):
        blocks.append(heading(rng, lang, len(blocks)))  # 番号は「今までに置いたブロックの数」= 0, 1, 2, ...
        for _ in range(rng.randint(1, 3)):
            builder = rng.choice([paragraph, paragraph, list_block])  # 段落を出やすく
            blocks.append(builder(rng, lang, len(blocks)))
    size = rng.choice(["10pt", "11pt", "12pt"])
    body = "\n\n".join(b.tex for b in blocks)
    return PREAMBLE % {"size": size} + body + "\n\\end{document}\n", blocks
