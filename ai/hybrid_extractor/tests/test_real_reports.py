"""本物のレポートに目印を入れる変換（real_reports.py）のテスト。コンパイルはしない（TeX の文字列だけを見る）"""
from pathlib import Path

from hybrid_extractor.real_reports import (
    annotate,
    insert_list_end,
    match_brace,
    parse_tabular,
    strip_comments,
    uses_lualatex,
)

PREAMBLE = "\\documentclass{jsarticle}\n\\usepackage{minted}\n\\title{題名}\n\\author{著者}\n\\date{日付}\n"


def run(body: str, tmp_path: Path = Path(".")) -> tuple[str, list]:
    tex, blocks, _ = annotate(PREAMBLE + "\\begin{document}\n" + body + "\n\\end{document}\n", tmp_path)
    return tex, blocks


def test_match_brace_skips_escaped():
    s = r"{a\}b{c}d}e"
    assert match_brace(s, 0) == len(s) - 2


def test_strip_comments_keeps_code_and_escaped_percent():
    s = "a % コメント\nb 100\\% です\n\\begin{verbatim}\nx % 消さない\n\\end{verbatim}\n\\lstinline|%y|"
    out = strip_comments(s)
    assert "コメント" not in out
    assert "100\\% です" in out
    assert "x % 消さない" in out
    assert "\\lstinline|%y|" in out


def test_headings_paragraphs_and_lists():
    tex, blocks = run("\\section{はじめに}\n本文です。\\textbf{太字}も。\n\n\\begin{itemize}\n  \\item 一つ目\n"
                      "  \\item 二つ目\n  \\begin{itemize}\n    \\item 入れ子\n  \\end{itemize}\n\\end{itemize}\n")
    assert [(b.label, b.level) for b in blocks] == [("heading", 1), ("paragraph", None), ("list", None)]
    # 目次には目印の無い題名を出す
    assert "\\section[はじめに]{\\mk{b0s}はじめに\\mk{b0e}}" in tex
    assert blocks[1].decorations == [("b1d0", "bold")]
    # 箇条書きの終わりの目印は、入れ子の最後の文の後ろ
    assert "入れ子\\mk{b2e}" in tex
    # minted をやめて listings を読む
    assert "\\usepackage{listings}" in tex and "minted" not in tex


def test_codefile_becomes_lstlisting(tmp_path: Path):
    (tmp_path / "_parent" / "src").mkdir(parents=True)
    (tmp_path / "_parent" / "src" / "a.py").write_text("x = 1\ndef f():\n\treturn x\ny = 2\n", encoding="utf-8")
    tex, blocks = run("説明。\n\\codefile{python}{../src/a.py}{2}{3}\n続きの文。", tmp_path)
    assert [b.label for b in blocks] == ["paragraph", "code", "paragraph"]
    assert blocks[1].text == "def f():\n    return x"   # タブは4文字に広げる
    assert "firstnumber=2" in tex and "language={Python}" in tex
    # 目印の前後で段落を終える（前の文・後ろの文の行に目印が乗らないように）
    assert "\\par\\mk{b1s}\\par" in tex and "\\par\\mk{b1e}\\par" in tex


def test_run_in_paragraph_heading():
    tex, blocks = run("\\paragraph{結果}\n期待される出力：\\lstinline|a{b|\n\n\\paragraph{見出しだけ}\n\n本文。")
    assert [(b.label, b.level) for b in blocks] == [("paragraph", None), ("heading", 3), ("paragraph", None)]
    assert "\\lstinline|a{b|\\mk{b0e}" in tex
    assert "{\\mk{b1s}見出しだけ\\mk{b1e}}\\leavevmode\\par" in tex


def test_center_title_dotfill_and_rule_only():
    _, blocks = run("\\begin{center}\n\\hrule\n\\textbf{手順}\n\\hrule\n\\end{center}\n\n"
                    "\\begin{center}\n\\makebox[\\linewidth]{\\dotfill}\n\\end{center}\n\n"
                    "\\begin{center}\n\\hrule\n\\end{center}\n")
    assert [b.label for b in blocks] == ["paragraph", "other"]


def test_figure_macro_and_title():
    _, blocks = run("\\maketitle\n\\tableofcontents\n\\resultimg{img/a.png}{結果}{0.5}\n")
    assert [b.label for b in blocks] == ["title", "author", "date", "other", "figure", "caption"]
    # 浮動する図とキャプションは、先に取り分ける（full_width）
    assert blocks[4].full_width and blocks[5].full_width


def test_parse_tabularx():
    tab = ("\\begin{tabularx}{\\linewidth}{l|X}\n\\hline\n\\textbf{名前} & 説明 \\\\\n\\hline\nPUT & 石を置く \\\\\n"
           "\\hline\n\\end{tabularx}")
    t = parse_tabular(tab)
    assert t["cells"] == [["名前", "説明"], ["PUT", "石を置く"]]
    assert t["aligns"] == ["l", "l"]
    assert t["col_rules"] == [False, True, False]
    assert t["row_rules"] == [True, True, True]


def test_insert_list_end_flat():
    src = "\\begin{enumerate}\n\\item a\n\\item b\n\\end{enumerate}"
    assert insert_list_end(src, "E") == "\\begin{enumerate}\n\\item a\n\\item bE\n\\end{enumerate}"


def test_uses_lualatex():
    assert uses_lualatex("\\documentclass[11pt]{ltjsarticle}")
    assert not uses_lualatex("\\documentclass[dvipdfmx]{jsarticle}")
