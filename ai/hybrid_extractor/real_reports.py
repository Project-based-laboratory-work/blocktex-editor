"""本物のレポート（TeX のソースがあるもの）に目印を入れてコンパイルし直し、正解付きの評価データセットにする。

    python -m hybrid_extractor.real_reports --reports ../docs/reports --out evaluation/data/real_reports
    python -m evaluation.run --model hybrid_operator --dataset evaluation/data/real_reports   # ルール
    python -m evaluation.run --model hybrid_ai --dataset evaluation/data/real_reports         # AI

DocBank（Li ら, COLING 2020）と同じ考え方で、中身は本物、正解は自動で付ける。DocBank は部品ごとに文字の色を変えたが、
ここではチームの方針どおり、評価用データ（evaluation/synth_doc.py）と同じ目印（\\special{pdf:dest}）を使う。
目印の名前・置き方・正解の作り方（evaluation/ground_truth.py の build_ground_truth）も synth_doc と同じ。

レポートは --reports の下の <名前>/report.tex。変換は自分たちのコピーの上でだけ行う（元のファイルは書き換えない）:

- コードはスコープ（listings）に合わせ、minted の \\codefile{言語}{ファイル}{開始行}{終了行} を lstlisting にする。
  設定は minted と同じ（行番号・上下の線・折り返し・\\small）。ファイルの「../」は、コピーでは「_parent/」の下に置く
- 見出し（\\section など）・段落・箇条書き（itemize / enumerate / description）・コード・図・表・キャプション・
  タイトル・著者・日付・目次・ページ番号に目印を置く。段落と箇条書きの中の \\textbf は、太字の装飾（runs）の正解にする
- 図・表・キャプションは full_width（段をまたぐもの）にする。1段組なので段の意味は無いが、build_ground_truth はこれらの
  文字を先に取り分けるので、ページをまたぐコードの続きの範囲（ページの上の端から）に、ページの上に浮いた図が入らない
- 迷うものの決め方（AI.md・実装メモにも書く）:
  - アルゴリズムの枠（center の中の \\textbf の題名）は paragraph（CENTER_TITLE_LABEL）。点線（\\dotfill）は other
  - \\paragraph{…} は、すぐ後ろに文が続く（同じ行に並ぶ）ので、その文と合わせて1つの paragraph。
    後ろに文が無いときだけ見出し（レベル 3）
  - 目次（\\tableofcontents）は、見出し「目次」も含めて丸ごと other（文書の中身ではなく、自動で作られるもの）
  - 罫線だけの center（\\hrule）は文字が無いので、ブロックにしない
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from common.ir import save_ir
from evaluation.ground_truth import build_ground_truth
from evaluation.images import SITE_DPI, render_pages
from evaluation.synth_doc import PAGE_NUMBER_FOOT, PAGE_STYLE, SourceBlock

CENTER_TITLE_LABEL = "paragraph"   # アルゴリズムの枠の題名（center の中の太字）のラベル
SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
LIST_ENVS = {"itemize", "enumerate", "description"}
CODE_ENVS = {"lstlisting", "verbatim"}
# 中身を TeX として読まない（目印や装飾を入れない・コメントを消さない）環境
VERBATIM_ENVS = CODE_ENVS
# 置いても文字が出ない命令（それだけの「段落」はブロックにしない）
SILENT = re.compile(
    r"\\(label|vspace\*?|hspace\*?|centering|noindent|clearpage|newpage|par|medskip|bigskip|smallskip|hrule|hfill)"
    r"(\[[^\]]*\])?(\{[^{}]*\})?"
)
# 段落の途中でも、そこで段落を切る命令（その命令が自分のブロックを作る）
DISPLAY_COMMANDS = {
    *SECTION_LEVELS, "paragraph", "codefile", "resultimg", "resultimgH", "RunResult", "maketitle", "tableofcontents",
    "clearpage", "newpage",
}
# minted の言語の名前 → listings の言語の名前（無いものは言語を付けない）
LISTINGS_LANGUAGES = {"cpp": "C++", "c": "C", "python": "Python", "java": "Java", "ocaml": "Caml"}
# minted の \codefile と同じ見た目の lstlisting の設定
LISTINGS_OPTIONS = (
    r"numbers=left,numberstyle=\tiny,firstnumber={first},frame=lines,breaklines=true,"
    r"basicstyle=\ttfamily\small,columns=fixed,keepspaces=true,showstringspaces=false,tabsize=4{language}"
)
MARK_COMMAND = r"\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}"
# LuaLaTeX（ltjsarticle など）では \special{pdf:dest} が効かないので、LuaTeX の命令で同じ名前付きの目印を置く。
# どちらも、目印を置いた位置（ベースラインの高さ）が記録される
MARK_COMMAND_LUA = r"\newcommand{\mk}[1]{\leavevmode\pdfextension dest name{#1} xyz\relax}"


def uses_lualatex(tex: str) -> bool:
    """LuaLaTeX 用の文書クラス（ltjsarticle など、luatexja の ltjs〜）か"""
    m = re.search(r"\\documentclass(\[[^\]]*\])?\{([^}]*)\}", tex)
    return bool(m) and m.group(2).startswith("ltjs")
COMMAND = re.compile(r"\\([a-zA-Z@]+)(\*?)")
BLANK_LINE = re.compile(r"\n[ \t]*\n")


# ---- TeX を読む小さな道具 ----


def match_brace(s: str, i: int, open_: str = "{", close: str = "}") -> int:
    """s[i] が開きかっこのとき、対応する閉じかっこの位置。\\{ のようにエスケープしたものは数えない"""
    assert s[i] == open_, (s[i:i + 20], open_)
    depth = 0
    k = i
    while k < len(s):
        c = s[k]
        if c == "\\":
            k += 2
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return k
        k += 1
    raise ValueError(f"かっこが閉じていない: {s[i:i + 40]!r}")


def read_args(s: str, i: int, n: int, optional: bool = True) -> tuple[str | None, list[str], int]:
    """s[i:] から [任意の引数] と {引数} を n 個読む。(任意の引数, 引数の並び, 読み終わった位置)"""
    k = skip_spaces(s, i)
    opt = None
    if optional and k < len(s) and s[k] == "[":
        end = match_brace(s, k, "[", "]")
        opt, k = s[k + 1:end], end + 1
    args = []
    for _ in range(n):
        k = skip_spaces(s, k)
        end = match_brace(s, k)
        args.append(s[k + 1:end])
        k = end + 1
    return opt, args, k


def skip_spaces(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\n":
        i += 1
    return i


def skip_inline_verbatim(s: str, i: int) -> int:
    """\\lstinline|…| や \\verb!…! の、命令の名前のすぐ後ろ（i）から、中身の終わりの次の位置まで飛ばす。
    中身には { や % がそのまま入るので、ふつうの TeX として読んではいけない"""
    if i < len(s) and s[i] == "[":
        i = match_brace(s, i, "[", "]") + 1
    delim = s[i]
    if delim == "{":
        return match_brace(s, i) + 1
    return s.index(delim, i + 1) + 1


def env_end(s: str, i: int, env: str) -> int:
    """\\begin{env} の後ろ（i）から、対応する \\end{env} の終わりの位置。同じ環境の入れ子も数える"""
    if env in VERBATIM_ENVS:
        end = s.index(rf"\end{{{env}}}", i)
        return end + len(rf"\end{{{env}}}")
    depth = 1
    pattern = re.compile(rf"\\(begin|end)\{{{re.escape(env)}\}}")
    for m in pattern.finditer(s, i):
        depth += 1 if m.group(1) == "begin" else -1
        if depth == 0:
            return m.end()
    raise ValueError(f"\\end{{{env}}} が無い")


def strip_comments(s: str) -> str:
    """% から行末までのコメントを、行末の改行ごと消す（TeX でもコメントの行の改行は無視されるので、意味は変わらない）。
    \\%、コードの環境、\\lstinline・\\verb の中は消さない"""
    out = []
    k = 0
    while k < len(s):
        c = s[k]
        if c == "\\":
            m = COMMAND.match(s, k)
            if m and m.group(1) == "begin":
                env = s[m.end() + 1:s.index("}", m.end())]
                if env in VERBATIM_ENVS:
                    end = env_end(s, s.index("}", m.end()) + 1, env)
                    out.append(s[k:end])
                    k = end
                    continue
            if m and m.group(1) in ("lstinline", "verb"):
                end = skip_inline_verbatim(s, m.end())
                out.append(s[k:end])
                k = end
                continue
            out.append(s[k:k + 2])
            k += 2
            continue
        if c == "%":
            newline = s.find("\n", k)
            k = len(s) if newline < 0 else newline + 1
            continue
        out.append(c)
        k += 1
    return "".join(out)


def inline_inputs(tex: str, root: Path) -> str:
    """\\input{tex/kadai01} を、そのファイルの中身で置き換える（目印を入れるには、全部が1つのソースにある方が楽）"""
    def replace(m: re.Match) -> str:
        path = root / m.group(1)
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        return inline_inputs(path.read_text(encoding="utf-8"), root)

    return re.sub(r"\\input\{([^}]*)\}", replace, tex)


def plain_text(tex: str) -> str:
    """表のセルの TeX から、見える文字だけを取り出す（\\textbf{x} → x、\\_ → _）"""
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", tex)
    text = re.sub(r"\\(.)", r"\1", text)
    return re.sub(r"\s+", " ", text.replace("{", "").replace("}", "")).strip()


def resolve_code_path(root: Path, path: str) -> Path:
    """\\codefile のファイル。元のレポートの「../」は、コピーでは「_parent/」の下に置いている"""
    return root / path.replace("../", "_parent/")


# ---- 目印を入れる ----


def mark(name: str) -> str:
    return rf"\mk{{{name}}}"


# コード・図・表の目印は、その上と下の別の行に置く（synth_doc と同じ）。始まりの目印の前でも段落を終える。
# 終わりの目印の後ろでも段落を終える（すぐ後ろの文の始まりの目印が、同じ行に乗らないように）。
# 空行なしでコードの直前に文を書いたとき（「\textbf{lexer.mll}（32--34行目）」の次の行に \codefile）、
# 目印が文の最後の行に乗って、コードの範囲に文が入ってしまうため


@dataclass
class Annotator:
    """レポートの本文を前から読み、ブロックごとに目印を入れた TeX と、ブロックの一覧（SourceBlock）を作る"""
    root: Path                                   # レポートのフォルダ（\codefile のファイルを読む）
    blocks: list[SourceBlock] = field(default_factory=list)
    title_ids: dict[str, int] = field(default_factory=dict)   # \maketitle のタイトル・著者・日付の番号
    warnings: list[str] = field(default_factory=list)

    def new_block(self, label: str, level: int | None = None, **kw) -> SourceBlock:
        block = SourceBlock(len(self.blocks), label, level, "", **kw)
        self.blocks.append(block)
        return block

    # -- 本文 --

    def body(self, s: str) -> str:
        """本文（または quote などの中身）に目印を入れる"""
        out = []
        k = 0
        while k < len(s):
            if s[k] in " \t\n":
                out.append(s[k])
                k += 1
                continue
            m = COMMAND.match(s, k)
            if m and (m.group(1) in DISPLAY_COMMANDS - {"paragraph"} or m.group(1) == "begin"):
                text, k = self.display(s, k, m)
                out.append(text)
                continue
            end = self.paragraph_end(s, k)
            out.append(self.paragraph(s[k:end]))
            k = end
        return "".join(out)

    def paragraph_end(self, s: str, i: int) -> int:
        """i から始まる段落の終わり。空行か、段落を切る命令（見出し・コード・環境）の手前まで"""
        k = i
        while k < len(s):
            blank = BLANK_LINE.match(s, k)
            if blank:
                return k
            c = s[k]
            if c == "\\":
                m = COMMAND.match(s, k)
                if not m:
                    k += 2
                    continue
                name = m.group(1)
                if k != i and (name in DISPLAY_COMMANDS or name == "begin"):
                    return k
                if name in ("lstinline", "verb"):
                    k = skip_inline_verbatim(s, m.end())
                    continue
                k = m.end()
                continue
            if c == "{":
                k = match_brace(s, k) + 1
                continue
            k += 1
        return k

    def paragraph(self, chunk: str) -> str:
        """段落1つ。文字が出ないもの（\\label だけなど）は、そのまま返す"""
        body = chunk.rstrip()
        trailing = chunk[len(body):]
        if not SILENT.sub("", body).strip():
            return chunk
        m = re.match(r"\\paragraph\*?(\[[^\]]*\])?\{", body)
        if m:
            end = match_brace(body, m.end() - 1)
            title, rest = body[m.end():end], body[end + 1:]
            only_heading = not SILENT.sub("", rest).strip()
            block = self.new_block("heading" if only_heading else "paragraph", 3 if only_heading else None)
            rest = self.decorate(rest, block)
            if only_heading:
                # \paragraph の見出しは、次の段落の頭に付けて組まれる。すぐ後ろがコードだと、コードの始まりの目印が
                # 見出しの行に乗ってしまうので、見出しだけで行を終える
                return (rf"\paragraph[{title}]{{{mark(f'b{block.id}s')}{title}{mark(f'b{block.id}e')}}}\leavevmode\par"
                        f"{rest}{trailing}")
            return rf"\paragraph[{title}]{{{mark(f'b{block.id}s')}{title}}}{rest}{mark(f'b{block.id}e')}{trailing}"
        block = self.new_block("paragraph")
        lead = re.match(r"(\\noindent\s*)?", body).group(0)   # \noindent の前に目印を置くと字下げが付いてしまう
        text = self.decorate(body[len(lead):], block)
        return f"{lead}{mark(f'b{block.id}s')}{text}{mark(f'b{block.id}e')}{trailing}"

    def decorate(self, tex: str, block: SourceBlock) -> str:
        """\\textbf{…} の前後に装飾の目印を置き、block.decorations に記録する（\\lstinline の中は見ない）"""
        out = []
        k = 0
        while k < len(tex):
            m = COMMAND.match(tex, k)
            if not m:
                out.append(tex[k:k + 2] if tex[k] == "\\" else tex[k])
                k += 2 if tex[k] == "\\" else 1
                continue
            if m.group(1) in ("lstinline", "verb"):
                end = skip_inline_verbatim(tex, m.end())
                out.append(tex[k:end])
                k = end
                continue
            if m.group(1) == "textbf" and m.end() < len(tex) and tex[m.end()] == "{":
                end = match_brace(tex, m.end())
                name = f"b{block.id}d{len(block.decorations)}"
                block.decorations.append((name, "bold"))
                inner = tex[m.end() + 1:end]
                out.append(f"{mark(name + 's')}\\textbf{{{inner}}}{mark(name + 'e')}")
                k = end + 1
                continue
            out.append(m.group(0))
            k = m.end()
        return "".join(out)

    # -- 段落を切るもの --

    def display(self, s: str, i: int, m: re.Match) -> tuple[str, int]:
        name, star = m.group(1), m.group(2)
        if name in SECTION_LEVELS:
            opt, (title,), end = read_args(s, m.end(), 1)
            block = self.new_block("heading", SECTION_LEVELS[name])
            marked = f"{mark(f'b{block.id}s')}{title}{mark(f'b{block.id}e')}"
            if star:
                return rf"\{name}*{{{marked}}}", end
            return rf"\{name}[{opt if opt is not None else title}]{{{marked}}}", end   # 目次には目印の無い題名を出す
        if name == "codefile":
            _, (lang, path, first, last), end = read_args(s, m.end(), 4)
            lines = resolve_code_path(self.root, path).read_text(encoding="utf-8").splitlines()
            return self.code(lines[int(first) - 1:int(last)], lang, int(first)), end
        if name in ("resultimg", "resultimgH"):
            _, (file, caption, width), end = read_args(s, m.end(), 3)
            place = "H" if name == "resultimgH" else "htbp"
            return self.figure_env(
                f"\\begin{{figure}}[{place}]\n\\centering\n\\includegraphics[width={width}\\linewidth]{{{file}}}\n"
                f"\\caption{{{caption}}}\n\\end{{figure}}"
            ), end
        if name == "RunResult":
            _, (file,), end = read_args(s, m.end(), 1, optional=False)
            # 浮動しない図なので full_width にしない（ページの境目で目印だけが前のページに残ると、
            # full_width の範囲はページ番号の文字まで拾ってしまう）
            block = self.new_block("figure", kind="figure", own_line_marks=True)
            return (f"\\begin{{center}}\n\\par{mark(f'b{block.id}s')}\\par\n"
                    f"\\includegraphics[width=0.85\\textwidth]{{screenshots/{file}.png}}\n"
                    f"\\par{mark(f'b{block.id}e')}\\par\n\\end{{center}}"), end
        if name == "maketitle":
            for part, label in (("title", "title"), ("author", "author"), ("date", "date")):
                self.title_ids[part] = self.new_block(label, full_width=True).id
            return m.group(0), m.end()
        if name == "tableofcontents":
            block = self.new_block("other")
            # 「目次」の見出しの位置に始まりの目印を置く（\tableofcontents の前に置くと、空の行が1つ増える）
            return (rf"\renewcommand{{\contentsname}}{{{mark(f'b{block.id}s')}目次}}\tableofcontents"
                    f"{mark(f'b{block.id}e')}"), m.end()
        if name in ("clearpage", "newpage"):
            return m.group(0), m.end()
        # \begin{環境}
        env_start = m.end()
        env = s[env_start + 1:s.index("}", env_start)]
        end = env_end(s, s.index("}", env_start) + 1, env)
        return self.environment(env, s[i:end]), end

    def environment(self, env: str, src: str) -> str:
        if env in LIST_ENVS:
            return self.list_env(src)
        if env in CODE_ENVS:
            content = src[src.index("\n") + 1:src.rindex("\\end")] if "\n" in src else ""
            block = self.new_block("code", own_line_marks=True,
                                   text="\n".join(line.expandtabs(4) for line in content.rstrip("\n").split("\n")))
            return f"\\par{mark(f'b{block.id}s')}\\par\n{src}\n\\par{mark(f'b{block.id}e')}\\par"
        if env in ("figure", "figure*"):
            return self.figure_env(src)
        if env in ("table", "table*"):
            return self.table_env(src)
        if env == "center":
            return self.center_env(src)
        if env == "quote":
            head = src.index("\n") + 1 if "\n" in src else len(r"\begin{quote}")
            tail = src.rindex(r"\end{quote}")
            return src[:head] + self.body(src[head:tail]) + src[tail:]
        if env == "titlepage":
            return self.titlepage_env(src)
        self.warnings.append(f"知らない環境 {env} はブロックにしない")
        return src

    def code(self, lines: list[str], lang: str, first: int) -> str:
        """\\codefile の行を、minted と同じ見た目の lstlisting にする。正解の文字はファイルの行（タブは4文字に広げる）"""
        lines = [line.expandtabs(4) for line in lines]
        block = self.new_block("code", own_line_marks=True, text="\n".join(lines))
        language = f",language={{{LISTINGS_LANGUAGES[lang]}}}" if lang in LISTINGS_LANGUAGES else ""
        options = LISTINGS_OPTIONS.replace("{first}", str(first)).replace("{language}", language)
        body = "\n".join(lines)
        return (f"\\par{mark(f'b{block.id}s')}\\par\n\\begin{{lstlisting}}[{options}]\n{body}\n\\end{{lstlisting}}\n"
                f"\\par{mark(f'b{block.id}e')}\\par")

    def list_env(self, src: str) -> str:
        """箇条書きは、入れ子も含めて丸ごと1つの list。始まりの目印は最初の \\item の後ろ、終わりの目印は最後の文の後ろ"""
        block = self.new_block("list")
        for env in ("lstlisting", "verbatim", "figure", "codefile", "center"):
            if env in src:
                self.warnings.append(f"箇条書きの中に {env} がある（箇条書きと重なる）")
        item = re.search(r"\\item\s*(\[[^\]]*\])?", src)
        src = src[:item.end()] + " " + mark(f"b{block.id}s") + src[item.end():]
        src = self.decorate(src, block)
        return insert_list_end(src, mark(f"b{block.id}e"))

    def figure_env(self, src: str) -> str:
        """図の環境。画像（\\includegraphics）の上下の別の行に目印を置き、キャプションは別のブロックにする"""
        parts = []
        k = 0
        for m in re.finditer(r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}|\\caption\{", src):
            parts.append(src[k:m.start()])
            if m.group(0).startswith(r"\caption"):
                end = match_brace(src, m.end() - 1)
                caption = src[m.end():end]
                block = self.new_block("caption", full_width=True)
                parts.append(rf"\caption[{caption}]{{{mark(f'b{block.id}s')}{caption}{mark(f'b{block.id}e')}}}")
                k = end + 1
            else:
                block = self.new_block("figure", kind="figure", own_line_marks=True, full_width=True)
                parts.append(f"\\par{mark(f'b{block.id}s')}\\par\n{m.group(0)}\n\\par{mark(f'b{block.id}e')}\\par")
                k = m.end()
        parts.append(src[k:])
        return "".join(parts)

    def table_env(self, src: str) -> str:
        """表の環境。キャプションと表（tabular / tabularx）を、書かれた順のブロックにする。表の正解の cells などはソースから"""
        m = re.search(r"\\begin\{(tabularx?)\}", src)
        env = m.group(1)
        end = env_end(src, m.end(), env)
        tabular = src[m.start():end]
        caption_m = re.search(r"\\caption\{", src)
        pieces = []

        def caption_part() -> str:
            c_end = match_brace(src, caption_m.end() - 1)
            caption = src[caption_m.end():c_end]
            block = self.new_block("caption", full_width=True)
            return rf"\caption[{caption}]{{{mark(f'b{block.id}s')}{caption}{mark(f'b{block.id}e')}}}", c_end + 1

        def table_part() -> str:
            block = self.new_block("table", kind="table", own_line_marks=True, full_width=True, **parse_tabular(tabular))
            return f"\\par{mark(f'b{block.id}s')}\\par\n{tabular}\n\\par{mark(f'b{block.id}e')}\\par"

        if caption_m and caption_m.start() < m.start():
            cap, c_end = caption_part()
            pieces = [src[:caption_m.start()], cap, src[c_end:m.start()], table_part(), src[end:]]
        else:
            pieces = [src[:m.start()], table_part()]
            rest = src[end:]
            if caption_m:
                cap, c_end = caption_part()
                rest = src[end:caption_m.start()] + cap + src[c_end:]
            pieces.append(rest)
        return "".join(pieces)

    def center_env(self, src: str) -> str:
        """center。画像なら図、点線なら other、太字の題名ならアルゴリズムの枠の題名、罫線だけならブロックにしない"""
        if r"\includegraphics" in src:
            return self.figure_env(src)
        head = src.index(r"\begin{center}") + len(r"\begin{center}")
        tail = src.rindex(r"\end{center}")
        lines = src[head:tail].split("\n")
        text_rows = [k for k, line in enumerate(lines) if SILENT.sub("", line).strip()]
        if not text_rows:
            return src
        label = "other" if r"\dotfill" in src else CENTER_TITLE_LABEL
        block = self.new_block(label)
        first, last = text_rows[0], text_rows[-1]
        indent = re.match(r"\s*", lines[first]).group(0)
        lines[first] = indent + mark(f"b{block.id}s") + lines[first][len(indent):]
        lines[last] = lines[last].rstrip() + mark(f"b{block.id}e")
        if label != "other":
            lines[first:last + 1] = self.decorate("\n".join(lines[first:last + 1]), block).split("\n")
        return src[:head] + "\n".join(lines) + src[tail:]

    def titlepage_env(self, src: str) -> str:
        """自分で組んだ表紙（{\\LARGE 題名\\par} {\\large 著者\\par} …）。最初のまとまりがタイトル、
        「提出日」を含むまとまりが日付、その間が著者"""
        groups = [(g.start(), match_brace(src, g.start())) for g in re.finditer(r"\{\\(LARGE|Large|large|huge|Huge)", src)]
        if not groups:
            return src
        roles = ["title"] + ["date" if "提出日" in src[a:b] else "author" for a, b in groups[1:]]
        spans: dict[str, list[tuple[int, int]]] = {}
        for role, span in zip(roles, groups):
            spans.setdefault(role, []).append(span)
        inserts: list[tuple[int, str]] = []
        for role in ("title", "author", "date"):
            if role not in spans:
                continue
            block = self.new_block(role, full_width=True)
            (a, _), (_, b) = spans[role][0], spans[role][-1]
            size = re.match(r"\{\\[a-zA-Z]+\s*", src[a:]).end()
            body_end = src.rfind(r"\par", a, b)
            inserts += [(a + size, mark(f"b{block.id}s")), (body_end if body_end >= 0 else b, mark(f"b{block.id}e"))]
        for pos, text in sorted(inserts, reverse=True):
            src = src[:pos] + text + src[pos:]
        return src


def insert_list_end(src: str, end_mark: str) -> str:
    """箇条書きの最後の文の後ろに、終わりの目印を入れる。最後の項目が入れ子の箇条書きで終わるなら、その中へ入る
    （\\end{itemize} の直前に置くと、入れ子の後ろに空の行が1つ増える）"""
    m = re.search(r"\\end\{(itemize|enumerate|description)\}\s*$", src)
    inner = src[:m.start()].rstrip()
    nested = re.search(r"\\end\{(itemize|enumerate|description)\}$", inner)
    if nested:
        inner = insert_list_end(inner, end_mark)
    else:
        inner += end_mark
    return inner + "\n" + src[m.start():]


def parse_tabular(tabular: str) -> dict:
    """tabular / tabularx の正解（cells・揃え・罫線）。X の列（tabularx の伸びる列）は左寄せ、列幅は持たない"""
    if tabular.startswith(r"\begin{tabularx}"):
        spec_start = match_brace(tabular, len(r"\begin{tabularx}")) + 1
    else:
        spec_start = len(r"\begin{tabular}")
    spec_end = match_brace(tabular, spec_start)
    spec = tabular[spec_start + 1:spec_end]
    body = tabular[spec_end + 1:tabular.rindex(r"\end")]
    cols = [c for c in re.sub(r"[pmb]\{[^}]*\}", "l", spec) if c in "lcrX|"]
    aligns, col_rules = [], [False]
    for c in cols:
        if c == "|":
            col_rules[-1] = True
        else:
            aligns.append({"X": "l"}.get(c, c))
            col_rules.append(False)
    rows, row_rules = [], [False]
    for raw in re.split(r"\\\\", body):
        if r"\hline" in raw:
            row_rules[-1] = True
        cells = raw.replace(r"\hline", "").strip()
        if not cells:
            continue
        rows.append([plain_text(cell) for cell in cells.split("&")])
        row_rules.append(False)
    return {"cells": rows, "aligns": aligns, "col_rules": col_rules, "row_rules": row_rules,
            "text": "\n".join(" ".join(row) for row in rows)}


def patch_preamble(preamble: str, title_ids: dict[str, int]) -> str:
    """プリアンブルを変える: minted をやめて listings を読む、目印の命令とページ番号の目印を足す、
    \\title・\\author・\\date に目印を入れる"""
    preamble = preamble.replace(r"\usepackage{minted}", r"\usepackage{listings}")
    for part, block_id in title_ids.items():
        m = re.search(rf"\\{part}\{{", preamble)
        if not m:
            continue
        end = match_brace(preamble, m.end() - 1)
        inner = preamble[m.end():end]
        preamble = preamble[:m.end()] + mark(f"b{block_id}s") + inner + mark(f"b{block_id}e") + preamble[end:]
    page_style = PAGE_STYLE % {"head": "", "foot": PAGE_NUMBER_FOOT}
    command = MARK_COMMAND_LUA if uses_lualatex(preamble) else MARK_COMMAND
    return preamble + "\n" + command + "\n" + page_style + "\n"


def annotate(tex: str, root: Path) -> tuple[str, list[SourceBlock], list[str]]:
    """レポートの TeX に目印を入れる。(目印を入れた TeX, ブロックの一覧, 気になったこと)"""
    tex = strip_comments(inline_inputs(tex, root))
    begin = tex.index(r"\begin{document}")
    end = tex.index(r"\end{document}")
    annotator = Annotator(root)
    body = annotator.body(tex[begin + len(r"\begin{document}"):end])
    preamble = patch_preamble(tex[:begin], annotator.title_ids)
    return preamble + r"\begin{document}" + body + tex[end:], annotator.blocks, annotator.warnings


# ---- コンパイルしてデータセットにする ----


def compile_report(tex: str, root: Path, work: Path) -> Path:
    """レポートのフォルダ（画像など）を work にコピーして、目印を入れた TeX をコンパイルする。
    目次と参照（\\ref）のために uplatex を2回。目印を残すため dvipdfmx に -C 0x0010"""
    shutil.copytree(root, work, dirs_exist_ok=True)
    (work / "doc.tex").write_text(tex, encoding="utf-8")
    if uses_lualatex(tex):
        for _ in range(2):
            subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                           cwd=work, check=True, capture_output=True, timeout=600)
        return work / "doc.pdf"
    for _ in range(2):
        subprocess.run(["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                       cwd=work, check=True, capture_output=True, timeout=300)
    subprocess.run(["dvipdfmx", "-q", "-C", "0x0010", "doc.dvi"], cwd=work, check=True, capture_output=True, timeout=300)
    return work / "doc.pdf"


def make_case(root: Path, case_dir: Path) -> dict:
    tex, blocks, warnings = annotate((root / "report.tex").read_text(encoding="utf-8"), root)
    for w in sorted(set(warnings)):
        print(f"  注意: {w}")
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        try:
            pdf = compile_report(tex, root, work)
        except subprocess.CalledProcessError:
            log = work / "doc.log"
            if log.exists():
                print("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
            raise
        doc = build_ground_truth(str(pdf), blocks, source="input.pdf")
        shutil.copy(pdf, case_dir / "input.pdf")
    render_pages(str(case_dir / "input.pdf"), case_dir, SITE_DPI)
    (case_dir / "source.tex").write_text(tex, encoding="utf-8")
    save_ir(doc, str(case_dir / "gt.labeled.json"))
    placed = sum(1 for page in doc.pages for _ in page.elements)
    print(f"  ブロック {len(blocks)}個 → 正解の要素 {placed}個、{len(doc.pages)}ページ")
    labels = {el.label for page in doc.pages for el in page.elements}
    return {
        "lang": "ja",
        "pages": len(doc.pages),
        "columns": 1,
        "real": True,
        "has_title": "title" in labels,
        "has_list": "list" in labels,
        "has_table": "table" in labels,
        "has_figure": "figure" in labels,
        "has_code": "code" in labels,
        "has_decoration": any(b.decorations for b in blocks),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本物のレポート（TeX）に目印を入れて、正解付きの評価データセットにする")
    parser.add_argument("--reports", required=True, help="<名前>/report.tex が並ぶフォルダ（例: ../docs/reports）")
    parser.add_argument("--out", required=True, help="例: evaluation/data/real_reports")
    args = parser.parse_args(argv)

    out = Path(args.out)
    cases = []
    for root in sorted(p.parent for p in Path(args.reports).glob("*/report.tex")):
        print(root.name)
        try:
            tags = make_case(root, out / "cases" / root.name)
        except subprocess.CalledProcessError:
            print(f"{root.name}: TeXのコンパイルに失敗したので飛ばす")
            continue
        cases.append({"id": root.name, "tags": tags})
    manifest = {
        "name": out.name,
        "created": date.today().isoformat(),
        "generator": {"script": "hybrid_extractor.real_reports", "reports": str(args.reports)},
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)}件を {out} に作った")


if __name__ == "__main__":
    main()
