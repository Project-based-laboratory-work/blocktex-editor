"""数式を読む係。数式の範囲（文字の並び）を受け取って、LaTeX を返す。読めなければ None。

    区切り・ラベルの専門家と同じく、差し替えられる形にしている:
        A. PdfTextReader   PDFの文字を並べるだけ（基準。添字・分数は崩れる）
        B. RuleReader      文字の位置・大きさ・フォントから組み立てる。読めなければ None
        C. OcrReader       範囲を画像に切り抜いて math_ocr で読む（学習済みの重み best.pt が要る）
        D. FallbackReader  B で読めなければ C

    対応させる数式は、math_ocr v2 が読める記号の一覧（ikumi さんのドキュメント「数式AI（math_ocr v2）が読める記号一覧」）。

    使い方（文中の数式を見つけて読む）:
        spans = inline_math_spans(chars)              # 段落の文字から、文中の数式の範囲を探す
        tex = reader.read(page, chars[start:end])     # 読む

文字（glyph）は extract.py の1文字の辞書 {"c", "bbox", "font", "size", "flags", "baseline"}。
B の下ごしらえでいくつかの文字を1つにまとめた文字（\\frac{a}{b} など）は "folded": True を持ち、"c" に LaTeX が入る。
"""

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import pymupdf

from hybrid_extractor.extract import MATH_FONT, MONO_FONT, union_all

# ---- 文字 → LaTeX ----
GREEK = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon", "ϵ": r"\epsilon", "ζ": r"\zeta",
    "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "µ": r"\mu", "ν": r"\nu",
    "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "ϕ": r"\phi", "φ": r"\varphi", "χ": r"\chi",
    "ψ": r"\psi", "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Π": r"\Pi",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega", "ℓ": r"\ell",
    "\u2206": r"\Delta", "\u2126": r"\Omega",  # 見た目が同じ別の文字（増分 U+2206・オーム U+2126）
}
SYMBOLS = {
    "−": "-", "×": r"\times", "·": r"\cdot", "÷": r"\div", "±": r"\pm", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "≈": r"\approx", "→": r"\to", "←": r"\leftarrow", "⇒": r"\Rightarrow", "∞": r"\infty", "∂": r"\partial",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "∪": r"\cup", "∩": r"\cap", "∀": r"\forall", "∃": r"\exists",
    "∇": r"\nabla", "′": "'", "…": r"\dots", "⋯": r"\cdots", "{": r"\{", "}": r"\}",
    "◦": r"\circ", "∘": r"\circ", "≡": r"\equiv", "∼": r"\sim", "∝": r"\propto",
}
BIG_OPERATORS = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap"}
FUNCTIONS = ("sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min", "sup", "inf", "det")
LIMIT_WORDS = ("lim", "max", "min", "sup", "inf")  # 別行立てでは範囲が真下に付く（\lim_{x \to 0}）
RADICAL = "√"
# アクセントは、土台の文字と重なる別の文字として出てくる（¯ と x）
ACCENTS = {"¯": r"\bar", "ˉ": r"\bar", "ˆ": r"\hat", "⃗": r"\vec", "˜": r"\tilde", "˙": r"\dot"}
# 大きいかっこ（CMEX）→ \left( \right)
OPENERS = {"(": "(", "[": "[", "{": r"\{", "|": "|"}
CLOSERS = {")": ")", "]": "]", "}": r"\}", "|": "|"}
BB_FONT = re.compile(r"MSBM|BBold|DoubleStruck", re.IGNORECASE)   # \mathbb
BF_FONT = re.compile(r"CMBX|CMMIB|Bold", re.IGNORECASE)            # \mathbf

SCRIPT_SIZE = 0.85      # 主な文字サイズのこの割合より小さい文字は、添字の候補
SCRIPT_SHIFT = 0.12     # ベースラインが主な文字サイズのこの割合以上ずれていれば、上付き・下付き
FRACTION_REACH = 1.2    # 分数の線から、主な文字サイズのこの倍までの文字を分子・分母にする
AXIS = 0.3              # 数式の軸（分数の線や ∫ の中心の高さ）は、ベースラインより主な文字サイズのこの割合だけ上


def to_latex_char(c: str) -> str:
    return GREEK.get(c) or SYMBOLS.get(c) or BIG_OPERATORS.get(c) or c


def glyph_tex(g: dict) -> str:
    """1文字（まとめた文字を含む）の LaTeX。書体はフォント名で見分ける（MSBM → \\mathbb、CMBX → \\mathbf）"""
    if g.get("folded"):
        return g["c"]
    if g["c"].isalnum() and BB_FONT.search(g["font"]):
        return rf"\mathbb{{{g['c']}}}"
    if g["c"].isalnum() and BF_FONT.search(g["font"]):
        return rf"\mathbf{{{g['c']}}}"
    return to_latex_char(g["c"])


def shown(glyphs: list[dict]) -> list[dict]:
    return [g for g in glyphs if g["bbox"] is not None and not g["c"].isspace()]


def center_x(g: dict) -> float:
    return (g["bbox"][0] + g["bbox"][2]) / 2


def center_y(g: dict) -> float:
    return (g["bbox"][1] + g["bbox"][3]) / 2


def folded(tex: str, parts: list[dict], size: float, baseline: float, op: str | None = None) -> dict:
    """いくつかの文字を1つにまとめた文字（\\frac{a}{b} など）。op は "limits"（範囲が付く lim など）か "function" """
    return {"c": tex, "bbox": union_all([g["bbox"] for g in parts]), "font": "", "size": size, "flags": 0,
            "baseline": baseline, "folded": True, "op": op}


# =====================================================================
# 読む係
# =====================================================================


class MathReader(Protocol):
    name: str

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        """数式の文字（glyph）の並びから LaTeX を返す。読めなければ None"""
        ...


class PdfTextReader:
    """A. PDFの文字を左から並べるだけ（基準）"""
    name = "pdf_text"

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        text = " ".join(to_latex_char(g["c"]) for g in sorted(shown(glyphs), key=lambda g: g["bbox"][0]))
        return text or None


@dataclass
class RuleReader:
    """B. 文字の位置・大きさ・フォントから組み立てる。複数行の数式は行ごとに組み立てて \\\\ でつなぐ。

    先に、いくつかの文字を1つにまとめる（下ごしらえ）:
        1. 重ねた文字: 否定（斜線と = → \\neq）、アクセント（¯ と x → \\bar{x}）
        2. 関数名・範囲の付く言葉（立体の l i m → \\lim）
        3. 横線: 分数（\\frac）と根号（\\sqrt、何乗根も）。短い線（内側）から
        4. 大きいかっこ（CMEX）: 中が横線なしの上下2段なら \\binom、それ以外は \\left( \\right)
    そのあと左から読み、添字・範囲を付ける。最後に点3つを \\dots / \\cdots にまとめる"""
    name: str = "rule"

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        gs = shown(glyphs)
        if not gs:
            return None
        gs = fold_words(fold_accents(fold_negation(gs)))
        gs = self.fold_bars(page, gs)
        if gs is None:
            return None
        gs = self.fold_parens(gs)
        if gs is None or any(g["c"] == RADICAL for g in gs):
            return None  # 組み立てられない分数・根号・かっこが残った
        lines = [self.read_line(line) for line in split_lines(gs)]
        if any(line is None for line in lines):
            return None
        return r" \\ ".join(lines)

    def fold_bars(self, page: pymupdf.Page, gs: list[dict]) -> list[dict] | None:
        """横線ごとに、分数（線の上が分子・下が分母）か根号（線の左端に √）を、1文字にまとめる。
        短い線（内側）から順にまとめるので、入れ子も読める。組み立てられなければ None"""
        main = main_size(gs)
        for x0, x1, y in fraction_bars(page, gs):
            # 線の左端に √ があれば根号の線。√ の枠は実際の位置より上にずれることがある（CMSY などのフォント）ので、
            # 線が √ の枠の上端〜下端のどこかにあればよい
            radical = next((g for g in gs if g["c"] == RADICAL and abs(g["bbox"][2] - x0) <= 2
                            and g["bbox"][1] - 3 <= y <= g["bbox"][3] + 3), None)
            near = [g for g in gs if g is not radical and x0 - 1 <= center_x(g) <= x1 + 1]
            # 線のすぐ下の段の文字。根号の中身・分母（背の高い f などは枠が線より上にはみ出すので、中心で見る）
            below = nearest_row([g for g in near if y < center_y(g) <= y + FRACTION_REACH * main], y, "below")
            if radical is not None:
                inside = self.read_line(below) if below else None
                if inside is None:
                    return None
                # √ の左上の小さい文字は、何乗根（\sqrt[3]{x}）
                # （√ の上半分の高さにあるものだけ。別の行の添字を拾わないように）
                index = [g for g in gs if g is not radical and g not in below and g["size"] < main * 0.75
                         and radical["bbox"][0] - 1 <= center_x(g) <= radical["bbox"][2] - 1
                         and radical["bbox"][1] - main * 0.5 <= center_y(g) <= center_y(radical)]
                root = f"[{self.read_line(index)}]" if index else ""
                parts, tex = [radical] + below + index, rf"\sqrt{root}{{{inside}}}"
                baseline = statistics.median(g["baseline"] for g in below)
            else:
                # 線のすぐ上の文字が分子。離れた文字（∑ の下限など）は入れない
                above = nearest_row([g for g in near if y - FRACTION_REACH * main <= center_y(g) < y], y, "above")
                if not above or not below:
                    continue
                top, bottom = self.read_line(above), self.read_line(below)
                if top is None or bottom is None:
                    return None
                parts, tex = above + below, rf"\frac{{{top}}}{{{bottom}}}"
                # 分数の線は、行のベースラインより文字サイズの 1/4 くらい上（数式の軸）にある
                baseline = y + main * 0.25
            taken = {id(g) for g in parts}
            gs = [g for g in gs if id(g) not in taken] + [folded(tex, parts, main, baseline)]
        return gs

    def fold_parens(self, gs: list[dict]) -> list[dict] | None:
        """大きいかっこ（CMEX）の組を、\\left( … \\right) か \\binom{上}{下} の1文字にまとめる（内側から）"""
        main = main_size(gs)
        while True:
            pair = None
            opens = [g for g in gs if g["c"] in OPENERS and "CMEX" in g["font"]]
            for o in sorted(opens, key=lambda g: -g["bbox"][0]):  # 右の（内側の）かっこから
                closes = [g for g in gs if g["c"] in CLOSERS and "CMEX" in g["font"] and g["bbox"][0] > o["bbox"][0]
                          and abs(center_y(g) - center_y(o)) < main]
                if closes:
                    pair = (o, min(closes, key=lambda g: g["bbox"][0]))
                    break
            if pair is None:
                return gs
            o, c = pair
            inside = [g for g in gs if g is not o and g is not c and o["bbox"][2] - 0.5 <= center_x(g) <= c["bbox"][0] + 0.5]
            if not inside:
                gs = [g for g in gs if g is not o and g is not c]
                continue
            top, bottom = stacked(inside)
            if o["c"] == "(" and c["c"] == ")" and top and not any(g.get("folded") for g in inside):
                upper, lower = self.read_line(top), self.read_line(bottom)
                tex = rf"\binom{{{upper}}}{{{lower}}}" if upper and lower else None
            else:
                content = self.read_line(inside)
                tex = rf"\left{OPENERS[o['c']]} {content} \right{CLOSERS[c['c']]}" if content else None
            if tex is None:
                return None
            inner_main = main_size(inside)
            mains = [g["baseline"] for g in inside if g["size"] >= inner_main * 0.9]
            baseline = statistics.median(mains or [g["baseline"] for g in inside])
            parts = [o, c] + inside
            taken = {id(g) for g in parts}
            gs = [g for g in gs if id(g) not in taken] + [folded(tex, parts, main, baseline)]

    def read_line(self, gs: list[dict]) -> str | None:
        gs = drop_equation_number(gs)
        if not gs:
            return None
        main = main_size(gs)
        # 行のベースライン。主な大きさの文字で決める（無ければ、∑ などを除いた文字、それも無ければ全部の文字）
        mains = [g["baseline"] for g in gs if g["size"] >= main * 0.9 and g["c"] not in BIG_OPERATORS]
        others = [g["baseline"] for g in gs if g["c"] not in BIG_OPERATORS]
        base = statistics.median(mains or others or [g["baseline"] for g in gs])
        # トークンは [土台, 下付き, 上付き]。上下に重なった添字も、いつも _{下}^{上} の順に出す
        tokens: list[list[str]] = []
        used: set[int] = set()
        for i in sorted(range(len(gs)), key=lambda i: gs[i]["bbox"][0]):
            if i in used:
                continue
            g = gs[i]
            if (g["c"] in BIG_OPERATORS and g["c"] != "∫") or g.get("op") == "limits":
                # ∑ や lim の上と下に置かれた文字は範囲（\sum_{下}^{上}）。文中のように右横に付く範囲はふつうの添字
                upper, lower = limits(gs, i)
                used |= set(upper) | set(lower)
                sub = (self.read_line([gs[j] for j in lower]) or "") if lower else ""
                sup = (self.read_line([gs[j] for j in upper]) or "") if upper else ""
                tokens.append([glyph_tex(g), sub, sup])
                continue
            role = "base" if g["c"] == "′" else script_role(g, main, base)  # f' のプライムは上付きではない
            if role == "base":
                tokens.append([glyph_tex(g), "", ""])
                continue
            if not tokens:
                tokens.append(["{}", "", ""])
            slot = 1 if role == "sub" else 2
            tex = glyph_tex(g)
            tokens[-1][slot] = join_tokens([tokens[-1][slot], tex]) if tokens[-1][slot] else tex
        out = [b + (f"_{{{sub}}}" if sub else "") + (f"^{{{sup}}}" if sup else "") for b, sub, sup in tokens]
        return join_tokens(collapse_dots(out))


def join_tokens(tokens: list[str]) -> str:
    """トークンをつなぐ。\\alpha のような命令の後に、文字か命令（\\beta など）が続くときだけ空白を入れる"""
    out = ""
    for t in tokens:
        if out and re.search(r"\\[A-Za-z]+$", out) and (t[:1].isalpha() or re.match(r"\\[A-Za-z]", t)):
            out += " "
        out += t
    return out


def collapse_dots(tokens: list[str]) -> list[str]:
    """点が3つ並んだら1つにまとめる（. . . → \\dots、· · · → \\cdots）"""
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if out[-3:] == ["."] * 3:
            out[-3:] = [r"\dots"]
        elif out[-3:] == [r"\cdot"] * 3:
            out[-3:] = [r"\cdots"]
    return out


NEGATION = "\u0338"  # ≠ は、斜線（U+0338）と = を重ねて描かれる


def fold_negation(gs: list[dict]) -> list[dict]:
    """重ねて描かれた否定（斜線＋=）を \\neq の1文字にする"""
    out = list(gs)
    for slash in [g for g in gs if g["c"] == NEGATION]:
        eq = min((g for g in out if g["c"] == "="), key=lambda g: abs(center_x(g) - center_x(slash)), default=None)
        if eq is None or abs(center_x(eq) - center_x(slash)) > eq["bbox"][2] - eq["bbox"][0]:
            out = [g for g in out if g is not slash]
            continue
        merged = {**eq, "c": r"\neq", "folded": True, "bbox": union_all([eq["bbox"], slash["bbox"]])}
        out = [merged if g is eq else g for g in out if g is not slash]
    return out


def fold_accents(gs: list[dict]) -> list[dict]:
    """アクセントの文字（¯ ˆ ⃗ ˜ ˙）を、重なっている文字と1つにまとめる（\\bar{x}）"""
    out = list(gs)
    for a in [g for g in gs if g["c"] in ACCENTS]:
        candidates = [g for g in out if g is not a and g["c"] not in ACCENTS]

        def score(g: dict) -> float:
            if a["bbox"][2] - a["bbox"][0] > 0.5:  # 横に重なる幅
                return min(a["bbox"][2], g["bbox"][2]) - max(a["bbox"][0], g["bbox"][0])
            return -abs(g["bbox"][0] - a["bbox"][0])  # 幅の無い ⃗ は、近くの右の文字（3pt ほど左にずれて置かれる）

        if not candidates:
            continue
        base = max(candidates, key=score)
        if score(base) < -5:
            continue
        merged = {**base, "c": f"{ACCENTS[a['c']]}{{{glyph_tex(base)}}}", "folded": True}
        out = [merged if g is base else g for g in out if g is not a]
    return out


def fold_words(gs: list[dict]) -> list[dict]:
    """立体の文字で並んだ関数名（sin・log など）と、範囲の付く言葉（lim・max など）を1つにまとめる。
    同じベースラインの文字の中で探す（複数行の数式で、別の行の文字が間に入らないように）"""
    lines: list[list[dict]] = []
    for g in sorted(gs, key=lambda g: g["baseline"]):
        if lines and abs(lines[-1][0]["baseline"] - g["baseline"]) <= 0.5:
            lines[-1].append(g)
        else:
            lines.append([g])
    out: list[dict] = []
    for line in lines:
        ordered = sorted(line, key=lambda g: g["bbox"][0])
        k = 0
        while k < len(ordered):
            for name in sorted(FUNCTIONS, key=len, reverse=True):
                run = ordered[k:k + len(name)]
                if "".join(g["c"] for g in run) != name or any(MATH_FONT.search(g["font"]) or g.get("folded") for g in run):
                    continue
                if any(b["bbox"][0] - a["bbox"][2] > 1.0 for a, b in zip(run, run[1:])):
                    continue
                op = "limits" if name in LIMIT_WORDS else "function"
                out.append(folded("\\" + name, run, run[0]["size"], run[0]["baseline"], op))
                k += len(name)
                break
            else:
                out.append(ordered[k])
                k += 1
    return out


def nearest_row(gs: list[dict], y: float, side: str) -> list[dict]:
    """横線（高さ y）の上（side="above"）か下（"below"）の文字のうち、線に一番近い段の文字だけ。
    入れ子の分数で、内側の分母に外側の分母が入らないように"""
    if not gs:
        return []
    if side == "below":
        first = min(gs, key=lambda g: g["bbox"][1] - y)
    else:
        first = min(gs, key=lambda g: y - g["bbox"][3])
    top, bottom = first["bbox"][1], first["bbox"][3]
    pad = (bottom - top) * 0.25
    return [g for g in gs if top - pad <= center_y(g) <= bottom + pad]


def stacked(gs: list[dict]) -> tuple[list[dict], list[dict]]:
    """文字を上下2段に分ける（\\binom の上と下）。1段しかなければ ([], [])"""
    ordered = sorted(gs, key=center_y)
    gaps = [(center_y(b) - center_y(a), k + 1) for k, (a, b) in enumerate(zip(ordered, ordered[1:]))]
    if not gaps:
        return [], []
    gap, cut = max(gaps)
    if gap < main_size(gs) * 0.5:
        return [], []
    return ordered[:cut], ordered[cut:]


def main_size(gs: list[dict]) -> float:
    """主な文字サイズ。添字より大きい、ふつうの文字の大きさ = 一番大きい文字の大きさ
    （∑ などの大きい演算子と、大きいかっこ（CMEX）は除く）。添字の文字のほうが多い x^{10} でも正しく決まる"""
    sizes = [g["size"] for g in gs if g["c"] not in BIG_OPERATORS and "CMEX" not in g["font"]]
    return round(max(sizes or [g["size"] for g in gs]), 1)


def script_role(g: dict, main: float, base: float) -> str:
    """"base"（ふつうの文字）/ "sup"（上付き）/ "sub"（下付き）"""
    if g["size"] >= main * SCRIPT_SIZE:
        return "base"
    if g["baseline"] < base - main * SCRIPT_SHIFT:
        return "sup"
    if g["baseline"] > base + main * SCRIPT_SHIFT * 0.5:
        return "sub"
    return "base"


def limits(gs: list[dict], op: int) -> tuple[list[int], list[int]]:
    """大きい演算子（∑ や lim）の上と下に置かれた文字（上限, 下限）。
    文字の中心が演算子の幅の中にあり、演算子の中心より上（下）にあるもの。右横に付く添字（文中の ∑）は入れない"""
    ob = gs[op]["bbox"]
    mid_y = (ob[1] + ob[3]) / 2
    upper, lower = [], []
    for j, g in enumerate(gs):
        if j == op:
            continue
        cx = center_x(g)
        if not ob[0] - 1 <= cx <= ob[2] - 0.5 or g["bbox"][0] >= ob[2] - 0.5:
            continue
        cy = center_y(g)
        if g["bbox"][3] <= mid_y and cy < ob[1] + (ob[3] - ob[1]) * 0.35:
            upper.append(j)
        elif g["bbox"][1] >= mid_y and cy > ob[3] - (ob[3] - ob[1]) * 0.35:
            lower.append(j)
    return upper, lower


def split_lines(gs: list[dict]) -> list[list[dict]]:
    """複数行の数式（align など）を行に分ける。主な大きさの文字のベースラインで行を決め、ほかの文字は近い行に入れる"""
    main = main_size(gs)
    bases = sorted({round(g["baseline"]) for g in gs if g["size"] >= main * 0.9 and g["c"] not in BIG_OPERATORS})
    if not bases:
        return [gs]
    rows: list[float] = []
    for b in bases:
        if not rows or b - rows[-1] > main * 1.5:
            rows.append(b)
    lines: list[list[dict]] = [[] for _ in rows]
    for g in gs:
        # 文字の中心を、各行の数式の軸（ベースラインより少し上）と比べて、近い行に入れる
        k = min(range(len(rows)), key=lambda r: abs(rows[r] - main * AXIS - center_y(g)))
        lines[k].append(g)
    return [line for line in lines if line]


def drop_equation_number(gs: list[dict]) -> list[dict]:
    """右端に離れて置かれた式番号「(1)」「(2.3)」を除く。
    文字を後ろから1つずつ見る（下ごしらえでまとめた \\frac{a}{b} などは何文字もある1文字なので、文字列の位置では数えない）"""
    ordered = sorted(gs, key=lambda g: g["bbox"][0])
    if len(ordered) < 4 or ordered[-1]["c"] != ")":
        return gs
    k = len(ordered) - 2
    while k >= 0 and re.fullmatch(r"[0-9.]", ordered[k]["c"]):
        k -= 1
    if k < 0 or k == len(ordered) - 2 or ordered[k]["c"] != "(":
        return gs
    number, rest = ordered[k:], ordered[:k]
    if rest and number[0]["bbox"][0] - rest[-1]["bbox"][2] > main_size(gs) * 3:
        return rest
    return gs


_DRAWINGS: dict[tuple, list] = {}


def page_drawings(page: pymupdf.Page) -> list:
    """ページの図形。1回 1ms ほどかかるので、同じページは使い回す（直近の数ページ分だけ覚える）"""
    parent = getattr(page, "parent", None)
    if parent is None:
        return page.get_drawings()  # 文書につながっていないページ（テストの作り物など）は覚えない
    key = (id(parent), parent.name, page.number)
    if key not in _DRAWINGS:
        if len(_DRAWINGS) > 8:
            _DRAWINGS.clear()
        _DRAWINGS[key] = page.get_drawings()
    return _DRAWINGS[key]


def fraction_bars(page: pymupdf.Page, gs: list[dict]) -> list[tuple[float, float, float]]:
    """数式の範囲の中の、分数・根号の線の候補（細い横線）を (x0, x1, y) で。短い線（内側）から順に"""
    box = union_all([g["bbox"] for g in gs])
    bars = []
    for d in page_drawings(page):
        r = d["rect"]
        if r.height > 1.5 or r.width < 3:
            continue
        # 分数の線は中身より少し長い（入れ子の外側の線は、中の分数より左右に 1〜2pt はみ出す）
        if box[0] - 3 <= r.x0 and r.x1 <= box[2] + 3 and box[1] - 1 <= r.y0 <= box[3] + 1:
            bars.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
    return sorted(bars, key=lambda b: b[1] - b[0])


@dataclass
class OcrReader:
    """C. 範囲を画像に切り抜いて math_ocr で読む。重み（best.pt）は最初に使うときに読み込む。
    複数行の数式は行ごとに読む（長い数式は math_ocr が苦手なため）"""
    checkpoint: Path
    dpi: int = 200
    name: str = "ocr"
    _model: object = field(default=None, repr=False)
    _vocab: object = field(default=None, repr=False)

    def load(self) -> None:
        if self._model is None:
            from math_ocr.predict import load_model  # torch が要るので、使うときだけ読み込む
            self._model, self._vocab = load_model(self.checkpoint)

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        gs = shown(glyphs)
        if not gs:
            return None
        self.load()
        lines = []
        for line in split_lines(gs):
            line = drop_equation_number(line)
            if not line:
                continue
            lines.append(self.read_box(page, union_all([g["bbox"] for g in line])))
        return r" \\ ".join(t for t in lines if t) or None

    def read_box(self, page: pymupdf.Page, box: tuple) -> str:
        import torch
        from PIL import Image

        from math_ocr.dataset import preprocess

        clip = pymupdf.Rect(box) + (-2, -2, 2, 2)  # 少し余白を付けて切り抜く
        pixmap = page.get_pixmap(clip=clip, dpi=self.dpi, colorspace=pymupdf.csGRAY)
        image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
        with torch.no_grad():
            ids = self._model.generate(preprocess(image).unsqueeze(0))[0]
        return balance_braces(self._vocab.decode(ids.tolist()[1:]))


def balance_braces(tex: str) -> str:
    """OCRの結果の { } の数をそろえる（閉じ忘れ・閉じすぎでエディタのコンパイルが壊れないように）"""
    depth, out = 0, []
    for c in tex:
        if c == "}" and depth == 0:
            continue
        depth += c == "{"
        depth -= c == "}"
        out.append(c)
    return "".join(out) + "}" * depth


@dataclass
class FallbackReader:
    """D. まず first（ルール）で読み、読めなければ second（OCR）"""
    first: MathReader
    second: MathReader
    name: str = "rule+ocr"

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        return self.first.read(page, glyphs) or self.second.read(page, glyphs)


def make_reader(kind: str, checkpoint: str | Path | None = None) -> MathReader | None:
    """名前から読む係を作る。"none" なら None（数式を読まない）。OCR を使う係は重みが無ければエラー"""
    if kind == "none":
        return None
    if kind == "pdf_text":
        return PdfTextReader()
    if kind == "rule":
        return RuleReader()
    if checkpoint is None or not Path(checkpoint).exists():
        raise FileNotFoundError(f"{kind} には math_ocr の重み（best.pt）が要る: {checkpoint}")
    if kind == "ocr":
        return OcrReader(Path(checkpoint))
    if kind == "rule+ocr":
        return FallbackReader(RuleReader(), OcrReader(Path(checkpoint)))
    raise ValueError(f"不明な読む係 {kind!r}（none / pdf_text / rule / ocr / rule+ocr）")


# =====================================================================
# 文中の数式の範囲を探す
# =====================================================================

# 数式の文字どうしの間にあってよい文字（= + - ( ) など、数字、空白）。英語の単語や日本語が来たら数式の終わり
BETWEEN = re.compile(r"[\s0-9=+\-−×·±<>≤≥≠≈()\[\],./|:;'′!]")
# 数式の後ろに続いてよい文字（閉じかっこ・プライム・階乗）
TRAILING = set(")]′'!")
FUNCTION_WORD = re.compile(r"(?<![A-Za-z])(" + "|".join(FUNCTIONS) + r")$")


def is_math_glyph(g: dict) -> bool:
    """数式の種になる文字: 数式のフォント（CMMI・CMSY・MSBM など）か、アクセントの記号"""
    if g["bbox"] is None or MONO_FONT.search(g["font"]):
        return False
    return bool(MATH_FONT.search(g["font"])) or g["c"] in ACCENTS


def inline_math_spans(chars: list[dict]) -> list[tuple[int, int]]:
    """段落の文字の並びから、文中の数式の範囲（chars の [start, end) の添字）を探す。

    数式のフォントの文字を種にして、間が「数式の中にあってよい文字」だけなら、つないで1つの数式にする。
    種のすぐ後ろの小さい文字（添字の数字 x2 の 2 など）・閉じかっこ・階乗、前の開きかっこ・数字・関数名も入れる"""
    seeds = [i for i, g in enumerate(chars) if is_math_glyph(g)]
    if not seeds:
        return []
    spans: list[list[int]] = []
    for i in seeds:
        if spans:
            start, end = spans[-1]
            between = chars[end:i]
            same_line = all(g["c"] != "\n" for g in between)
            between_text = re.sub("|".join(FUNCTIONS), "", "".join(g["c"] for g in between))  # x + \sin y の sin
            if same_line and all(BETWEEN.fullmatch(c) for c in between_text):
                spans[-1][1] = i + 1
                continue
        spans.append([i, i + 1])
    result = []
    for start, end in spans:
        # 後ろに続く、小さい文字（添字）と閉じかっこ・階乗を入れる
        size = max(chars[k]["size"] for k in range(start, end))
        while end < len(chars) and chars[end]["bbox"] is not None and chars[end]["c"] != "\n" and (
            chars[end]["size"] < size * SCRIPT_SIZE or chars[end]["c"] in TRAILING
        ):
            end += 1
        # 前の開きかっこ・数字（2x の 2、(x の ( など）を入れる
        while start > 0 and chars[start - 1]["bbox"] is not None and re.fullmatch(r"[0-9(\[]", chars[start - 1]["c"]) \
                and not MONO_FONT.search(chars[start - 1]["font"]):
            start -= 1
        start = include_function_name(chars, start, size)
        result.append((start, end))
    return result


def include_function_name(chars: list[dict], start: int, size: float) -> int:
    """数式の前にある関数名（sin x の sin、log₂ x の log、lim の範囲の前の lim）を数式に入れた、新しい始まり。
    関数名は立体の文字なので数式のフォントではない。関数名と数式の間の空白・小さい添字は飛ばして探す"""
    k = start
    while k > 0 and chars[k - 1]["c"] != "\n" and (chars[k - 1]["c"] == " " or (
            chars[k - 1]["bbox"] is not None and chars[k - 1]["size"] < size * SCRIPT_SIZE)):
        k -= 1
    head = "".join(g["c"] for g in chars[max(0, k - 3):k])
    m = FUNCTION_WORD.search(head)
    if m is None:
        return start
    return k - (len(head) - m.start())
