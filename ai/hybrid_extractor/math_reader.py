"""数式を読む係。数式の範囲（文字の並び）を受け取って、LaTeX を返す。読めなければ None。

    区切り・ラベルの専門家と同じく、差し替えられる形にしている:
        A. PdfTextReader   PDFの文字を並べるだけ（基準。添字・分数は崩れる）
        B. RuleReader      文字の位置と大きさから組み立てる（上付き・下付き・∑ の上下・分数）。根号は読めない → None
        C. OcrReader       範囲を画像に切り抜いて math_ocr で読む（学習済みの重み best.pt が要る）
        D. FallbackReader  B で読めなければ C

    使い方（文中の数式を見つけて読む）:
        spans = inline_math_spans(chars)              # 段落の文字から、文中の数式の範囲を探す
        tex = reader.read(page, glyphs_of(span))      # 読む

文字（glyph）は extract.py の1文字の辞書 {"c", "bbox", "font", "size", "flags", "baseline"}。
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
    "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "ξ": r"\xi",
    "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "ϕ": r"\phi", "χ": r"\chi", "ψ": r"\psi",
    "ω": r"\omega", "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Π": r"\Pi", "Σ": r"\Sigma",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
}
SYMBOLS = {
    "−": "-", "×": r"\times", "·": r"\cdot", "÷": r"\div", "±": r"\pm", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq",
    "≈": r"\approx", "→": r"\to", "←": r"\leftarrow", "⇒": r"\Rightarrow", "∞": r"\infty", "∂": r"\partial",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "∪": r"\cup", "∩": r"\cap", "∀": r"\forall", "∃": r"\exists",
    "∇": r"\nabla", "′": "'", "…": r"\dots", "⋯": r"\cdots", "{": r"\{", "}": r"\}",
}
BIG_OPERATORS = {"∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "⋃": r"\bigcup", "⋂": r"\bigcap"}
FUNCTIONS = ("sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min", "sup", "inf", "det")
HARD = {"√"}  # 根号はルールでは組み立てない

SCRIPT_SIZE = 0.85      # 主な文字サイズのこの割合より小さい文字は、添字の候補
SCRIPT_SHIFT = 0.12     # ベースラインが主な文字サイズのこの割合以上ずれていれば、上付き・下付き
FRACTION_REACH = 1.2    # 分数の線から、主な文字サイズのこの倍までの文字を分子・分母にする
AXIS = 0.3              # 数式の軸（分数の線や ∫ の中心の高さ）は、ベースラインより主な文字サイズのこの割合だけ上


def to_latex_char(c: str) -> str:
    return GREEK.get(c) or SYMBOLS.get(c) or BIG_OPERATORS.get(c) or c


def shown(glyphs: list[dict]) -> list[dict]:
    return [g for g in glyphs if g["bbox"] is not None and not g["c"].isspace()]


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
    """B. 文字の位置と大きさから組み立てる。複数行の数式は行ごとに組み立てて \\\\ でつなぐ"""
    name: str = "rule"

    def read(self, page: pymupdf.Page, glyphs: list[dict]) -> str | None:
        gs = shown(glyphs)
        if not gs or any(g["c"] in HARD for g in gs):
            return None
        gs = self.fold_fractions(page, gs)
        if gs is None:
            return None
        lines = [self.read_line(line) for line in split_lines(gs)]
        if any(line is None for line in lines):
            return None
        return r" \\ ".join(lines)

    def fold_fractions(self, page: pymupdf.Page, gs: list[dict]) -> list[dict] | None:
        """分数の横線ごとに、線の上の文字（分子）と下の文字（分母）を \\frac{分子}{分母} の1文字にまとめる。
        短い線（内側の分数）から順にまとめるので、入れ子の分数も読める。組み立てられなければ None"""
        main = main_size(gs)
        for bar in fraction_bars(page, gs):
            x0, x1, y = bar
            near = [g for g in gs if x0 - 1 <= (g["bbox"][0] + g["bbox"][2]) / 2 <= x1 + 1]
            # 線のすぐ上・すぐ下（文字1つ分くらい）の文字だけ。離れた文字（∑ の下限など）は入れない
            num = [g for g in near if y - FRACTION_REACH * main <= g["bbox"][3] <= y + 1]
            den = [g for g in near if y - 1 <= g["bbox"][1] <= y + FRACTION_REACH * main]
            if not num or not den:
                continue
            top, bottom = self.read_line(num), self.read_line(den)
            if top is None or bottom is None:
                return None
            box = union_all([g["bbox"] for g in num + den])
            # 分数の線は、行のベースラインより文字サイズの 1/4 くらい上（数式の軸）にある
            folded = {"c": f"\\frac{{{top}}}{{{bottom}}}", "bbox": box, "font": "", "size": main,
                      "flags": 0, "baseline": y + main * 0.25}
            taken = {id(g) for g in num + den}
            gs = [g for g in gs if id(g) not in taken] + [folded]
        return gs

    def read_line(self, gs: list[dict]) -> str | None:
        gs = drop_equation_number(gs)
        if not gs:
            return None
        main = main_size(gs)
        # 行のベースライン。主な大きさの文字で決める（無ければ、∑ などを除いた文字、それも無ければ全部の文字）
        mains = [g["baseline"] for g in gs if g["size"] >= main * 0.9 and g["c"] not in BIG_OPERATORS]
        others = [g["baseline"] for g in gs if g["c"] not in BIG_OPERATORS]
        base = statistics.median(mains or others or [g["baseline"] for g in gs])
        tokens: list[str] = []
        used: set[int] = set()
        order = sorted(range(len(gs)), key=lambda i: gs[i]["bbox"][0])
        k = 0
        while k < len(order):
            i = order[k]
            if i in used:
                k += 1
                continue
            g = gs[i]
            if g["c"] in BIG_OPERATORS and g["c"] != "∫":
                # ∑ の真上・真下にある小さい文字は、上限・下限（\sum_{下}^{上}）
                upper, lower = limits(gs, i)
                used |= set(upper) | set(lower)
                tok = BIG_OPERATORS[g["c"]]
                if lower:
                    tok += "_{" + join_tokens([to_latex_char(gs[j]["c"]) for j in sorted(lower, key=lambda j: gs[j]["bbox"][0])]) + "}"
                if upper:
                    tok += "^{" + join_tokens([to_latex_char(gs[j]["c"]) for j in sorted(upper, key=lambda j: gs[j]["bbox"][0])]) + "}"
                tokens.append(tok)
                k += 1
                continue
            role = script_role(g, main, base)
            if role == "base":
                tokens.append(to_latex_char(g["c"]))
                k += 1
                continue
            # 同じ向きの添字が続く間を1つにまとめる（x^{10}、a_{ij}）
            group = []
            while k < len(order) and order[k] not in used and script_role(gs[order[k]], main, base) == role \
                    and gs[order[k]]["c"] not in BIG_OPERATORS:
                group.append(to_latex_char(gs[order[k]]["c"]))
                k += 1
            mark = "^" if role == "sup" else "_"
            if not tokens:
                tokens.append("{}")
            tokens[-1] += mark + "{" + join_tokens(group) + "}"
        return join_tokens(name_functions(tokens))


def join_tokens(tokens: list[str]) -> str:
    """トークンをつなぐ。\\alpha のような命令の後に文字が続くときだけ空白を入れる"""
    out = ""
    for t in tokens:
        if out and re.search(r"\\[A-Za-z]+$", out) and t[:1].isalpha():
            out += " "
        out += t
    return out


def name_functions(tokens: list[str]) -> list[str]:
    """s i n のように並んだ1文字を、\\sin にまとめる"""
    out: list[str] = []
    k = 0
    while k < len(tokens):
        for name in FUNCTIONS:
            n = len(name)
            if "".join(tokens[k:k + n]) == name and all(len(t) == 1 for t in tokens[k:k + n]):
                out.append("\\" + name + " ")
                k += n
                break
        else:
            out.append(tokens[k])
            k += 1
    return [t.rstrip() if i == len(out) - 1 else t for i, t in enumerate(out)]


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
    """大きい演算子（∑ など）の真上・真下にある文字の添字（上限, 下限）"""
    ob = gs[op]["bbox"]
    upper, lower = [], []
    for j, g in enumerate(gs):
        if j == op:
            continue
        cx = (g["bbox"][0] + g["bbox"][2]) / 2
        if not ob[0] - 2 <= cx <= ob[2] + 2:
            continue
        cy = (g["bbox"][1] + g["bbox"][3]) / 2
        if cy < ob[1] + (ob[3] - ob[1]) * 0.2:
            upper.append(j)
        elif cy > ob[3] - (ob[3] - ob[1]) * 0.2:
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
        cy = (g["bbox"][1] + g["bbox"][3]) / 2
        k = min(range(len(rows)), key=lambda r: abs(rows[r] - main * AXIS - cy))
        lines[k].append(g)
    return [line for line in lines if line]


def drop_equation_number(gs: list[dict]) -> list[dict]:
    """右端に離れて置かれた式番号「(1)」を除く"""
    ordered = sorted(gs, key=lambda g: g["bbox"][0])
    text = "".join(g["c"] for g in ordered)
    m = re.search(r"\(\d+(\.\d+)?\)$", text)
    if not m:
        return gs
    number = ordered[m.start():]
    rest = ordered[:m.start()]
    if rest and number[0]["bbox"][0] - rest[-1]["bbox"][2] > main_size(gs) * 3:
        return rest
    return gs


def fraction_bars(page: pymupdf.Page, gs: list[dict]) -> list[tuple[float, float, float]]:
    """数式の範囲の中の、分数の線の候補（細い横線）を (x0, x1, y) で。短い線（内側の分数）から順に"""
    box = union_all([g["bbox"] for g in gs])
    bars = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.height > 1.5 or r.width < 3:
            continue
        if box[0] - 1 <= r.x0 and r.x1 <= box[2] + 1 and box[1] - 1 <= r.y0 <= box[3] + 1:
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
BETWEEN = re.compile(r"[\s0-9=+\-−×·±<>≤≥≠≈()\[\],./|:;'′]")


def is_math_glyph(g: dict) -> bool:
    return g["bbox"] is not None and bool(MATH_FONT.search(g["font"])) and not MONO_FONT.search(g["font"])


def inline_math_spans(chars: list[dict]) -> list[tuple[int, int]]:
    """段落の文字の並びから、文中の数式の範囲（chars の [start, end) の添字）を探す。

    数式のフォントの文字を種にして、間が「数式の中にあってよい文字」だけなら、つないで1つの数式にする。
    種のすぐ後ろの小さい文字（添字の数字 x2 の 2 など）も数式に入れる"""
    seeds = [i for i, g in enumerate(chars) if is_math_glyph(g)]
    if not seeds:
        return []
    spans: list[list[int]] = []
    for i in seeds:
        if spans:
            start, end = spans[-1]
            between = chars[end:i]
            same_line = all(g["c"] != "\n" for g in between)
            if same_line and all(BETWEEN.fullmatch(g["c"]) for g in between):
                spans[-1][1] = i + 1
                continue
        spans.append([i, i + 1])
    result = []
    for start, end in spans:
        # 後ろに続く、小さい文字（添字）と閉じかっこ・数字を入れる
        size = max(chars[k]["size"] for k in range(start, end))
        while end < len(chars) and chars[end]["bbox"] is not None and (
            chars[end]["size"] < size * SCRIPT_SIZE or (chars[end]["c"] in ")]′'" )
        ) and chars[end]["c"] != "\n":
            end += 1
        # 前の開きかっこ・数字（2x の 2、(x の ( など）を入れる
        while start > 0 and chars[start - 1]["bbox"] is not None and re.fullmatch(r"[0-9(\[]", chars[start - 1]["c"]) \
                and not MONO_FONT.search(chars[start - 1]["font"]):
            start -= 1
        result.append((start, end))
    return result
