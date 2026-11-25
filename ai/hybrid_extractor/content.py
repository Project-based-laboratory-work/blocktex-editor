"""文書の「中身の材料」（単語・見出し・キャプションの語・数式・コード）のパック。

学習データ（doc_gen.py）と評価データを同じ材料で作ると、AI が言い回しを覚えて点数が高めに出る
（前に見つけた「表1 で始まればキャプション」の近道も、これが原因だった）。そこで材料をパックにして差し替えられるようにし、
「見たことのない中身」で評価できるようにする（AI.md の 7.7）。

    default  今までの材料（学習に使う）
    alt      手で用意した別の材料（単語・見出し・キャプションの語・数式の部品・コード）。評価だけに使う
    real     alt に、本物の文（青空文庫・Project Gutenberg）と本物の数式（im2latex-100k）を足したもの。評価だけに使う

real の材料は外から取ってきたもので、git には入れない（data/content/、build_real で作る）:

    python -m hybrid_extractor.content --build   # data/content/raw/ のファイルから、文と数式の一覧を作る
"""

import argparse
import html
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

CONTENT_DIR = Path(__file__).parent / "data" / "content"
RAW_DIR = CONTENT_DIR / "raw"


@dataclass(frozen=True)
class Content:
    name: str
    words: dict                      # 言語 → {"nouns", "verbs", "ends", "headings", "authors", "captions", "leads", "sep"}
    cell_words: dict                 # 言語 → 表の見出しの語
    math_parts: list[str]            # 別行立ての数式の部品（いくつかを + でつなぐ）
    inline_parts: list[str]          # 段落の中の文中の数式
    code_snippets: list[list[str]]
    title_kinds: list[str] = field(default_factory=lambda: ["調査", "記録", "検討"])  # 日本語のタイトルの終わり
    sentences: dict | None = None    # 言語 → 本物の文の一覧（あれば、文はここから選ぶ）
    display_formulas: list[str] | None = None  # 本物の別行立ての数式（あれば math_parts の代わり）
    inline_formulas: list[str] | None = None   # 本物の文中の数式（あれば inline_parts と、文中の数式の実験の数式の代わり）


DEFAULT = Content(
    name="default",
    words={
        "ja": {
            "nouns": ["料理", "旅行", "天気", "電車", "図書館", "植物", "音楽", "地図", "時計", "温度", "距離", "速度",
                      "観測", "記録", "材料", "工程", "地域", "季節", "道具", "計画", "予算", "費用", "人数", "期間"],
            "verbs": ["を調べた", "を集めた", "を比べた", "をまとめた", "を見直した", "を記録した", "を計算した", "を予測した"],
            "ends": ["。", "。", "。", "ことが分かった。", "必要がある。", "と考えられる。"],
            "headings": ["背景", "目的", "準備", "手順", "観察", "考察", "まとめ", "課題", "方法", "結果の整理", "今後の予定", "参考"],
            "authors": ["鈴木一郎", "高橋花子", "田中次郎", "伊藤美咲"],
            "captions": ["の変化", "の比較", "の一覧", "の分布", "の流れ"],
            "leads": ["注意", "補足", "例", "ポイント"],
            "sep": "",
        },
        "en": {
            "nouns": ["the recipe", "the journey", "the weather", "the train", "the library", "the plant", "the song",
                      "the map", "the clock", "the temperature", "the distance", "the budget", "the season", "the tool"],
            "verbs": ["changes", "affects", "follows", "records", "explains", "limits", "supports", "reflects"],
            "ends": [".", ".", ".", " in most cases.", " over time.", " as expected."],
            "headings": ["Background", "Goals", "Setup", "Procedure", "Observations", "Analysis", "Summary",
                         "Open Problems", "Approach", "Findings", "Next Steps", "Notes"],
            "authors": ["Ken Suzuki", "Mary Brown", "John Lee", "Emma Davis"],
            "captions": [" over time", " by region", " in detail", " at a glance", " per step"],
            "leads": ["Note.", "Remark.", "Example.", "Tip."],
            "sep": " ",
        },
    },
    cell_words={"ja": ["項目", "回数", "時刻", "場所", "合計", "割合", "平均値", "番号"],
                "en": ["Item", "Count", "Time", "Place", "Total", "Rate", "Mean", "No."]},
    math_parts=[r"a^{2} + b^{2}", r"\frac{x}{y}", r"\sqrt{2g h}", r"\sum_{k=0}^{m} c_{k}", r"\prod_{j=1}^{n} p_{j}",
                r"\cos\theta", r"\lim_{t \to 0} h(t)", r"\mu + \sigma", r"m v^{2}", r"\int_{a}^{b} u(s) \, ds"],
    inline_parts=[r"$x_{i}$", r"$\alpha + \beta$", r"$n^{2}$", r"$f(t)$"],
    code_snippets=[
        ["def mean(xs):", "    total = 0", "    for x in xs:", "        total += x", "    return total / len(xs)"],
        ["while queue:", "    node = queue.pop()", "    visit(node)"],
        ["for (int k = 0; k < n; k++) {", "    sum += v[k];", "}"],
        ["SELECT name, age", "FROM users", "WHERE age > 20;"],
        ["if x > limit:", "    x = limit", "else:", "    x += 1"],
    ],
)

# 学習に一度も使わない材料。分野（理工系のレポート）も、言い回しも変える
ALT = Content(
    name="alt",
    words={
        "ja": {
            "nouns": ["回路", "電圧", "抵抗", "試料", "溶液", "濃度", "周波数", "波形", "測定値", "誤差", "装置", "センサ",
                      "反応", "触媒", "荷重", "変位", "信号", "雑音", "標本", "母集団", "分散", "閾値", "アルゴリズム", "計算量"],
            "verbs": ["を測定した", "を推定した", "を評価した", "を検証した", "を設計した", "を実装した", "を解析した", "を調整した"],
            "ends": ["。", "。", "ことを確認した。", "ためである。", "と推察される。", "ことが示された。"],
            "headings": ["はじめに", "実験方法", "実験装置", "実験結果", "考察", "結論", "関連研究", "理論", "評価方法",
                         "謝辞", "付録", "問題設定", "提案手法", "結果と考察"],
            "authors": ["山本健太", "中村結衣", "小林大輝", "加藤沙織"],
            "captions": ["の測定結果", "の構成図", "の推移", "と誤差の関係", "の実験条件"],
            "leads": ["定義", "備考", "手順", "仮定"],
            "sep": "",
        },
        "en": {
            "nouns": ["the circuit", "the voltage", "the sample", "the solution", "the frequency", "the waveform",
                      "the sensor", "the signal", "the noise", "the estimator", "the variance", "the algorithm",
                      "the load", "the threshold"],
            "verbs": ["measures", "estimates", "validates", "reduces", "amplifies", "predicts", "dominates", "bounds"],
            "ends": [".", ".", " in all trials.", " under these conditions.", " within the error.", " for large inputs."],
            "headings": ["Introduction", "Method", "Experimental Setup", "Results", "Discussion", "Conclusion",
                         "Related Work", "Theory", "Evaluation", "Acknowledgments", "Appendix", "Problem Statement"],
            "authors": ["Kenta Yamamoto", "Yui Nakamura", "Daiki Kobayashi", "Saori Kato"],
            "captions": [" measured at each step", " and its error", " under load", " for all samples", " in the setup"],
            "leads": ["Definition.", "Assumption.", "Procedure.", "Caution."],
            "sep": " ",
        },
    },
    cell_words={"ja": ["条件", "試行", "電圧", "電流", "誤差", "平均", "標準偏差", "試料"],
                "en": ["Condition", "Trial", "Voltage", "Current", "Error", "Avg.", "Std.", "Sample"]},
    math_parts=[r"\frac{a + b}{2}", r"\sqrt{x^{2} + y^{2}}", r"\sum_{i=1}^{n} x_{i}^{2}", r"e^{-\lambda t}",
                r"\int_{0}^{\infty} f(t) \, dt", r"\log_{2} N", r"\max_{k} w_{k}", r"\binom{n}{k}", r"\bar{x}",
                r"\alpha \beta"],
    inline_parts=[r"$y = ax + b$", r"$\theta$", r"$k_{B} T$", r"$\sigma^{2}$", r"$N - 1$"],
    code_snippets=[
        ["#include <stdio.h>", "int main(void) {", "    printf(\"%d\\n\", 42);", "    return 0;", "}"],
        ["function sum(a) {", "  let s = 0;", "  for (const x of a) s += x;", "  return s;", "}"],
        ["for f in *.csv; do", "  wc -l \"$f\"", "done"],
        ["class Stack:", "    def __init__(self):", "        self.items = []", "    def push(self, x):",
         "        self.items.append(x)"],
    ],
    title_kinds=["実験", "報告", "解析"],
)


@lru_cache(maxsize=None)
def real_content() -> Content:
    """alt に、本物の文と本物の数式を足したもの（data/content/ の一覧を読む。無ければ build_real で作る）"""
    files = {name: CONTENT_DIR / f"{name}.txt" for name in ("ja_sentences", "en_sentences", "display_formulas",
                                                            "inline_formulas")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"本物の材料が無い: {missing}（python -m hybrid_extractor.content --build で作る）")
    read = {name: [line for line in p.read_text(encoding="utf-8").splitlines() if line] for name, p in files.items()}
    return replace(ALT, name="real", sentences={"ja": read["ja_sentences"], "en": read["en_sentences"]},
                   display_formulas=read["display_formulas"], inline_formulas=read["inline_formulas"])


def get_content(name: str) -> Content:
    if name == "default":
        return DEFAULT
    if name == "alt":
        return ALT
    if name == "real":
        return real_content()
    raise ValueError(f"知らない材料のパック: {name}（default / alt / real）")


# =====================================================================
# 本物の材料を作る（data/content/raw/ → data/content/*.txt）
# =====================================================================

# TeX で意味を持つ記号（文の中に入れると壊れる）。こういう文字を含む文は使わない
TEX_SPECIAL = re.compile(r"[\\%&#$_{}~^|<>]")
# 数式を読む係（math_reader.py）と数式OCR の対応表で扱える命令
FORMULA_COMMANDS = {
    r"\frac", r"\sqrt", r"\sum", r"\prod", r"\int", r"\lim", r"\infty", r"\partial", r"\nabla", r"\cdot", r"\times",
    r"\pm", r"\leq", r"\le", r"\geq", r"\ge", r"\neq", r"\approx", r"\equiv", r"\in", r"\subset", r"\cup", r"\cap",
    r"\to", r"\rightarrow", r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp", r"\max", r"\min", r"\hat", r"\bar",
    r"\vec", r"\tilde", r"\dot", r"\mathbb", r"\mathbf", r"\left", r"\right", r"\binom", r"\dots", r"\cdots",
    r"\ldots", r"\prime", r"\ell", r"\,", r"\;", r"\!", r"\quad",
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta", r"\eta", r"\theta", r"\kappa",
    r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\phi", r"\varphi", r"\chi", r"\psi",
    r"\omega", r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega",
}
FORMULA_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=<>()[],./|'!_^{}")
FORMULA_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|\S")
INLINE_MAX_TOKENS = 20
DISPLAY_MAX_TOKENS = 40


def formula_ok(tokens: list[str]) -> bool:
    """対応している命令と文字だけでできていて、かっこの数が合っているか"""
    if not tokens:
        return False
    for t in tokens:
        if t.startswith("\\") and len(t) > 1:
            if t not in FORMULA_COMMANDS:
                return False
        elif t not in FORMULA_CHARS:
            return False
    return tokens.count("{") == tokens.count("}") and tokens.count(r"\left") == tokens.count(r"\right")


def build_formulas(lines: list[str]) -> tuple[list[str], list[str]]:
    """im2latex の数式（トークンが空白で区切ってある）から、文中の数式と別行立ての数式の一覧を作る。
    末尾の句読点（. ,）は外す。短いもの（20トークン以下）は文中、それより長く40トークン以下は別行立て"""
    inline, display = [], []
    seen = set()
    for line in lines:
        tokens = FORMULA_TOKEN.findall(line)
        while tokens and tokens[-1] in (".", ",", r"\,", r"\;", r"\quad"):
            tokens.pop()
        if not formula_ok(tokens) or "&" in tokens:
            continue
        tex = " ".join(tokens)
        if tex in seen:
            continue
        seen.add(tex)
        n = sum(t not in "{}" for t in tokens)
        if 2 <= n <= INLINE_MAX_TOKENS:
            inline.append(tex)
        elif n <= DISPLAY_MAX_TOKENS:
            display.append(tex)
    return inline, display


def aozora_sentences(raw_html: bytes) -> list[str]:
    """青空文庫の XHTML（Shift_JIS）の本文から、文の一覧。ルビ（読みがな）と注記は外す"""
    text = raw_html.decode("cp932", errors="ignore")
    m = re.search(r'<div class="main_text">(.*?)<div class="bibliographical_information">', text, re.S)
    body = m.group(1) if m else text
    body = re.sub(r"<rp>.*?</rp>|<rt>.*?</rt>", "", body, flags=re.S)   # ルビの読みがなを外す
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body).replace("\r", "").replace("\n", "").replace("　", "")
    body = re.sub(r"［＃[^］]*］", "", body)                             # 注記
    out = []
    for s in re.split(r"(?<=。)", body):
        s = s.strip()
        if 15 <= len(s) <= 80 and not TEX_SPECIAL.search(s) and "※" not in s and "「" not in s:
            out.append(s)
    return out


def gutenberg_sentences(text: str) -> list[str]:
    """Project Gutenberg のテキストから、本文の文の一覧（前後の説明と、記号の多い行は使わない）"""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    body = text[text.find("\n", start) + 1:end if end > 0 else None]
    body = re.sub(r"\s+", " ", body.replace("_", ""))
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body):
        s = s.strip()
        if 40 <= len(s) <= 200 and s.isascii() and not TEX_SPECIAL.search(s) and s[0].isupper() \
                and not re.search(r"[\[\]*\"]|[A-Z]{4,}", s):
            out.append(s)
    return out


def build_real() -> None:
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    ja = [s for p in sorted(RAW_DIR.glob("aozora_*.html")) for s in aozora_sentences(p.read_bytes())]
    en = [s for p in sorted(RAW_DIR.glob("gutenberg_*.txt")) for s in gutenberg_sentences(p.read_text(encoding="utf-8"))]
    lines = (RAW_DIR / "im2latex_test_formulas.txt").read_text(encoding="utf-8").splitlines()
    inline, display = build_formulas(lines)
    for name, items in (("ja_sentences", ja), ("en_sentences", en), ("inline_formulas", inline),
                        ("display_formulas", display)):
        (CONTENT_DIR / f"{name}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
        print(f"{name}: {len(items)}個  例: {items[:2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中身の材料のパック")
    parser.add_argument("--build", action="store_true", help="data/content/raw/ から本物の材料の一覧を作る")
    if parser.parse_args().build:
        build_real()
