"""
文中の装飾（太字・斜体・下線）を見分けるAIの学習データ: 文の1行をTeXで組んで画像にし、装飾の付いた範囲（正解）と一緒に保存する。

    python -m image_deco.generate --out image_deco/data/train --lines 30000 --seed 0 --workers 10

  1. 1行に収まる文を作り、ところどころの語に \\textbf・\\underline・\\textit（英語だけ）を付ける。
     評価用データ（evaluation/synth_doc.py）と同じく、日本語は太字・下線、英語は太字・斜体・下線
  2. 行の始まり・終わりと、装飾の範囲の前後に目印 \\mk{...} を入れる
  3. PDFの文字（rawdict の char）と下線（図形）から行の外枠を決め、ランダムな解像度で画像にする。
     日本語のフォント（原ノ味・IPAex）は埋め込む（埋め込まないと明朝と太字のゴシックが同じ見た目になる）
  4. 装飾の範囲は、目印の x 座標を画像の画素にしたもの

出力: <out>/images/<名前>.png と <out>/labels/<名前>.json
  labels: {"width", "height", "lang", "spans": [[x0, x1, "bold" | "italic" | "underline"], ...], "text"}
"""

import argparse
import json
import random
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from evaluation.synth_doc import WORDS

LINES_PER_DOC = 60
SIZES = {"10pt": 1.0, "11pt": 1.095, "12pt": 1.2}
DPI_RANGE = (110, 260)
MARKS = {"ja": ["bold", "underline"], "en": ["bold", "italic", "underline"]}
TEX = {"bold": r"\textbf", "italic": r"\textit", "underline": r"\underline"}
NO_DECORATION_RATE = 0.3  # 装飾の無い行の割合（何も無い行で「無い」と答えられるように）
# 日本語のフォントを埋め込む対応表。埋め込まないと、画像にしたとき明朝とゴシック（太字）が同じ代わりのフォントになり、
# 日本語の太字が見分けられない（#63）。ai/CONSTRAINTS.md の画像読み込みの制約「埋め込みは IPAex・原ノ味 など」に合わせる
FONT_MAPS = ["uptex-haranoaji.map", "uptex-ipaex.map"]
PREAMBLE = r"""\documentclass[uplatex,a4paper,%s]{jsarticle}
\usepackage[dvipdfmx,paperwidth=32cm,paperheight=40cm,margin=1cm]{geometry}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
\pagestyle{empty}
\setlength{\parskip}{6pt}
\begin{document}
"""
EXTRA = {
    "ja": ["この", "結果", "提案", "方法", "重要", "注意", "次の", "実験", "比較", "とても", "改善", "必ず", "条件", "評価値", "データ"],
    "en": ["this", "result", "method", "important", "note", "next", "very", "always", "model", "value", "data", "simple"],
}


def tokens(rng: random.Random, lang: str, n: int) -> list[str]:
    words = WORDS[lang]["nouns"] + WORDS[lang]["verbs"] + EXTRA[lang]
    out = [rng.choice(words) for _ in range(n)]
    if rng.random() < 0.3:
        out.insert(rng.randrange(len(out) + 1), f"{rng.uniform(0, 100):.1f}")
    return out


def random_line(rng: random.Random, lang: str) -> tuple[str, list[tuple[int, int, str]]]:
    """(1行のTeX, 装飾 [(開始の語, 終わりの語, 種類)]) 。目印はあとで付ける"""
    n = rng.randint(4, 14) if lang == "ja" else rng.randint(5, 16)
    words = tokens(rng, lang, n)
    spans = []
    if rng.random() > NO_DECORATION_RATE:
        free = list(range(n))
        for _ in range(rng.randint(1, 3)):
            if not free:
                break
            start = rng.choice(free)
            end = min(n, start + rng.choice([1, 1, 1, 2, 3]))
            if any(k not in free for k in range(start, end)):
                continue
            spans.append((start, end, rng.choice(MARKS[lang])))
            for k in range(max(0, start - 1), min(n, end + 1)):  # 隣どうしの装飾はくっつけない
                if k in free:
                    free.remove(k)
    return words, sorted(spans)


def line_tex(k: int, words: list[str], spans: list[tuple[int, int, str]], lang: str) -> str:
    sep = WORDS[lang]["sep"]
    parts, i, d = [], 0, 0
    while i < len(words):
        span = next((s for s in spans if s[0] == i), None)
        if span is None:
            parts.append(words[i])
            i += 1
            continue
        start, end, kind = span
        inner = sep.join(words[start:end])
        # 目印は装飾の命令の外に置く（命令の中だと下線などに目印が混ざる。synth_doc と同じ）
        parts.append(rf"\mk{{l{k}d{d}s}}{TEX[kind]}{{{inner}}}\mk{{l{k}d{d}e}}")
        d += 1
        i = end
    body = sep.join(parts) + WORDS[lang]["end"]
    return rf"\noindent\mk{{l{k}s}}{body}\mk{{l{k}e}}" + "\n\n"


def build_doc(args: tuple[Path, int, int]) -> int:
    out, doc_index, seed = args
    rng = random.Random(seed * 1_000_003 + doc_index)
    size, lang, font_map = rng.choice(list(SIZES)), rng.choice(["ja", "en"]), rng.choice(FONT_MAPS)
    lines = [random_line(rng, lang) for _ in range(LINES_PER_DOC)]
    tex = PREAMBLE % size + "".join(line_tex(k, w, s, lang) for k, (w, s) in enumerate(lines)) + "\\end{document}\n"
    saved = 0
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "doc.tex").write_text(tex, encoding="utf-8")
        try:
            for cmd in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                        ["dvipdfmx", "-q", "-C", "0x0010", "-f", font_map, "doc.dvi"]):  # fmt: skip
                subprocess.run(cmd, cwd=tmp, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0
        with pymupdf.open(Path(tmp) / "doc.pdf") as pdf:
            height = pdf[0].rect.height
            marks: dict[int, dict] = {}
            for name, d in pdf.resolve_names().items():
                marks.setdefault(d["page"], {})[name] = (d["to"][0], height - d["to"][1])
            pages = {}
            for page_no, page_marks in marks.items():
                page = pdf[page_no]
                chars = []
                for block in page.get_text("rawdict")["blocks"]:
                    for ln in block.get("lines", []):
                        for span in ln["spans"]:
                            chars += [ch["bbox"] for ch in span["chars"] if ch["c"].strip()]
                pages[page_no] = (page, page_marks, chars, [d["rect"] for d in page.get_drawings()])
            for k, (words, spans) in enumerate(lines):
                where = next(((p, v) for p, v in pages.items() if f"l{k}s" in v[1]), None)
                if where is None:
                    continue
                page, page_marks, chars, rules = where[1]
                s, e = page_marks[f"l{k}s"], page_marks.get(f"l{k}e")
                if e is None or abs(s[1] - e[1]) > 1:  # 折り返された行は使わない
                    continue
                inside = [c for c in chars if s[0] - 1 <= (c[0] + c[2]) / 2 <= e[0] + 1 and c[1] <= s[1] + 1 <= c[3] + 4]
                inside += [tuple(r) for r in rules if s[0] - 1 <= r.x0 and r.x1 <= e[0] + 1 and s[1] - 1 <= r.y0 <= s[1] + 6]
                if not inside:
                    continue
                pad = [rng.uniform(0.5, 4) for _ in range(4)]
                clip = pymupdf.Rect(min(c[0] for c in inside) - pad[0], min(c[1] for c in inside) - pad[1],
                                    max(c[2] for c in inside) + pad[2], max(c[3] for c in inside) + pad[3])  # fmt: skip
                dpi = rng.randint(*DPI_RANGE)
                pix = page.get_pixmap(clip=clip, dpi=dpi, colorspace=pymupdf.csGRAY)
                f = dpi / 72
                label_spans = []
                for d, (_, _, kind) in enumerate(spans):
                    a, b = page_marks.get(f"l{k}d{d}s"), page_marks.get(f"l{k}d{d}e")
                    if a and b:
                        label_spans.append([round((a[0] - clip.x0) * f, 2), round((b[0] - clip.x0) * f, 2), kind])
                if len(label_spans) != len(spans):
                    continue
                name = f"d{seed}_{doc_index:05d}_{k:02d}"
                pix.save(str(out / "images" / f"{name}.png"))
                label = {"width": pix.width, "height": pix.height, "lang": lang, "spans": label_spans,
                         "text": WORDS[lang]["sep"].join(words)}  # fmt: skip
                (out / "labels" / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
                saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="文中の装飾を見分けるAIの学習データを作る")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lines", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    docs = -(-args.lines // LINES_PER_DOC)
    with ProcessPoolExecutor(args.workers) as pool:
        saved = sum(pool.map(build_doc, [(args.out, i, args.seed) for i in range(docs)]))
    print(f"{saved} 行を保存しました → {args.out}")


if __name__ == "__main__":
    main()
