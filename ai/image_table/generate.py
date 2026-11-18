"""
表AIの学習データ: ランダムな表をTeXで組んで画像にし、行・列の位置（正解）と一緒に保存する。

    python -m image_table.generate --out image_table/data/train --tables 20000 --seed 0 --workers 10

表の材料（罫線の種類・セルの語・折り返す列）は評価用データ（evaluation/synth_doc.py の table_blocks）に合わせ、
それより行・列を多く、中身も多様にする（数値の形・語・文・空のセル・太字の見出し）。

正解の作り方:
  1. 表を1ページに1つずつ置き、各セルの中身を目印 \\mk{t<表>_<行>_<列>s} 〜 \\mk{...e} で挟む
  2. PDFの文字（PyMuPDF の rawdict の char）のうち、目印の間にあるものを集め、その外枠をセルの四角にする
  3. 行の範囲 = その行のセルの四角の縦の範囲を合わせたもの。列も同じ（横の範囲）
  4. 表の部分（セルと罫線を囲む四角に、ランダムな余白を足したもの）を、ランダムな解像度で画像にする

出力: <out>/images/<名前>.png と <out>/labels/<名前>.json（座標は画像の画素）
"""

import argparse
import json
import random
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from evaluation.synth_doc import TABLE_RULE_STYLES, TABLE_WORDS, WORDS, sentence

TABLES_PER_DOC = 40
SIZES = {"10pt": 1.0, "11pt": 1.095, "12pt": 1.2}  # jsarticle の \mag（evaluation/synth_doc.py の MAGNIFICATION と同じ）
WRAP_CM = [3.0, 4.0, 5.0]
DPI_RANGE = (130, 260)  # 評価は 200dpi。スクショは解像度がまちまちなので幅を持たせる
CM = 72 / 2.54
PREAMBLE = r"""\documentclass[uplatex,a4paper,%s]{jsarticle}
\usepackage{booktabs}
\usepackage{array}
\usepackage[margin=1.5cm]{geometry}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
\pagestyle{empty}
\begin{document}
"""
EXTRA_WORDS = {
    "ja": ["提案手法", "従来手法", "学習時間", "正解率", "データ数", "パラメータ", "画像", "文書", "備考", "合計", "なし", "あり"],
    "en": ["Proposed", "Baseline", "Total", "Note", "Params", "Speed", "Recall", "F1", "Ours", "None", "Yes", "No"],
}


def number(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.35:
        return f"{rng.uniform(0, 100):.{rng.choice([1, 1, 2, 3])}f}"
    if kind < 0.55:
        return str(rng.randint(0, 10 ** rng.randint(1, 6)))
    if kind < 0.7:
        return f"{rng.uniform(0, 100):.1f}%"
    if kind < 0.8:
        return f"{rng.uniform(-50, 50):.2f}"
    if kind < 0.9:
        return f"{rng.randint(1, 999)} {rng.choice(['ms', 's', 'MB', 'px', 'pt'])}"
    return f"{rng.uniform(0, 1):.3f}"


def word(rng: random.Random, lang: str) -> str:
    words = TABLE_WORDS[lang] + EXTRA_WORDS[lang]
    if rng.random() < 0.3:
        return WORDS[lang]["sep"].join(rng.sample(words, 2))
    return rng.choice(words)


def random_table(rng: random.Random, lang: str) -> dict:
    """表1つの中身と形。cells[0] が見出しの行"""
    cols = rng.randint(2, 7)
    n_body = rng.randint(1, 9)
    wrap_col = rng.randrange(cols) if cols <= 5 and rng.random() < 0.25 else None
    text_cols = {c for c in range(cols) if rng.random() < 0.35}  # 数値ではなく語を入れる列
    cells = [[word(rng, lang) for _ in range(cols)]]
    for _ in range(n_body):
        row = []
        for c in range(cols):
            if c == wrap_col:
                row.append(WORDS[lang]["sep"].join(sentence(rng, lang) for _ in range(rng.randint(1, 3))))
            elif rng.random() < 0.04:
                row.append("")  # 空のセル
            else:
                row.append(word(rng, lang) if c in text_cols or c == 0 else number(rng))
        cells.append(row)
    aligns = [rng.choice("lcr") for _ in range(cols)]
    if wrap_col is not None:
        aligns[wrap_col] = "l"
    return {
        "cells": cells,
        "aligns": aligns,
        "style": rng.choice(TABLE_RULE_STYLES),
        "wrap_col": wrap_col,
        "wrap_cm": rng.choice(WRAP_CM),
        "bold_header": rng.random() < 0.3,
    }


def table_tex(t: int, table: dict) -> str:
    cells, style, wrap_col = table["cells"], table["style"], table["wrap_col"]
    specs = [
        rf">{{\raggedright\arraybackslash}}p{{{table['wrap_cm']}cm}}" if c == wrap_col else a
        for c, a in enumerate(table["aligns"])
    ]
    rule = "|" if style == "grid" else ""
    top, middle, bottom = {
        "booktabs": (r"\toprule", r"\midrule", r"\bottomrule"),
        "grid": (r"\hline", r"\hline", r"\hline"),
        "hline": (r"\hline", r"\hline", r"\hline"),
        "none": (None, None, None),
    }[style]

    def row_tex(r: int, row: list[str]) -> str:
        parts = []
        for c, cell in enumerate(row):
            body = cell.replace("%", r"\%")  # cells には見たままの文字を持ち、TeXでだけ逃がす
            body = rf"\textbf{{{body}}}" if r == 0 and table["bold_header"] and cell else body
            parts.append(rf"\mk{{t{t}_{r}_{c}s}}{body}\mk{{t{t}_{r}_{c}e}}")
        return " & ".join(parts) + r" \\"

    lines = [*([top] if top else []), row_tex(0, cells[0]), *([middle] if middle else [])]
    for r in range(1, len(cells)):
        lines.append(row_tex(r, cells[r]))
        if style == "grid" and r < len(cells) - 1:
            lines.append(r"\hline")
    lines += [bottom] if bottom else []
    return "\n".join([r"\begin{center}", rf"\begin{{tabular}}{{{rule + rule.join(specs) + rule}}}", *lines,
                      r"\end{tabular}", r"\end{center}", r"\clearpage", ""])  # fmt: skip


def rules_of(table: dict) -> tuple[list[bool], list[bool]]:
    """(行の境目ごとの横線, 列の境目ごとの縦線)。evaluation/synth_doc.py の table_blocks と同じ決まり"""
    style, n_rows, n_cols = table["style"], len(table["cells"]), len(table["cells"][0])
    has = style != "none"
    row_rules = [has, has, *([style == "grid"] * (n_rows - 2)), has]
    return row_rules, [style == "grid"] * (n_cols + 1)


def cell_boxes(page: pymupdf.Page, marks: dict, t: int, table: dict, mag: float) -> list[list[list[float] | None]] | None:
    """セルごとの文字の外枠 [x0, y0, x1, y1]（pt、上から）。空のセルは None。目印が欠けていたら None"""
    chars = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                chars += [(ch["origin"], ch["bbox"]) for ch in span["chars"] if ch["c"].strip()]
    boxes = []
    for r, row in enumerate(table["cells"]):
        out_row = []
        for c in range(len(row)):
            s, e = marks.get(f"t{t}_{r}_{c}s"), marks.get(f"t{t}_{r}_{c}e")
            if s is None or e is None:
                return None
            if c == table["wrap_col"]:
                x_lo, x_hi = s[0] - 1, s[0] + table["wrap_cm"] * CM * mag + 1
            else:
                x_lo, x_hi = s[0] - 0.5, e[0] + 0.5
            inside = [b for (ox, oy), b in chars if x_lo <= ox <= x_hi and s[1] - 1 <= oy <= e[1] + 1 and ox < x_hi]
            if not inside:
                out_row.append(None)
                continue
            out_row.append([min(b[0] for b in inside), min(b[1] for b in inside),
                            max(b[2] for b in inside), max(b[3] for b in inside)])  # fmt: skip
        boxes.append(out_row)
    return boxes


def spans(boxes: list, axis: int) -> list[list[float]] | None:
    """行（axis=1）または列（axis=0）ごとの範囲 [始まり, 終わり]。空の行・列があれば None"""
    groups = boxes if axis == 1 else [list(col) for col in zip(*boxes)]
    out = []
    for group in groups:
        present = [b for b in group if b is not None]
        if not present:
            return None
        out.append([min(b[axis] for b in present), max(b[axis + 2] for b in present)])
    return out


def build_doc(args: tuple[Path, int, int]) -> int:
    out, doc_index, seed = args
    rng = random.Random(seed * 1_000_003 + doc_index)
    size = rng.choice(list(SIZES))
    lang = rng.choice(["ja", "en"])
    tables = [random_table(rng, lang) for _ in range(TABLES_PER_DOC)]
    tex = PREAMBLE % size + "".join(table_tex(t, table) for t, table in enumerate(tables)) + "\\end{document}\n"
    saved = 0
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "doc.tex").write_text(tex, encoding="utf-8")
        try:
            for cmd in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], ["dvipdfmx", "-q", "-C", "0x0010", "doc.dvi"]):
                subprocess.run(cmd, cwd=tmp, check=True, capture_output=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return 0
        with pymupdf.open(Path(tmp) / "doc.pdf") as pdf:
            height = pdf[0].rect.height
            marks = {}
            for name, d in pdf.resolve_names().items():
                marks.setdefault(d["page"], {})[name] = (d["to"][0], height - d["to"][1])
            for t, table in enumerate(tables):
                page_no = next((p for p, m in marks.items() if f"t{t}_0_0s" in m), None)
                if page_no is None:
                    continue
                page = pdf[page_no]
                boxes = cell_boxes(page, marks[page_no], t, table, SIZES[size])
                rows, cols = (spans(boxes, 1), spans(boxes, 0)) if boxes else (None, None)
                if not rows or not cols:
                    continue
                # 表の部分 = セルと罫線を囲む四角 + ランダムな余白
                # 罫線は高さ（幅）0の四角なので、Rect の結合では無視される。座標で広げる
                boxes = [(cols[0][0], rows[0][0], cols[-1][1], rows[-1][1])] + [tuple(d["rect"]) for d in page.get_drawings()]
                area = pymupdf.Rect(min(b[0] for b in boxes), min(b[1] for b in boxes),
                                    max(b[2] for b in boxes), max(b[3] for b in boxes))  # fmt: skip
                if area.x1 > page.rect.x1 - 5:
                    continue  # ページからはみ出した表
                pad = [rng.uniform(1, 12) for _ in range(4)]
                clip = pymupdf.Rect(area.x0 - pad[0], area.y0 - pad[1], area.x1 + pad[2], area.y1 + pad[3])
                dpi = rng.randint(*DPI_RANGE)
                pix = page.get_pixmap(clip=clip, dpi=dpi, colorspace=pymupdf.csGRAY)
                f = dpi / 72

                def px(v: float, origin: float) -> float:
                    return round((v - origin) * f, 2)

                row_rules, col_rules = rules_of(table)
                name = f"t{seed}_{doc_index:05d}_{t:02d}"
                pix.save(str(out / "images" / f"{name}.png"))
                label = {
                    "width": pix.width, "height": pix.height, "dpi": dpi, "lang": lang, "style": table["style"],
                    "rows": [[px(a, clip.y0), px(b, clip.y0)] for a, b in rows],
                    "cols": [[px(a, clip.x0), px(b, clip.x0)] for a, b in cols],
                    "cells": table["cells"], "aligns": table["aligns"], "row_rules": row_rules, "col_rules": col_rules,
                    "wrap_col": table["wrap_col"],
                }  # fmt: skip
                (out / "labels" / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
                saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="表AIの学習データ（表の画像 + 行・列の位置）を作る")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=int, default=2000, help="作る表の数（組めなかった表・はみ出した表の分だけ少なくなる）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    docs = -(-args.tables // TABLES_PER_DOC)
    with ProcessPoolExecutor(args.workers) as pool:
        saved = sum(pool.map(build_doc, [(args.out, i, args.seed) for i in range(docs)]))
    print(f"{saved} 個の表を保存しました → {args.out}")


if __name__ == "__main__":
    main()
