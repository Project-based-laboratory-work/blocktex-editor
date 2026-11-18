"""
評価用データ（evaluation/synth_doc.py の table_blocks）と同じ作り方の表で、表AIを試すためのデータを作る。

学習データ（generate.py）とは別のプログラムで作った表なので、「学習データの作り方に慣れただけ」でないかを確かめられる。
正解は、行の数・列の数・セルの中身（cells）・列の揃え・罫線（セルの位置は無い）。

    python -m image_table.team_tables --out image_table/data/team --tables 500

表の部分は、表の上と下の目印（b<番号>s / b<番号>e）の間で、文字と罫線を囲む四角を 200dpi（評価と同じ）で画像にする。
"""

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from evaluation.synth_doc import preamble, table_blocks

DPI = 200
PAD = 4  # pt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    saved, per_doc = 0, 50
    for d in range(-(-args.tables // per_doc)):
        size, lang = rng.choice(["10pt", "11pt", "12pt"]), rng.choice(["ja", "en"])
        two_column = rng.random() < 0.5
        tables = [table_blocks(rng, lang, 2 * k, two_column, size)[0] for k in range(per_doc)]
        tex = preamble(size, two_column) + "".join(t.tex + "\n\\clearpage\n" for t in tables) + "\\end{document}\n"
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "doc.tex").write_text(tex, encoding="utf-8")
            try:
                for cmd in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                            ["dvipdfmx", "-q", "-C", "0x0010", "doc.dvi"]):  # fmt: skip
                    subprocess.run(cmd, cwd=tmp, check=True, capture_output=True, timeout=300)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
            with pymupdf.open(Path(tmp) / "doc.pdf") as pdf:
                h = pdf[0].rect.height
                marks = {n: (v["page"], h - v["to"][1]) for n, v in pdf.resolve_names().items()}
                for k, table in enumerate(tables):
                    s, e = marks.get(f"b{2 * k}s"), marks.get(f"b{2 * k}e")
                    if not s or not e or s[0] != e[0]:
                        continue
                    page = pdf[s[0]]
                    # 目印の間にある文字と罫線を集める。罫線は高さ0の四角なので、Rect の交差・結合ではなく座標で判定する
                    boxes = [w[:4] for w in page.get_text("words") if s[1] - 2 < w[1] and w[3] < e[1] + 2]
                    boxes += [tuple(dr["rect"]) for dr in page.get_drawings() if s[1] - 2 < dr["rect"].y0 and dr["rect"].y1 < e[1] + 2]
                    if not boxes:
                        continue
                    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
                    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
                    clip = pymupdf.Rect(x0 - PAD, y0 - PAD, x1 + PAD, y1 + PAD)
                    pix = page.get_pixmap(clip=clip, dpi=DPI, colorspace=pymupdf.csGRAY)
                    name = f"team_{d:03d}_{k:02d}"
                    pix.save(str(args.out / "images" / f"{name}.png"))
                    style = "grid" if table.col_rules[0] else "横線のみ" if table.row_rules[0] else "none"
                    label = {"cells": table.cells, "aligns": table.aligns, "row_rules": table.row_rules, "col_rules": table.col_rules,
                             "style": style, "wrap": table.col_widths is not None, "two_column": two_column}  # fmt: skip
                    (args.out / "labels" / f"{name}.json").write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
                    saved += 1
    print(f"{saved} 個 → {args.out}")


if __name__ == "__main__":
    main()
