"""
数式を本物のLaTeX（pdflatex）で描画し、synth_data.py と同じ形式のデータセットを作る。

synth_data.py の画像は matplotlib の mathtext で描いたもので、LaTeXが組んだ数式とは字形や
記号の配置が少し違う。実際のPDFに近い画像で評価・学習するために、同じ数式をLaTeXで描き直す。

1ページに数式を1つだけ置いた文書をまとめてコンパイルし、ページごとに画像へ変換する。
数式1つごとにLaTeXを起動するより、はるかに速い（1000件で数十秒）。

    python -m math_ocr.render_latex --src math_ocr/data/math_synth_60k --split val --out math_ocr/data/math_latex

pdflatex と pdftoppm（poppler-utils）が必要。
"""

import argparse
import random
import shutil
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

from math_ocr.dataset import read_labels

CHUNK_SIZE = 500  # 1つの文書に入れる数式の数

# 用紙を横長にして、長い数式がはみ出したり折り返されたりしないようにする。
# ページ番号が入ると数式と一緒に切り出されてしまうので \pagestyle{empty} で消す。
TEMPLATE = r"""\documentclass{article}
\usepackage[paperwidth=50cm,paperheight=8cm,margin=1cm]{geometry}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
%s
\end{document}
"""


def _render_chunk(args: tuple[int, list[str], Path, str, int]) -> list[tuple[str, str]]:
    chunk_index, formulas, image_dir, split, seed = args
    # 解像度をチャンクごとに変えて、文字の太さ・にじみ方にばらつきを持たせる
    dpi = random.Random(seed * 1_000_003 + chunk_index).randint(100, 200)
    body = "\n\\newpage\n".join(rf"\[ {formula} \]" for formula in formulas)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "doc.tex").write_text(TEMPLATE % body, encoding="utf-8")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
            cwd=tmp_dir, check=True, capture_output=True,
        )  # fmt: skip
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-gray", "-png", "doc.pdf", "page"],
            cwd=tmp_dir, check=True, capture_output=True,
        )  # fmt: skip
        # pdftoppm は page-001.png のような連番で出力する。名前順に並べるとページ順になる
        pages = sorted(tmp_dir.glob("page-*.png"))
        if len(pages) != len(formulas):
            raise RuntimeError(f"ページ数 {len(pages)} が数式の数 {len(formulas)} と合わない")

        results = []
        for offset, (page, formula) in enumerate(zip(pages, formulas)):
            filename = f"{split}_{chunk_index * CHUNK_SIZE + offset:06d}.png"
            shutil.move(page, image_dir / filename)
            results.append((filename, formula))
    return results


def render_dataset(src: Path, out: Path, split: str, limit: int | None, seed: int, workers: int) -> None:
    formulas = [latex for _, latex in read_labels(src, split)][:limit]
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    chunks = [formulas[i : i + CHUNK_SIZE] for i in range(0, len(formulas), CHUNK_SIZE)]
    jobs = [(i, chunk, image_dir, split, seed) for i, chunk in enumerate(chunks)]
    with Pool(workers) as pool:
        results = pool.map(_render_chunk, jobs)

    with open(out / f"{split}.tsv", "w", encoding="utf-8") as f:
        for filename, latex in (pair for chunk in results for pair in chunk):
            f.write(f"{filename}\t{latex}\n")
    print(f"{split}: {len(formulas)} 件をLaTeXで描画 → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="数式の一覧（{split}.tsv）があるディレクトリ")
    parser.add_argument("--out", type=Path, default=Path("math_ocr/data/math_latex"))
    parser.add_argument("--split", nargs="+", default=["val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for seed, split in enumerate(args.split):
        render_dataset(args.src, args.out, split, args.limit, seed, args.workers)


if __name__ == "__main__":
    main()
