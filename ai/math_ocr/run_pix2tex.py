"""
既存の数式OCR pix2tex（LaTeX-OCR, https://github.com/lukas-blecher/LaTeX-OCR）で、画像のフォルダを読む。

pix2tex はチームの環境とは別の環境に入れて動かす（依存するライブラリが多いため）。このファイルは math_ocr の他のファイルを
読み込まないので、その環境からそのまま実行できる:

    python3 -m venv ~/.venvs/pix2tex && ~/.venvs/pix2tex/bin/pip install pix2tex
    ~/.venvs/pix2tex/bin/python math_ocr/run_pix2tex.py <images のフォルダ> <出力の tsv>

出力: 1行に「画像のファイル名 \\t 読んだLaTeX \\t 秒数」。秒数は画像1枚ずつ読んだときの時間（CPU）
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from pix2tex.cli import LatexOCR


def main() -> None:
    image_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    model = LatexOCR()
    # モデルの大きさ（パラメータ数）。数式を読む本体と、画像の大きさを決める前段の2つに分かれている
    parts = {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("model", "image_resizer")
             if getattr(model, name, None) is not None}  # fmt: skip
    out.with_suffix(".params").write_text(" ".join(f"{k}={v}" for k, v in parts.items()) + "\n", encoding="utf-8")
    paths = sorted(image_dir.glob("*.png"))
    done = set()
    if out.exists():  # 途中で止まっても続きから
        done = {line.split("\t", 1)[0] for line in out.read_text(encoding="utf-8").splitlines()}
    with open(out, "a", encoding="utf-8") as f:
        for k, path in enumerate(paths, 1):
            if path.name in done:
                continue
            image = Image.open(path).convert("RGB")
            started = time.perf_counter()
            try:
                latex = model(image)
            except Exception as e:  # 読めなかった画像は空の答えにする
                latex = ""
                print(f"{path.name}: {e}", file=sys.stderr)
            seconds = time.perf_counter() - started
            f.write(f"{path.name}\t{' '.join(latex.split())}\t{seconds:.4f}\n")
            f.flush()
            if k % 100 == 0:
                print(f"{k}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
