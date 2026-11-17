"""
自作の数式OCRと、既存の数式OCR pix2tex（LaTeX-OCR）を、同じ画像・同じ物差しで比べる（正確さ・大きさ・CPUでの速さ）。

    # 1. 試験データを作る（画像と正解）。対応範囲は自作モデルの語彙で決める
    python -m math_ocr.compare_pix2tex prepare --out math_ocr/data/pix2tex_cmp \\
        --checkpoint math_ocr/checkpoints/v4_im2latex/best.pt --im2latex math_ocr/data/im2latex_parquet/test.parquet --tex <レポートのTeX...>
    # 2. pix2tex で読む（別の環境。math_ocr/run_pix2tex.py を参照）
    ~/.venvs/pix2tex/bin/python math_ocr/run_pix2tex.py math_ocr/data/pix2tex_cmp/real/images math_ocr/data/pix2tex_cmp/real/pix2tex.tsv
    # 3. 自作モデルでも読み、採点する
    python -m math_ocr.compare_pix2tex score --out math_ocr/data/pix2tex_cmp --checkpoint math_ocr/checkpoints/v4_im2latex/best.pt

試験データ（どれも、自作モデルの語彙で書けて50トークン以下の数式＝対応範囲だけ）:
  - real:     自分のレポートのTeXの数式（real_eval と同じ取り出し方・組み方）
  - pdf:      im2latex の数式を日本語の文書に埋め込んでPDFにし、切り抜いたもの（pdf_eval と同じ 600個）
  - im2latex: IM2LATEX-100K の test に付いている画像。pix2tex は im2latex を含むデータで学習しているので、pix2tex に有利

採点:
  - 文字列の一致: 書き方をそろえて（im2latex_eval.canonical）から完全に同じか
  - 見た目の一致: 答えと正解を、同じように書き方をそろえてから同じ方法（pdflatex、150dpi）で画像にして、ほぼ同じ画像になるか。
    \\mathrm{d} と d のように、書き方は違っても見た目が同じ答えを正解にできる（Deng et al. の image match と同じ考え方）
"""

import argparse
import io
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from math_ocr import real_eval as R
from math_ocr.dataset import preprocess
from math_ocr.im2latex_eval import canonical
from math_ocr.pdf_eval import PER_DOC, compile_doc, crops, document
from math_ocr.predict import load_model
from math_ocr.render_latex import _compile_pages
from math_ocr.tokenizer import detokenize

SETS = ["real", "pdf", "im2latex"]
PDF_COUNT = 600  # pdf_eval と同じ数・同じ選び方
RENDER_DPI = 150
SAME_SHAPE = 2  # 画像の縦横の差がこの画素数以内なら同じ大きさとみなす
SAME_INK = 0.02  # 違う画素が、文字の画素のこの割合以下なら同じ見た目とみなす
SPEED_IMAGES = 100  # 速さは、この枚数を1枚ずつ読んだ時間で測る


def save_set(out: Path, name: str, items: list[tuple[Image.Image, str, bool]]) -> None:
    """items: (画像, 正解のLaTeX, 別行立てか)"""
    (out / name / "images").mkdir(parents=True, exist_ok=True)
    with open(out / name / "gold.tsv", "w", encoding="utf-8") as f:
        for k, (image, latex, display) in enumerate(items):
            image.convert("L").save(out / name / "images" / f"{k:05d}.png")
            f.write(f"{k:05d}.png\t{int(display)}\t{latex}\n")
    print(f"{name}: {len(items)} 個")


def prepare(args: argparse.Namespace) -> None:
    vocab_tokens = set(load_model(args.checkpoint)[1].itos)

    # real: 自分のレポートの数式。同じ数式（同じ形）は1回だけ
    items, seen = [], set()
    with tempfile.TemporaryDirectory() as tmp:
        for path in args.tex:
            for formula in R.extract_formulas(path):
                gt = canonical(formula.latex)
                key = (formula.display, " ".join(gt))
                if key in seen or not R.in_scope(gt, vocab_tokens):
                    continue
                seen.add(key)
                image = R.render(formula, Path(tmp))
                if image is not None:
                    items.append((image, detokenize(gt), formula.display))
    save_set(args.out, "real", items)

    # im2latex: test に付いている画像（対応範囲だけ）
    rows = pq.read_table(args.im2latex, columns=["formula", "image"]).to_pylist()
    scoped = []
    for row in rows:
        gt = canonical(row["formula"])
        if R.in_scope(gt, vocab_tokens):
            scoped.append((detokenize(gt), row["image"]["bytes"]))
    save_set(args.out, "im2latex", [(Image.open(io.BytesIO(data)), gt, True) for gt, data in scoped])

    # pdf: pdf_eval と同じ選び方で、文書に埋め込んで切り抜く（半分が別行立て、半分が文中）
    rng = random.Random(0)
    chosen = rng.sample([gt for gt, _ in scoped], min(PDF_COUNT, len(scoped)))
    pdf_items = [(k, latex, k % 2 == 0) for k, latex in enumerate(chosen)]
    images = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(pdf_items), PER_DOC):
            chunk = pdf_items[i : i + PER_DOC]
            pdf = compile_doc(document(chunk, rng), Path(tmp))
            if pdf is None:  # 組めない数式があると文書ごと失敗するので、1つずつ組み直す
                for item in chunk:
                    pdf = compile_doc(document([item], rng), Path(tmp))
                    if pdf is not None:
                        images.update(crops(pdf, [item]))
                continue
            images.update(crops(pdf, chunk))
    save_set(args.out, "pdf", [(images[k], latex, display) for k, latex, display in pdf_items if k in images])


def render_all(latexes: list[str], displays: list[bool]) -> list[np.ndarray | None]:
    """LaTeX の並び → 「文字=True」の画像の並び（組めなければ None）"""
    pages = [rf"\[ {s} \]" if d else rf"\noindent ${s}$" for s, d in zip(latexes, displays)]
    out = []
    for i in range(0, len(pages), 200):
        for image in _compile_pages(pages[i : i + 200], "", RENDER_DPI):
            out.append(None if image is None else np.asarray(image) < 128)
    return out


def same_look(a: np.ndarray | None, b: np.ndarray | None) -> bool:
    if a is None or b is None or abs(a.shape[0] - b.shape[0]) > SAME_SHAPE or abs(a.shape[1] - b.shape[1]) > SAME_SHAPE:
        return False
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa, pb = np.zeros((h, w), bool), np.zeros((h, w), bool)
    pa[: a.shape[0], : a.shape[1]], pb[: b.shape[0], : b.shape[1]] = a, b
    return (pa ^ pb).sum() <= SAME_INK * max(pb.sum(), 1)


@torch.no_grad()
def ours(checkpoint: Path, image_dir: Path, names: list[str]) -> tuple[list[str], float]:
    """自作モデルの答えと、1枚あたりの秒数（SPEED_IMAGES 枚を1枚ずつ読んだ平均）"""
    model, vocab = load_model(checkpoint)
    images = [preprocess(Image.open(image_dir / n)) for n in names]
    preds = []
    for i in range(0, len(images), 64):
        preds += [vocab.decode(ids.tolist()[1:]) for ids in model.generate(torch.stack(images[i : i + 64]))]
    started = time.perf_counter()
    for image in images[:SPEED_IMAGES]:
        model.generate(image.unsqueeze(0))
    return preds, (time.perf_counter() - started) / max(1, min(SPEED_IMAGES, len(images)))


def score(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    model, _ = load_model(args.checkpoint)
    our_params = sum(p.numel() for p in model.parameters())
    params_file = next(args.out.glob("*/pix2tex.params"), None)
    pix_params = sum(int(kv.split("=")[1]) for kv in params_file.read_text().split()) if params_file else None
    print(f"パラメータ数: 自作 {our_params:,} / pix2tex {pix_params:,}" if pix_params else f"パラメータ数: 自作 {our_params:,}")
    print(f"\n{'':<10}{'':<10}{'文字列の一致':>10}{'見た目の一致':>10}{'s/枚(CPU)':>11}{'件数':>7}")
    for name in args.sets:
        gold_path = args.out / name / "gold.tsv"
        if not gold_path.exists():
            continue
        rows = [line.split("\t") for line in gold_path.read_text(encoding="utf-8").splitlines()]
        names, displays, golds = [r[0] for r in rows], [r[1] == "1" for r in rows], [r[2] for r in rows]
        answers = {}
        answers["自作"], our_seconds = ours(args.checkpoint, args.out / name / "images", names)
        seconds = {"自作": our_seconds}
        pix_path = args.out / name / "pix2tex.tsv"
        if pix_path.exists():
            pix = {}
            for line in pix_path.read_text(encoding="utf-8").splitlines():
                n, latex, sec = (line.split("\t") + ["", "0"])[:3]
                pix[n] = (latex, float(sec))
            if all(n in pix for n in names):
                answers["pix2tex"] = [pix[n][0] for n in names]
                seconds["pix2tex"] = sum(pix[n][1] for n in names[:SPEED_IMAGES]) / max(1, min(SPEED_IMAGES, len(names)))
        gold_images = render_all(golds, displays)
        usable = [k for k, g in enumerate(gold_images) if g is not None]  # 正解が組めない数式は数えない
        for who, preds in answers.items():
            # 正解と同じく、書き方をそろえて（空白の指定などを除いて）から画像にする
            pred_images = render_all([detokenize(canonical(p)) or "\\relax" for p in preds], displays)
            exact = sum(canonical(preds[k]) == canonical(golds[k]) for k in usable)
            look = sum(same_look(pred_images[k], gold_images[k]) for k in usable)
            print(f"{name:<10}{who:<10}{exact / len(usable):>12.1%}{look / len(usable):>12.1%}{seconds[who]:>11.3f}{len(usable):>7}",
                  flush=True)  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="自作の数式OCRと pix2tex を比べる")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--im2latex", type=Path, required=True, help="IM2LATEX-100K の test の parquet")
    p.add_argument("--tex", type=Path, nargs="+", required=True, help="自分のレポートのTeX")
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--threads", type=int, default=4, help="CPUのスレッド数（pix2tex と同じにする）")
    s.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else score(args)


if __name__ == "__main__":
    main()
