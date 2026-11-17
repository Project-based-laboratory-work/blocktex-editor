"""
数式を「文書の中に」埋め込んだTeXをPDFにし、PDFから数式の部分を切り抜いて読ませる評価。

実際の PDF → JSON の流れ（PDFのページから数式の範囲を200dpiで切り抜いて数式OCRに渡す）に近い条件で測る。
数式の範囲は、ブロック分類などの誤りを混ぜないよう、TeXに埋め込んだ目印（\\special の pdf:dest）の位置から決める。

  - 別行立て: 前の文の終わりの目印と、後ろの文の始まりの目印の間（その行の帯）を切り抜く
  - 文中:     数式の前後の目印の間を、その行の高さで切り抜く（行をまたいで折り返された数式は除く）

    python -m math_ocr.pdf_eval <im2latex_eval の scoped.tsv> <作業dir> <件数> <ckpt...>
"""

import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pymupdf
import torch
from PIL import Image

from math_ocr.im2latex_eval import canonical
from math_ocr.dataset import preprocess
from math_ocr.predict import load_model

DPI = 200  # block_classifier/ir_adapter.py の MATH_DPI と同じ
PER_DOC = 20
FILLER = [
    "この関係は次のように書ける。", "ここで、各変数は前節と同じ意味で用いる。", "以上の議論から、次の式が得られる。",
    "簡単な計算により確かめられる。", "条件を満たすとき、", "これを整理すると", "なお、境界条件は省略した。",
    "具体的な値を代入して確認する。", "この結果は直観とも合う。", "次に、もう一つの場合を考える。",
]  # fmt: skip
PREAMBLE = r"""\documentclass[uplatex]{jsarticle}
\usepackage{amsmath,amssymb}
\newcommand{\mk}[1]{\leavevmode\special{pdf:dest (#1) [@thispage /XYZ @xpos @ypos null]}}
\begin{document}
"""


def document(items: list[tuple[int, str, bool]], rng: random.Random) -> str:
    parts = [PREAMBLE]
    for k, latex, display in items:
        before, after = rng.choice(FILLER), rng.choice(FILLER)
        if display:
            parts.append(f"{before}\\mk{{a{k}}}\n\\[ {latex} \\]\n\\mk{{b{k}}}{after}\n\n")
        else:
            parts.append(f"{before}\\mk{{s{k}}}${latex}$\\mk{{e{k}}}{after}\n\n")
    parts.append("\\end{document}\n")
    return "".join(parts)


def compile_doc(tex: str, work: Path) -> Path | None:
    (work / "doc.tex").write_text(tex, encoding="utf-8")
    try:
        for cmd in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                    ["dvipdfmx", "-q", "-C", "0x0010", "doc.dvi"]):  # fmt: skip
            subprocess.run(cmd, cwd=work, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return work / "doc.pdf"


def crops(pdf: Path, items: list[tuple[int, str, bool]]) -> dict[int, Image.Image]:
    """目印の位置から数式の部分を切り抜く。座標は pt（上から）"""
    out = {}
    with pymupdf.open(pdf) as doc:
        h = doc[0].rect.height
        marks = {name: (d["page"], d["to"][0], h - d["to"][1]) for name, d in doc.resolve_names().items()}
        for k, _, display in items:
            if display:
                a, b = marks.get(f"a{k}"), marks.get(f"b{k}")
                if not a or not b or a[0] != b[0]:
                    continue
                page = doc[a[0]]
                # 前の行の下端〜次の行の上端。目印は行の基準線にあり、和文は基準線の約9pt上まであるので、下は11pt空ける
                rect = pymupdf.Rect(page.rect.x0, a[2] + 3, page.rect.x1, b[2] - 11)
            else:
                s, e = marks.get(f"s{k}"), marks.get(f"e{k}")
                if not s or not e or s[0] != e[0] or abs(s[2] - e[2]) > 1:  # 折り返された
                    continue
                page = doc[s[0]]
                rect = pymupdf.Rect(s[1] - 1, s[2] - 11, e[1] + 1, s[2] + 5)
            if rect.is_empty:
                continue
            pix = page.get_pixmap(clip=rect, dpi=DPI, colorspace=pymupdf.csGRAY)
            out[k] = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    return out


def main() -> None:
    scoped_tsv, work_dir, count, checkpoints = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    rng = random.Random(0)
    rows = [line.rstrip("\n").split("\t", 1)[1] for line in open(scoped_tsv, encoding="utf-8")]
    rows = rng.sample(rows, min(count, len(rows)))
    items = [(k, latex, k % 2 == 0) for k, latex in enumerate(rows)]  # 半分を別行立て、半分を文中に

    images: dict[int, Image.Image] = {}
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(items), PER_DOC):
            chunk = items[i : i + PER_DOC]
            pdf = compile_doc(document(chunk, rng), Path(tmp))
            if pdf is None:  # 1つでも組めない数式があると文書ごと失敗するので、1つずつ組み直す
                for item in chunk:
                    pdf = compile_doc(document([item], rng), Path(tmp))
                    if pdf is None:
                        failed += 1
                        continue
                    images.update(crops(pdf, [item]))
                continue
            images.update(crops(pdf, chunk))
    usable = [item for item in items if item[0] in images]
    print(f"{len(items)} 個 → 組めなかった {failed} 個、切り抜けた {len(usable)} 個"
          f"（別行立て {sum(d for _, _, d in usable)}・文中 {sum(not d for _, _, d in usable)}）")
    work_dir.mkdir(parents=True, exist_ok=True)
    for k, image in list(images.items())[:20]:
        image.save(work_dir / f"crop_{k:04d}.png")

    print("model\t全体\t別行立て\t文中\t10トークン以上")
    for checkpoint in checkpoints:
        model, vocab = load_model(Path(checkpoint))
        results = []
        with torch.no_grad():
            for i in range(0, len(usable), 64):
                batch = usable[i : i + 64]
                x = torch.stack([preprocess(images[k]) for k, _, _ in batch])
                for (k, latex, display), ids in zip(batch, model.generate(x)):
                    gt = canonical(latex)
                    results.append((display, len(gt), canonical(vocab.decode(ids.tolist()[1:])) == gt))
        groups = [results, [r for r in results if r[0]], [r for r in results if not r[0]], [r for r in results if r[1] >= 10]]
        cells = [f"{sum(r[2] for r in g) / max(len(g), 1):.1%} ({len(g)})" for g in groups]
        print(Path(checkpoint).parent.name + "\t" + "\t".join(cells), flush=True)


if __name__ == "__main__":
    main()
