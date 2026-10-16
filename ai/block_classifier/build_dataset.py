"""
ランダムな文書をPDFにし、PDFから取り出したテキストブロックに正解ラベルを自動で付ける。

    python -m block_classifier.build_dataset --out block_classifier/data/synth --docs 2000

流れ（1文書ごと）:
  1. doc_gen.random_document で、ブロックの始まり/終わりに目印を埋め込んだTeXを作る
  2. uplatex → dvipdfmx でPDFにする。dvipdfmx は目印の位置をPDFの座標で記録する
     （-C 0x0010: 使われていない目印も消さずに残す）
  3. rule_based.pdf_extract でテキストブロックを取り出す（実際の変換と同じ処理）
  4. 各テキストブロックが、どのブロックの目印の範囲に一番近いかでラベルを決める

出力（チームのラベル形式 rule_based/labels.py に合わせる。source は "auto"）:
    <out>/pdfs/doc_000000.pdf
    <out>/labels/doc_000000.json
"""

import argparse
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import pymupdf

from block_classifier.doc_gen import FONT_MAPS, SourceBlock, random_document
from rule_based.labels import LabeledBlock, save_labels
from rule_based.pdf_extract import TextBlock, extract

ASCENT = 12.0  # 目印（ベースライン）から文字の上端までの余裕（pt）
DESCENT = 5.0  # 目印から文字の下端までの余裕（pt）
MAX_DISTANCE = 10.0  # どのブロックの範囲からもこれ以上離れていたら other（ページ番号など）


@dataclass
class Segment:
    """ブロックが占める、1つの段（ページ・段組の列）の中の縦の範囲（上からの座標）"""

    label: str
    page: int
    column: int  # -1 は段をまたいで横に広がるもの（2段組のタイトルなど）
    top: float
    bottom: float


def _column(x: float, page_width: float, two_column: bool) -> int:
    return int(two_column and x >= page_width / 2)


def block_segments(
    blocks: list[SourceBlock], marks: dict[str, tuple[int, float, float]], page_size: tuple[float, float], two_column: bool
) -> list[Segment]:
    """目印の位置から、各ブロックが占める範囲を作る。

    ページや段をまたぐブロック（段落の途中で改ページなど）は、始まりの段は下端まで、
    終わりの段は上端から、という範囲に分ける。
    """
    width, height = page_size
    segments = []
    for block in blocks:
        start, end = marks.get(f"b{block.id}s"), marks.get(f"b{block.id}e")
        if start is None or end is None:  # 目印が出力されなかった（ページ外に流れたなど）
            continue
        (p0, x0, y0), (p1, x1, y1) = start, end
        c0, c1 = _column(x0, width, two_column), _column(x1, width, two_column)
        if (p0, c0) == (p1, c1):
            segments.append(Segment(block.label, p0, c0, min(y0, y1), max(y0, y1)))
            continue
        if p0 == p1 and abs(y0 - y1) < 3.0:  # 左の段で始まり、同じ高さの右の段で終わる = 段をまたぐ1行
            segments.append(Segment(block.label, p0, -1, y0, y1))
            continue
        segments.append(Segment(block.label, p0, c0, y0, height))
        # 間に挟まった段は丸ごと（2段組で1ページ以上続くことはほぼ無いので、同じページの右段だけ見る）
        if two_column and p0 == p1 - 1 and c0 == 0:
            segments.append(Segment(block.label, p0, 1, 0, height))
        segments.append(Segment(block.label, p1, c1, 0, y1))
    return segments


def label_of(text_block: TextBlock, segments: list[Segment], page_width: float, two_column: bool) -> str:
    """テキストブロックの縦の中心が、どのブロックの範囲に一番近いかでラベルを決める"""
    x0, y0, x1, y1 = text_block.bbox
    center_y = (y0 + y1) / 2
    column = _column((x0 + x1) / 2, page_width, two_column)

    best_label, best_key = "other", (MAX_DISTANCE, 0.0)
    for seg in segments:
        if seg.page != text_block.page or seg.column not in (column, -1):
            continue
        top, bottom = seg.top - ASCENT, seg.bottom + DESCENT
        distance = 0.0 if top <= center_y <= bottom else min(abs(center_y - top), abs(center_y - bottom))
        # 同じ距離なら、範囲の狭いブロックを優先する（広いブロックの中に別のブロックが入っている場合）
        key = (distance, bottom - top)
        if key < best_key:
            best_label, best_key = seg.label, key
    return best_label


def read_marks(pdf_path: Path) -> tuple[dict[str, tuple[int, float, float]], tuple[float, float]]:
    """PDFの名前付きリンク先（目印）を読む。座標はPDFの下からなので、上からに直す"""
    with pymupdf.open(pdf_path) as doc:
        height = doc[0].rect.height
        marks = {name: (d["page"], d["to"][0], height - d["to"][1]) for name, d in doc.resolve_names().items()}
        return marks, (doc[0].rect.width, height)


def compile_tex(tex: str, work_dir: Path, font_map: str | None = None) -> Path:
    (work_dir / "doc.tex").write_text(tex, encoding="utf-8")
    dvipdfmx = ["dvipdfmx", "-q", "-C", "0x0010", *(["-f", font_map] if font_map else []), "doc.dvi"]
    for command in (["uplatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], dvipdfmx):
        subprocess.run(command, cwd=work_dir, check=True, capture_output=True, timeout=120)
    return work_dir / "doc.pdf"


def build_one(args: tuple[int, int, Path]) -> dict[str, int] | None:
    index, seed, out = args
    rng = random.Random(seed * 1_000_003 + index)
    tex, blocks = random_document(rng)
    two_column = "twocolumn" in tex.split("\n", 1)[0]
    font_map = rng.choice(FONT_MAPS)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            pdf = compile_tex(tex, Path(tmp), font_map)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        marks, page_size = read_marks(pdf)
        segments = block_segments(blocks, marks, page_size, two_column)

        labeled = []
        for page in extract(str(pdf)):
            for text_block in page.text_blocks:
                label = label_of(text_block, segments, page.width, two_column)
                labeled.append(LabeledBlock(text_block.page, text_block.bbox, label, text_block.text, "auto"))

        name = f"doc_{index:06d}"
        shutil.copy(pdf, out / "pdfs" / f"{name}.pdf")
        (out / "tex").mkdir(exist_ok=True)
        (out / "tex" / f"{name}.tex").write_text(tex, encoding="utf-8")  # ラベルの確認・デバッグ用
        save_labels(f"{name}.pdf", labeled, str(out / "labels" / f"{name}.json"))

    counts: dict[str, int] = {}
    for block in labeled:
        counts[block.label] = counts.get(block.label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("block_classifier/data/synth"))
    parser.add_argument("--docs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    for sub in ["pdfs", "labels"]:
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        results = pool.map(build_one, [(i, args.seed, args.out) for i in range(args.docs)])

    ok = [r for r in results if r is not None]
    total: dict[str, int] = {}
    for counts in ok:
        for label, n in counts.items():
            total[label] = total.get(label, 0) + n
    print(f"{len(ok)}/{args.docs} 文書を生成（コンパイル失敗 {args.docs - len(ok)}）")
    print("ラベルごとのブロック数:", dict(sorted(total.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
