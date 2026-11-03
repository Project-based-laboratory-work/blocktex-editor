"""
ランダムな文書をPDFにし、ページごとの「正解ブロック」（段落単位のbbox・ラベル・文字列）と画像を作る。

文書の生成とPDFへの変換は block_classifier（doc_gen.py / build_dataset.py）をそのまま使う。
block_classifier との違いは、正解の単位:
  - block_classifier: pdf_extract が取り出したテキストブロック1つずつにラベルを付ける（PDFの抽出結果が基準）
  - ここ: TeXで置いたブロック1つ = 正解1つ。OCRの結果に依存しない、画像の上での「あるべき段落」
画像から読む抽出器（EasyOCR など）は、どんな塊で文字を返すか分からないので、抽出器に依存しない正解にしておく。

正解ブロックの作り方:
  1. 各ブロックの目印（始まり/終わり）から、ブロックが占める縦の範囲を作る（build_dataset.block_segments）
  2. PDFの文字の行（PyMuPDF の line）を、一番近い範囲のブロックに割り当てる（build_dataset.label_of と同じ規則）
  3. 同じブロック・同じページ・同じ段の行をまとめて、bbox を合わせ、文字列をつなげる
  4. 図（\\fbox）・表の罫線・コードの枠は文字が無いので、線や四角（get_drawings）も同じ規則で割り当てる
ページ番号と、どのブロックの範囲にも近くない行は、1行ずつ other にする。
"""

import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf

from block_classifier.build_dataset import ASCENT, DESCENT, MAX_DISTANCE, Segment, block_segments, compile_tex, read_marks
from block_classifier.doc_gen import FONT_MAPS, SourceBlock, random_document

Bbox = tuple[float, float, float, float]

# 線や四角も範囲に含めるブロック（それ以外のブロックでは、下線などの線は文字の bbox に入っているので見ない）
DRAWING_LABELS = {"figure", "table", "code"}
# ページの大部分を覆う線や四角（ページ枠など）は無視する
MAX_DRAWING_SHARE = 0.9
HEADING_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
# ページ番号とみなす行：数字だけで、ページの下からこの割合の範囲にある
PAGE_NUMBER_AREA = 0.1


@dataclass
class TruthBlock:
    """正解ブロック1つ。座標はPDFの pt（左上原点）"""

    label: str  # rule_based/labels.py の FINE_LABELS のどれか
    bbox: Bbox
    text: str | None  # 図は None
    kind: str  # common/ir.py の Kind（"text" / "table" / "figure"）
    level: int | None = None  # 見出しのレベル（section=1, subsection=2, subsubsection=3）
    tex: str | None = None  # 数式のLaTeX（目印を除いたもの）


@dataclass
class TruthPage:
    page: int  # 0始まり
    width: float  # pt
    height: float
    blocks: list[TruthBlock]  # 文書の中の順（＝読む順）


def _column(x: float, page_width: float, two_column: bool) -> int:
    return int(two_column and x >= page_width / 2)


def _nearest(bbox: Bbox, page: int, segments: list[Segment], page_width: float, two_column: bool) -> int | None:
    """bbox の縦の中心が一番近い範囲の番号。どれからも MAX_DISTANCE 以上離れていれば None（build_dataset.label_of と同じ規則）"""
    x0, y0, x1, y1 = bbox
    center_y = (y0 + y1) / 2
    column = _column((x0 + x1) / 2, page_width, two_column)
    best, best_key = None, (MAX_DISTANCE, 0.0)
    for i, seg in enumerate(segments):
        if seg.page != page or seg.column not in (column, -1):
            continue
        top, bottom = seg.top - ASCENT, seg.bottom + DESCENT
        distance = 0.0 if top <= center_y <= bottom else min(abs(center_y - top), abs(center_y - bottom))
        key = (distance, bottom - top)
        if key < best_key:
            best, best_key = i, key
    return best


def _union(boxes: list[Bbox]) -> Bbox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _join_lines(lines: list[tuple[Bbox, str]]) -> str:
    """行を上から順に並べてつなげる。同じ高さに並んだ行（箇条書きの記号と本文など）は空白でつなぐ（pdf_extract と同じ）"""
    rows: list[list[tuple[Bbox, str]]] = []
    for bbox, text in sorted(lines, key=lambda line: (line[0][1], line[0][0])):
        if rows and abs(bbox[1] - rows[-1][0][0][1]) <= (bbox[3] - bbox[1]) / 2:
            rows[-1].append((bbox, text))
        else:
            rows.append([(bbox, text)])
    return "\n".join(" ".join(text for _, text in sorted(row, key=lambda line: line[0][0])) for row in rows)


def _is_page_number(bbox: Bbox, text: str, page_height: float) -> bool:
    """ページ番号の行か。次のページに続くブロックの範囲はページの下端まで伸びるので、
    範囲で割り当てるとページ番号がそのブロックに入ってしまう。先に取り除いて other にする"""
    return text.isdigit() and bbox[1] > page_height * (1 - PAGE_NUMBER_AREA)


def _heading_level(block: SourceBlock) -> int | None:
    match = re.match(r"\\(\w+)\{", block.tex)
    return HEADING_LEVELS.get(match.group(1)) if match else None


def _math_tex(block: SourceBlock) -> str:
    tex = re.sub(r"\\mk\{[^}]*\}", "", block.tex)
    return tex.replace("\\begin{align*}\n", "").replace("\n\\end{align*}", "").strip()


def _page_lines(page: pymupdf.Page) -> list[tuple[Bbox, str]]:
    lines = []
    for raw_block in page.get_text("dict")["blocks"]:
        for line in raw_block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((tuple(line["bbox"]), text))
    return lines


def _page_drawings(page: pymupdf.Page) -> list[Bbox]:
    width, height = page.rect.width, page.rect.height
    boxes = []
    for drawing in page.get_drawings():
        x0, y0, x1, y1 = drawing["rect"]
        if x1 - x0 > width * MAX_DRAWING_SHARE and y1 - y0 > height * MAX_DRAWING_SHARE:
            continue
        boxes.append((x0, y0, x1, y1))
    return boxes


def truth_pages(pdf_path: Path, blocks: list[SourceBlock], two_column: bool) -> list[TruthPage]:
    """PDFと、置いたブロックの一覧から、ページごとの正解ブロックを作る"""
    marks, (width, height) = read_marks(pdf_path)
    # 範囲ごとに、どのブロックのものかを覚えておく（ページや段をまたぐブロックは範囲が複数になる）
    segments: list[Segment] = []
    owners: list[int] = []
    for index, block in enumerate(blocks):
        for seg in block_segments([block], marks, (width, height), two_column):
            segments.append(seg)
            owners.append(index)

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc):
            # 範囲の番号 → その範囲に割り当てた行・線
            lines: dict[int, list[tuple[Bbox, str]]] = {}
            drawings: dict[int, list[Bbox]] = {}
            strays: list[tuple[Bbox, str]] = []  # どのブロックにも近くない行
            for bbox, text in _page_lines(page):
                seg = None if _is_page_number(bbox, text, height) else _nearest(bbox, page_no, segments, width, two_column)
                if seg is None:
                    strays.append((bbox, text))
                else:
                    lines.setdefault(seg, []).append((bbox, text))
            for bbox in _page_drawings(page):
                seg = _nearest(bbox, page_no, segments, width, two_column)
                if seg is not None and segments[seg].label in DRAWING_LABELS:
                    drawings.setdefault(seg, []).append(bbox)

            truth = []
            for seg in sorted(set(lines) | set(drawings), key=lambda s: (owners[s], s)):  # 文書の中の順
                source = blocks[owners[seg]]
                seg_lines = lines.get(seg, [])
                boxes = [bbox for bbox, _ in seg_lines] + drawings.get(seg, [])
                label = source.label
                kind = label if label in ("table", "figure") else "text"
                text = _join_lines(seg_lines) if seg_lines else None
                if kind != "figure" and text is None:  # 文字の無い表・段落（罫線だけが割り当てられた）は正解にしない
                    continue
                truth.append(
                    TruthBlock(
                        label=label,
                        bbox=_union(boxes),
                        text=text if kind != "figure" else None,
                        kind=kind,
                        level=_heading_level(source) if label == "heading" else None,
                        tex=_math_tex(source) if label == "math" else None,
                    )
                )
            truth += [TruthBlock("other", bbox, text, "text") for bbox, text in strays]
            pages.append(TruthPage(page_no, page.rect.width, page.rect.height, truth))
    return pages


def render_page(pdf: bytes, page_no: int, dpi: float) -> np.ndarray:
    """PDFの1ページを画像（高さ × 幅 × RGB の uint8）にする"""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        scale = dpi / 72
        pixmap = doc[page_no].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3).copy()


@dataclass
class RenderedDocument:
    tex: str
    pdf: bytes
    pages: list[TruthPage]


def render_document(rng: random.Random) -> RenderedDocument | None:
    """ランダムな文書を1つ作り、PDFと正解ブロックを返す。コンパイルに失敗したら None"""
    tex, blocks = random_document(rng)
    two_column = "twocolumn" in tex.split("\n", 1)[0]
    font_map = rng.choice(FONT_MAPS)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pdf = compile_tex(tex, Path(tmp), font_map)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return RenderedDocument(tex, pdf.read_bytes(), truth_pages(pdf, blocks, two_column))
