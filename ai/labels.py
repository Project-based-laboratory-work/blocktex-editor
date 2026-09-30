"""
4-1-3: 学習データのラベル定義と保存形式。

ラベルは細かく付け（FINE_LABELS）、学習・評価時に粗くまとめる（to_coarse）。
ラベルはbboxと一緒に保存し、抽出方法（PyMuPDF / pdf.js）が変わっても座標で対応付けられるようにする。

保存形式: PDF1つにつきJSON1つ
    {"pdf": "xxx.pdf", "blocks": [{"page": 0, "bbox": [...], "label": "heading", "text": "...", "source": "auto"}, ...]}
"""
import json
from dataclasses import asdict, dataclass
from typing import Literal, get_args

Bbox = tuple[float, float, float, float]

Label = Literal["heading", "paragraph", "list", "math", "caption", "table", "figure", "other"]
Source = Literal["auto", "manual", "prelabel"]

FINE_LABELS = list(get_args(Label))
COARSE_LABELS = ["heading", "paragraph", "caption", "table", "figure", "other"]
VALID_SOURCES = list(get_args(Source))

# 細かいラベル → 学習・評価用の粗いラベル
COARSE_MAP = {
    "heading": "heading",
    "paragraph": "paragraph",
    "list": "paragraph",
    "math": "paragraph",
    "caption": "caption",
    "table": "table",
    "figure": "figure",
    "other": "other",
}


@dataclass
class LabeledBlock:
    page: int
    bbox: Bbox
    label: Label  # FINE_LABELS のどれか
    text: str | None
    source: Source  # "auto"（TeXの色付けコンパイルから自動取得）| "manual"（人手）| "prelabel"（ルールベースの下書き、未確認）


def to_coarse(label: str) -> str:
    if label not in COARSE_MAP:
        raise ValueError(f"不明なラベルです: {label!r}（使えるのは {FINE_LABELS}）")
    return COARSE_MAP[label]


def validate(blocks: list[LabeledBlock]) -> list[str]:
    problems = []
    for i, block in enumerate(blocks):
        if block.label not in FINE_LABELS:
            problems.append(f"ブロック{i}: 不明なラベル {block.label!r}")

        if block.source not in VALID_SOURCES:
            problems.append(f"ブロック{i}: 不明なsource {block.source!r}")

        if len(block.bbox) != 4:
            problems.append(f"ブロック{i}: bboxの要素数が{len(block.bbox)}個（4個必要）")
        else:
            x0, y0, x1, y1 = block.bbox
            if x0 >= x1:
                problems.append(f"ブロック{i}: bboxの左右が逆 (x0={x0}, x1={x1})")
            if y0 >= y1:
                problems.append(f"ブロック{i}: bboxの上下が逆 (y0={y0}, y1={y1})")

    return problems


def save_labels(pdf_name: str, blocks: list[LabeledBlock], out_path: str) -> None:
    data = {
        "pdf": pdf_name,
        "blocks": [asdict(block) for block in blocks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_labels(json_path: str, check: bool = False) -> tuple[str, list[LabeledBlock]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    blocks = []
    for d in data["blocks"]:
        d["bbox"] = tuple(d["bbox"])
        blocks.append(LabeledBlock(**d))

    if check:
        problems = validate(blocks)
        if problems:
            raise ValueError(f"{json_path} に問題があります:\n" + "\n".join(problems))

    return data["pdf"], blocks
