"""
共通中間表現（IR）JSONの型定義・保存・読み込み・形式チェック。

全ての抽出器（PDF-A, IMG-A, ...）はこの形式で出力し、構造化器はこの形式を入力にする。
構造化器（分類器）の出力も同じ形式で、各要素に label を付けたもの（ラベル付きIR）とする。
学習・評価用の正解ラベルもラベル付きIRで保存する。

別途JSON Schemaは用意せず、このファイルの dataclass をスキーマの正とする（web/src/model.ts と同じ方針）。
仕様の説明は common/README.md を参照。

使い方（形式チェックだけ行う）:
    python -m common.ir out/sample.ir.json
"""

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

# IRの形を変えたらインクリメントする
IR_VERSION = 1

# (x0, y0, x1, y1)。左上原点で、ページの幅・高さで割った0〜1の値
Bbox = tuple[float, float, float, float]

# 抽出器が見つけた領域の種類。意味（見出し・段落など）は含めない
Kind = Literal["text", "table", "figure"]
# 構造化器が付ける意味ラベル。rule_based/labels.py の FINE_LABELS と同じ
Label = Literal["heading", "paragraph", "list", "math", "caption", "table", "figure", "other", "code"]
# 正解ラベルの出どころ。rule_based/labels.py の Source と同じ
LabelSource = Literal["auto", "manual", "prelabel"]
# ページの幅・高さ・文字サイズの単位。PDFはpt、画像はpx
Unit = Literal["pt", "px"]

KINDS = list(get_args(Kind))
LABELS = list(get_args(Label))
LABEL_SOURCES = list(get_args(LabelSource))
UNITS = list(get_args(Unit))


@dataclass
class Element:
    kind: Kind
    bbox: Bbox
    text: str | None = None  # kind="text" では必須。行の区切りは "\n"
    font_size: float | None = None  # ページの unit での文字サイズ。取れない抽出器は bbox の高さから推定する
    bold: bool | None = None  # 判定できない抽出器は None
    confidence: float | None = None  # 参考値。モデルごとに意味が違うので比較には使わない
    cells: list[list[str]] | None = None  # 表の中身（行 × 列）。取れる抽出器のみ
    # ↓ ここから下はラベル付きIRでだけ使う
    label: Label | None = None
    level: int | None = None  # label="heading" の見出しレベル（1〜3）。分からなければ None
    tex: str | None = None  # label="math" のLaTeX（数式OCRなどで得られた場合）
    label_source: LabelSource | None = None  # 正解ラベルのときだけ付ける


@dataclass
class Page:
    page: int  # 1始まり
    width: float  # 元の単位（unit）での幅
    height: float
    unit: Unit
    elements: list[Element] = field(default_factory=list)


@dataclass
class IRDocument:
    source: str  # 入力ファイル名
    extractor: str  # 抽出器の名前（例: "rule_based", "tesseract"）
    pages: list[Page]
    params: dict[str, Any] = field(default_factory=dict)  # 前処理などのパラメータ（比較時の区別に使う）
    structurer: str | None = None  # ラベルを付けた構造化器の名前。抽出しただけのIRでは None
    ir_version: int = IR_VERSION


def normalize_bbox(bbox: tuple[float, float, float, float], width: float, height: float) -> Bbox:
    """元の単位のbboxを0〜1に正規化する。ページから少しはみ出した分は0〜1に収める。"""
    def clamp(v: float) -> float:
        return round(min(1.0, max(0.0, v)), 5)

    x0, y0, x1, y1 = bbox
    return (clamp(x0 / width), clamp(y0 / height), clamp(x1 / width), clamp(y1 / height))


def denormalize_bbox(bbox: Bbox, width: float, height: float) -> tuple[float, float, float, float]:
    """正規化したbboxを元の単位に戻す（画像の切り出しなどに使う）。"""
    x0, y0, x1, y1 = bbox
    return (x0 * width, y0 * height, x1 * width, y1 * height)


def to_dict(doc: IRDocument) -> dict[str, Any]:
    """JSONにする形の dict。None の項目はキーごと省く（取れなかった値と区別がつくように）。"""
    data = asdict(doc)
    for page in data["pages"]:
        page["elements"] = [{k: v for k, v in el.items() if v is not None} for el in page["elements"]]
    return {k: v for k, v in data.items() if v is not None}


def from_dict(data: dict[str, Any]) -> IRDocument:
    pages = []
    for p in data["pages"]:
        elements = [Element(**{**el, "bbox": tuple(el["bbox"])}) for el in p["elements"]]
        pages.append(Page(page=p["page"], width=p["width"], height=p["height"], unit=p["unit"], elements=elements))
    return IRDocument(
        source=data["source"],
        extractor=data["extractor"],
        pages=pages,
        params=data.get("params", {}),
        structurer=data.get("structurer"),
        ir_version=data["ir_version"],
    )


def save_ir(doc: IRDocument, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_dict(doc), f, ensure_ascii=False, indent=2)


def load_ir(json_path: str, check: bool = False) -> IRDocument:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = from_dict(json.load(f))

    if check:
        problems = validate(doc)
        if problems:
            raise ValueError(f"{json_path} に問題があります:\n" + "\n".join(problems))
    return doc


def _validate_element(where: str, el: Element) -> list[str]:
    problems = []
    if el.kind not in KINDS:
        problems.append(f"{where}: 不明なkind {el.kind!r}（使えるのは {KINDS}）")

    if len(el.bbox) != 4:
        problems.append(f"{where}: bboxの要素数が{len(el.bbox)}個（4個必要）")
    else:
        x0, y0, x1, y1 = el.bbox
        if not all(0.0 <= v <= 1.0 for v in el.bbox):
            problems.append(f"{where}: bboxが0〜1の範囲外 {el.bbox}（正規化を忘れていないか）")
        if x0 >= x1:
            problems.append(f"{where}: bboxの左右が逆 (x0={x0}, x1={x1})")
        if y0 >= y1:
            problems.append(f"{where}: bboxの上下が逆 (y0={y0}, y1={y1})")

    if el.kind == "text" and el.text is None:
        problems.append(f"{where}: kind='text' なのに text がない")
    if el.cells is not None and not all(isinstance(row, list) and all(isinstance(c, str) for c in row) for row in el.cells):
        problems.append(f"{where}: cells は文字列の二次元配列にする")

    if el.label is not None and el.label not in LABELS:
        problems.append(f"{where}: 不明なlabel {el.label!r}（使えるのは {LABELS}）")
    if el.level is not None:
        if el.label != "heading":
            problems.append(f"{where}: level は label='heading' のときだけ付ける")
        elif el.level not in (1, 2, 3):
            problems.append(f"{where}: level は1〜3（{el.level}）")
    if el.label_source is not None and el.label_source not in LABEL_SOURCES:
        problems.append(f"{where}: 不明なlabel_source {el.label_source!r}（使えるのは {LABEL_SOURCES}）")
    return problems


def validate(doc: IRDocument) -> list[str]:
    """形式の問題を日本語のメッセージの一覧で返す。問題がなければ空のリスト。"""
    problems = []
    if doc.ir_version != IR_VERSION:
        problems.append(f"ir_version が {doc.ir_version}（現行は {IR_VERSION}）")

    for i, page in enumerate(doc.pages):
        if page.page != i + 1:
            problems.append(f"{i + 1}番目のページの page が {page.page}（1始まりの連番にする）")
        if page.width <= 0 or page.height <= 0:
            problems.append(f"ページ{page.page}: 幅・高さが0以下 ({page.width}, {page.height})")
        if page.unit not in UNITS:
            problems.append(f"ページ{page.page}: 不明なunit {page.unit!r}（使えるのは {UNITS}）")
        for j, el in enumerate(page.elements):
            problems += _validate_element(f"ページ{page.page}の要素{j}", el)

    return problems


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("使い方: python -m common.ir <IRのJSON> ...")
    ok = True
    for path in sys.argv[1:]:
        problems = validate(load_ir(path))
        print(f"{path}: {'OK' if not problems else f'{len(problems)}件の問題'}")
        for p in problems:
            print(f"  {p}")
        ok = ok and not problems
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
