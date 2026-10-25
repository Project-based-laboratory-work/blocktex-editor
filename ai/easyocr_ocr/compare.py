"""
プリセットごとの文字誤り率（CER）を、正解テキストのある画像で比べる。

画像と同じ名前の .txt（正解テキスト）がある画像だけが対象。位置（bbox）は見ず、
IRの要素を読む順に並べてつなげた文字列を、正解テキストと比べる。
CERの計算は evaluation/metrics.py と同じ関数を使う（空白・改行は取り除いて比べる）。

使い方（ai/ で）:
    python -m easyocr_ocr.compare                          # 全プリセット × 全画像
    python -m easyocr_ocr.compare --presets base up2       # プリセットを絞る
    python -m easyocr_ocr.compare --category jphorizon     # カテゴリ（sample/ の下のフォルダ名）を絞る
      → easyocr_ocr/data/compare/summary.csv  プリセットごと・カテゴリごとのCERと時間
        easyocr_ocr/data/compare/details.csv  画像1枚ごとの結果
        easyocr_ocr/data/compare/ir/          読み取った結果（IR）。あれば読み直さずに再利用する
"""

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from common.ir import Element, load_ir, save_ir
from easyocr_ocr.ir_adapter import extract_ir
from easyocr_ocr.presets import PRESETS
from evaluation.metrics import char_errors, char_errors_by_script, ratio_or_none

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = REPO_ROOT / "ocr" / "sample"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "compare"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
# 縦長の箱（高さが幅のこの倍以上）がこの割合以上なら縦書きとみなす
VERTICAL_RATIO = 1.5
VERTICAL_SHARE = 0.5


@dataclass
class Sample:
    category: str
    image: Path
    truth: Path


@dataclass
class Row:
    preset: str
    category: str
    image: str
    errors: int
    chars: int
    ja_errors: int
    ja_chars: int
    en_errors: int
    en_chars: int
    elements: int
    seconds: float | None

    @property
    def cer(self) -> float | None:
        return ratio_or_none(self.errors, self.chars)


def find_samples(sample_dir: Path, categories: list[str] | None = None) -> list[Sample]:
    """sample_dir/<カテゴリ>/ の中から、同じ名前の .txt がある画像を探す。"""
    samples = []
    for truth in sorted(sample_dir.glob("*/*.txt")):
        category = truth.parent.name
        if categories and category not in categories:
            continue
        for suffix in IMAGE_SUFFIXES:
            image = truth.with_suffix(suffix)
            if image.is_file():
                samples.append(Sample(category, image, truth))
                break
    return samples


def is_vertical(elements: list[Element]) -> bool:
    """縦長の箱が多ければ縦書きとみなす。"""
    boxes = [el.bbox for el in elements]
    if not boxes:
        return False
    tall = sum(1 for x0, y0, x1, y1 in boxes if (y1 - y0) >= (x1 - x0) * VERTICAL_RATIO)
    return tall / len(boxes) >= VERTICAL_SHARE


def _group(elements: list[Element], center, size) -> list[list[Element]]:
    """中心の位置が近い要素を同じ行（または列）にまとめる。近さの基準は箱の大きさの中央値の半分。"""
    if not elements:
        return []
    limit = statistics.median(size(el) for el in elements) / 2
    groups: list[list[Element]] = []
    for el in sorted(elements, key=center):
        if groups and abs(center(el) - center(groups[-1][-1])) <= limit:
            groups[-1].append(el)
        else:
            groups.append([el])
    return groups


def reading_order(elements: list[Element], vertical: bool) -> list[Element]:
    """要素を読む順に並べる。横書きは上の行から・行の中は左から、縦書きは右の列から・列の中は上から。"""
    if vertical:
        columns = _group(
            elements,
            center=lambda el: (el.bbox[0] + el.bbox[2]) / 2,
            size=lambda el: el.bbox[2] - el.bbox[0],
        )
        return [el for column in reversed(columns) for el in sorted(column, key=lambda el: el.bbox[1])]
    lines = _group(
        elements,
        center=lambda el: (el.bbox[1] + el.bbox[3]) / 2,
        size=lambda el: el.bbox[3] - el.bbox[1],
    )
    return [el for line in lines for el in sorted(line, key=lambda el: el.bbox[0])]


def page_text(elements: list[Element]) -> str:
    """1ページ分の要素を読む順につなげた文字列。"""
    ordered = reading_order([el for el in elements if el.text], is_vertical(elements))
    return "".join(el.text for el in ordered)


def score(preset: str, sample: Sample, elements: list[Element], seconds: float | None) -> Row:
    pred = page_text(elements)
    truth = sample.truth.read_text(encoding="utf-8")
    errors, chars = char_errors(pred, truth)
    by_script = char_errors_by_script(pred, truth)
    return Row(
        preset=preset,
        category=sample.category,
        image=sample.image.name,
        errors=errors,
        chars=chars,
        ja_errors=by_script["ja"][0],
        ja_chars=by_script["ja"][1],
        en_errors=by_script["en"][0],
        en_chars=by_script["en"][1],
        elements=len(elements),
        seconds=seconds,
    )


def run(presets: list[str], samples: list[Sample], out_dir: Path, force: bool = False) -> list[Row]:
    """プリセット × 画像を全部読み、採点する。読み取り結果（IR）と時間は out_dir に残し、次回は再利用する。"""
    times_path = out_dir / "times.json"
    times = json.loads(times_path.read_text(encoding="utf-8")) if times_path.is_file() else {}
    rows = []
    total = len(presets) * len(samples)
    for i, (preset, sample) in enumerate(((p, s) for p in presets for s in samples), start=1):
        ir_path = out_dir / "ir" / sample.category / f"{sample.image.stem}.{preset}.ir.json"
        key = f"{sample.category}/{sample.image.stem}.{preset}"
        if ir_path.is_file() and not force:
            doc = load_ir(str(ir_path))
            note = "保存済みを再利用"
        else:
            start = time.perf_counter()
            doc = extract_ir(str(sample.image), preset)
            times[key] = round(time.perf_counter() - start, 2)
            ir_path.parent.mkdir(parents=True, exist_ok=True)
            save_ir(doc, str(ir_path))
            times_path.write_text(json.dumps(times, ensure_ascii=False, indent=2), encoding="utf-8")
            note = f"{times[key]}秒"
        row = score(preset, sample, doc.pages[0].elements, times.get(key))
        rows.append(row)
        print(f"[{i}/{total}] {preset:15s} {key:40s} CER={_fmt(row.cer)}（{note}）", flush=True)
    return rows


def summarize(rows: list[Row], categories: list[str]) -> list[dict]:
    """プリセットごとに、カテゴリ別・全体のCER（誤りの合計 ÷ 文字数の合計）と、1枚あたりの平均時間を出す。"""
    summary = []
    for preset in dict.fromkeys(row.preset for row in rows):
        mine = [row for row in rows if row.preset == preset]
        entry: dict = {"preset": preset}
        for category in categories:
            rs = [row for row in mine if row.category == category]
            entry[category] = ratio_or_none(sum(r.errors for r in rs), sum(r.chars for r in rs))
        entry["全体"] = ratio_or_none(sum(r.errors for r in mine), sum(r.chars for r in mine))
        entry["日本語"] = ratio_or_none(sum(r.ja_errors for r in mine), sum(r.ja_chars for r in mine))
        entry["英語"] = ratio_or_none(sum(r.en_errors for r in mine), sum(r.en_chars for r in mine))
        seconds = [r.seconds for r in mine if r.seconds is not None]
        entry["秒/枚"] = round(sum(seconds) / len(seconds), 1) if seconds else None
        summary.append(entry)
    return sorted(summary, key=lambda e: (e["全体"] is None, e["全体"] or 0))


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def write_csv(path: Path, records: list[dict]) -> None:
    """Excelで開いても文字化けしないように、BOM付きUTF-8で書く。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if v is None else v) for k, v in record.items()})


def print_table(summary: list[dict]) -> None:
    columns = list(summary[0])
    print("\t".join(columns))
    for entry in summary:
        print("\t".join(entry["preset"] if c == "preset" else _fmt(entry[c]) for c in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="プリセットごとのCERを正解テキストと比べる（EasyOCR）")
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS), help="比べるプリセット")
    parser.add_argument("--category", nargs="+", help="比べるカテゴリ（sample/ の下のフォルダ名）")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES, help="画像と正解テキストのフォルダ")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT, help="出力先")
    parser.add_argument("--force", action="store_true", help="保存済みの読み取り結果を使わず、読み直す")
    args = parser.parse_args()

    samples = find_samples(args.samples, args.category)
    if not samples:
        raise SystemExit(f"エラー: {args.samples} に、正解テキスト（.txt）のある画像がありません。")
    print(f"画像 {len(samples)} 枚 × プリセット {len(args.presets)} 個", flush=True)

    rows = run(args.presets, samples, args.out_dir, args.force)
    categories = list(dict.fromkeys(s.category for s in samples))
    summary = summarize(rows, categories)

    write_csv(args.out_dir / "details.csv", [{**asdict(r), "cer": r.cer} for r in rows])
    write_csv(args.out_dir / "summary.csv", summary)
    print()
    print_table(summary)
    print(f"\n保存先: {args.out_dir}")


if __name__ == "__main__":
    main()
