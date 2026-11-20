"""
オペレータAIの進め方1: image_dataset の画像を、OCRの全部の設定で読んで採点し、採点表（CSV）を作る。

使い方（ai/ で）:
    python -m operator_ai.score_all image_dataset/data/val --limit 5          # まず5枚で試す
    python -m operator_ai.score_all image_dataset/data/val                    # 全部の画像（時間がかかるので計算機サーバで）
    python -m operator_ai.score_all image_dataset/data/val --settings tesseract:base easyocr:up2
      → operator_ai/data/<データセット名>/scores.csv       採点表（1行 = 画像1枚 × 設定1つ）
        operator_ai/data/<データセット名>/ocr/<engine>/<preset>/<case>.ir.json   OCRの結果（次回はこれを使う。読み直すときは --force）

列の名前は、チームの評価（evaluation/run.py の report.json と evaluation/metrics.py の summarize）に揃えている。
列の意味は operator_ai/README.md の「採点表の列」を参照。
"""

import argparse
import csv
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from common.ir import Bbox, Element, IRDocument, Page, load_ir, save_ir
from evaluation.import_dataset import guess_lang
from evaluation.metrics import Counts, char_errors, char_errors_by_script, count_document, normalize_text, summarize

# 最初に比べる設定。base から1か所だけ変えたもの（tesseract_ocr/presets.py・easyocr_ocr/presets.py にある名前）
DEFAULT_SETTINGS = [
    "tesseract:base",
    "tesseract:invert",
    "tesseract:up2",
    "tesseract:psm6",
    "tesseract:psm11",
    "easyocr:base",
    "easyocr:invert",
    "easyocr:up2",
    "easyocr:lowtext",
    "easyocr:up2_lowtext",
]
ENGINES = ("tesseract", "easyocr")

# ページ全体のCERで数えない正解のラベル（と kind）。表・図は文字の読み方がモデルごとに違い、
# 数式の正解の text はPDFの文字を並べただけ（分数などの並びが崩れている）なので、OCRの文字の比べ先にならない
PAGE_CER_EXCLUDED = ("table", "figure", "math")

SCORE_KEYS = [
    # ページ全体の文字の誤り率（取りこぼしも、余計に読んだ文字も誤りに数える）
    "cer_page", "cer_page_ja", "cer_page_en",
    "errors", "chars", "ja_errors", "ja_chars", "en_errors", "en_chars", "extra_chars",
    # チームの評価と同じ値（evaluation/metrics.py の summarize）。EasyOCR は段落にまとめてから比べる
    "precision", "recall", "f1", "miou", "cer",
    "sec_per_page",
]
TAG_KEYS = ["style", "dpi", "scale", "jpeg_quality", "dark_mode", "crop", "viewer", "background", "lang", "columns"]
COLUMNS = ["case", "setting", "engine", "preset", *SCORE_KEYS, *TAG_KEYS, "error"]


# ---------------------------------------------------------------------------
# ページ全体のCER
# ---------------------------------------------------------------------------

@dataclass
class Line:
    """予測の1行。Tesseract の段落は行に分け、箱を行の数で上から等分した位置を、その行の位置とする"""

    text: str
    bbox: Bbox

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2


def split_lines(elements: list[Element]) -> list[Line]:
    """予測の要素を行に分ける。段落と正解のブロックの区切りが違っても、行ごとに正しいブロックへ割り当てられるように"""
    lines = []
    for el in elements:
        texts = [t for t in (el.text or "").split("\n") if normalize_text(t)]
        if not texts:
            continue
        x0, y0, x1, y1 = el.bbox
        step = (y1 - y0) / len(texts)
        for i, text in enumerate(texts):
            lines.append(Line(text, (x0, y0 + step * i, x1, y0 + step * (i + 1))))
    return lines


def contains(bbox: Bbox, point: tuple[float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    x, y = point
    return x0 <= x <= x1 and y0 <= y <= y1


def area(bbox: Bbox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def reading_order(lines: list[Line]) -> list[Line]:
    """1つのブロックの中の行を、上の行から・同じ高さなら左から並べる。高さの近さは、行の高さの中央値の半分以内"""
    if not lines:
        return []
    tolerance = statistics.median(ln.bbox[3] - ln.bbox[1] for ln in lines) / 2
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda ln: ln.center[1]):
        if rows and abs(ln.center[1] - rows[-1][-1].center[1]) <= tolerance:
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln.bbox[0])]


def page_char_errors(pred: Page, truth: Page) -> dict[str, int]:
    """ページ全体の文字の誤りを数える。

    予測の各行を、行の中心を含む正解のブロックに割り当て、ブロックごとに文字列を比べる
    （ブロックの並び順や、段落の区切り方の違いを誤りに数えないため）。
    - 正解の表・図・数式の中にある行は捨てる（PAGE_CER_EXCLUDED）
    - どのブロックにも入らない行は、余計に読んだ文字として全部を誤りに数える（extra_chars）
    - 何も割り当てられなかったブロックは、全部の文字が取りこぼし
    """
    targets = [el for el in truth.elements if el.text and (el.label or el.kind) not in PAGE_CER_EXCLUDED
               and el.kind not in PAGE_CER_EXCLUDED]
    excluded = [el.bbox for el in truth.elements if (el.label or el.kind) in PAGE_CER_EXCLUDED or el.kind in PAGE_CER_EXCLUDED]

    assigned: list[list[Line]] = [[] for _ in targets]
    extra: list[Line] = []
    for ln in split_lines(pred.elements):
        if any(contains(b, ln.center) for b in excluded):
            continue
        hits = [i for i, el in enumerate(targets) if contains(el.bbox, ln.center)]
        if hits:
            assigned[min(hits, key=lambda i: area(targets[i].bbox))].append(ln)
        else:
            extra.append(ln)

    counts = dict.fromkeys(["errors", "chars", "ja_errors", "ja_chars", "en_errors", "en_chars", "extra_chars"], 0)
    for el, lines in zip(targets, assigned):
        pred_text = "".join(ln.text for ln in reading_order(lines))
        errors, chars = char_errors(pred_text, el.text)
        counts["errors"] += errors
        counts["chars"] += chars
        by_script = char_errors_by_script(pred_text, el.text)
        for script in ("ja", "en"):
            counts[f"{script}_errors"] += by_script[script][0]
            counts[f"{script}_chars"] += by_script[script][1]
    for ln in extra:
        by_script = char_errors_by_script(ln.text, "")
        for script in ("ja", "en"):
            counts[f"{script}_errors"] += by_script[script][0]
        extra = len(normalize_text(ln.text))
        counts["errors"] += extra
        counts["extra_chars"] += extra
    return counts


def rate(errors: int, chars: int) -> float | None:
    return round(errors / chars, 4) if chars > 0 else None


# ---------------------------------------------------------------------------
# OCR（結果は保存して使い回す）
# ---------------------------------------------------------------------------

def parse_setting(setting: str) -> tuple[str, str]:
    """"tesseract:invert" → ("tesseract", "invert")。プリセットの名前が無いものはエラー"""
    engine, _, preset = setting.partition(":")
    if engine not in ENGINES or not preset:
        raise ValueError(f"設定は <engine>:<preset> の形で書く（engine は {ENGINES}）: {setting!r}")
    if engine == "tesseract":
        from tesseract_ocr.presets import PRESETS
    else:
        from easyocr_ocr.presets import PRESETS
    if preset not in PRESETS:
        raise ValueError(f"{engine} に {preset!r} というプリセットはありません（使えるのは {list(PRESETS)}）")
    return engine, preset


@dataclass
class OcrResult:
    """1回読んだ結果を2つの形で持つ。paragraphs はチームの評価と同じ形（f1 など）、lines はページ全体のCER に使う"""

    paragraphs: IRDocument
    lines: IRDocument


def run_tesseract(preset: str, image_path: str) -> OcrResult:
    """Tesseract を1回だけ動かし、段落単位（tesseract_ocr.ir_adapter.extract_ir と同じ出力）と行単位の両方を作る。

    extract_ir の IR は段落の箱しか持たず、psm6 のようにページ全体が1段落になる設定では行の位置が分からない。
    tesseract_ocr のコードは変えず、同じ部品（前処理・run_tesseract・to_elements）を呼んで行単位も作る。
    extract_ir を変えたら、ここも合わせる。
    """
    import pytesseract
    from PIL import Image

    from image_common.preprocess import apply_preprocess
    from tesseract_ocr import ir_adapter as tess
    from tesseract_ocr.presets import PRESETS

    settings = PRESETS[preset]
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size
    processed, scale = apply_preprocess(rgb, settings["preprocess"])
    oem = settings["tesseract"].get("oem", tess.DEFAULT_OEM)
    psm = settings["tesseract"].get("psm", tess.DEFAULT_PSM)
    run = tess.run_tesseract(processed, tess.DEFAULT_LANG, psm, oem)
    params = {"preset": preset, **settings, "lang": run.lang, "psm": run.psm, "oem": oem,
              "tesseract_version": str(pytesseract.get_tesseract_version())}

    # 段落の番号を行ごとに別にすると、to_elements が1行ずつの要素を作る
    data = run.data
    per_line = {**data, "par_num": [p * 10_000 + ln for p, ln in zip(data["par_num"], data["line_num"])]}
    line_run = tess.OcrRun(run.lang, run.psm, per_line)

    def doc(elements: list[Element]) -> IRDocument:
        page = Page(page=1, width=width, height=height, unit="px", elements=elements)
        return IRDocument(source=os.path.basename(image_path), extractor=tess.NAME, pages=[page], params=dict(params))

    return OcrResult(doc(tess.to_elements(run, width, height, scale)), doc(tess.to_elements(line_run, width, height, scale)))


def run_easyocr(preset: str, image_path: str) -> OcrResult:
    """EasyOCR は行単位で返すので、それを lines にし、段落にまとめたもの（チームの評価と同じ）を paragraphs にする"""
    from easyocr_ocr.grouping import group_document
    from easyocr_ocr.ir_adapter import extract_ir

    lines = extract_ir(image_path, preset=preset)
    return OcrResult(group_document(lines), lines)


def run_ocr(engine: str, preset: str, image_path: str) -> OcrResult:
    return run_tesseract(preset, image_path) if engine == "tesseract" else run_easyocr(preset, image_path)


class OcrCrashed(RuntimeError):
    """前回この画像・設定を読んでいる途中で、プロセスごと落ちた"""


def load_or_run_ocr(cache_dir: Path, engine: str, preset: str, case: str, image_path: str, force: bool) -> OcrResult:
    """保存したOCRの結果を読む。無ければ読んで、かかった秒数を params["seconds"] に入れて保存する。

    EasyOCR はライブラリの中で Segmentation fault を起こすことがあり、例外で拾えずにプロセスごと止まる。
    読む前に目印（<case>.running）を置き、読み終えたら消す。目印が残っていれば前回そこで落ちたので、
    もう一度読まずに OcrCrashed にする（error 列に残り、もう一度実行すれば続きから進む）。--force なら読み直す。
    """
    base = cache_dir / engine / preset
    para_path, line_path = base / f"{case}.ir.json", base / f"{case}.lines.ir.json"
    marker = base / f"{case}.running"
    if para_path.exists() and line_path.exists() and not force:
        return OcrResult(load_ir(str(para_path)), load_ir(str(line_path)))
    if marker.exists() and not force:
        raise OcrCrashed("前回この画像を読んでいる途中でOCRが落ちた（Segmentation fault など）。読み直すには --force")
    base.mkdir(parents=True, exist_ok=True)
    marker.touch()
    start = time.perf_counter()
    result = run_ocr(engine, preset, image_path)
    seconds = round(time.perf_counter() - start, 3)
    for doc, path in ((result.paragraphs, para_path), (result.lines, line_path)):
        doc.params["seconds"] = seconds
        save_ir(doc, str(path))
    marker.unlink()
    return result


# ---------------------------------------------------------------------------
# 採点
# ---------------------------------------------------------------------------

def tags(truth: IRDocument, dataset_dir: Path) -> dict:
    """切り口（分析専用。オペレータの振り分けには使わない）。image_dataset の params と、正解から推定した言語・段組み"""
    p = truth.params
    tex = dataset_dir / "tex" / f"{p.get('doc', '')}.tex"
    columns = None
    if tex.exists():
        with open(tex, encoding="utf-8") as f:
            columns = 2 if "twocolumn" in f.readline() else 1
    return {
        "style": p.get("style"),
        "dpi": p.get("dpi"),
        "scale": p.get("scale", 1.0),
        "jpeg_quality": p.get("jpeg_quality"),
        "dark_mode": "dark_mode" in p,
        "crop": "crop" in p,
        "viewer": "viewer" in p,
        "background": "background" in p,
        "lang": guess_lang(truth),
        "columns": columns,
    }


def score(ocr: OcrResult, truth: IRDocument) -> dict:
    """1枚分の点数。列の名前は SCORE_KEYS"""
    gt = truth.pages[0]
    line_page = ocr.lines.pages[0] if ocr.lines.pages else Page(1, gt.width, gt.height, gt.unit)
    page = page_char_errors(line_page, gt)
    row: dict = {
        "cer_page": rate(page["errors"], page["chars"]),
        "cer_page_ja": rate(page["ja_errors"], page["ja_chars"]),
        "cer_page_en": rate(page["en_errors"], page["en_chars"]),
        **page,
    }
    counts = Counts()
    count_document(ocr.paragraphs, truth, counts)
    s = summarize(counts)
    for key in ("precision", "recall", "f1", "miou", "cer"):
        row[key] = None if s[key] is None else round(s[key], 4)
    row["sec_per_page"] = ocr.paragraphs.params.get("seconds")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:  # Excel で開けるよう BOM 付き（tesseract_ocr と同じ）
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)  # 途中で止めても、前の表が壊れないように


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="image_dataset の画像を全部の設定で読んで採点し、採点表（CSV）を作る")
    parser.add_argument("dataset", type=Path, help="image_dataset.build の出力先（images/・truth/・tex/ があるディレクトリ）")
    parser.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help="比べる設定（<engine>:<preset>）")
    parser.add_argument("--limit", type=int, help="先頭からこの枚数だけ使う（試すとき）")
    parser.add_argument("--out", type=Path, help="出力先（既定は operator_ai/data/<データセット名>）")
    parser.add_argument("--force", action="store_true", help="保存したOCRの結果を使わず、読み直す")
    args = parser.parse_args(argv)

    settings = [(s, *parse_setting(s)) for s in args.settings]
    truth_paths = sorted((args.dataset / "truth").glob("*.ir.json"))[: args.limit]
    if not truth_paths:
        raise SystemExit(f"エラー: {args.dataset}/truth/ に正解がありません。")
    out = args.out or Path("operator_ai/data") / args.dataset.name
    csv_path = out / "scores.csv"

    rows: list[dict] = []
    total = len(truth_paths) * len(settings)
    for n_case, truth_path in enumerate(truth_paths):
        truth = load_ir(str(truth_path))
        case = truth_path.name.removesuffix(".ir.json")
        case_tags = tags(truth, args.dataset)
        image_path = str(args.dataset / "images" / truth.source)
        for n_setting, (setting, engine, preset) in enumerate(settings):
            row = {"case": case, "setting": setting, "engine": engine, "preset": preset, **case_tags, "error": None}
            try:
                ocr = load_or_run_ocr(out / "ocr", engine, preset, case, image_path, args.force)
                row.update(score(ocr, truth))
            except Exception as e:  # 1枚の失敗で全体を止めない（report.json の error と同じ扱い）
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            done = n_case * len(settings) + n_setting + 1
            shown = row["error"] or f"cer_page {row['cer_page']}  f1 {row['f1']}"
            print(f"[{done}/{total}] {case} {setting:22s} {shown}", flush=True)
        write_csv(csv_path, rows)

    errors = sum(1 for r in rows if r["error"])
    print(f"\n{csv_path} に {len(rows)} 行を書きました（エラー {errors} 行）")
    return csv_path


if __name__ == "__main__":
    main()
