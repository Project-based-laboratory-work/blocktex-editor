"""
Tesseract の認識結果を正解テキストと比べて CER（文字誤り率）と処理時間を出す。

サンプルの置き方（画像と同じ名前の .txt が正解。.txt が無い画像は飛ばす）:
    ocr/sample/eng/eng_001.jpg, eng_001.txt, ...
    ocr/sample/jphorizon/jphorizontal_001.jpg, jphorizontal_001.txt, ...
    ocr/sample/jpvertical/jpvertical_001.jpg, jpvertical_001.txt, ...

使い方（ai/ で）:
    python -m tesseract_ocr.evaluate
    # プリセットを比べる（presets.py の名前を並べる）
    python -m tesseract_ocr.evaluate --presets base gray otsu up2
    # データセットを絞る・向きの自動判定を試す
    python -m tesseract_ocr.evaluate --datasets jpvertical --direction auto

向きと言語は、--direction / --lang を省略するとフォルダ名から決める:
    eng* → horizontal・eng / *vert* → vertical・jpn_vert+eng / それ以外 → horizontal・jpn+eng

出力（-o の下）:
    <プリセット>/<データセット>/<画像名>.<プリセット>.ir.json  … 共通IR ①
    <プリセット>/<データセット>/<画像名>.ocr.txt                … 認識した全文（正解と見比べる用）
    results.csv   … 画像ごとの CER・処理時間
    summary.json  … プリセット × データセットごとの集計

CER は evaluation/metrics.py の char_errors（空白・改行を除いて編集距離）を使い、
EasyOCR などほかのモデルと同じ数え方にする。データセットの CER は「誤りの合計 ÷ 正解の文字数の合計」。
cer_ja / cer_en は日本語の文字・英語（ASCII）の文字だけで数えたもの。
"""

import argparse
import csv
import json
import time
from pathlib import Path

from common.ir import IRDocument, save_ir
from evaluation.metrics import char_errors, char_errors_by_script, ratio_or_none
from tesseract_ocr.ir_adapter import check_langs, extract_ir
from tesseract_ocr.presets import PRESETS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def default_setting(dataset: str) -> tuple[str, str | None]:
    """フォルダ名から (向き, 言語) の既定値を決める。言語 None は ir_adapter の既定値を使う。"""
    name = dataset.lower()
    if "vert" in name:
        return "vertical", None
    if name.startswith("eng"):
        return "horizontal", "eng"
    return "horizontal", None


def find_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """画像と、同じ名前の正解 .txt の組"""
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [(img, img.with_suffix(".txt")) for img in images if img.with_suffix(".txt").exists()]


def ir_text(doc: IRDocument) -> str:
    """IR の text 要素を並び順につないだ全文"""
    return "\n".join(el.text for page in doc.pages for el in page.elements if el.text)


def rate(errors: int, chars: int) -> float | None:
    r = ratio_or_none(errors, chars)
    return None if r is None else round(r, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tesseract の CER 評価")
    parser.add_argument("--sample-root", type=Path, default=Path("../ocr/sample"), help="データセットのフォルダが並ぶ場所")
    parser.add_argument("--datasets", nargs="+", default=None, help="評価するフォルダ名（省略時は正解 .txt がある全フォルダ）")
    parser.add_argument("--presets", nargs="+", default=["base"], choices=list(PRESETS), help="比べるプリセット")
    parser.add_argument("--direction", default=None, choices=["horizontal", "vertical", "auto", "mixed"],
                        help="全データセットの向き（省略時はフォルダ名から）")
    parser.add_argument("--lang", default=None, help="全データセットの言語（省略時はフォルダ名から）")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("out/tesseract_eval"), help="出力先")
    args = parser.parse_args()

    datasets = args.datasets or sorted(
        p.name for p in args.sample_root.iterdir() if p.is_dir() and find_pairs(p)
    )
    if not datasets:
        raise SystemExit(f"正解 .txt のある画像が {args.sample_root} に見つかりません。")

    rows = []
    for ds in datasets:
        direction, lang = default_setting(ds)
        direction, lang = args.direction or direction, args.lang or lang
        check_langs(direction, lang)
        pairs = find_pairs(args.sample_root / ds)

        for preset in args.presets:
            out_dir = args.out_dir / preset / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{preset}] {ds}（{direction}, lang={lang or '既定'}, {len(pairs)}枚）")

            for img, txt in pairs:
                t0 = time.perf_counter()
                doc = extract_ir(str(img), preset, direction, lang)
                sec = time.perf_counter() - t0

                pred, gt = ir_text(doc), txt.read_text(encoding="utf-8")
                save_ir(doc, str(out_dir / f"{img.stem}.{preset}.ir.json"))
                (out_dir / f"{img.stem}.ocr.txt").write_text(pred, encoding="utf-8")

                errors, chars = char_errors(pred, gt)
                by_script = char_errors_by_script(pred, gt)
                row = {
                    "preset": preset,
                    "dataset": ds,
                    "image": img.name,
                    "direction": doc.params["chosen_direction"],
                    "lang": doc.params["lang"],
                    "psm": doc.params["psm"],
                    "cer": rate(errors, chars),
                    "cer_ja": rate(*by_script["ja"]),
                    "cer_en": rate(*by_script["en"]),
                    "errors": errors,
                    "gt_chars": chars,
                    "ja_errors": by_script["ja"][0], "ja_chars": by_script["ja"][1],
                    "en_errors": by_script["en"][0], "en_chars": by_script["en"][1],
                    "sec": round(sec, 3),
                }
                rows.append(row)
                print(f"  {img.name}: CER {row['cer']:.2%}（誤り {errors} / {chars} 文字）, {sec:.2f}秒")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:  # Excel で開けるよう BOM 付き
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for preset in args.presets:
        for ds in datasets:
            group = [r for r in rows if r["preset"] == preset and r["dataset"] == ds]
            total = lambda key: sum(r[key] for r in group)  # noqa: E731
            summary.append({
                "preset": preset,
                "dataset": ds,
                "images": len(group),
                "cer": rate(total("errors"), total("gt_chars")),
                "cer_ja": rate(total("ja_errors"), total("ja_chars")),
                "cer_en": rate(total("en_errors"), total("en_chars")),
                "mean_sec": round(total("sec") / len(group), 3),
            })
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fmt = lambda v: "-" if v is None else f"{v:.2%}"  # noqa: E731
    print("\n=== まとめ（CER = 誤りの合計 ÷ 正解の文字数の合計） ===")
    print(f"{'preset':<14}{'dataset':<12}{'枚数':>4}{'CER':>9}{'CER_ja':>9}{'CER_en':>9}{'秒/枚':>8}")
    for s in summary:
        print(f"{s['preset']:<14}{s['dataset']:<12}{s['images']:>4}{fmt(s['cer']):>9}{fmt(s['cer_ja']):>9}"
              f"{fmt(s['cer_en']):>9}{s['mean_sec']:>8.2f}")
    print(f"\n詳細: {args.out_dir / 'results.csv'}, {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
