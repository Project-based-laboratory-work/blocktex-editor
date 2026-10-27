"""
Tesseract の認識結果を正解テキストと比べて CER（文字誤り率）と処理時間を出す。

サンプルの置き方（画像と同じ名前の .txt が正解。.txt が無い画像は飛ばす）:
    ocr/sample/eng/eng_001.jpg, eng_001.txt, ...
    ocr/sample/jphorizon/jphorizontal_001.jpg, jphorizontal_001.txt, ...

使い方（ai/ で）:
    # 全プリセットを比べる（既定。プリセットの数 × 画像の数だけ読むので時間がかかる。横書きに固定して10枚で約9分）
    python -m tesseract_ocr.evaluate
    # プリセットを絞る（presets.py の名前を並べる）
    python -m tesseract_ocr.evaluate --presets base gray otsu up2
    # データセットを絞る・向きを決め打ちする
    python -m tesseract_ocr.evaluate --datasets eng --direction horizontal

向きは、--direction を省略すると auto（ir_adapter が横書き・縦書きの両方で読み、「文字数 × 信頼度」が
大きい方を採る）。言語は、--lang を省略すると向きごとの既定値（横 jpn+eng / 縦 jpn_vert+eng）。
フォルダ名は結果をまとめる単位に使うだけで、読み方の判断には使わない。
画像ごとに選ばれた向きと、横・縦それぞれの点数は results.csv に残す。

出力（-o の下。既定は tesseract_ocr/data/eval/、git管理外）:
    <プリセット>/<データセット>/<画像名>.<プリセット>.ir.json  … 共通IR ①
    <プリセット>/<データセット>/<画像名>.ocr.txt                … 認識した全文（正解と見比べる用）
    results.csv   … 画像ごとの CER・処理時間
    summary.json  … プリセット × データセットごとの集計
    compare.csv   … 比較表（行＝プリセット、列＝データセットごとの CER と全体の CER・秒/枚。全体の CER が小さい順）
最後に、データセットごとに CER が一番小さかったプリセットも表示する。

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
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "eval"


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


def compare_table(rows: list[dict], presets: list[str], datasets: list[str]) -> list[dict]:
    """プリセットごとに、データセット別の CER・全体の CER・1枚あたりの秒数を1行にまとめる。全体の CER が小さい順。
    CER はどれも「誤りの合計 ÷ 正解の文字数の合計」（画像ごとの CER の平均ではない）。"""
    table = []
    for preset in presets:
        mine = [r for r in rows if r["preset"] == preset]
        entry: dict = {"preset": preset}
        for ds in datasets:
            group = [r for r in mine if r["dataset"] == ds]
            entry[ds] = rate(sum(r["errors"] for r in group), sum(r["gt_chars"] for r in group))
        entry["全体"] = rate(sum(r["errors"] for r in mine), sum(r["gt_chars"] for r in mine))
        entry["秒/枚"] = round(sum(r["sec"] for r in mine) / len(mine), 2) if mine else None
        table.append(entry)
    return sorted(table, key=lambda e: (e["全体"] is None, e["全体"] or 0))


def best_presets(table: list[dict], datasets: list[str]) -> dict[str, tuple[str, float]]:
    """データセットごとに、CER が一番小さかったプリセットとその CER。同じ CER なら表で上（全体の CER が小さい方）を採る。"""
    best = {}
    for ds in [*datasets, "全体"]:
        scored = [(e[ds], e["preset"]) for e in table if e[ds] is not None]
        if scored:
            cer, preset = min(scored, key=lambda s: s[0])
            best[ds] = (preset, cer)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tesseract の CER 評価")
    parser.add_argument("--sample-root", type=Path, default=Path("../ocr/sample"), help="データセットのフォルダが並ぶ場所")
    parser.add_argument("--datasets", nargs="+", default=None, help="評価するフォルダ名（省略時は正解 .txt がある全フォルダ）")
    parser.add_argument("--presets", nargs="+", default=["all"], choices=[*PRESETS, "all"],
                        help="比べるプリセット（既定の all は presets.py の全部）")
    parser.add_argument("--direction", default="auto", choices=["horizontal", "vertical", "auto", "mixed"],
                        help="全画像の向き（既定の auto は画像ごとに横・縦の両方で読んで選ぶ）")
    parser.add_argument("--lang", default=None, help="全画像の言語（省略時は向きごとの既定値）")
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT, help="出力先")
    args = parser.parse_args()
    if "all" in args.presets:
        args.presets = list(PRESETS)

    datasets = args.datasets or sorted(
        p.name for p in args.sample_root.iterdir() if p.is_dir() and find_pairs(p)
    )
    if not datasets:
        raise SystemExit(f"正解 .txt のある画像が {args.sample_root} に見つかりません。")

    check_langs(args.direction, args.lang)
    rows = []
    for ds in datasets:
        pairs = find_pairs(args.sample_root / ds)

        for preset in args.presets:
            out_dir = args.out_dir / preset / ds
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{preset}] {ds}（{args.direction}, lang={args.lang or '既定'}, {len(pairs)}枚）")

            for img, txt in pairs:
                t0 = time.perf_counter()
                doc = extract_ir(str(img), preset, args.direction, args.lang)
                sec = time.perf_counter() - t0

                pred, gt = ir_text(doc), txt.read_text(encoding="utf-8")
                save_ir(doc, str(out_dir / f"{img.stem}.{preset}.ir.json"))
                (out_dir / f"{img.stem}.ocr.txt").write_text(pred, encoding="utf-8")

                errors, chars = char_errors(pred, gt)
                by_script = char_errors_by_script(pred, gt)
                # auto のときだけある、横・縦それぞれの点数（向きを選んだ根拠）
                scores = doc.params.get("direction_scores", {})
                row = {
                    "preset": preset,
                    "dataset": ds,
                    "image": img.name,
                    "direction": doc.params["chosen_direction"],
                    "score_horizontal": scores.get("horizontal"),
                    "score_vertical": scores.get("vertical"),
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
                print(f"  {img.name}: {row['direction']}・{row['lang']} CER {row['cer']:.2%}"
                      f"（誤り {errors} / {chars} 文字）, {sec:.2f}秒")

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

    table = compare_table(rows, args.presets, datasets)
    with open(args.out_dir / "compare.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    print("\n=== 比較表（全体の CER が小さい順） ===")
    print(f"{'preset':<14}" + "".join(f"{ds:>12}" for ds in datasets) + f"{'全体':>10}{'秒/枚':>8}")
    for e in table:
        print(f"{e['preset']:<14}" + "".join(f"{fmt(e[ds]):>12}" for ds in datasets)
              + f"{fmt(e['全体']):>10}{e['秒/枚']:>8.2f}")
    print("\n=== データセットごとに一番よかったプリセット ===")
    for ds, (preset, cer) in best_presets(table, datasets).items():
        print(f"  {ds:<12} {preset:<14} CER {fmt(cer)}")
    print(f"\n詳細: {args.out_dir / 'results.csv'}, {args.out_dir / 'summary.json'}, {args.out_dir / 'compare.csv'}")


if __name__ == "__main__":
    main()
