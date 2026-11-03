"""
評価用のデータセットを作る。1ケース = ランダムな文書1つ（PDF + 正解のラベル付きIR）。

    python -m evaluation.make_dataset --out evaluation/data/synth_v1 --docs 20
    python -m evaluation.make_dataset --out evaluation/data/synth_all --docs 40 --features all            # 全部入り（各機能は半分くらいの文書に）
    python -m evaluation.make_dataset --out evaluation/data/synth_table --docs 20 --features table --rate 1  # 全文書に表

--features で文書に入れるもの（synth_doc.FEATURES）を選び、--rate でそれぞれを入れる文書の割合を決める。
入れたかどうかは tags（has_table など）に残すので、評価で「表がある文書」「無い文書」に分けて比べられる。

出力:
    <out>/manifest.json               全ケースの一覧と tags（評価で「英語だけ」などに分けて集計するのに使う）
    <out>/cases/<id>/input.pdf        モデルに渡すPDF
    <out>/cases/<id>/gt.labeled.json  正解（ラベル付きIR）
    <out>/cases/<id>/source.tex       元のTeX（デバッグ用）
    <out>/cases/<id>/page-<n>.png     ページの画像（評価サイトで、正解と予測の枠を重ねる下絵）
"""
import argparse
import json
import random
import shutil
import subprocess
import tempfile
from datetime import date
from pathlib import Path

from common.ir import save_ir
from evaluation.ground_truth import build_ground_truth, compile_tex
from evaluation.images import SITE_DPI, render_pages
from evaluation.synth_doc import FEATURES, document_files, random_document

# 機能ごとの tags の名前（入れたかどうか）。two_column だけは段の数にする
FEATURE_TAGS = {
    "title": "has_title",
    "decoration": "has_decoration",
    "table": "has_table",
    "figure": "has_figure",
    "math": "has_math",
    "code": "has_code",
    "page_number": "has_page_number",
}


def make_case(rng: random.Random, lang: str, case_dir: Path, features: set[str] | None = None, enabled: set[str] | None = None) -> dict:
    """1ケース分を case_dir に作り、そのケースの tags を返す。コンパイルに失敗したら subprocess.CalledProcessError。
    features はこの文書に入れるもの（None なら synth_doc の既定）。enabled はデータセット全体で選んだもので、
    その分の tags（has_table など）を付ける"""
    tex, blocks = random_document(rng, lang, features)
    two_column = features is not None and "two_column" in features
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # .aux や .log などは、ここに出して最後に消す
        pdf = compile_tex(tex, Path(tmp), document_files(blocks))
        doc = build_ground_truth(str(pdf), blocks, source="input.pdf", two_column=two_column)
        shutil.copy(pdf, case_dir / "input.pdf")
    render_pages(str(case_dir / "input.pdf"), case_dir, SITE_DPI)
    (case_dir / "source.tex").write_text(tex, encoding="utf-8")
    save_ir(doc, str(case_dir / "gt.labeled.json"))
    labels = {el.label for page in doc.pages for el in page.elements}
    tags = {
        "lang": lang,
        "pages": len(doc.pages),
        "has_list": "list" in labels,
        "has_title": "title" in labels,
        "has_decoration": any(b.decorations for b in blocks),
    }
    for feature in sorted(enabled or set()):
        if feature == "two_column":
            tags["columns"] = 2 if two_column else 1
        elif feature not in ("title", "decoration"):
            tags[FEATURE_TAGS[feature]] = feature in (features or set())
    return tags


def parse_features(text: str | None) -> set[str] | None:
    """--features の値。"all" は全部、None（指定なし）は synth_doc の既定"""
    if text is None:
        return None
    features = set(FEATURES) if text == "all" else {f for f in text.split(",") if f}
    unknown = features - set(FEATURES)
    if unknown:
        raise SystemExit(f"不明な features {sorted(unknown)}（使えるのは {FEATURES} と all）")
    return features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="評価用のデータセット（PDF + 正解のラベル付きIR）を作る")
    parser.add_argument("--out", required=True, help="データセットのフォルダ（例: evaluation/data/synth_v1）")
    parser.add_argument("--docs", type=int, default=20, help="作る文書の数")
    parser.add_argument("--seed", type=int, default=0, help="同じ seed なら同じデータセットになる")
    parser.add_argument("--langs", default="ja,en", help="文書の言語。カンマ区切りで、順番に交互に使う")
    parser.add_argument(
        "--features",
        help=f"文書に入れるもの。カンマ区切りで {','.join(FEATURES)}、または all。"
        "指定しなければ、ときどきタイトル・著者・日付と、文中の装飾だけ",
    )
    parser.add_argument("--rate", type=float, default=0.5, help="--features のそれぞれを入れる文書の割合（1 なら全文書）")
    args = parser.parse_args(argv)

    out = Path(args.out)
    langs = args.langs.split(",")
    enabled = parse_features(args.features)
    cases = []
    for i in range(args.docs):
        case_id = f"{i:04d}"
        lang = langs[i % len(langs)]
        rng = random.Random(args.seed * 1_000_003 + i)  # ケースごとに種を変える（1件だけ作り直しても同じ文書になる）
        features = None if enabled is None else {f for f in sorted(enabled) if rng.random() < args.rate}
        try:
            tags = make_case(rng, lang, out / "cases" / case_id, features, enabled)
        except subprocess.CalledProcessError:
            print(f"{case_id}: TeXのコンパイルに失敗したので飛ばす")
            continue
        cases.append({"id": case_id, "tags": tags})

    manifest = {
        "name": out.name,
        "created": date.today().isoformat(),
        "generator": {
            "script": "evaluation.make_dataset",
            "seed": args.seed,
            "langs": langs,
            "features": sorted(enabled) if enabled is not None else None,
            "rate": args.rate if enabled is not None else None,
        },
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)}件を {out} に作った")


if __name__ == "__main__":
    main()
