"""
評価用のデータセットを作る。1ケース = ランダムな文書1つ（PDF + 正解のラベル付きIR）。

    python -m evaluation.make_dataset --out evaluation/data/synth_v1 --docs 20

出力:
    <out>/manifest.json               全ケースの一覧と tags（評価で「英語だけ」などに分けて集計するのに使う）
    <out>/cases/<id>/input.pdf        モデルに渡すPDF
    <out>/cases/<id>/gt.labeled.json  正解（ラベル付きIR）
    <out>/cases/<id>/source.tex       元のTeX（デバッグ用）
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
from evaluation.synth_doc import random_document


def make_case(rng: random.Random, lang: str, case_dir: Path) -> dict:
    """1ケース分を case_dir に作り、そのケースの tags を返す。コンパイルに失敗したら subprocess.CalledProcessError"""
    tex, blocks = random_document(rng, lang)
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:  # .aux や .log などは、ここに出して最後に消す
        pdf = compile_tex(tex, Path(tmp))
        doc = build_ground_truth(str(pdf), blocks, source="input.pdf")
        shutil.copy(pdf, case_dir / "input.pdf")
    (case_dir / "source.tex").write_text(tex, encoding="utf-8")
    save_ir(doc, str(case_dir / "gt.labeled.json"))
    labels = {el.label for page in doc.pages for el in page.elements}
    return {"lang": lang, "pages": len(doc.pages), "has_list": "list" in labels}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="評価用のデータセット（PDF + 正解のラベル付きIR）を作る")
    parser.add_argument("--out", required=True, help="データセットのフォルダ（例: evaluation/data/synth_v1）")
    parser.add_argument("--docs", type=int, default=20, help="作る文書の数")
    parser.add_argument("--seed", type=int, default=0, help="同じ seed なら同じデータセットになる")
    parser.add_argument("--langs", default="ja,en", help="文書の言語。カンマ区切りで、順番に交互に使う")
    args = parser.parse_args(argv)

    out = Path(args.out)
    langs = args.langs.split(",")
    cases = []
    for i in range(args.docs):
        case_id = f"{i:04d}"
        lang = langs[i % len(langs)]
        rng = random.Random(args.seed * 1_000_003 + i)  # ケースごとに種を変える（1件だけ作り直しても同じ文書になる）
        try:
            tags = make_case(rng, lang, out / "cases" / case_id)
        except subprocess.CalledProcessError:
            print(f"{case_id}: TeXのコンパイルに失敗したので飛ばす")
            continue
        cases.append({"id": case_id, "tags": tags})

    manifest = {
        "name": out.name,
        "created": date.today().isoformat(),
        "generator": {"script": "evaluation.make_dataset", "seed": args.seed, "langs": langs},
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)}件を {out} に作った")


if __name__ == "__main__":
    main()
