"""評価サイト用のデータセットを、学習データと同じ作り方（doc_gen.py）で作る。ルール版と AI 版の差を見るため。

    python -m hybrid_extractor.make_eval_dataset --out evaluation/data/hybrid_varied --docs 60 --seed 3
    python -m evaluation.run --model hybrid_operator --dataset evaluation/data/hybrid_varied   # ルール
    python -m evaluation.run --model hybrid_ai --dataset evaluation/data/hybrid_varied         # AI
    python -m hybrid_extractor.make_eval_dataset --out evaluation/data/hybrid_fonts --docs 60 --seed 4   # 書体も混ぜた版

評価用の synth_all（evaluation/synth_doc.py）は作りがそろっていて、ルールでラベルが全部当たる。
こちらは見た目をばらつかせた文書（番号の無い見出し、記号の違う箇条書き、キャプションが下の表…）なので、差が見える。
学習・検証・テスト（seed 0〜2）と重ならない seed にすること。ただし作り方は学習データと同じなので、
AI に少し有利（本物のPDFでの確認は evaluation.import_dataset で別にする）。

出力の形は evaluation/make_dataset.py と同じ（manifest.json、cases/<id>/input.pdf・gt.labeled.json・page-<n>.png）。
"""

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

from common.ir import save_ir
from evaluation.ground_truth import build_ground_truth, compile_tex
from evaluation.images import SITE_DPI, render_pages
from hybrid_extractor.doc_gen import add_variant_args, document_files, random_document, variant_from_args


def make_case(rng: random.Random, lang: str, case_dir: Path, variant=None) -> dict:
    tex, blocks, style = random_document(rng, lang, variant)
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = compile_tex(tex, Path(tmp), document_files(blocks))
        doc = build_ground_truth(str(pdf), blocks, source="input.pdf", two_column=style.two_column)
        shutil.copy(pdf, case_dir / "input.pdf")
    render_pages(str(case_dir / "input.pdf"), case_dir, SITE_DPI)
    (case_dir / "source.tex").write_text(tex, encoding="utf-8")
    save_ir(doc, str(case_dir / "gt.labeled.json"))
    labels = {el.label for page in doc.pages for el in page.elements}
    s = asdict(style)
    return {
        "lang": lang,
        "pages": len(doc.pages),
        "columns": 2 if style.two_column else 1,
        "doc_class": s["cls"],
        "numbered_headings": s["numbered"],
        "gothic_body": s["gothic_body"],
        "text_font": s["text_font"],
        "t1": s["t1"],
        "mono_font": s["mono_font"],
        "ja_font": s["ja_font"],
        "has_title": "title" in labels,
        "has_list": "list" in labels,
        "has_table": "table" in labels,
        "has_figure": "figure" in labels,
        "has_math": "math" in labels,
        "has_code": "code" in labels,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="評価サイト用の、見た目をばらつかせたデータセットを作る")
    parser.add_argument("--out", required=True, help="例: evaluation/data/hybrid_varied")
    parser.add_argument("--docs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3, help="学習データ（0〜2）と重ならない seed")
    add_variant_args(parser)
    args = parser.parse_args(argv)
    variant = variant_from_args(args)

    out = Path(args.out)
    cases = []
    for i in range(args.docs):
        case_id = f"{i:04d}"
        lang = ["ja", "en"][i % 2]
        rng = random.Random(args.seed * 1_000_003 + i)
        try:
            tags = make_case(rng, lang, out / "cases" / case_id, variant)
        except subprocess.CalledProcessError:
            print(f"{case_id}: TeXのコンパイルに失敗したので飛ばす")
            continue
        cases.append({"id": case_id, "tags": tags})
    manifest = {
        "name": out.name,
        "created": date.today().isoformat(),
        "generator": {"script": "hybrid_extractor.make_eval_dataset", "seed": args.seed, "variant": asdict(variant)},
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)}件を {out} に作った")


if __name__ == "__main__":
    main()
