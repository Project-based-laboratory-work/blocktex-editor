"""学習データを作る。1文書 = 1つのJSON（抽出器のブロック + 正解のラベル + ルールの答え）。

    python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/train --docs 1000 --seed 0
    python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/val --docs 200 --seed 1

流れ（1文書ごと）:
    doc_gen でランダムな文書の TeX を作る
      → PDF にする（uplatex → dvipdfmx。目印を残す）
      → 目印から正解のラベル付きIRを作る（evaluation.ground_truth）
      → 自分の抽出器（extract.py）でブロックを取り出す
      → ブロックごとに、一番重なる正解の要素のラベルを付ける（ブロックの面積の半分以上が重なるときだけ）
      → ルール分類器（rules.py）の答えも一緒に保存する（AIと比べるため）

出力（<out>/）:
    docs/<id>.json   ブロックの一覧（下の形）
    pdfs/<id>.pdf    元のPDF（目で確かめる用）
    manifest.json    作った文書の一覧と、ラベルごとの数

docs/<id>.json の形:
    {
      "id": "0000", "lang": "ja", "style": {...},
      "pages": [
        {"page": 1, "width": 595.28, "height": 841.89,
         "blocks": [
           {"kind": "text", "bbox": [0.13, 0.16, 0.5, 0.18],   ← ページの幅・高さで割った 0〜1
            "text": "1 はじめに", "font_size": 14.5, "bold": true, "font": "HaranoAjiGothic-Medium",
            "mono_ratio": 0.0, "math_ratio": 0.0, "n_rows": 1, "first_indent": 0.0, "list_head": false,
            "label": "heading", "level": 1, "overlap": 0.98, "rule_label": "heading"},
           ...
         ]}
      ]
    }
    label が null のブロックは、どの正解ともあまり重ならなかったもの（学習・評価には使わない）。
"""

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from common.ir import IRDocument, normalize_bbox
from evaluation.ground_truth import build_ground_truth, compile_tex
from hybrid_extractor.doc_gen import document_files, random_document
from hybrid_extractor.extract import extract_blocks
from hybrid_extractor.rules import label_pages

MIN_OVERLAP = 0.5  # ブロックの面積のこの割合以上が正解の要素と重なっていれば、その要素のラベルを付ける


def overlap_ratio(inner: tuple, outer: tuple) -> float:
    """inner の面積のうち、outer と重なっている割合（0〜1）"""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return w * h / area if area > 0 else 0.0


def match_labels(gt: IRDocument, page_no: int, bbox: tuple) -> tuple[str | None, int | None, float]:
    """ブロック（正規化した bbox）に、一番重なる正解の要素のラベル・見出しレベル・重なりの割合を返す"""
    best, best_ratio = None, 0.0
    for el in gt.pages[page_no - 1].elements:
        r = overlap_ratio(bbox, el.bbox)
        if r > best_ratio:
            best, best_ratio = el, r
    if best is None or best_ratio < MIN_OVERLAP:
        return None, None, round(best_ratio, 3)
    return best.label, best.level, round(best_ratio, 3)


def make_doc(args: tuple[int, int, str, str]) -> dict | None:
    """1文書分を作って保存し、manifest に載せる情報を返す。コンパイルに失敗したら None"""
    index, seed, lang, out = args
    doc_id = f"{index:05d}"
    rng = random.Random(seed * 1_000_003 + index)
    tex, blocks, style = random_document(rng, lang)
    out_dir = Path(out)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pdf = compile_tex(tex, Path(tmp), document_files(blocks))
        except subprocess.CalledProcessError:
            return None
        gt = build_ground_truth(str(pdf), blocks, source=f"{doc_id}.pdf", two_column=style.two_column)
        pages = extract_blocks(str(pdf))
        rule_labels = label_pages(pages)
        shutil.copy(pdf, out_dir / "pdfs" / f"{doc_id}.pdf")

    record_pages = []
    counts: Counter = Counter()
    for page, labeled in zip(pages, rule_labels):
        records = []
        for block, rule_label in labeled:
            bbox = normalize_bbox(block.bbox, page.width, page.height)
            label, level, ratio = match_labels(gt, page.page, bbox)
            counts[label or "(none)"] += 1
            records.append({
                "kind": block.kind,
                "bbox": list(bbox),
                "text": block.text,
                "font_size": block.font_size,
                "bold": block.bold,
                "font": block.font,
                "mono_ratio": block.mono_ratio,
                "math_ratio": block.math_ratio,
                "n_rows": block.n_rows,
                "first_indent": block.first_indent,
                "list_head": block.list_head,
                "label": label,
                "level": level,
                "overlap": ratio,
                "rule_label": rule_label,
            })
        record_pages.append({"page": page.page, "width": page.width, "height": page.height, "blocks": records})

    doc = {"id": doc_id, "lang": lang, "style": asdict(style), "pages": record_pages}
    (out_dir / "docs" / f"{doc_id}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return {"id": doc_id, "lang": lang, "pages": len(record_pages), "counts": dict(counts)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="hybrid_extractor の学習データを作る")
    parser.add_argument("--out", required=True, help="出力先（例: hybrid_extractor/data/train）")
    parser.add_argument("--docs", type=int, default=100, help="作る文書の数")
    parser.add_argument("--seed", type=int, default=0, help="同じ seed なら同じデータになる。train と val は変える")
    parser.add_argument("--workers", type=int, default=8, help="同時に作る数（CPUのコア数まで）")
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed, ["ja", "en"][i % 2], str(out)) for i in range(args.docs)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, result in enumerate(pool.map(make_doc, jobs), start=1):
            if result is None:
                print(f"{jobs[k - 1][0]:05d}: TeXのコンパイルに失敗したので飛ばす")
            else:
                results.append(result)
            if k % 50 == 0:
                print(f"{k}/{args.docs}")

    total: Counter = Counter()
    for r in results:
        total.update(r["counts"])
    manifest = {"seed": args.seed, "docs": results, "label_counts": dict(total.most_common())}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(results)}文書を {out} に作った")
    for label, n in total.most_common():
        print(f"  {label:10} {n}")


if __name__ == "__main__":
    main()
