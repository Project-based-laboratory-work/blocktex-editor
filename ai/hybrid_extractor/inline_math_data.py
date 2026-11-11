"""文中の数式の専門家（AI版）の学習データを作る。段落の1文字ずつに「数式か」の正解を付ける（系列ラベリング）。

    python -m hybrid_extractor.inline_math_data --out hybrid_extractor/data/inline_train --docs 600 --seed 10
    python -m hybrid_extractor.inline_math_data --out hybrid_extractor/data/inline_val --docs 120 --seed 11
    python -m hybrid_extractor.inline_math_data --out hybrid_extractor/data/inline_test --docs 120 --seed 12

文書の作り方と正解の付け方は、比較実験（inline_math_eval.py）と同じ（$...$ の前後の目印）。
比較実験は seed 7 なので、学習・検証・テストはそれと重ならない seed にする。

出力（<out>/）:
    docs/<id>.json   ブロックごとの文字の並びと正解（下の形）
    pdfs/<id>.pdf    元のPDF
    manifest.json    文書の一覧と、文字の数・数式の文字の割合

docs/<id>.json の形:
    {"id": "0000", "lang": "ja", "blocks": [
        {"page": 1, "page_width": 595.28, "page_height": 841.89, "bbox": [...],      ← ブロックの枠（0〜1）
         "font_size": 10.0,                                                         ← ブロックの文字サイズ（一番多いもの）
         "chars": [
            {"c": "x", "font": "CMMI10", "size": 10.0, "bbox": [...], "baseline": 0.123,  ← bbox・baseline は 0〜1
             "space": false,      ← 抽出器が入れた空白・改行（PDFの文字ではない）
             "math": 1,           ← 正解: 数式の文字なら 1
             "math_id": 3,        ← 正解: どの数式か（数式でなければ null）。隣の文字と同じ数式かが分かる
             "rule": 1},          ← ルール版の専門家の答え
            ...]}]}

文字の正解: 目印の間の文字なら 1。抽出器が入れた空白（"space": true）は、前後の文字が同じ数式なら 1。
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common.ir import normalize_bbox
from hybrid_extractor.experts import INLINE_MATH_EXPERTS
from hybrid_extractor.extract import extract_blocks
from hybrid_extractor.inline_math_eval import glyph_key, make_case


def char_labels(chars: list[dict], owner: dict[tuple, int]) -> list[int | None]:
    """文字ごとの数式の番号（数式でなければ None）。PDFの文字は目印から、空白は前後の文字から決める"""
    ids: list[int | None] = [owner.get(glyph_key(ch)) if ch["bbox"] is not None else None for ch in chars]
    for k, ch in enumerate(chars):
        if ch["bbox"] is not None and not ch["c"].isspace():
            continue
        before = next((ids[j] for j in range(k - 1, -1, -1) if chars[j]["bbox"] is not None and not chars[j]["c"].isspace()), None)
        after = next((ids[j] for j in range(k + 1, len(chars)) if chars[j]["bbox"] is not None and not chars[j]["c"].isspace()), None)
        ids[k] = before if before is not None and before == after else None
    return ids


def make_doc(args: tuple[int, int, str, str]) -> dict | None:
    index, seed, lang, out = args
    case = make_case((index, seed, lang, out))
    if case is None:
        return None
    pdf = Path(out) / "pdfs" / f"{case['id']}.pdf"
    owner = {tuple(key): n for n, gt in enumerate(case["gts"]) for key in gt["glyph_keys"]}
    rule = INLINE_MATH_EXPERTS["rule"]
    blocks = []
    for page in extract_blocks(str(pdf)):
        w, h = page.width, page.height
        for b in page.blocks:
            if b.kind != "text" or not b.chars:
                continue
            ids = char_labels(b.chars, owner)
            predicted = [0] * len(b.chars)
            for start, end in rule.find(b.chars):
                predicted[start:end] = [1] * (end - start)
            chars = []
            for ch, math_id, r in zip(b.chars, ids, predicted):
                synthetic = ch["bbox"] is None
                chars.append({
                    "c": ch["c"],
                    "font": ch["font"],
                    "size": round(ch["size"], 2),
                    "bbox": None if synthetic else list(normalize_bbox(ch["bbox"], w, h)),
                    "baseline": None if synthetic else round(ch["baseline"] / h, 5),
                    "space": synthetic,
                    "math": int(math_id is not None),
                    "math_id": math_id,
                    "rule": r,
                })
            blocks.append({"page": page.page, "page_width": w, "page_height": h,
                           "bbox": list(normalize_bbox(b.bbox, w, h)), "font_size": b.font_size, "chars": chars})
    doc = {"id": case["id"], "lang": lang, "blocks": blocks}
    (Path(out) / "docs" / f"{case['id']}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    counts = Counter()
    for block in blocks:
        for ch in block["chars"]:
            counts["chars"] += 1
            counts["math"] += ch["math"]
            counts["rule_correct"] += ch["math"] == ch["rule"]
    return {"id": case["id"], "lang": lang, "n_math": len(case["gts"]), **counts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="文中の数式の専門家（AI版）の学習データを作る")
    parser.add_argument("--out", required=True)
    parser.add_argument("--docs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10, help="比較実験（7）と重ならない種にする")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed, ["ja", "en"][i % 2], str(out)) for i in range(args.docs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        docs = [d for d in pool.map(make_doc, jobs) if d is not None]
    total = Counter()
    for d in docs:
        total.update({k: d[k] for k in ("chars", "math", "rule_correct", "n_math")})
    manifest = {"seed": args.seed, "docs": docs, "total": dict(total)}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(docs)}文書、文字 {total['chars']}、うち数式の文字 {total['math']}（{total['math'] / max(1, total['chars']):.1%}）、"
          f"文中の数式 {total['n_math']}個")
    print(f"ルール版の専門家の、1文字ずつの正解率: {total['rule_correct'] / max(1, total['chars']):.3f}")


if __name__ == "__main__":
    main()
