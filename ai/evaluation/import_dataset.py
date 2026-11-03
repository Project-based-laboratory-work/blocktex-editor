"""
本物のPDF（論文・資料など）を、評価のデータセットに取り込む。

    python -m evaluation.import_dataset --out evaluation/data/real_v1 papers/a.pdf papers/b.pdf
    python -m evaluation.import_dataset --out evaluation/data/real_v1 --draft rule_based papers/*.pdf

正解は、PDFの隣の <名前>.labeled.json（ラベル付きIR）を使う。人が付けた正解には label_source: "manual" を付ける。
正解が無いPDFは、--draft に登録したモデル（models.json の id）の出力を下書きの正解にする（label_source: "prelabel"）。
下書きは、そのままでは「そのモデル自身の答え」なので、人が直してから使う（直した要素は label_source を "manual" にする）。
tags の label_source で、下書きのままのケースを分けて見られる。

出力の形は make_dataset.py と同じ（manifest.json、cases/<id>/input.pdf・gt.labeled.json・page-<n>.png）。
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

from common.ir import IRDocument, load_ir, save_ir, validate
from evaluation.images import SITE_DPI, render_pages
from evaluation.models import load_registry, predict

# tags に has_<label> として残すブロックの種類（make_dataset の tags と同じ名前になるものを選ぶ）
TAGGED_LABELS = ["list", "title", "table", "figure", "math", "code"]


def guess_lang(doc: IRDocument) -> str:
    """正解の文字のうち、ASCII以外（日本語など）が3割以上なら "ja"、それ以外は "en" """
    text = "".join(el.text or "" for page in doc.pages for el in page.elements)
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return "en"
    return "ja" if sum(not c.isascii() for c in chars) / len(chars) >= 0.3 else "en"


def main_label_source(doc: IRDocument) -> str:
    """正解の要素の label_source のうち、一番多いもの（無ければ "manual" とみなす）"""
    sources: dict[str, int] = {}
    for page in doc.pages:
        for el in page.elements:
            key = el.label_source or "manual"
            sources[key] = sources.get(key, 0) + 1
    return max(sources, key=lambda k: sources[k]) if sources else "manual"


def draft_ground_truth(model_id: str, pdf_path: Path) -> IRDocument:
    """登録したモデルの出力を、下書きの正解にする（全部の要素に label_source: "prelabel"）"""
    doc = predict(load_registry()[model_id], str(pdf_path))
    for page in doc.pages:
        for el in page.elements:
            el.label_source = "prelabel"
    doc.structurer = f"draft:{model_id}"
    return doc


def import_case(pdf_path: Path, case_dir: Path, draft: str | None) -> dict:
    """1つのPDFを case_dir に取り込み、tags を返す。正解が無く、--draft も無ければ FileNotFoundError"""
    gt_path = pdf_path.with_suffix(".labeled.json")
    if gt_path.exists():
        gt = load_ir(str(gt_path), check=True)
    elif draft is not None:
        gt = draft_ground_truth(draft, pdf_path)
    else:
        raise FileNotFoundError(f"{gt_path} が無い（--draft <モデル> で下書きを作れる）")
    problems = validate(gt)
    if problems:
        raise ValueError(f"{gt_path} の形式に問題がある:\n" + "\n".join(problems))

    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdf_path, case_dir / "input.pdf")
    render_pages(str(case_dir / "input.pdf"), case_dir, SITE_DPI)
    gt.source = "input.pdf"
    save_ir(gt, str(case_dir / "gt.labeled.json"))

    labels = {el.label for page in gt.pages for el in page.elements}
    tags: dict = {"lang": guess_lang(gt), "pages": len(gt.pages), "label_source": main_label_source(gt)}
    for label in TAGGED_LABELS:
        tags[f"has_{label}"] = label in labels
    return tags


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本物のPDFを、評価のデータセットに取り込む")
    parser.add_argument("pdfs", nargs="+", help="取り込むPDF。正解は隣の <名前>.labeled.json")
    parser.add_argument("--out", required=True, help="データセットのフォルダ（例: evaluation/data/real_v1）")
    parser.add_argument("--draft", help="正解が無いPDFの下書きを作るモデル（models.json の id。例: rule_based）")
    args = parser.parse_args(argv)

    out = Path(args.out)
    cases = []
    for n, pdf in enumerate(sorted(Path(p) for p in args.pdfs)):
        case_id = f"{n:04d}"
        try:
            tags = import_case(pdf, out / "cases" / case_id, args.draft)
        except (FileNotFoundError, ValueError) as err:
            print(f"{pdf}: 飛ばす（{err}）")
            continue
        cases.append({"id": case_id, "tags": tags, "source": pdf.name})
        print(f"{case_id}: {pdf.name}（正解: {tags['label_source']}）")

    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": out.name,
        "created": date.today().isoformat(),
        "generator": {"script": "evaluation.import_dataset", "draft": args.draft},
        "cases": cases,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)}件を {out} に取り込んだ")


if __name__ == "__main__":
    main()
