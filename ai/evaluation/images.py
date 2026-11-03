"""
PDFのページを画像（PNG）にする。

- 画像を入力にするモデル（OCR）に渡す（models.py）
- 評価サイトの「ケース詳細」で、正解と予測の枠を重ねる下絵にする（make_dataset.py が各ケースに保存する）

使い方（作り直さずに、今あるデータセットへ画像だけ足すとき）:
    python -m evaluation.images evaluation/data/synth_v1
"""
import argparse
import json
from pathlib import Path

import pymupdf

SITE_DPI = 100      # サイトに出す画像の解像度（A4 で 827 × 1170 ピクセル）
MODEL_DPI = 200     # OCRに渡す画像の解像度の既定値（models.json の params の "dpi" で変えられる）


def page_image_name(page_no: int) -> str:
    """ページ番号（1始まり）の画像のファイル名。サーバーの窓口もこの名前で探す"""
    return f"page-{page_no}.png"


def render_pages(pdf_path: str, out_dir: Path, dpi: int) -> list[Path]:
    """PDFの全ページを out_dir/page-<n>.png にして、画像のパスをページ順に返す"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            path = out_dir / page_image_name(i + 1)
            page.get_pixmap(dpi=dpi).save(str(path))
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="データセットの各ケースに、ページの画像（サイト用）を足す")
    parser.add_argument("dataset", help="データセットのフォルダ（例: evaluation/data/synth_v1）")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        case_dir = dataset_dir / "cases" / case["id"]
        render_pages(str(case_dir / "input.pdf"), case_dir, SITE_DPI)
    print(f"{len(manifest['cases'])}件に画像を足した")


if __name__ == "__main__":
    main()
