"""
学習済みモデルでPDFのブロックを分類する。--draw を付けると、ラベルごとに色分けした枠を描いた
ページ画像を出力する（目で確認する用）。

    python -m block_classifier.predict block_classifier/checkpoints/transformer/best.pt report.pdf --draw out/
"""

import argparse
from pathlib import Path

import pymupdf
import torch

from block_classifier.features import page_features
from block_classifier.model import ModelConfig, build_model
from rule_based.pdf_extract import extract

COLORS = {
    "heading": (1, 0, 0), "paragraph": (0, 0, 1), "list": (0, 0.6, 0), "math": (0.7, 0, 0.7),
    "caption": (1, 0.5, 0), "table": (0, 0.7, 0.7), "figure": (0.6, 0.4, 0), "other": (0.5, 0.5, 0.5), "code": (0.9, 0.6, 0),
}  # fmt: skip


def load(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path)
    model = build_model(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["labels"]


@torch.no_grad()
def classify_pdf(model: torch.nn.Module, labels: list[str], pdf_path: Path) -> list[tuple[int, tuple, str, str]]:
    """PDFの全テキストブロックを分類し、(ページ, bbox, ラベル, テキスト) のリストを返す"""
    with pymupdf.open(pdf_path) as doc:
        page_height = doc[0].rect.height
    results = []
    for page in extract(str(pdf_path)):
        if not page.text_blocks:
            continue
        features = page_features(page, page_height).unsqueeze(0)  # 1ページ = バッチサイズ1
        padding_mask = torch.zeros(features.shape[:2], dtype=torch.bool)
        predicted = model(features, padding_mask)[0].argmax(-1)
        for block, label_id in zip(page.text_blocks, predicted.tolist()):
            results.append((block.page, block.bbox, labels[label_id], block.text))
    return results


def draw(pdf_path: Path, results: list[tuple[int, tuple, str, str]], out_dir: Path, dpi: int = 80) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        for page_no, bbox, label, _ in results:
            page = doc[page_no]
            page.draw_rect(pymupdf.Rect(bbox), color=COLORS[label], width=1.2)
            page.insert_text((bbox[0], bbox[1] - 1), label[:4], fontsize=5, color=COLORS[label])
        for page in doc:
            page.get_pixmap(dpi=dpi).save(out_dir / f"{pdf_path.stem}_p{page.number + 1}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--draw", type=Path, default=None, help="色分けしたページ画像の出力先")
    args = parser.parse_args()

    model, labels = load(args.checkpoint)
    results = classify_pdf(model, labels, args.pdf)
    for page_no, _, label, text in results:
        print(f"p{page_no + 1} [{label:<9}] {text[:50]!r}")
    if args.draw:
        draw(args.pdf, results, args.draw)
        print(f"→ {args.draw} に画像を出力")


if __name__ == "__main__":
    main()
