"""
ラベル付きPDF（build_dataset.py の出力）から、ページ単位の学習データを作る。

1サンプル = 1ページ = (そのページの全テキストブロックの特徴 (N, F), 正解ラベル (N,))。
ページ単位にしているのは、Transformerに同じページの他のブロックを参照させるため。

PDFからの抽出（特に pdfplumber の表検出）は遅いので、最初に1回だけ計算してファイルに保存する。

    python -m block_classifier.dataset block_classifier/data/train
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import pymupdf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from block_classifier.features import page_features
from rule_based.labels import FINE_LABELS
from rule_based.pdf_extract import PageBlocks, extract
from rule_based.rule_based_classifier import (
    CAPTION_MAX_CHARS, CAPTION_PATTERN, HEADING_MAX_CHARS, HEADING_SIZE_RATIO, OVERLAP_RATIO_THRESHOLD,
    _body_font_size, _overlap_ratio,
)  # fmt: skip

LABELS = FINE_LABELS  # 並びの番号がクラスの番号になる（rule_based/labels.py の Label の順）
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100  # パディングの位置。CrossEntropyLoss の ignore_index の既定値と同じ


def rule_based_labels(page: PageBlocks) -> list[str]:
    """チームのルールベース分類（rule_based_classifier.classify_page）を、テキストブロック1つずつの
    予測として出す（比較用）。表・図の領域に吸収されるテキストは table / figure とする。"""
    body_size = _body_font_size(page.text_blocks)
    labels = []
    for block in page.text_blocks:
        if any(_overlap_ratio(block.bbox, t.bbox) >= OVERLAP_RATIO_THRESHOLD for t in page.table_blocks):
            labels.append("table")
        elif any(_overlap_ratio(block.bbox, f.bbox) >= OVERLAP_RATIO_THRESHOLD for f in page.figure_blocks):
            labels.append("figure")
        elif CAPTION_PATTERN.match(block.text) and len(block.text) <= CAPTION_MAX_CHARS:
            labels.append("caption")
        elif (block.font_size >= body_size * HEADING_SIZE_RATIO or block.bold) and len(block.text) <= HEADING_MAX_CHARS:
            labels.append("heading")
        else:
            labels.append("paragraph")
    return labels


def _doc_pages(label_path: Path) -> list[dict]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    pdf_path = label_path.parent.parent / "pdfs" / data["pdf"]
    with pymupdf.open(pdf_path) as doc:
        page_height = doc[0].rect.height

    labels = iter(data["blocks"])
    pages = []
    for page in extract(str(pdf_path)):
        if not page.text_blocks:
            continue
        page_labels = []
        for block in page.text_blocks:
            labeled = next(labels)
            # ラベルは同じ抽出処理の出力に同じ順番で付けているので、位置が一致するはず
            assert labeled["page"] == block.page and tuple(labeled["bbox"]) == tuple(block.bbox), label_path
            page_labels.append(LABEL_TO_ID[labeled["label"]])
        pages.append({
            "doc": data["pdf"], "page": page.page,
            "features": page_features(page, page_height),
            "labels": torch.tensor(page_labels),
            "texts": [b.text for b in page.text_blocks],
            "rule_based": torch.tensor([LABEL_TO_ID[label] for label in rule_based_labels(page)]),
        })  # fmt: skip
    return pages


def build_cache(data_dir: Path, workers: int = 6) -> Path:
    label_paths = sorted((data_dir / "labels").glob("*.json"))
    with Pool(workers) as pool:
        pages = [page for doc in pool.map(_doc_pages, label_paths, chunksize=8) for page in doc]
    cache = data_dir / "pages.pt"
    torch.save(pages, cache)
    print(f"{data_dir}: {len(label_paths)} 文書 → {len(pages)} ページ, {sum(len(p['labels']) for p in pages)} ブロック")
    return cache


class PageDataset(Dataset):
    def __init__(self, data_dir: Path):
        cache = data_dir / "pages.pt"
        self.pages = torch.load(cache if cache.exists() else build_cache(data_dir))

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        page = self.pages[index]
        return page["features"], page["labels"]


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ページごとにブロック数が違うので、一番多いページに合わせて埋める。

    戻り値: features (B, N, F), labels (B, N)（埋めた位置は IGNORE）, padding_mask (B, N)（埋めた位置が True）
    """
    features, labels = zip(*batch)
    features = pad_sequence(features, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE)
    return features, labels, labels == IGNORE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for data_dir in args.data_dirs:
        build_cache(data_dir)


if __name__ == "__main__":
    main()
