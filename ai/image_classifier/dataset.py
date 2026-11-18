"""
image_dataset の画像・正解と、そのOCR結果（run_ocr.py で <dir>/ocr/ に保存したもの）から、
ページ単位の学習データを作る。

1サンプル = 1ページ = (全要素の特徴 (N, F), 画像 (N, H, W), 正解ラベル (N,))。

正解ラベルの付け方: OCRの段落と正解のブロックは区切り方が違うことがある（OCRが2つのブロックを1つにまとめる、など）。
そこで、OCRの段落の面積の半分以上を占める正解のブロックがあれば、そのラベルを正解とする。
無ければ（複数のブロックにまたがる段落など）学習・採点から外す（IGNORE）。

    python -m image_classifier.dataset image_classifier/data/train image_classifier/data/val
"""

import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from common.ir import LABELS, load_ir
from image_classifier.features import page_inputs

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
IGNORE = -100  # CrossEntropyLoss の ignore_index の既定値と同じ
MIN_OVERLAP = 0.5


def overlap_ratio(inner: tuple, outer: tuple) -> float:
    """inner の面積のうち、outer と重なっている割合"""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    w, h = min(ix1, ox1) - max(ix0, ox0), min(iy1, oy1) - max(iy0, oy0)
    area = (ix1 - ix0) * (iy1 - iy0)
    return max(0.0, w) * max(0.0, h) / area if area > 0 else 0.0


def target_labels(ocr_bboxes: list[tuple], truth_elements: list) -> list[int]:
    labels = []
    for bbox in ocr_bboxes:
        best = max(truth_elements, key=lambda g: overlap_ratio(bbox, g.bbox), default=None)
        ok = best is not None and best.label and overlap_ratio(bbox, best.bbox) >= MIN_OVERLAP
        labels.append(LABEL_TO_ID[best.label] if ok else IGNORE)
    return labels


def _page_sample(args: tuple[str, str]) -> dict | None:
    data_dir, truth_path = args
    ocr_path = os.path.join(data_dir, "ocr", os.path.basename(truth_path))
    if not os.path.exists(ocr_path):
        return None
    truth, ocr = load_ir(truth_path), load_ir(ocr_path)
    page, gt_page = ocr.pages[0], truth.pages[0]
    if not page.elements:
        return None
    features, crops = page_inputs(page, os.path.join(data_dir, "images", truth.source))
    return {
        "name": os.path.basename(truth_path),
        "style": truth.params.get("style", "-"),
        "features": features,
        "crops": crops,
        "labels": torch.tensor(target_labels([el.bbox for el in page.elements], gt_page.elements)),
    }


def build_cache(data_dir: Path, workers: int = 4) -> Path:
    truth_paths = sorted(glob.glob(os.path.join(data_dir, "truth", "*.ir.json")))
    with Pool(workers) as pool:
        samples = [s for s in pool.map(_page_sample, [(str(data_dir), p) for p in truth_paths], chunksize=16) if s]
    cache = Path(data_dir) / "pages.pt"
    torch.save(samples, cache)
    n = sum(len(s["labels"]) for s in samples)
    ignored = sum((s["labels"] == IGNORE).sum().item() for s in samples)
    print(f"{data_dir}: {len(samples)} ページ, OCRの段落 {n} 個（正解が決まらず外した {ignored} 個）")
    return cache


class PageDataset(Dataset):
    def __init__(self, data_dir: Path):
        cache = Path(data_dir) / "pages.pt"
        self.pages = torch.load(cache if cache.exists() else build_cache(data_dir))

    def __len__(self) -> int:
        return len(self.pages)

    def __getitem__(self, index: int):
        page = self.pages[index]
        return page["features"], page["crops"], page["labels"]


def collate(batch):
    """ページごとに要素数が違うので、一番多いページに合わせて埋める。

    戻り値: features (B, N, F), crops (B, N, H, W) float 0〜1, labels (B, N)（埋めた位置は IGNORE）,
    padding_mask (B, N)（埋めた位置が True）
    """
    features, crops, labels = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    features = pad_sequence(features, batch_first=True)
    crops = pad_sequence(crops, batch_first=True).float() / 255.0
    labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE)
    padding_mask = torch.arange(features.size(1))[None, :] >= lengths[:, None]
    return features, crops, labels, padding_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dirs", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for data_dir in args.data_dirs:
        build_cache(data_dir, args.workers)


if __name__ == "__main__":
    main()
