"""
画像とLaTeXの組を、モデルに入力できるテンソルに変換する。

PyTorchのデータ読み込みは2段構え:
  - Dataset: 「i番目のサンプルを1件返す」だけを担当する（__len__ と __getitem__ を実装する）
  - DataLoader: Datasetから複数件取り出して「バッチ」にまとめる。まとめ方は collate_fn で指定する

画像は全サンプル同じ大きさ（IMAGE_HEIGHT × IMAGE_WIDTH）に揃えるので、そのまま積み重ねればよい。
一方、トークン列は数式ごとに長さが違うので、バッチ内で一番長いものに合わせて末尾を <pad> で埋める。
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from math_ocr.tokenizer import Vocab

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 320
MARGIN = 4


def preprocess(image: Image.Image) -> torch.Tensor:
    """
    任意サイズの数式画像 → (1, IMAGE_HEIGHT, IMAGE_WIDTH) の float テンソル。

    1. 白黒を反転して「背景=0, 文字=1」にする。余白を0で埋めたとき、それが「何も無い」の意味になる
    2. 文字の外側の余白を切り落とす
    3. 縦横比を保ったままキャンバスに収まるよう拡大縮小し、左寄せ・上下中央に置く
    """
    ink = 1.0 - np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    rows, cols = np.nonzero(ink > 0.05)
    if len(rows) == 0:
        return torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH)
    ink = ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    h, w = ink.shape
    scale = min((IMAGE_HEIGHT - 2 * MARGIN) / h, (IMAGE_WIDTH - 2 * MARGIN) / w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = Image.fromarray((ink * 255).astype(np.uint8)).resize((new_w, new_h), Image.BILINEAR)

    canvas = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
    top = (IMAGE_HEIGHT - new_h) // 2
    canvas[top : top + new_h, MARGIN : MARGIN + new_w] = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(canvas).unsqueeze(0)  # チャンネル次元を足す: (H, W) → (1, H, W)


def read_labels(data_dir: Path, split: str) -> list[tuple[str, str]]:
    """{split}.tsv を読み、(画像ファイル名, LaTeX) のリストを返す"""
    pairs = []
    with open(data_dir / f"{split}.tsv", encoding="utf-8") as f:
        for line in f:
            filename, latex = line.rstrip("\n").split("\t", 1)
            pairs.append((filename, latex))
    return pairs


class MathFormulaDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, vocab: Vocab, limit: int | None = None):
        self.image_dir = data_dir / "images"
        self.samples = read_labels(data_dir, split)[:limit]
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename, latex = self.samples[index]
        image = preprocess(Image.open(self.image_dir / filename))
        token_ids = torch.tensor(self.vocab.encode(latex), dtype=torch.long)
        return image, token_ids


class Collator:
    """DataLoaderの collate_fn。サンプルのリストを1つのバッチにまとめる。

    （関数ではなくクラスにしているのは、pad_id を持たせつつ、DataLoaderの num_workers>0 で
    別プロセスに渡せる（pickleできる）ようにするため）
    """

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        images, sequences = zip(*batch)
        images = torch.stack(images)  # (B, 1, H, W)
        sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_id)  # (B, T)
        return images, sequences
