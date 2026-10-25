"""
学習済みモデルで数式画像をLaTeXに変換する。

    python -m math_ocr.predict checkpoints/math_ocr/best.pt 画像1.png 画像2.png ...
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from math_ocr.dataset import preprocess
from math_ocr.model import MathOCRModel, ModelConfig
from math_ocr.tokenizer import Vocab


def load_model(checkpoint_path: Path) -> tuple[MathOCRModel, Vocab]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = MathOCRModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, Vocab(checkpoint["vocab"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()

    model, vocab = load_model(args.checkpoint)
    batch = torch.stack([preprocess(Image.open(path)) for path in args.images])
    for path, ids in zip(args.images, model.generate(batch)):
        print(f"{path}\t{vocab.decode(ids.tolist()[1:])}")


if __name__ == "__main__":
    main()
