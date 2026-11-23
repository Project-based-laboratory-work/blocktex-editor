"""
文中の装飾（太字・斜体・下線）を見分けるAI。1行の画像の、横の位置ごとに「太字か・斜体か・下線か」を当てる。

    1行の画像 (B, 1, 40, W)
      │ CNN: 文字の太さ・傾き・下の線を見る。縦を1つにまとめ、横は 1/4 にする
      ▼
    (B, C, W/4)
      │ 双方向GRU: 左右両方から読み、前後の文字と比べる（太字かどうかは、まわりの文字より太いかで決まる）
      ▼
    (B, 3, W/4)   ← 位置ごとの [太字, 斜体, 下線] の点数（それぞれ独立。太字かつ下線もありうる）
"""

from dataclasses import dataclass

import torch
from torch import nn

MARKS = ("bold", "italic", "underline")


@dataclass
class ModelConfig:
    base: int = 32
    hidden: int = 128


def conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class DecorationTagger(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config.base
        self.cnn = nn.Sequential(
            conv(1, c), conv(c, c), nn.MaxPool2d(2),  # 20 × W/2
            conv(c, 2 * c), conv(2 * c, 2 * c), nn.MaxPool2d(2),  # 10 × W/4
            conv(2 * c, 4 * c), conv(4 * c, 4 * c), nn.MaxPool2d((2, 1)),  # 5 × W/4
            conv(4 * c, 4 * c),
        )
        self.gru = nn.GRU(8 * c, config.hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * config.hidden, len(MARKS))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f = self.cnn(images)  # (B, C, 5, W/4)
        f = torch.cat([f.mean(2), f.amax(2)], dim=1)  # 縦をまとめる（平均と最大）→ (B, 2C, W/4)
        x, _ = self.gru(f.transpose(1, 2))
        return self.out(x).transpose(1, 2)  # (B, 3, W/4)
