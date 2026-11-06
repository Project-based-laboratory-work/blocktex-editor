"""
表AIのモデル。表の画像から、縦の位置ごとの「行の外らしさ」と、横の位置ごとの「列の外らしさ」を出す。

    画像 (B, 1, 768, 768)
      │ CNN: 畳み込みで、文字・罫線・空白の形を見る。1/4 に縮める
      ▼
    (B, C, 192, 192)
      ├─ 横方向に平均・最大をとる → (B, 2C, 192)  縦の位置ごとの特徴（その高さの横一列に何があるか）
      │    → 1次元の畳み込み ＋ 双方向GRU（上下の遠くまで見る） → 2倍に広げる → (B, 384)  行の外らしさ
      └─ 縦方向に平均・最大をとる → (B, 2C, 192)
           → 同じ形の別の部品 → (B, 384)  列の外らしさ

双方向GRU は、並びを上から下・下から上の両方向に読みながら、前後の情報を覚えていく部品。
「この空白の次の行では、他の列にも文字が始まる → ここは行の境目」のような、離れた位置の関係を使える。
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    base: int = 32  # 最初の段のチャンネル数
    hidden: int = 128  # 1次元の部分の大きさ


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )  # fmt: skip


class AxisHead(nn.Module):
    """(B, 2C, L) の並び → (B, 2L) の点数"""

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.ReLU(inplace=True),
        )  # fmt: skip
        self.gru = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.up = nn.ConvTranspose1d(hidden, hidden // 2, 2, stride=2)
        self.out = nn.Conv1d(hidden // 2, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x, _ = self.gru(x.transpose(1, 2))
        return self.out(torch.relu(self.up(x.transpose(1, 2))))[:, 0]


class TableSplitter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config.base
        self.cnn = nn.Sequential(
            conv_block(1, c, stride=2),  # 1/2
            conv_block(c, 2 * c, stride=2),  # 1/4
            conv_block(2 * c, 4 * c),
        )
        self.rows = AxisHead(8 * c, config.hidden)
        self.cols = AxisHead(8 * c, config.hidden)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.cnn(images)  # (B, C, H/4, W/4)
        along_y = torch.cat([f.mean(3), f.amax(3)], dim=1)  # 横方向にまとめる → 縦の位置ごと
        along_x = torch.cat([f.mean(2), f.amax(2)], dim=1)
        return self.rows(along_y), self.cols(along_x)  # (B, H/2), (B, W/2)
