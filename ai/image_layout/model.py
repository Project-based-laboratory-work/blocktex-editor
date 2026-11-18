"""
レイアウト解析（座標AI）のモデル。画像の各画素を「背景・見出し・段落・表・図…」に塗り分ける U-Net。

    画像 (B, 1, 768, 576)
      │ エンコーダ: 「畳み込み×2 → 半分に縮める」を繰り返し、広い範囲の手がかり（段組み・表の形）を集める
      ▼
    (B, 256, 48, 36)
      │ デコーダ: 2倍に広げ、エンコーダの同じ大きさの特徴（細かい位置の情報）をつなげて、畳み込み
      ▼
    (B, クラス数, 192, 144)   ← 入力の 1/4 の大きさで、画素ごとのクラスの点数

U-Net の「つなげる（skip connection）」のおかげで、縮めて失われた細かい位置（ブロックの端）を取り戻せる。
数式OCRのCNNとの違いは、最後に縮めたまま使うのではなく、もう一度広げて「画素ごと」に答えること。
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_classes: int
    base: int = 32  # 最初の段のチャンネル数。段ごとに2倍にする
    boundary: bool = False  # True なら、クラスの点数に加えて「ブロックのふち」の点数を1枚、最後のチャンネルに出す


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )  # fmt: skip


class LayoutUNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config.base
        # 768×576 →(stem 1/2)→ 384×288 → 1/4: 192×144 → 1/8: 96×72 → 1/16: 48×36
        self.stem = nn.Sequential(nn.Conv2d(1, c, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.enc1 = conv_block(c, c)  # 1/2
        self.enc2 = conv_block(c, 2 * c)  # 1/4
        self.enc3 = conv_block(2 * c, 4 * c)  # 1/8
        self.enc4 = conv_block(4 * c, 8 * c)  # 1/16
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = conv_block(8 * c, 4 * c)  # 1/8（広げたもの + enc3）
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = conv_block(4 * c, 2 * c)  # 1/4（広げたもの + enc2）
        self.head = nn.Conv2d(2 * c, config.num_classes + int(config.boundary), 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.stem(images))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        return self.head(d2)  # (B, クラス数 (+ふち1), 192, 144)
