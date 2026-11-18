"""
画像の構造化器（分類）のモデル。

    要素ごとの数値の特徴 (B, N, F) ─ Linear ─┐
                                            ├─ 足す ─▶ (B, N, d) ─ TransformerEncoder ─▶ Linear ─▶ (B, N, クラス数)
    要素ごとの画像 (B, N, H, W) ─ CNN ──────┘

  - CNN: 要素の部分の画像から、太字・等幅・数式の記号・表の線などを読み取る。use_image=False にすると使わない
    （画像を見る効果を比べるため）
  - TransformerEncoder: 同じページの要素同士を参照させる（PDF版のブロック分類の PageTransformer と同じ）。
    causal mask は使わない（全要素が互いを見てよい）。位置は特徴の座標に入っているので、位置埋め込みも足さない

記号: B=ページ数（バッチ）, N=ページ内の要素数, F=特徴の数, d=d_model
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_features: int
    num_classes: int
    use_image: bool = True
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1


class CropEncoder(nn.Module):
    """(M, 1, 32, 256) → (M, d)。「畳み込み → BatchNorm → ReLU → MaxPool」を3回のあと、
    縦をまとめ、横は4つの区間に分けて平均する（左寄せの画像なので、文字の詰まり方の左右差を少し残す）"""

    def __init__(self, d_model: int):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )  # fmt: skip

        self.net = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), nn.AdaptiveAvgPool2d((1, 4)))
        self.out = nn.Linear(128 * 4, d_model)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(crops).flatten(1))


class PageClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.config = config
        self.features = nn.Sequential(nn.Linear(config.num_features, d), nn.ReLU(), nn.Linear(d, d))
        self.crops = CropEncoder(d) if config.use_image else None
        layer = nn.TransformerEncoderLayer(d, config.nhead, 4 * d, config.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.output = nn.Linear(d, config.num_classes)

    def forward(self, features: torch.Tensor, crops: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.features(features)  # (B, N, d)
        if self.crops is not None:
            B, N, H, W = crops.shape
            keep = ~padding_mask  # 埋めた位置の画像はCNNに通さない（計算の節約）
            image = torch.zeros_like(x)
            image[keep] = self.crops(crops[keep].unsqueeze(1))  # (要素数, d)
            x = x + image
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output(x)  # (B, N, クラス数)
