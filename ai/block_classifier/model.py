"""
ブロック分類のモデル。比較のために2種類を用意する。

BlockMLP:
    ブロック1つの特徴だけを見て分類する（周りのブロックは見ない）。比較の基準。

PageTransformer:
    同じページの全ブロックを1つの列としてTransformerEncoderに入れ、self-attentionで
    ブロック同士を参照させてから分類する。

        特徴 (B, N, F) ─ Linear ─▶ (B, N, d) ─ TransformerEncoder ─▶ (B, N, d) ─ Linear ─▶ (B, N, クラス数)

    数式OCRのデコーダと違い、ここでは causal mask を使わない（全ブロックが互いを見てよい）。
    また、位置埋め込みも足さない。各ブロックの座標が特徴に入っているので、それ自体が位置の情報になる。

記号: B=ページ数（バッチ）, N=ページ内のブロック数, F=特徴の数, d=d_model
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    num_features: int
    num_classes: int
    kind: str = "transformer"  # "transformer" | "mlp"
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1


class BlockMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.num_features, d), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(d, config.num_classes),
        )  # fmt: skip

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # 各ブロックを独立に処理するので padding_mask は不要


class PageTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d = config.d_model
        self.input = nn.Sequential(nn.Linear(config.num_features, d), nn.ReLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d, config.nhead, 4 * d, config.dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.output = nn.Linear(d, config.num_classes)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.input(features)  # (B, N, d)
        # padding_mask (B, N): 埋めた位置（True）を、他のブロックから参照されないようにする
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.output(x)  # (B, N, クラス数)


def build_model(config: ModelConfig) -> nn.Module:
    return {"transformer": PageTransformer, "mlp": BlockMLP}[config.kind](config)
