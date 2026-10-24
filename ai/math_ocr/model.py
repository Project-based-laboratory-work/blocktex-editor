"""
数式画像 → LaTeXトークン列 を出力するエンコーダ・デコーダモデル。

    画像 (B,1,64,320)
      │ ConvEncoder: CNNで画像を縮めながら特徴を取り出す
      ▼
    特徴マップ (B,d,4,40) ── 4×40=160マスを「160個のベクトルの列」とみなす
      │ + 2次元位置埋め込み（どのマスのベクトルかを教える）
      │ TransformerEncoder: マス同士が互いを参照し、画像全体の文脈を取り込む
      ▼
    memory (B,160,d)
      │
      │       これまでに出力したトークン (B,T)
      │         │ トークン埋め込み + 位置埋め込み
      ▼         ▼
    TransformerDecoder:
      - self-attention: これまでのトークン同士を参照（未来のトークンはcausal maskで見えなくする）
      - cross-attention: 画像のどのマスを見るべきかを決めて memory から情報を取る
      ▼
    次のトークンの確率分布 (B,T,語彙数)

記号: B=バッチサイズ, T=トークン列の長さ, d=d_model（各ベクトルの次元数）, V=語彙数
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 3
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 64  # 出力トークン列の最大長（<bos>/<eos> 込み）


class ConvEncoder(nn.Module):
    """(B, 1, 64, 320) → (B, d_model, 4, 40)

    「畳み込み → BatchNorm → ReLU → MaxPool」を4回繰り返す。MaxPoolのたびに縦横が半分になり、
    1マスが元画像のより広い範囲を表すようになる。最後だけ横を縮めない（pool=(2,1)）のは、
    数式は横に長く、横方向の細かい位置（どの文字が何番目か）を残したいため。
    """

    def __init__(self, d_model: int):
        super().__init__()

        def block(in_ch: int, out_ch: int, pool: int | tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                # bias=False: 直後のBatchNormが平均を引くので、biasを足しても打ち消されて無意味
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.net = nn.Sequential(
            block(1, 32, 2),  # → (B, 32, 32, 160)
            block(32, 64, 2),  # → (B, 64, 16, 80)
            block(64, 128, 2),  # → (B, 128, 8, 40)
            block(128, d_model, (2, 1)),  # → (B, d, 4, 40)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class MathOCRModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        # --- エンコーダ側 ---
        self.cnn = ConvEncoder(d)
        # Transformerは「列の中の順番」を知らない（入力を並べ替えても計算が変わらない）ので、
        # 位置の情報を足してやる必要がある。行と列の埋め込みを別々に学習し、足し合わせて2次元の位置を表す
        self.row_embed = nn.Embedding(16, d)
        self.col_embed = nn.Embedding(128, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout,
            batch_first=True,  # 入力を (B, 列の長さ, d) の順で受け取る（デフォルトは (列の長さ, B, d)）
            norm_first=True,  # Pre-LN: LayerNormを各サブ層の前に置く。深くしても学習が安定しやすい
        )  # fmt: skip
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_encoder_layers, norm=nn.LayerNorm(d), enable_nested_tensor=False
        )

        # --- デコーダ側 ---
        self.token_embed = nn.Embedding(config.vocab_size, d, padding_idx=config.pad_id)
        self.pos_embed = nn.Embedding(config.max_len, d)
        decoder_layer = nn.TransformerDecoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers, norm=nn.LayerNorm(d))
        self.output = nn.Linear(d, config.vocab_size)  # 各位置のベクトル → 語彙ごとのスコア（logits）

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 1, H, W) → memory (B, S, d)。S = 特徴マップのマス数（4×40=160）"""
        features = self.cnn(images)  # (B, d, h, w)
        _, _, h, w = features.shape
        rows = self.row_embed(torch.arange(h, device=images.device))  # (h, d)
        cols = self.col_embed(torch.arange(w, device=images.device))  # (w, d)
        # (h,1,d) + (1,w,d) → ブロードキャストで (h, w, d)。マス(i,j)には「i行目 + j列目」の埋め込みが入る
        pos = rows[:, None, :] + cols[None, :, :]

        # (B, d, h, w) → (B, h, w, d) → (B, h*w, d): マスを左上から1列に並べる
        x = features.permute(0, 2, 3, 1) + pos
        x = x.flatten(1, 2)
        return self.encoder(x)

    def decode(self, memory: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """memory (B, S, d) とこれまでのトークン tgt_in (B, T) → logits (B, T, V)

        出力の位置 t は「t番目までのトークンを見たうえで、t+1番目のトークンを予測した結果」。
        """
        T = tgt_in.size(1)
        positions = torch.arange(T, device=tgt_in.device)
        x = self.token_embed(tgt_in) + self.pos_embed(positions)  # (B, T, d)

        # causal mask (T, T): True のところは参照禁止。位置 t は 0..t だけを見られる。
        # 学習時は正解の列を丸ごと入力するので、これが無いと「答え（未来のトークン）を見てカンニング」できてしまう
        #   T=4 のとき:  [[F, T, T, T],
        #                [F, F, T, T],
        #                [F, F, F, T],
        #                [F, F, F, F]]
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        # padding mask (B, T): <pad> の位置は参照しない（長さを揃えるための埋め草なので意味が無い）
        padding_mask = tgt_in == self.config.pad_id

        h = self.decoder(
            x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask, tgt_is_causal=True
        )  # (B, T, d)
        return self.output(h)  # (B, T, V)

    def forward(self, images: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images), tgt_in)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """greedy decoding: 毎ステップ最もスコアの高いトークンを選び、それを入力に足して次を予測する。

        学習時（forward）は正解の列を一度に入れて全位置を並列に計算できたが、推論時は正解が無いので
        1トークンずつ順番に生成するしかない。戻り値は <bos> から始まるID列 (B, ≦max_len)。

        毎ステップ列全体をデコーダに通し直しているので、長さTに対してO(T^2)の計算になる。
        過去の計算結果を使い回す「KVキャッシュ」で速くできるが、まずは分かりやすさを優先している。
        """
        max_len = max_len or self.config.max_len
        memory = self.encode(images)  # 画像側の計算は1回だけで良い
        B = images.size(0)
        ys = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            logits = self.decode(memory, ys)[:, -1]  # 最後の位置の予測だけ使う: (B, V)
            next_token = logits.argmax(dim=-1)  # (B,)
            # 既に <eos> を出し終えたサンプルは <pad> で埋めておく
            next_token = torch.where(finished, self.config.pad_id, next_token)
            ys = torch.cat([ys, next_token[:, None]], dim=1)
            finished |= next_token == self.config.eos_id
            if finished.all():
                break
        return ys


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
