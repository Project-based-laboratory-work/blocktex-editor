"""自作AIの専門家のモデルと、学習の共通部分（PyTorch）。

    MLP          : 特徴の数字の並び → 「はい」の自信（0〜1）。区切りの専門家・ラベルの専門家で使う
    CharTagger   : 段落の文字の並び → 1文字ずつ「数式か」の自信。文中の数式の専門家で使う（小さい Transformer）
    Standardizer : 特徴を「平均0・ばらつき1」にそろえる（学習しやすくするため）
    train_binary / train_tagger : 学習のループ（検証用データの損失が下がらなくなったら止める）

学習が終わったモデルは save_checkpoint で .pt に保存し、load_checkpoint で読み戻す。
"""

import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """乱数の種をそろえる（同じ条件なら同じ結果になるように）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================================
# 特徴をそろえる
# =====================================================================


@dataclass
class Standardizer:
    """特徴ごとに「(値 - 平均) / 標準偏差」にする。平均・標準偏差は学習用データだけから求める"""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        std = x.std(axis=0)
        std[std < 1e-6] = 1.0  # ずっと同じ値の特徴は割らない
        return cls(x.mean(axis=0).astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(np.array(d["mean"], dtype=np.float32), np.array(d["std"], dtype=np.float32))


# =====================================================================
# モデル
# =====================================================================


class MLP(nn.Module):
    """多層パーセプトロン: 入力 → [全結合 → ReLU → ドロップアウト] × 2 → 全結合 → 1つの値（ロジット）。
    ロジットを sigmoid に通すと「はい」の確率（自信）になる"""

    def __init__(self, n_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CharTagger(nn.Module):
    """文字の並びに、1文字ずつ「数式か」のロジットを付ける小さい Transformer。

        文字（ID）→ 埋め込み（d/2 次元）──────┐
        文字の手がかり（数字）→ 全結合（d/2 次元）─┴ つなぐ（d 次元）＋ 位置の埋め込み
        → Transformer エンコーダ（自己注意で、前後の文字を見比べる）× layers
        → 全結合 → 1文字ずつ1つの値
    """

    def __init__(self, vocab_size: int, n_feat: int, d_model: int = 64, layers: int = 2, heads: int = 4,
                 max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model // 2, padding_idx=0)
        self.feat = nn.Linear(n_feat, d_model - d_model // 2)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model * 2, dropout=dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids: torch.Tensor, feats: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) 文字のID、feats: (B, T, F) 手がかり、pad: (B, T) 埋め草なら True → (B, T) のロジット"""
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        x = torch.cat([self.embed(ids), self.feat(feats)], dim=-1) + self.pos(positions)
        x = self.encoder(x, src_key_padding_mask=pad)
        return self.head(x).squeeze(-1)


# =====================================================================
# 成績
# =====================================================================


def binary_scores(prob: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """二値分類の成績: 正解率・適合率（はいと答えたうち本当にはい）・再現率（本当のはいのうち見つけた）・F1"""
    pred = prob >= threshold
    yb = y.astype(bool)
    tp = int((pred & yb).sum())
    fp = int((pred & ~yb).sum())
    fn = int((~pred & yb).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"acc": float((pred == yb).mean()), "precision": precision, "recall": recall, "f1": f1, "n": int(len(y))}


# =====================================================================
# MLP の学習
# =====================================================================


def train_binary(x: np.ndarray, y: np.ndarray, xv: np.ndarray, yv: np.ndarray, hidden: int = 64,
                 epochs: int = 60, lr: float = 1e-3, batch: int = 256, patience: int = 6, seed: int = 0,
                 log: bool = True) -> tuple[MLP, Standardizer, list[dict]]:
    """MLP を二値分類で学習する。

    1. 特徴を、学習用データの平均・標準偏差でそろえる
    2. ミニバッチ（batch 件ずつ）で、予測 → 損失（2値の交差エントロピー）→ 誤差逆伝播 → Adam で重みを更新。
       「はい」が少ないときは、はいの見逃しを重く数える（pos_weight = いいえの数 / はいの数）
    3. 1周（エポック）ごとに検証用データの損失を測り、一番良かったときの重みを残す。
       patience 回続けて良くならなければ止める（早期終了。学習用データの覚えすぎ＝過学習を防ぐ）"""
    set_seed(seed)
    std = Standardizer.fit(x)
    xt = torch.tensor(std(x))
    yt = torch.tensor(y, dtype=torch.float32)
    xvt = torch.tensor(std(xv))
    yvt = torch.tensor(yv, dtype=torch.float32)
    model = MLP(x.shape[1], hidden)
    positives = max(1.0, float(y.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - positives) / positives))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_loss, waited, history = copy.deepcopy(model.state_dict()), float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(xvt)
            val_loss = loss_fn(logits, yvt).item()
            val = binary_scores(torch.sigmoid(logits).numpy(), yv)
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_loss": val_loss, **val})
        if log:
            print(f"  epoch {epoch:2d}  train_loss {total / len(xt):.4f}  val_loss {val_loss:.4f}  "
                  f"val_acc {val['acc']:.4f}  val_f1 {val['f1']:.4f}")
        if val_loss < best_loss - 1e-4:
            best, best_loss, waited = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best)
    model.eval()
    return model, std, history


def predict_binary(model: MLP, std: Standardizer, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(std(x)))).numpy()


# =====================================================================
# CharTagger の学習
# =====================================================================


@dataclass
class Sequences:
    """文字の並びのデータ（長さはばらばら）。ids[k]: (T,)、feats[k]: (T, F)、labels[k]: (T,)"""
    ids: list[np.ndarray]
    feats: list[np.ndarray]
    labels: list[np.ndarray]


def pad_batch(seqs: Sequences, idx: list[int], std: Standardizer) -> tuple[torch.Tensor, ...]:
    """長さの違う並びを、一番長いものに合わせて埋め草（0）でそろえる。pad は埋め草の位置で True"""
    t = max(len(seqs.ids[i]) for i in idx)
    n_feat = seqs.feats[idx[0]].shape[1]
    ids = torch.zeros(len(idx), t, dtype=torch.long)
    feats = torch.zeros(len(idx), t, n_feat)
    labels = torch.zeros(len(idx), t)
    pad = torch.ones(len(idx), t, dtype=torch.bool)
    for row, i in enumerate(idx):
        n = len(seqs.ids[i])
        ids[row, :n] = torch.tensor(seqs.ids[i])
        feats[row, :n] = torch.tensor(std(seqs.feats[i]))
        labels[row, :n] = torch.tensor(seqs.labels[i], dtype=torch.float32)
        pad[row, :n] = False
    return ids, feats, labels, pad


def predict_tagger(model: CharTagger, std: Standardizer, seqs: Sequences, batch: int = 32,
                   logits: bool = False) -> list[np.ndarray]:
    """並びごとに、1文字ずつの「数式か」の確率（logits=True なら、確率にする前のロジット。蒸留の先生の答えに使う）"""
    out: list[np.ndarray] = []
    with torch.no_grad():
        for k in range(0, len(seqs.ids), batch):
            idx = list(range(k, min(k + batch, len(seqs.ids))))
            ids, feats, _, pad = pad_batch(seqs, idx, std)
            z = model(ids, feats, pad)
            value = z if logits else torch.sigmoid(z)
            out += [value[row, :len(seqs.ids[i])].numpy() for row, i in enumerate(idx)]
    return out


def pad_values(values: list[np.ndarray], idx: list[int]) -> torch.Tensor:
    """並びごとの値（先生のロジットなど）を、pad_batch と同じ長さにそろえる"""
    t = max(len(values[i]) for i in idx)
    out = torch.zeros(len(idx), t)
    for row, i in enumerate(idx):
        out[row, :len(values[i])] = torch.tensor(values[i], dtype=torch.float32)
    return out


def distill_loss(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """蒸留の損失（1文字ずつ）。先生と生徒のロジットを温度 T で割ってなめらかにし、生徒の確率を先生の確率に近づける。

    2値なので、ソフトマックスの代わりにシグモイド: 先生の答え p = σ(z_先生 / T)、損失 = 交差エントロピー(σ(z_生徒 / T), p)。
    T が大きいほど 0.99 と 0.6 のような自信の差がなだらかになり、「どれくらい迷ったか」が生徒に伝わる。
    T² を掛けるのは、T で割ったぶん勾配が 1/T² に小さくなるのを戻すため（Hinton ら 2015）"""
    target = torch.sigmoid(teacher / temperature)
    loss = nn.functional.binary_cross_entropy_with_logits(student / temperature, target, reduction="none")
    return loss * temperature ** 2


def train_tagger(train: Sequences, val: Sequences, vocab_size: int, epochs: int = 30, lr: float = 1e-3,
                 batch: int = 32, patience: int = 4, seed: int = 0, log: bool = True,
                 teacher: list[np.ndarray] | None = None, temperature: float = 2.0, alpha: float = 0.5,
                 **model_args) -> tuple[CharTagger, Standardizer, list[dict]]:
    """CharTagger を学習する（流れは train_binary と同じ。損失は埋め草でない文字だけで数える）。

    teacher（並びごとの先生のロジット）を渡すと蒸留: 損失 = alpha × 正解との損失 + (1 - alpha) × 先生との損失（distill_loss）。
    早く止めるかどうかは、どちらでも検証用データの「正解との損失」で決める（比べられるように）"""
    set_seed(seed)
    std = Standardizer.fit(np.concatenate(train.feats))
    model = CharTagger(vocab_size, train.feats[0].shape[1], **model_args)
    labels = np.concatenate(train.labels)
    positives = max(1.0, float(labels.sum()))
    pos_weight = torch.tensor((len(labels) - positives) / positives)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def masked_loss(logits: torch.Tensor, y: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        keep = (~pad).float()
        return (loss * keep).sum() / keep.sum()

    best, best_loss, waited, history = copy.deepcopy(model.state_dict()), float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train.ids)))
        random.shuffle(order)
        total, count = 0.0, 0
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            ids, feats, y, pad = pad_batch(train, idx, std)
            optimizer.zero_grad()
            logits = model(ids, feats, pad)
            loss = masked_loss(logits, y, pad)
            if teacher is not None:
                keep = (~pad).float()
                soft = (distill_loss(logits, pad_values(teacher, idx), temperature) * keep).sum() / keep.sum()
                loss = alpha * loss + (1 - alpha) * soft
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for k in range(0, len(val.ids), batch):
                idx = list(range(k, min(k + batch, len(val.ids))))
                ids, feats, y, pad = pad_batch(val, idx, std)
                val_loss += masked_loss(model(ids, feats, pad), y, pad).item() * len(idx)
        val_loss /= len(val.ids)
        scores = binary_scores(np.concatenate(predict_tagger(model, std, val)), np.concatenate(val.labels))
        history.append({"epoch": epoch, "train_loss": total / count, "val_loss": val_loss, **scores})
        if log:
            print(f"  epoch {epoch:2d}  train_loss {total / count:.4f}  val_loss {val_loss:.4f}  "
                  f"val_acc {scores['acc']:.4f}  val_f1 {scores['f1']:.4f}")
        if val_loss < best_loss - 1e-4:
            best, best_loss, waited = copy.deepcopy(model.state_dict()), val_loss, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best)
    model.eval()
    return model, std, history


# =====================================================================
# 保存と読み込み
# =====================================================================


def save_checkpoint(path: Path, model: nn.Module, std: Standardizer, config: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "std": std.to_dict(), "config": config, "extra": extra or {}}, path)


def load_checkpoint(path: Path) -> dict:
    """{"model": 重み, "std": Standardizer, "config": モデルの大きさなど, "extra": そのほか（語彙など）}"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["std"] = Standardizer.from_dict(ckpt["std"])
    return ckpt
