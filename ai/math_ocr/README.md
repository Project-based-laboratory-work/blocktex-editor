# math_ocr — 数式画像 → LaTeX

ReTeXのパイプライン「② 数式・文字の認識」のうち、数式画像をLaTeXに変換するモデル。
CNNエンコーダとTransformerデコーダを組み合わせたエンコーダ・デコーダ型で、PyTorchで実装している。

```
数式画像 ─▶ CNN ─▶ Transformer Encoder ─▶ memory ─┐
                                                  ▼ cross-attention
           <bos> \frac { a ... ─▶ Transformer Decoder ─▶ 次のトークン
```

## 読む順番

| # | ファイル | 内容 | ここで学べること |
|---|---|---|---|
| 1 | [tokenizer.py](tokenizer.py) | LaTeX ⇔ トークン列 ⇔ ID列 | 系列モデルの入出力の作り方、特殊トークン |
| 2 | [synth_data.py](synth_data.py) | ランダムな数式を描画して学習データを作る | ラベル付けを自動化する、表記ゆれの統一 |
| 3 | [dataset.py](dataset.py) | 画像の前処理、`Dataset`、バッチ化 | `Dataset` / `DataLoader` / `collate_fn`、padding |
| 4 | [model.py](model.py) | モデル本体（**本命**） | `nn.Module`、CNN、位置埋め込み、causal mask、greedy decoding |
| 5 | [train.py](train.py) | 学習ループ | teacher forcing、損失、optimizer、学習率スケジュール、`train()`と`eval()` |
| 6 | [metrics.py](metrics.py) | 評価指標 | 完全一致率、トークン誤り率（編集距離） |
| 7 | [predict.py](predict.py) | 学習済みモデルで推論 | チェックポイントの保存と読み込み |

テスト（[../tests/test_math_ocr.py](../tests/test_math_ocr.py)）も読み物として役に立つ。特に以下の2つは、
系列モデルを書いたときに必ず確認すべき性質をテストにしたもの。

- `test_causal_mask_hides_future_tokens`: 未来のトークンを書き換えても過去の位置の出力が変わらない
  （=カンニングしていない）
- `test_can_overfit_tiny_batch`: 数件のデータを暗記できる（=勾配が流れ、入出力のずらし方が正しい）

## 実行方法

`ai/` ディレクトリで、仮想環境を有効にしてから実行する。

```bash
cd ai
source .venv/bin/activate
pip install -r requirements.txt   # GPUが無い環境では torch を CPU版にすると軽い（下記）

python -m pytest                                     # テスト
python -m math_ocr.synth_data --train 20000 --val 1000   # data/math_synth/ にデータ生成（約2分）
python -m math_ocr.train --overfit-batch             # 1バッチ過学習チェック
python -m math_ocr.train --epochs 10                 # 学習 → checkpoints/math_ocr/best.pt
python -m math_ocr.predict checkpoints/math_ocr/best.pt data/math_synth/images/val_000000.png
```

CPU版のtorchは `pip install torch --index-url https://download.pytorch.org/whl/cpu` で入る。

**計算時間の目安:** 手元のCPU（WSL2、12コア）では、デフォルト設定（d_model=256、約520万パラメータ）で
1ステップ（バッチ16）あたり約1.4秒かかる。2万件×10エポックの学習はCPUでは数時間かかるので、
手元では `--d-model 128 --limit-train 6000` などで縮めて動作を確認し、本番の学習はGPU
（IS計算機サーバ）で行う。

## 押さえておきたい概念

### テンソルの形（shape）を追う

PyTorchのコードを読み書きするときは、各行でテンソルの形がどう変わるかを追うのが基本。
[model.py](model.py) では各行の右に形をコメントで書いている。

| 変数 | 形 | 意味 |
|---|---|---|
| `images` | (B, 1, 64, 320) | バッチ × チャンネル × 高さ × 幅 |
| CNNの出力 | (B, d, 4, 40) | 画像を縦16分の1・横8分の1に縮めた特徴マップ |
| `memory` | (B, 160, d) | 4×40=160マスを1列に並べたもの。デコーダはここを参照する |
| `tgt_in` | (B, T) | デコーダに入れるトークンID列 |
| `logits` | (B, T, V) | 各位置での「次のトークン」のスコア（V=語彙数） |

### 学習と推論で、デコーダの動かし方が違う

- **学習時（teacher forcing）**: 正解の列を1つずらして入力し、全位置の「次のトークン」を一度に予測する。
  causal mask で未来を隠しているので、各位置は過去だけを見て予測している。
- **推論時（`generate`）**: 正解が無いので、`<bos>` から始めて1トークンずつ予測し、出したトークンを
  入力に足して次を予測する。`<eos>` が出たら終わり。

学習時は常に正しい入力を見ているが、推論時は自分の誤りが後ろに伝わっていく。このずれ
（exposure bias）が、検証データでの損失は低いのに生成結果は間違う、という現象の原因の1つになる。

### `model.train()` と `model.eval()`

Dropout と BatchNorm は、学習時と推論時で動きが変わる。評価・推論の前には必ず `model.eval()`、
学習に戻るときは `model.train()` を呼ぶ。勾配が要らない計算は `@torch.no_grad()` で囲み、
メモリと時間を節約する。

### 損失だけでなく生成結果を見る

`val_loss` が下がっていても、生成結果（exact_match / TER）が良くなっているとは限らない。
[train.py](train.py) では毎エポック、検証データの一部を実際に生成して評価し、正解と予測を3件表示している。

## 次にやること（学習課題を兼ねる）

1. **attention を自分で実装する**: `nn.TransformerDecoderLayer` を、自分で書いた
   Multi-Head Attention（`softmax(QKᵀ/√d)V`）で置き換える。上の2つのテストが通れば正しく書けている
2. **beam search**: greedyは毎ステップ最善の1つしか残さない。上位k個の候補を残す探索に変えて、精度の変化を見る
3. **uplatexで描画したデータ**: mathtextはLaTeXと字形が少し違う。`docker/` のTeX環境で描画したデータを足し、
   自分のTeXレポートから切り出した実際の数式で評価する
4. **既存モデルとの比較**: 学習済みモデル（pix2texなど）と、パラメータ数・CPUでの推論時間・精度を比べる
   （スライドの「精度と計算資源のトレードオフ」）
5. **ONNXへの書き出し**: ブラウザ（`frontend/`）で推論する場合に必要になる
