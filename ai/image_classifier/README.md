# image_classifier — 画像の構造化器（ブロック分類AI）

[image_rule_based](../image_rule_based/README.md) の AI 版。OCR で取り出した段落ごとに、**12種類のラベル**
（見出し・段落・箇条書き・数式・キャプション・表・図・コード・タイトル・著者・日付・その他）を付ける。
入力・出力の形はルールベース版と同じ（段落単位のIR → ラベル付きIR）なので、入れ替えて比べられる。

```
画像 ─ easyocr_ocr（--group）─→ IR（段落単位）─ image_classifier（ここ）─→ ラベル付きIR
```

## しくみ

段落1つずつを別々に判定するのではなく、**ページの段落を全部まとめて**見て判定する。

1. 段落ごとに、数値の特徴（22個）を作る（[features.py](features.py)）: 位置・大きさ・字下げ・本文と比べた文字の大きさ・行数・
   「1.2 」「図3」「・」で始まるか・数字/英字/日本語/数式記号の割合 など
2. 段落ごとに、その部分の画像を 32×256 に切り抜き、小さなCNNで128個の数にする（文字の形・罫線・数式の見た目を見る）
3. 1と2を足したものを、ページの段落の数だけ並べて Transformer に入れる。前後の段落を見比べて判定できる
   （例: 「図の直後の短い行はキャプション」「同じ字下げが続くのは箇条書き」）

## 使い方

`ai/` で実行する。

```bash
# 学習データ（image_dataset で自動生成。既定の崩し方 clean:screenshot = 1:4、ビューアの背景付きのスクショを含む）
python -m image_dataset.build --out image_classifier/data/train --docs 2000 --seed 0 --workers 10
python -m image_dataset.build --out image_classifier/data/val --docs 300 --seed 1 --workers 10
# OCR（GPUがあると速い。--shard 0/4 のように分けて並べられる）→ 学習用にまとめる
python -m image_classifier.run_ocr image_classifier/data/train
python -m image_classifier.run_ocr image_classifier/data/val
python -m image_classifier.dataset image_classifier/data/train image_classifier/data/val
# 学習（計算機サーバのGPUで約15分。画像を見ない版は約4分）と採点
python -m image_classifier.train               # → checkpoints/image/best.pt
python -m image_classifier.train --no-image    # → checkpoints/noimage/best.pt（画像を見ない版）
python -m image_classifier.evaluate image_classifier/data/val \
    --checkpoint image_classifier/checkpoints/noimage/best.pt --checkpoint image_classifier/checkpoints/image/best.pt
# 1枚を分類する
python -m image_classifier.ir_adapter <段落単位のIR> --image <元の画像> -o out/
python -m pytest image_classifier/tests
```

`data/` と `checkpoints/` は git の管理外。

## 結果（検証 854枚、ルールベースと同じOCR結果を使用）

image_dataset の今の既定（ビューアの背景付きのスクショを含む、#45 以降）で作ったデータで学習・採点したもの。

| | ラベル正解率 | 3種類だけで見た正解率※ | 時間/枚 |
|---|---|---|---|
| ルールベース（image_rule_based） | 0.653 | 0.849 | 0.3 ms |
| **AI（特徴＋画像）** | **0.990** | **0.993** | 89 ms |

※ ルールベースが出せる見出し・段落・キャプションの3種類の正解だけで数えたもの

画像を見る効果（#45 より前のデータで比べたもの）: 数値の特徴だけの版 0.977 → 特徴＋画像 0.994。特にキャプション・数式・箇条書きが良くなる。
前のデータで学習したモデルも、今のデータで 0.989 と、ビューアの背景にはほとんど影響されない（OCRの段落を見るため）。

- 検出のF1（段落の区切り方）はどれも 0.541 で同じ。区切り方は OCR の段落まとめで決まり、この分類器は変えないため。
  区切り方そのものを良くするのは [image_layout](../image_layout/README.md)
- 注意: 学習も検証も image_dataset が作った画像なので、AIに有利な条件。人が正解を付けた本物の画像での確認はこれから

## ファイル

| ファイル | 内容 |
|---|---|
| [run_ocr.py](run_ocr.py) | 学習データの画像を EasyOCR（GPU）で読み、段落にまとめたIRを `<dir>/ocr/` に保存する |
| [features.py](features.py) | 段落ごとの数値の特徴と、画像の切り抜き |
| [dataset.py](dataset.py) | OCRの段落に正解のラベルを対応付け（半分以上重なる正解のブロック）、`pages.pt` にまとめる |
| [model.py](model.py) | 切り抜きのCNN ＋ ページの段落を見比べる Transformer |
| [train.py](train.py) | 学習。best.pt はラベルごとのF1の平均で選ぶ |
| [evaluate.py](evaluate.py) | ルールベースと同じOCR結果・同じ正解で比べる |
| [ir_adapter.py](ir_adapter.py) | IRを読み、分類してラベル付きIRを出す |
