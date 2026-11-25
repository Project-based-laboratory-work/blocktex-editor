# image_deco — 文中の装飾（太字・斜体・下線）を見分けるAI

1行の画像を見て、横の位置ごとに「太字か・斜体か・下線か」を当てる。[image_ai](../image_ai/README.md) が、
段落・箇条書き・キャプションの行に使い、IR の runs（装飾の付いた文字の並び）にする。

```
1行の画像 ─ 文字のある高さに縮め、上下に余白（文字の高さの 0.25 倍）を足して、高さ 40 にそろえる
          ─ CNN（横を 1/4 に縮める）─ 双方向GRU（行の左右を見渡す）─ 位置ごとに [太字, 斜体, 下線] の確率
          ─ 0.5 を超える位置が続くところを範囲にし、行の幅の割合から文字に対応させる（英語は語の切れ目にそろえる）
```

- 上下の余白をそろえるのは、切り出し方で文字の大きさが変わると線の太さの見え方も変わり、太字と間違えるため。
  学習のときは余白を 0.1〜0.4 倍でばらつかせる
- 学習のときだけ、スクリーンショットに近づける崩し（縮小・ぼかし・JPEG・ダークモード）を入れる

## 学習データ

[generate.py](generate.py) が、TeX で1行の文を組み、装飾の前後に入れた目印の位置から正解の範囲を作る。
評価用データと同じく、日本語は太字・下線、英語は太字・斜体・下線。日本語フォントは原ノ味・IPAex を埋め込むか、
埋め込まない（評価用データと同じ）かを文書ごとに選ぶ。埋め込まないと日本語の太字が見た目に出ないので、正解から外す。

## 使い方

`ai/` で実行する。

```bash
python -m image_deco.generate --out image_deco/data/train --lines 40000 --seed 0 --workers 10
python -m image_deco.generate --out image_deco/data/val --lines 4000 --seed 1 --workers 10
python -m image_deco.train --train image_deco/data/train --val image_deco/data/val --epochs 12
python -m pytest image_deco/tests
```

## 結果

- 学習データと同じ作りの検証データ（4,020行）で、位置ごとの F1 は 太字 0.996・斜体 0.997・下線 0.967
- チームの評価用データでの結果（image_ai に組み込んだもの）は [image_ai の README](../image_ai/README.md) の「装飾」。
  1行だけでは、フォントが埋め込まれずゴシックで描かれた日本語の本文を太字と区別できないので、
  image_ai の側で「線の太さがページの本文の 1.25 倍以上」の太字だけを残している

## ファイル

| ファイル | 内容 |
|---|---|
| [generate.py](generate.py) | 学習データを作る |
| [dataset.py](dataset.py) | 読み込み・余白をそろえる・崩し・正解の作り方 |
| [model.py](model.py) | DecorationTagger（CNN＋双方向GRU） |
| [train.py](train.py) | 学習（best.pt は検証データの位置ごとの F1 の平均で選ぶ） |
| [predict.py](predict.py) | 1行の画像 → 装飾の範囲 → 文字の範囲 |
| [tests/](tests/) | pytest |
