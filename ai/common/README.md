# common — 共通のJSON形式（IR・ラベル付きIR・エディタ用JSON）

PDF・画像の読み込みで、どのモデルも同じ形のJSONを出すための定義と変換。モデルごとに出力の形が違うと、評価ハーネスでの比較もオペレータAIもできないため、ここで形を1つに決める。

```
入力(PDF/画像)
   ↓ 抽出器（rule_based, tesseract, 自作AI …）
① IR                    座標とテキストだけ。意味（見出し・段落など）は付けない
   ↓ 構造化器（ルールベース分類、Transformer …）
② ラベル付きIR          ①の各要素に label を付けたもの。学習・評価の正解ラベルもこの形
   ↓ common/blocks.py（全モデル共通の変換）
③ エディタ用JSON        web/src/model.ts の BlockDocument（エディタの保存形式と同じ）
```

- ①②は同じ形式で、②は `label` などの項目が増えるだけ。定義は [ir.py](ir.py) の dataclass が正（JSON Schemaは作らない）。
- ③は [web/src/model.ts](../../web/src/model.ts) が正。②→③の変換は [blocks.py](blocks.py) の1本だけにして、モデルごとに書かない。
- 新しい抽出器・構造化器を作るときは、独自の出力を①または②に変換するアダプタを自分のフォルダに書く（例: [rule_based/ir_adapter.py](../rule_based/ir_adapter.py)）。

## ① IR（抽出器の出力）

```json
{
  "ir_version": 1,
  "source": "sample.pdf",
  "extractor": "rule_based",
  "params": { "preprocess": "otsu", "scale": 1.5 },
  "pages": [
    {
      "page": 1, "width": 595.28, "height": 841.89, "unit": "pt",
      "elements": [
        { "kind": "text", "bbox": [0.131, 0.164, 0.31, 0.186], "text": "1.1 はじめに", "font_size": 18.0, "bold": false },
        { "kind": "table", "bbox": [0.131, 0.276, 0.5, 0.33], "cells": [["項目", "値"], ["工数", "15時間"]] },
        { "kind": "figure", "bbox": [0.131, 0.4, 0.6, 0.5] }
      ]
    }
  ]
}
```

| 項目 | 内容 |
|---|---|
| `ir_version` | IRの版数。形を変えたら `ir.py` の `IR_VERSION` を上げる |
| `source` / `extractor` | 入力ファイル名 / 抽出器の名前 |
| `params` | 前処理などのパラメータ。同じ抽出器でもパラメータ違いを別物として比較するために書く |
| `structurer` | ラベルを付けた構造化器の名前。①では書かない |
| `page` | **1始まり**の連番（PyMuPDFの0始まりから+1する） |
| `width` / `height` / `unit` | 元の単位でのページの大きさ。PDFは `"pt"`、画像は `"px"` |
| `kind` | `"text"` / `"table"` / `"figure"`。抽出器が見つけた領域の種類だけで、意味は含めない |
| `bbox` | `[x0, y0, x1, y1]`。**左上原点・ページの幅と高さで割った0〜1の値**。PDF(pt)と画像(px、DPI違い)を同じ基準でIoU比較するため。元の単位に戻すときは `denormalize_bbox` |
| `text` | `kind="text"` では必須。行の区切りは `\n` |
| `font_size` | `unit` での文字サイズ。取れない抽出器は bbox の高さから推定して入れる |
| `bold` | 太字か。判定できなければ書かない |
| `confidence` | 参考値。モデルごとに意味が違うので比較には使わない |
| `cells` | 表の中身（行×列の文字列）。取れる抽出器だけ |

取れなかった値は `null` にせず、**キーごと書かない**（`save_ir` が自動でそうする）。

要素の並び順は①では決めない（読み順に並べるのは構造化器の役目）。

## ② ラベル付きIR（構造化器の出力・正解ラベル）

①の形に、要素ごとに次の項目を足したもの。要素は**ページ内の読み順**に並べる。

| 項目 | 内容 |
|---|---|
| `label` | `heading` / `paragraph` / `list` / `math` / `caption` / `table` / `figure` / `other`（[rule_based/labels.py](../rule_based/labels.py) の `FINE_LABELS` と同じ） |
| `level` | `label="heading"` の見出しレベル（1〜3）。分からなければ書かない（③で文字サイズの大きい順に決める） |
| `tex` | `label="math"` のLaTeX。数式OCR（math_ocr）で読めたときに入れる |
| `label_source` | 正解ラベルのときだけ。`auto`（色付きTeXから自動）/ `manual`（人手）/ `prelabel`（ルールベースの下書き） |

```json
{ "kind": "text", "bbox": [0.131, 0.164, 0.31, 0.186], "text": "1.1 はじめに", "font_size": 18.0, "label": "heading", "level": 2 }
```

## ③ エディタ用JSON

エディタの `BlockDocument`（[web/src/model.ts](../../web/src/model.ts)、`schemaVersion: 3`）そのもの。②からの変換ルール（[blocks.py](blocks.py)）:

| label | ブロック | 補足 |
|---|---|---|
| `heading` | `heading` | `level` が無ければ、文字サイズの大きい順に 1, 2, 3 |
| `paragraph` | `paragraph` | PDFの折り返しの改行はつなぐ（英数字どうしの間だけ空白を入れる） |
| `list` | `list` | 連続した `list` 要素は1つにまとめる。行頭の `・` `•` `-` `1.` `(1)` で項目を分け、記号の種類から `marker` を決める |
| `math` | `math`（別行立て） | `tex` が無ければPDFの文字をそのまま入れる（仮の値。コンパイルが通らないことがある） |
| `table` | `table` | `cells` があれば中身を入れる。全部左寄せ |
| `figure` | `image` | 画像ファイルは無いので `src` は空（エディタで選び直す）。幅は本文の幅に対する比率 |
| `caption` | 隣の表・図の `caption` | 同じページの隣に表・図が無ければ段落として残す |
| `other` | なし | ページ番号・ヘッダーなどは捨てる |
| （label なし） | `kind` から | `text` は段落、`table` / `figure` はそのまま |

`model.ts` の `SCHEMA_VERSION` を上げたら、`blocks.py` の `EDITOR_SCHEMA_VERSION` と変換も直すこと（`tests/test_blocks.py` が食い違いを検出する）。

## 使い方

`ai/` ディレクトリで実行する。

```bash
# ルールベース（PDF-A）で ①②③ をまとめて出す
python -m rule_based.ir_adapter rule_based/sample_data/sample.pdf -o out/

# ①②の形式チェック
python -m common.ir out/sample.ir.json out/sample.labeled.json

# ②→③ だけ行う（構造化器を自作したとき）
python -m common.blocks out/sample.labeled.json -o out/sample.blocks.json
```

Pythonから使うとき:

```python
from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate

page = Page(page=1, width=img.width, height=img.height, unit="px")
page.elements.append(Element(kind="text", bbox=normalize_bbox((x0, y0, x1, y1), img.width, img.height), text="…", font_size=y1 - y0))
doc = IRDocument(source="input.png", extractor="tesseract", params={"lang": "jpn+eng"}, pages=[page])
assert validate(doc) == []   # 問題があれば日本語のメッセージが返る
save_ir(doc, "out/input.ir.json")
```

## 各モデルの対応状況

| モデル | 対応 |
|---|---|
| rule_based（抽出・分類・学習ラベル） | 抽出は [pdf_extract.py](../rule_based/pdf_extract.py) が①を直接出す。②③は [ir_adapter.py](../rule_based/ir_adapter.py) で変換する |
| tesseract_ocr | 未対応。`pytesseract.image_to_data` の `left` / `top` / `width` / `height` と行番号を使えば、行ごとの `text` 要素（`unit: "px"`）にできる |
| math_ocr | ページ全体は扱わないので①は出さない。②の `label="math"` の要素を切り出して読み、結果を `tex` に入れる使い方を想定 |
