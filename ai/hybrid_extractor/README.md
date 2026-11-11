# hybrid_extractor — ルールベース＋自作AIのPDF抽出器

PDFから文字・表・図をブロックとして取り出し（抽出器・ルール）、ブロックの種類（見出し・段落・数式…）を当てる（分類器・ルール＋自作AI）。

```
PDF ─ extract.py（ルール）─▶ ブロック（文字・位置・大きさ・書体、表のセル、図の範囲）
    ─ rules.py（ルール）───▶ ラベル付きIR（ベースライン。確実に分かるもの: 表・図・キャプション・ページ番号）
    ─ 自作AI（Step 6〜）───▶ ラベル付きIR（ルールが苦手なもの: 番号の無い見出し、記号の違う箇条書き、行番号つきのコード…）
```

入出力はチーム共通のIR（[common/README.md](../common/README.md)）。評価は [evaluation/](../evaluation/README.md) に `hybrid_rules` として登録している。

## 使い方

`ai/` で実行する。

```bash
# 抽出だけ（IR ①）
python -m hybrid_extractor.extract hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.ir.json
# 抽出＋ルール分類（ラベル付きIR ②）
python -m hybrid_extractor.rules hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.labeled.json
# 評価用データセットで採点（既存の rule_based と比べる）
python -m evaluation.run --model hybrid_rules --dataset evaluation/data/synth_all
# 学習データを作る（12並列で 40文書 15秒くらい）
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/train --docs 1500 --seed 0
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/val --docs 300 --seed 1
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/test --docs 300 --seed 2
# テスト
python -m pytest hybrid_extractor
```

サンプルのPDFは [sample_data/sample.tex](sample_data/sample.tex) から作る（PDFは git に入れない）:

```bash
cd hybrid_extractor/sample_data && uplatex sample.tex && uplatex sample.tex && dvipdfmx sample.dvi
```

## ファイル

| ファイル | 内容 |
|---|---|
| [look.py](look.py) | PDFの中身をのぞく練習用（Step 1） |
| [extract.py](extract.py) | 抽出器。行 → 横につなぐ → 表・図 → 縦につないでブロックにする |
| [rules.py](rules.py) | ルール分類器（ベースライン） |
| [doc_gen.py](doc_gen.py) | 学習用のランダムな文書（TeX）を作る。評価用とは別の作り方 |
| [build_dataset.py](build_dataset.py) | 学習データ（ブロック＋正解のラベル＋ルールの答え）を作る |
| [sample_data/sample.tex](sample_data/sample.tex) | 開発用の固定サンプル（IR版3の項目がひと通り入る） |
| [tests/](tests/) | pytest |

## 抽出器（extract.py）のしくみ

| 段階 | やること | 手がかり |
|---|---|---|
| 行 | PyMuPDF の line を1文字ずつの情報つきで取る | `get_text("rawdict")` |
| 横につなぐ | 同じ高さに並ぶ line をつなぐ（「1」＋「はじめに」） | 縦の範囲の重なり（`same_row`）。2段組では左右の段をつながない |
| 表 | 左右の端がそろった横線が2本以上 → 表。罫線の無い表は「表1」の下の文字の並び | 列の境目は縦線か「どの行にも文字の無いすき間」。空のセルがある行は上の行の折り返し |
| 図 | 画像と、ベクターの図形のまとまり | 直線だけで中に文字がある囲みは、図ではなく文字の枠（枠付きのコード） |
| 縦につなぐ | 段落の続きの行・数式の断片・コードの行をまとめる | 行の間のすき間、字下げ（段の左端と比べる）、文字の大きさ・書体の変化、数式・等幅フォントの割合 |
| 装飾 | 太字・斜体・下線 → `runs` | 書体名（本文が明朝ならゴシックは太字）、ベースラインのすぐ下の横線 |
| コードの字下げ | 行の左端のずれ ÷ 1文字の幅 → 空白の数 | 等幅フォント |
| 読み順 | 2段組は、段をまたぐもので帯に区切り、左の段 → 右の段 | |

## 結果

### 評価用データセット（evaluation/data/synth_all、40文書）

| 指標 | rule_based（既存） | hybrid_rules |
|---|---|---|
| ブロックの検出 F1 | 0.414 | **0.980** |
| ラベルの正解率 | 0.511 | **1.000** |
| 読み順 | 0.943 | **0.969** |
| 装飾の F1 | − | **1.000** |
| 表のセル / 揃え / 罫線 / 列幅 | 0.817 / 0.830 / 0.974 / 0.167 | **1.000** / 0.807 / **1.000** / **1.000** |
| コードの字下げ | 0.333 | **0.750** |

この評価用データセットは作りがそろっている（jsarticle、番号付きの見出し、明朝の本文）ので、ルールだけでラベルが全部当たる。
AIの効果は、作りをばらつかせた学習データの検証用・テスト用（下）と、本物のPDFで見る。

### 学習データ（doc_gen.py の文書）でのルールの成績

学習用 1500文書（28,550ブロック）・検証用 300文書（5,654ブロック）。正解は目印から作り、抽出器のブロックの面積の半分以上が重なる正解のラベルを付けた（どれとも重ならない 1% 弱のブロックは使わない）。

ルールの正解率: 学習用 **0.785**、検証用 **0.775**。検証用のラベルごと:

| ラベル | ブロック数 | ルールの正解率 | ルールの主な間違い |
|---|---|---|---|
| paragraph | 1270 | 0.998 | |
| heading | 1011 | 0.790 | 段落（番号の無い、本文と同じ大きさの太字の見出し） |
| list | 779 | 0.389 | 段落（★・―・(a) の項目、description） |
| code | 599 | 0.514 | 段落（行番号つき・小さい文字・枠つきのコード） |
| other | 446 | 0.774 | 段落（ページ番号・ヘッダー） |
| caption | 372 | 0.895 | 段落 |
| math | 340 | 0.818 | 段落（式番号など） |
| table | 269 | 0.680 | 段落（罫線が無く、キャプションが下の表） |
| title / author / date / figure | | 1.000 | |

ここが自作AIの目標: **ルールの 0.775 を超え、とくに list・code・heading・table を当てる**。

## 分かっている課題

- 罫線の無い表で、キャプションが表の下にあるものは見つけられない（ブロックごとに段落になる）
- 列の揃えは、セルの文字の幅が全部同じ列では余白で決めるので、ときどき外れる（評価の揃えの正解率 0.807）
- 数式の文字は、PDFの文字を読み順（上から）に並べただけで、LaTeX にはしていない
