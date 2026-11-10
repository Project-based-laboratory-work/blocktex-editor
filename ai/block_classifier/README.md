# block_classifier — PDFのブロック分類

PDFから取り出したテキストブロック（`rule_based/pdf_extract.py` の出力）を、
見出し / 段落 / 箇条書き / 数式 / キャプション / 表 / その他 / コード に分類するモデル。
ラベルの定義はチーム共通の [rule_based/labels.py](../rule_based/labels.py) に従う（`code` はこのモデルのために追加した）。

```
PDF ─ pdf_extract（ルールベース）─▶ テキストブロック（位置・文字サイズ・フォント・文字列）
    ─ features ─▶ 特徴ベクトル ─ PageTransformer ─▶ ブロックごとのラベル
```

## PDF → エディタ用JSON（ir_adapter.py）

学習したモデルで分類し、チーム共通の形式（[common/](../common/README.md)）のラベル付きIRとエディタ用JSONを出す。
数式は、数式OCR（[math_ocr/](../math_ocr/README.md)）で LaTeX にして `tex` に入れる。

```bash
python -m block_classifier.ir_adapter report.pdf -o out/
#   → out/report.labeled.json（ラベル付きIR）, out/report.blocks.json（エディタ用JSON）
```

| 処理 | 内容 |
|---|---|
| 分類 | `pdf_extract` のテキストブロックを PageTransformer で分類する。表・図の領域に入る文字は、その表・図のものとする |
| 数式をまとめる | 細切れの断片（`f(x) =` `∑` `i=1`）を1つにまとめる。隙間がごく小さい・添字のような小さい文字・同じ高さの帯に並ぶ、のどれかならまとめる。複数行の数式は行ごとに分ける |
| 数式を読む | まとめた範囲をPDFから200dpiで切り抜き、数式OCRで LaTeX にする |
| 行をつなぐ | 1行ずつに分かれた段落・箇条書きを、字下げと行の長さを見てつなぐ |
| 見出し | 番号（「1.2 」）を文字から取り除いてレベルにする（番号はエディタのTeX出力で自動で振られる） |

自分のレポート（stego_report）では、見出し・段落・箇条書き・コードはほぼ正しく変換できた。
長い数式（50トークンを超えるもの）は、数式OCRが学習していない長さなので後半が崩れる。文中の数式（`$...$`）は、
段落の文字のまま（PDFの文字）で、数式OCRでは読んでいない。

## 学習データの作り方（色を使わない自動ラベル付け）

人手でラベルを付ける代わりに、**正解が分かっている文書をTeXで作ってPDFにする**。

1. `doc_gen.py` がランダムな文書（見出し・段落・箇条書き・数式・表・図・コード）のTeXを作る。
   各ブロックの始まりと終わりに「目印」を埋め込む
   （`\special{pdf:dest ...}`。dvipdfmx がその位置をPDFの座標でそのまま記録する）
2. `build_dataset.py` が uplatex → dvipdfmx でPDFにし（`-C 0x0010` で目印を残す）、
   `pdf_extract` で取り出した各テキストブロックが、どのブロックの目印の範囲に入るかでラベルを決める

文字の内容で対応付けないのは、別行立ての数式が `f(x) =` `∑` `i=1` のように細切れに取り出され、
元のTeXと文字が一致しないため。色で見分ける方法（`auto_label.py`）は、チームの方針で使わない。

実際の文書の多様さに近づけるため、文書クラス（jsarticle / bxjsarticle）、文字サイズ、2段組、
本文の書体（明朝/ゴシック）、見出しの書体、埋め込みフォント（既定 / IPAex / 原ノ味 / IPA）を散らしている。

## 実行方法

`ai/` で実行する。uplatex と dvipdfmx が必要。

```bash
python -m block_classifier.build_dataset --out block_classifier/data/train --docs 2000 --seed 0   # 約15分
python -m block_classifier.build_dataset --out block_classifier/data/val --docs 300 --seed 1
python -m block_classifier.dataset block_classifier/data/train block_classifier/data/val          # 特徴を計算して保存
python -m block_classifier.train --model transformer --epochs 30                                  # CPUで約30分
python -m block_classifier.evaluate block_classifier/checkpoints/transformer/best.pt              # ルールベースと比較
python -m block_classifier.predict block_classifier/checkpoints/transformer/best.pt report.pdf --draw out/  # 色分け画像
python -m pytest block_classifier/tests
```

## ファイル

| ファイル | 内容 |
|---|---|
| [doc_gen.py](doc_gen.py) | ランダムな文書のTeX（目印入り）を作る |
| [build_dataset.py](build_dataset.py) | PDFにして、目印の位置から正解ラベルを付ける |
| [features.py](features.py) | テキストブロック → 特徴ベクトル（26次元） |
| [dataset.py](dataset.py) | ページ単位のデータセット。ルールベースの予測も比較用に保存する |
| [model.py](model.py) | `PageTransformer`（ページ内のブロック同士を参照）と `BlockMLP`（1ブロックだけ見る） |
| [train.py](train.py) / [evaluate.py](evaluate.py) / [predict.py](predict.py) | 学習・評価・推論 |

## 結果（2026-10-16、生成した検証データ300文書・18,756ブロック）

| モデル | パラメータ数 | 正解率 | macro F1 |
|---|---|---|---|
| ルールベース（`rule_based_classifier`、粗いラベルで評価） | − | 65.5% | 0.393 |
| BlockMLP | 約2万 | 97.2% | 0.971 |
| **PageTransformer** | 約62万 | **99.3%** | **0.992** |

ルールベースは「その他」「コード」を出さないので、その分は不利な比較になっている。それ以外の主な差は、
表（縦線のない booktabs の表を pdfplumber が検出できない）と見出し（日本語の見出しのゴシック体が
「太字」と判定されない）。

## 分かっている課題

自分のレポート2本（`bxjsarticle`、`ltjsarticle`）で目視確認した結果:

- **改善済み:** 本文がゴシック体の文書で、本文を全部「見出し」と判定していた。学習データで
  「ゴシック体なら見出し」だったため、モデルがその近道を覚えていた（ショートカット学習）。
  フォントの特徴を「本文と同じか」に変え、学習データの書体を散らして解消した
- **未解決:** コードのフォントが Type3（名前の読めないビットマップフォント）で埋め込まれている文書では、
  コードを段落や数式と判定する。フォント以外の手がかり（記号の並び、等幅らしさ）が必要
- **未解決:** 表のセルに等幅の関数名などが入っていると、表をコードや段落と判定する
- 上の99.3%は自分で生成した文書での成績で、実際の文書ではこれより下がる
