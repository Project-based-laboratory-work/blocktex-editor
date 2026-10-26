# easyocr_ocr — EasyOCRによる文字認識（IMG-A）

学習なしの既存OCR（EasyOCR）で画像の文字を読み取り、共通のIR形式（[common/README.md](../common/README.md) の①）で出力する抽出器。`tesseract_ocr` との比較対象になる。

| | 内容 |
|---|---|
| 入力 | 画像1枚（スクリーンショットを想定）。1枚＝1ページ、単位は `px` |
| 出力 | IR（`kind="text"` の要素。`bbox`・`text`・`font_size`・`confidence`） |
| 入口 | `ir_adapter.extract_ir(image_path)` |

## 実行方法

```bash
cd ai
pip install easyocr   # 初回はモデルのダウンロードがある
python -m easyocr_ocr.ir_adapter ../ocr/data/easy1.jpg -o out/               # → out/easy1.base.ir.json
python -m easyocr_ocr.ir_adapter ../ocr/data/easy1.jpg --preset up2 -o out/  # → out/easy1.up2.ir.json
python -m easyocr_ocr.ir_adapter ../ocr/data/easy1.jpg --group -o out/      # → out/easy1.base.group.ir.json（段落にまとめる）
python -m common.ir out/easy1.base.ir.json   # 形式チェック
```

## ファイル

| ファイル | 内容 |
|---|---|
| [ir_adapter.py](ir_adapter.py) | 解析の流れ：画像を開く → 前処理 → EasyOCR → IRに変換 |
| [presets.py](presets.py) | 設定の組（プリセット）。実験はここに名前を付けて足す |
| [preprocess.py](preprocess.py) | 前処理の部品（拡大・グレースケール・反転・二値化など）。EasyOCRに依存しない |
| [compare.py](compare.py) | プリセットごとの文字誤り率（CER）を、正解テキストのある画像で比べる |
| [grouping.py](grouping.py) | 行（EasyOCRの箱）を段落（ブロック）にまとめる。`--group` で使う。EasyOCRに依存しない |

使ったプリセットの中身はIRの `params` に記録されるので、出力を見ればどの設定の結果か分かる。
前処理で拡大した場合も、`bbox`・`font_size` は元の画像の大きさに戻してから書く。

## プリセットの比較

`ocr/sample/<カテゴリ>/` の中で、画像と同じ名前の `.txt`（正解テキスト）がある画像だけを対象に、全プリセットで読み取ってCERを出す。

```bash
cd ai
python -m easyocr_ocr.compare                       # 全プリセット × 全画像（CPUだと1時間以上かかる）
python -m easyocr_ocr.compare --presets base up2    # プリセットを絞る
python -m easyocr_ocr.compare --category jphorizon  # カテゴリを絞る
```

| 出力（`easyocr_ocr/data/compare/`、git管理外） | 内容 |
|---|---|
| `summary.csv` | プリセットごとの、カテゴリ別・全体・日本語・英語のCERと、1枚あたりの平均時間。全体のCERが小さい順 |
| `details.csv` | 画像1枚ごとの結果 |
| `ir/<カテゴリ>/<画像名>.<プリセット>.ir.json` | 読み取った結果。次回はこれを再利用する（読み直すときは `--force`） |
| `text/<カテゴリ>/<画像名>.<プリセット>.txt` | 比べた文字列（要素を読む順につなげたもの）。正解の `.txt` と並べるか、VS Code の「選択項目の比較」で違いを見る |

- 位置（bbox）は見ず、要素を読む順（横書きは上の行から、縦書きは右の列から）につなげた文字列を比べる。縦書きかどうかは箱の形で判定する
- CERの計算は `evaluation/metrics.py` と同じ（空白・改行は取り除く）。カテゴリごとのCERは、画像ごとのCERの平均ではなく「誤りの合計 ÷ 文字数の合計」

## 現状の制限

- 要素は既定ではEasyOCRが返した行（または行の一部）単位。`--group` を付けると段落にまとめる（同じ行の箱をつなぎ、字下げ・短い行・行間で段落を区切る）。
- `font_size` は箱の短い辺（横書きなら高さ、縦書きなら幅）から推定している。`paragraph=True` では箱が段落全体になるため書かない。
- 縦書き・数式・表の中身（`cells`）には対応していない。
