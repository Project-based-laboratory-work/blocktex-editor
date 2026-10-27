# tesseract_ocr — Tesseractによる文字認識

学習なしの既存OCR（Tesseract）で画像の文字を読み取り、共通のIR形式（[common/README.md](../common/README.md) の①）で出力する抽出器。`easyocr_ocr` と同じ前処理・同じ評価（CER）で比較する。

| | 内容 |
|---|---|
| 入力 | 画像1枚。1枚＝1ページ、単位は `px` |
| 対応 | 英語・日本語の横書き・日本語の縦書き（`--direction` で指定、既定は `auto`） |
| 出力 | IR（`kind="text"` の要素を Tesseract の段落ごとに1つ。`bbox`・`text`・`font_size`・`confidence`） |
| 入口 | `ir_adapter.extract_ir(image_path, preset, direction, lang)` |

## 準備

Python 3.10 以上（`common/ir.py` が `str | None` を使うため）。Tesseract 本体と言語データ `eng` / `jpn` / `jpn_vert` をOSに入れる。

```bash
brew install tesseract tesseract-lang   # macOS（Ubuntu は apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert）
tesseract --list-langs                  # eng, jpn, jpn_vert があるか確認
pip install pytesseract opencv-python-headless pillow numpy
```

言語データを別の場所（例: [tessdata/](tessdata/)。Gitには上げない）に置いた場合は、`TESSDATA_PREFIX` にそのフォルダを指定する。

## 実行方法

```bash
cd ai
# 1枚 → IR
python -m tesseract_ocr.ir_adapter ../ocr/sample/jphorizon/jphorizontal_001.jpg -o out/   # → out/jphorizontal_001.base.ir.json
python -m tesseract_ocr.ir_adapter ../ocr/sample/jpvertical/jpvertical_001.jpg --direction vertical --preset up2 -o out/
python -m tesseract_ocr.ir_adapter ../ocr/sample/eng/eng_001.jpg --lang eng -o out/
python -m common.ir out/jphorizontal_001.base.ir.json   # 形式チェック

# 正解 .txt がある画像すべてで、全プリセットの CER を比べる（時間はプリセットの数 × 画像の数に比例）
python -m tesseract_ocr.evaluate
# プリセットを絞る
python -m tesseract_ocr.evaluate --presets base up2 otsu
```

`evaluate` は `ocr/sample/<データセット>/` の画像と同じ名前の `.txt` を正解とする。フォルダ名は結果をまとめる単位に使うだけで、読み方の判断には使わない。
- 向き：`--direction` の既定は `auto`。画像ごとに横書き・縦書きの両方で読み、「文字数 × 信頼度」が大きい方を採る（1枚につき2回読むので、時間は約2〜3倍）。`--direction vertical` などで決め打ちもできる
- 言語：`--lang` を省略すると向きごとの既定値（横 `jpn+eng` / 縦 `jpn_vert+eng`）。英語の画像も `jpn+eng` で読む

- プリセット：`--presets` の既定は `all`（[presets.py](presets.py) の全部）。最後に、データセットごとに CER が一番小さかったプリセットを表示する

出力は `out/tesseract_eval/` に、IR・認識した全文（`.ocr.txt`）・画像ごとの `results.csv`・集計の `summary.json`・比較表の `compare.csv`（行＝プリセット、列＝データセットごとの CER と全体の CER・秒/枚。全体の CER が小さい順）。`results.csv` の `score_horizontal` / `score_vertical` は、`auto` で向きを選んだときの横・縦それぞれの点数。

## ファイル

| ファイル | 内容 |
|---|---|
| [ir_adapter.py](ir_adapter.py) | 解析の流れ：画像を開く → 前処理 → Tesseract（auto なら横・縦の2回）→ IRに変換 |
| [presets.py](presets.py) | 設定の組（プリセット）。前処理は [easyocr_ocr/preprocess.py](../easyocr_ocr/preprocess.py) の名前を使い、EasyOCRと同じ名前なら同じ前処理 |
| [evaluate.py](evaluate.py) | 正解テキストとの CER（[evaluation/metrics.py](../evaluation/metrics.py) の `char_errors`）と処理時間 |
| [predict.py](predict.py) | 旧版。画像/PDFの文字の高さからサイズクラス（大/中/小）を表示するだけ（IRは出さない） |

- 向き `auto` は横書き（`jpn+eng`, psm 3）と縦書き（`jpn_vert+eng`, psm 5）の両方で読み、「文字数 × 信頼度」が大きい方を採る。時間は約2倍。
- 縦書きの段落は、列を右→左、列の中を上→下に並べ直して `text` にする（列の区切りは `\n`）。
- `font_size` は横書きなら行の高さ、縦書きなら列の幅の中央値（元の画像の px）。`bold` は取れないので書かない。
- 日本語の単語の間には空白を入れない（英数字どうしの間だけ入れる）。
- 前処理で拡大しても、`bbox`・`font_size` は元の画像の大きさに戻してから書く。使った設定はIRの `params` に記録される。

## 結果（2026-10-25, Tesseract 5.5.3, 正解 .txt のある各2枚）

CER は空白・改行を除いた編集距離 ÷ 正解の文字数（データセットごとに合計してから割る）。

| preset | eng | jphorizon | jpvertical |
|---|---|---|---|
| base | 1.12% | **1.57%** | 5.68% |
| up2 | **1.06%** | 2.18% | **2.33%** |
| otsu | 1.35% | 1.90% | 8.48% |

- 縦書きは拡大（up2）で大きく改善し、二値化（otsu）では悪化した。
- 上の表は、向き・言語をフォルダ名から決めていたとき（英語は `eng`）の結果。
- フォルダ名を使わない `--direction auto` にしても（2026-10-26）、6枚すべてで正しい向きを選んだ。`base` の CER は英語 1.14%（`jpn+eng` で読むため 1.12% からわずかに変化）、横書き 1.57%、縦書き 5.68% で、日本語は変わらなかった。6枚の評価時間は約15秒 → 約44秒。
- 英語の `CER_ja` が数百%になるのは、正解に非ASCII文字が数個（“ ” だけ）しかなく、Tesseract が出した `—` などで割合が大きく見えるため。英語のデータセットでは `CER_ja` は見なくてよい。
