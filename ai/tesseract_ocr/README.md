# tesseract_ocr — Tesseractによる文字認識・サイズ判定

学習なしの既存OCR（Tesseract）を使うモデル。`math_ocr`（自作の学習モデル）との比較対象になる。

| | tesseract_ocr | math_ocr |
|---|---|---|
| 課題 | 文字認識＋文字サイズ判定（大/中/小） | 数式画像 → LaTeX |
| 学習 | なし（既製のTesseract） | あり（自作CNN+Transformer） |
| 入口 | `predict.predict(image)` | `math_ocr.predict` |

## 実行方法

事前に `tesseract-ocr`（日本語データ `tesseract-ocr-jpn` 含む）と `poppler-utils` をOSに入れておく。

```bash
cd ai
python -m tesseract_ocr.predict rule_based/sample_data/sample.pdf
```
