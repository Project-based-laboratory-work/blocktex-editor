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
python -m common.ir out/easy1.base.ir.json   # 形式チェック
```

## ファイル

| ファイル | 内容 |
|---|---|
| [ir_adapter.py](ir_adapter.py) | 解析の流れ：画像を開く → 前処理 → EasyOCR → IRに変換 |
| [presets.py](presets.py) | 設定の組（プリセット）。実験はここに名前を付けて足す |
| [preprocess.py](preprocess.py) | 前処理の部品（拡大・グレースケール・反転・二値化など）。EasyOCRに依存しない |

使ったプリセットの中身はIRの `params` に記録されるので、出力を見ればどの設定の結果か分かる。
前処理で拡大した場合も、`bbox`・`font_size` は元の画像の大きさに戻してから書く。

## 現状の制限

- 要素はEasyOCRが返した行（または単語）単位のまま。正解データはブロック（段落）単位なので、評価の前に行をブロックにまとめる処理が必要。
- `font_size` は箱の短い辺（横書きなら高さ、縦書きなら幅）から推定している。`paragraph=True` では箱が段落全体になるため書かない。
- 縦書き・数式・表の中身（`cells`）には対応していない。
