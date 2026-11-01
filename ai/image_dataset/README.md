# image_dataset — 画像の構造化器のための学習データ生成

画像（スクリーンショット・写真）から文書の構造（見出し・段落・数式・表…）を判定する**構造化器**の、
学習データを自動で大量に作る。作るのは「画像」と「その画像の正解ブロック」の組。

| | 内容 |
|---|---|
| 作るもの | 画像（PNG / JPEG）と、正解のラベル付きIR（[common/ir.py](../common/ir.py)、`unit="px"`、`label_source="auto"`） |
| 正解の単位 | 段落（ブロック）単位。`easyocr_ocr` の `--group`（[grouping.py](../easyocr_ocr/grouping.py)）が目指す単位と同じ |
| OCR | **使わない**。正解はOCRに依存しないので、EasyOCR・Tesseract のどちらの結果とも比べられる |

## 全体の中での位置

```
image_dataset（ここ）                         構造化器の担当（easyocr_ocr など）
──────────────────────                       ─────────────────────────────────
ランダムな文書(TeX) → PDF → 画像 → 崩す ──▶ 画像 ──▶ EasyOCR → grouping → 構造化器 ─▶ 予測のラベル付きIR
                                  │                                                     │
                                  └──────────▶ 正解のラベル付きIR ◀──── bbox の重なり（IoU）で対応付け ──┘
                                                                        学習: 対応した正解のラベルを教師にする
                                                                        評価: evaluation/metrics.py で比べる
```

文書の生成とPDFへの変換は、ブロック分類（[block_classifier](../block_classifier/README.md)）の
`doc_gen.py` / `build_dataset.py` をそのまま使っている。違いは正解の単位で、block_classifier は
「PDFから取り出したテキストブロック1つずつ」、ここは「TeXで置いたブロック1つ = 正解1つ」。

## 実行方法

`ai/` で実行する。uplatex と dvipdfmx（TeX Live）が必要。Python は 3.10 以上（`str | None` の書き方を使っているため）。

```bash
cd ai
pip install pymupdf pdfplumber matplotlib pillow opencv-python-headless numpy pytest   # または pip install -r requirements.txt

python -m image_dataset.build --out image_dataset/data/train --docs 2000 --seed 0   # 学習用（約7分・約600MB）
python -m image_dataset.build --out image_dataset/data/val --docs 300 --seed 1      # 検証用（seed を変える）
python -m image_dataset.preview image_dataset/data/train --n 20                     # 正解を色分けして描く → data/train/preview/
python -m pytest image_dataset/tests
```

時間の目安：10コアのMacで、50文書（画像122枚）が約10秒。`image_dataset/data/` は git の管理外（`.gitignore` の `ai/*/data/`）。

### オプション

| オプション | 既定 | 内容 |
|---|---|---|
| `--docs` | 20 | 作る文書の数。1文書は1〜4ページくらいで、1ページから画像を作る |
| `--seed` | 0 | 乱数の種。**学習用と検証用で必ず変える**（同じ文書が両方に入らないように） |
| `--variants` | 1 | 1ページから作る画像の数。崩し方を変えて何枚も作る |
| `--styles` | `clean=1,screenshot=2,photo=2` | 崩し方と、その割合。`photo` だけなら `--styles photo` |
| `--dpi MIN MAX` | 90 200 | PDFを画像にするときの解像度の範囲（文字の大きさが変わる） |
| `--workers` | 6 | 並列に作る数 |

## 出力

```
<out>/images/doc_000000_p1_v0.png      画像（JPEGの劣化を付けたものは .jpg）。p1 = 1ページ目、v0 = 1枚目の崩し方
<out>/truth/doc_000000_p1_v0.ir.json   正解。画像1枚 = 1ページのラベル付きIR
<out>/tex/doc_000000.tex               元のTeX（確認・デバッグ用）
```

正解のIRの例（一部）:

```json
{
  "source": "doc_000000_p1_v0.jpg",
  "extractor": "image_dataset",
  "params": {"doc": "doc_000000", "page": 1, "variant": 0, "seed": 0, "dpi": 100.9, "style": "photo",
             "paper": [244, 244, 244], "angle": -0.36, "skew": 0.03, "margin": 85, "desk": [200, 200, 195],
             "lighting": 0.112, "blur": 0.67, "noise": 1.71, "scale": 0.916, "jpeg_quality": 82},
  "pages": [{
    "page": 1, "width": 919, "height": 1236, "unit": "px",
    "elements": [
      {"kind": "text", "bbox": [0.16997, 0.20752, 0.29728, 0.22492], "text": "1 変換の比較", "label": "heading", "level": 1, "label_source": "auto"},
      {"kind": "text", "bbox": [0.19695, 0.23714, 0.63482, 0.28762], "text": "• さらに、特徴についてアルゴリズムの精度をシステムとして評価した。\n• 抽出\n• データと数式として述べる。", "label": "list", "label_source": "auto"},
      {"kind": "table", "bbox": [0.42546, 0.40534, 0.56736, 0.4661], "text": "計算量 記号 誤差\n32.4 11.5 58.7\n33.5 19.0 1.6", "label": "table", "label_source": "auto"},
      {"kind": "text", "bbox": [0.46844, 0.55453, 0.52427, 0.58131], "text": "P : δ →t\n{p, i, iu}", "label": "math", "tex": "P : \\delta \\to t \\\\\n\\{ p , i , i_{u} \\}", "label_source": "auto"}
    ]
  }]
}
```

- `bbox` は他のIRと同じく 0〜1 に正規化した値（左上原点）。px に戻すときは `common.ir.denormalize_bbox(bbox, width, height)`
- `params` に、どう崩したか（style・角度・ぼかし・JPEG品質など）が全部入っている。「写真風だけ成績が悪い」などを調べるときに使う
- `level` は見出しのレベル（`\section`=1, `\subsection`=2, `\subsubsection`=3）、`tex` は数式の元のLaTeX
- `text` はPDFに書かれている文字そのもの（OCRの結果ではない）。OCRの文字誤り率を測る正解にも使える

## 構造化器の学習・評価で使う例

```python
from common.ir import load_ir
from easyocr_ocr.grouping import group_document
from easyocr_ocr.ir_adapter import extract_ir
from evaluation.metrics import Counts, count_document, match, summarize

truth = load_ir("image_dataset/data/val/truth/doc_000000_p1_v0.ir.json")
pred = group_document(extract_ir("image_dataset/data/val/images/" + truth.source))

# 学習：OCRの段落と正解ブロックを IoU で対応付け、正解のラベルを教師にする
gt_page, pred_page = truth.pages[0], pred.pages[0]
for i, j in match([el.bbox for el in pred_page.elements], [el.bbox for el in gt_page.elements]):
    pred_page.elements[i]  # 入力（OCRの段落）
    gt_page.elements[j].label  # 教師（正解のラベル）

# 評価：構造化器がラベルを付けた予測のIRと、正解を比べる
counts = Counts()
count_document(labeled_pred, truth, counts)   # labeled_pred = 構造化器の出力
print(summarize(counts))   # precision / recall / f1（段落の切り方）, label_acc, cer など
```

- どの正解とも対応しなかったOCRの段落は、grouping が2つのブロックを1つにまとめてしまった（または分けすぎた）もの。
  学習から外すか、別に数えると grouping の改善点が分かる
- EasyOCRはCPUだと1枚に数秒〜十数秒かかる。OCRの結果は一度保存して使い回すと良い（`easyocr_ocr.compare` と同じ考え方）

## 崩し方（style）

[augment.py](augment.py) の `augment()`。正解の bbox も同じように動かす。

| style | 何をするか |
|---|---|
| `clean` | 何もしない。PDFをそのまま画像にしたもの |
| `screenshot` | 半分の確率でページの一部だけを切り出す（ブロックの間で切るので、ブロックを途中で切らない）。30%でダークモード、30%で背景色（セピア・灰色など）。0.5〜1倍に縮小。半分の確率でJPEG（品質50〜95） |
| `photo` | 70%で紙の色。±3度の傾きと、斜めから撮ったようなゆがみ（周りに机の色が写る）。明るさのむら、ぼかし、ノイズ。0.6〜1倍に縮小。JPEG（品質60〜90） |

傾け・ゆがみでは、bbox の四隅を同じ変換で動かし、それを囲む四角を新しい bbox にしている。
そのため写真風では、bbox の周りに少し余白が増え、隣のブロックと少し重なることがある。

## 正解ブロックの決めごと

[render.py](render.py) の `truth_pages()`。各ブロックの始まり・終わりに埋め込んだ目印（block_classifier と同じ仕組み）
から、ブロックが占める縦の範囲を作り、PDFの文字の行をいちばん近い範囲のブロックに割り当てる。

- **箇条書き**は、項目ごとではなく箇条書き全体で1つ（`list`）
- **表**は `kind="table"`。`text` はセルの文字を行ごとにつなげたもの（`cells` は入れていない）。キャプションは別のブロック（`caption`）
- **図**は `kind="figure"`、`text` なし。今の文書生成では、図は中身が空の枠（`\fbox`）
- **ページをまたぐ段落**は、ページごとに別の正解になる（画像が1ページずつなので）。2段組で段をまたぐものも段ごとに分かれる
- **タイトル・著者・日付**は `other`（block_classifier に合わせている）。**ページ番号**も `other`
- 並び順は文書の中の順（TeXに書いた順）。ただし表と図は、キャプションが上にあっても表・図が先に来る

## 分かっている制限・今後の課題

- 文書の見た目は、TeXで作ったレポート（jsarticle / bxjsarticle）だけ。本・配布資料・Webページ・スライドのような
  レイアウトは無い。足りなければ、TeX以外で文書を作る生成を足す
- 縦書きは無い（grouping も縦書きをまとめないため、当面は対象外）
- 数式の bbox は、PDFが返す文字の範囲なので、記号の端（積分記号の上下など）が少しはみ出すことがある
- 自動生成なので、この正解だけで測った成績は実際より高く出る（block_classifier の 99.3% と同じ）。
  本当の精度は、実際の画像に人がラベルを付けたデータ（`label_source="manual"`）で測る

## 変更したいとき

| 変えたいこと | 場所 |
|---|---|
| 崩し方の種類・強さ・確率 | [augment.py](augment.py) の `augment()`（色の候補はファイル先頭の定数） |
| style の割合の既定値 | [build.py](build.py) の `DEFAULT_STYLE_WEIGHTS` |
| 正解ブロックの作り方（割り当て規則・ページ番号の判定） | [render.py](render.py) |
| 文書の中身（ブロックの種類・文章・フォント・2段組の割合） | [block_classifier/doc_gen.py](../block_classifier/doc_gen.py)。**ブロック分類の学習データも変わる**ので、ikumiさんと相談する |

崩し方を足したら、[tests/test_image_dataset.py](tests/test_image_dataset.py) の
`test_augment_keeps_boxes_inside_image` が全 style で bbox が画像の中に収まっているかを確かめる。
見た目は `preview` で描いて確かめる。

## ファイル

| ファイル | 内容 |
|---|---|
| [build.py](build.py) | 入口。文書を作り、画像と正解を保存する（並列） |
| [render.py](render.py) | TeX → PDF → ページごとの正解ブロック、ページの画像化 |
| [augment.py](augment.py) | 画像を崩す（正解の bbox も一緒に動かす） |
| [preview.py](preview.py) | 正解を色分けして画像に描く（確認用） |
| [tests/](tests/test_image_dataset.py) | pytest。`build_one` のテストは uplatex が無ければ飛ばす |
