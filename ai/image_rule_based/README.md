# image_rule_based — 画像の構造化器（ルールベース）

OCR（EasyOCR・Tesseract）で画像から取り出したIRに、**見出し・段落・キャプション**のラベルを付け、
ラベル付きIRとエディタ用JSONにする。PDF版の [rule_based](../rule_based/) の画像版。

```
画像 ─ easyocr_ocr / tesseract_ocr ─→ IR（段落単位）─ image_rule_based（ここ）─→ ラベル付きIR ─ common/blocks.py ─→ エディタ用JSON → TeX
```

## 使い方

`ai/` で実行する。入力は**段落（ブロック）単位のIR**。EasyOCR は行単位で返すので `--group` を付けて段落にまとめる
（Tesseract は最初から段落単位）。

```bash
python -m easyocr_ocr.ir_adapter ../ocr/sample/eng/eng_003.jpg --group -o image_rule_based/data/out/
python -m image_rule_based.ir_adapter image_rule_based/data/out/eng_003.base.group.ir.json -o image_rule_based/data/out/
  # → image_rule_based/data/out/eng_003.base.group.labeled.json  ラベル付きIR
  #   image_rule_based/data/out/eng_003.base.group.blocks.json   エディタ用JSON
python -m image_rule_based.ir_adapter <IR> --stdout   # {"labeled": …, "document": …} を標準出力に出す（rule_based と同じ形）
python -m pytest image_rule_based/tests
```

### 採点する

[image_dataset](../image_dataset/README.md) で作った画像と正解で採点する。EasyOCR の結果は `<dir>/ocr/` に保存して使い回す
（1回目は83枚で約11分、2回目からは数秒）。

```bash
python -m image_dataset.build --out image_rule_based/data/val --docs 30 --seed 1   # 30文書 → 画像83枚
python -m image_rule_based.evaluate image_rule_based/data/val
```

`image_rule_based/data/` は git の管理外（`.gitignore` の `ai/*/data/`）。

## ファイル

| ファイル | 内容 |
|---|---|
| [image_classifier.py](image_classifier.py) | 分類のルール（`classify_page`）。PDF版の `rule_based_classifier.py` に当たる |
| [ir_adapter.py](ir_adapter.py) | IRを読み、分類してラベル付きIR・エディタ用JSONを出す。PDF版の `ir_adapter.py` に当たる |
| [evaluate.py](evaluate.py) | image_dataset の正解で採点する |

## 判定のルール

判定の流れは PDF版と同じ。上から順に当てはめる。

| ラベル | 条件 |
|---|---|
| （捨てる） | 表・図の箱と半分以上重なる文字（表・図の一部とみなす） |
| caption | 「図1」「表2」「Figure 3」「Table 4」などで始まり、100文字以内 |
| heading | （本文の文字の大きさの **1.3倍以上** または 太字）かつ 50文字以内。<br>または、**番号で始まる1行**（「3 実験」「4.1 CPUの評価」）で、30文字以内・文の終わり方（。.、など）をしていない |
| paragraph | それ以外 |

読み順は、両方の段にまたがる要素で区切った区間ごとに「左の段 → 右の段」、それぞれ上から（2段組みに対応）。

### PDF版から変えたところ

画像では成り立たない前提だけを変えている。しきい値（1.3倍・50文字など）は PDF版と同じ値から始めている。

| | PDF版 | 画像版 | 理由 |
|---|---|---|---|
| 本文の文字の大きさ | 同じ大きさの文字数が最も多い値 | 文字数で重みを付けた**中央値** | OCRで推定した大きさは 18.3, 18.7, … とばらつき、同じ値がほとんど出ない |
| 長さの単位 | pt の固定値（段組みのすき間 10pt） | **ページの幅に対する割合** | 画像は解像度がばらばら。同じレイアウトなら画像の大きさによらず同じ結果にする |
| 文字の大きさが無い要素 | （PDFには必ずある） | 箱の高さ ÷ 行数 | EasyOCR の `paragraph=True` などは大きさを出さない |
| 見出し | 大きさ・太字 | 大きさ・太字 ＋ **番号で始まる1行** | OCRの大きさでは差が出ない（image_dataset で見出しの大きさの比の中央値は 1.02）。太字も取れない |

## 成績

image_dataset（30文書・画像83枚、`--seed 1`）、EasyOCR base + `--group`。見出しは、対応が取れた箱の中で数えた値。

| | 見出しの適合率 | 見出しの再現率 | label_acc |
|---|---|---|---|
| 大きさ・太字だけ（PDF版と同じ） | 0.40 | 0.15 | 0.471 |
| ＋ 番号で始まる1行（今） | 0.78 | 0.82 | 0.657 |

- 見出しの誤検出30のうち16はタイトル（大きい文字のため）。タイトルの判定は未対応
- 番号の無い見出し（「まとめ」など）は取りこぼす。「短い1行」も見出しにすると再現率は 0.99 になるが、
  OCRが崩したキャプション・数式も見出しになり、適合率が 0.64 に下がるので入れていない
- 段落の切り方（f1 0.521）は grouping（easyocr_ocr）側の課題。style 別では photo が最も低い（f1 0.472）
- 手で正解を付けた実際の画像10枚（英語・日本語）では、番号の規則を足す前と同じ成績（label_acc 0.982）

## 今の限界

- **太字が取れない**（OCRは太字を返さない）。番号の無い見出しは大きさだけで判定するので、
  英語の章見出し「CHAPTER II」（本文の1.24倍）のようなものは見出しにならない
- **キャプションの多くを取りこぼす**。OCRが「図」「表」を別の字（「川」「町」など）に読み違えるため
- タイトル・著者・日付（PDF版にはある）は判定しない
- 表・図は、OCRが検出しないので今は出てこない（IRに表・図があれば PDF版と同じく扱う）
- 箇条書き・数式・コードのラベルは付けない（PDF版と同じ）
