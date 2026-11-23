# operator_ai — 画像ごとにOCRの設定を選ぶオペレータAI

**状態：進め方1（採点表を作る `score_all.py`）まで。** この README は、何を作るか・どう進めるか・誰が何を担当するかの共有用。

画像（スクリーンショット）を1枚受け取り、その画像に一番合う OCR の設定を1つ選ぶ。
選ぶ候補は、Tesseract の psm・前処理（[tesseract_ocr/presets.py](../tesseract_ocr/presets.py)）と EasyOCR（[easyocr_ocr/presets.py](../easyocr_ocr/presets.py)）。
文字を読むのは今ある Tesseract・EasyOCR で、ブロックの種類（見出し・段落など）を決めるのは構造化器（[image_rule_based](../image_rule_based/README.md)）。
オペレータAIは、その手前で「どの設定で読むか」だけを決める。

| | 内容 |
|---|---|
| 入力 | 画像1枚（スクリーンショットを想定）。1枚＝1ページ |
| 出力 | 選んだ設定の名前（例 `tesseract:invert`）と、その設定で読んだ IR（[common/README.md](../common/README.md) の①）。選んだ理由は IR の `params` に残す |
| 学習データ | [image_dataset](../image_dataset/README.md) の画像と正解。全部の設定で読ませて採点した表を作り、それを材料にする |

## 全体の流れ

```
準備（前もって1回）
  image_dataset の画像 ─▶ 全部の設定で読む ─▶ 採点表（画像 × 設定 → CER・秒数）─▶ 分析 ─▶ 振り分けのルール / モデル
                                                                                                   │
本番（画像が来るたび）                                                                              ▼
  スクショ画像 ─▶ 特徴量を測る（明るさ・文字の高さなど）─▶ オペレータAI ─▶ 選んだ設定で OCR ─▶ 構造化器 ─▶ ラベル付きIR
```

## 採点表を作る（進め方1）

`ai/` で実行する。Tesseract 本体（言語データ jpn・eng）と easyocr が要る。

```bash
python -m image_dataset.build --out image_dataset/data/val --docs 300 --seed 1   # 画像と正解（無ければ）
python -m operator_ai.score_all image_dataset/data/val --limit 5                  # まず5枚で試す
python -m operator_ai.score_all image_dataset/data/val                            # 全部（CPU だと1枚 × 10設定で約20秒。計算機サーバで）
python -m operator_ai.score_all image_dataset/data/val --settings tesseract:base easyocr:up2   # 設定を選ぶ
python -m pytest operator_ai
```

| 出力（`operator_ai/data/<データセット名>/`、git管理外。`--out` で変えられる） | 内容 |
|---|---|
| `scores.csv` | 採点表。1行 = 画像1枚 × 設定1つ。画像を1枚読み終えるたびに書き直すので、途中で止めても使える |
| `ocr/<engine>/<preset>/<case>.ir.json` | 段落単位のOCRの結果（チームの評価と同じ形） |
| `ocr/<engine>/<preset>/<case>.lines.ir.json` | 行単位のOCRの結果（ページ全体のCERに使う） |

OCRの結果は保存して使い回す（読み直すときは `--force`）。設定を足したときは、足した設定だけ読む。
OCR がプロセスごと落ちた（Segmentation fault など）ときは、もう一度同じコマンドを実行すれば続きから読む。落ちた画像 × 設定は `error` 列に `OcrCrashed` と書いて飛ばす（読んでいる間だけ `<case>.running` という目印を置いて見分ける）。
既定の設定は `score_all.py` の `DEFAULT_SETTINGS`（Tesseract・EasyOCR それぞれ base・invert・up2 など5つずつ）。

### 採点表の列

列の名前は、チームの評価（[evaluation/run.py](../evaluation/run.py) の `report.json` と [metrics.py](../evaluation/metrics.py) の `summarize`）に揃えている。

| 列 | 内容 |
|---|---|
| `case` | 画像の名前（`doc_000003_p1_v0`）。report.json の `cases[].id` に当たる |
| `setting` / `engine` / `preset` | `tesseract:invert` / `tesseract` / `invert`。`setting` がオペレータの選ぶ名前 |
| `cer_page` / `cer_page_ja` / `cer_page_en` | **ページ全体の文字の誤り率。設定を比べるときはこれを使う**（下の「ページ全体のCER」） |
| `errors` / `chars` / `ja_*` / `en_*` | `cer_page` の分子と分母。全体の値は「足してから割る」（画像ごとの CER の平均にしない） |
| `extra_chars` | どの正解のブロックにも入らなかった文字の数（ビューアの文字・ゴミなど）。`errors` に含まれる |
| `precision` / `recall` / `f1` / `miou` | チームの評価と同じ値（段落の切り方・枠の位置）。EasyOCR は `easyocr_image_rule_based` と同じく段落にまとめてから比べる |
| `cer` | チームの評価と同じ値（対応が取れた箱の中だけの CER）。参考。取りこぼしても下がらないので、設定の比較には使わない |
| `sec_per_page` | 1枚を読むのにかかった秒数（OCRの結果を保存したときの値） |
| `style` `dpi` `scale` `jpeg_quality` `dark_mode` `crop` `viewer` `background` | 正解IRの `params` から。崩し方が入っていれば True |
| `lang` | 正解の文字から推定（ASCII以外が3割以上なら `ja`。`evaluation.import_dataset.guess_lang`） |
| `columns` | 元のTeX（`<dataset>/tex/`）が twocolumn なら 2 |
| `error` | OCRが失敗したときのエラー。空でなければ点数は空欄 |

`style` から `columns` までは**分析専用**（下の「決めごと」）。オペレータの振り分けには、画像から測った特徴量だけを使う。

### ページ全体のCER

チームの `cer` は、正解の箱と対応が取れた要素だけで数える。これだと、文字を見落とした設定ほど CER が良く見えてしまう。
そこで `cer_page` は、取りこぼしも余計な文字も誤りに数える。

1. OCRの結果を行単位にする（Tesseract は内部の行の位置、EasyOCR は返した行をそのまま使う）
2. 行の中心を含む正解のブロックに、その行を割り当てる（いくつも含むときは小さいほう）
3. 正解の**表・図・数式**の中にある行は捨てる。正解のこの3つも数えない（表はセルの読み方が違い、数式の正解の text はPDFの文字を並べただけのため）
4. ブロックごとに、割り当てた行を上から・左から並べてつなぎ、正解の文字と比べる（`metrics.char_errors`。空白・改行は無視）
5. どのブロックにも入らなかった行は、全部の文字を誤りに数える（`extra_chars`）

ブロック単位で比べるので、ブロックの読む順や段落の区切り方の違いは誤りにならない。文字そのものの読み取りの良さだけを比べられる。

## 進め方

いきなり学習させず、次の順に進める。各段階で動くものができるので、途中で止まっても結果を出せる。

| # | やること | できあがるもの |
|---|---|---|
| 1 | `image_dataset` の画像を、全部の設定で読んで採点する。OCRの結果は保存して使い回す | 採点表（CSV。1行 = 画像1枚 × 設定1つ。画像名・設定名・CER・f1・秒数） |
| 2 | 基準の2つの数字を出す | **単独ベスト**（全画像を同じ設定で読んだときの一番良い設定）と、**オラクル**（画像ごとに一番良い設定を選べたときの点数＝上限） |
| 3 | 採点表に画像の `params`（ダークモード・dpi・縮小・JPEG品質・切り出し・言語・段組み）を付けて、条件ごとにどの設定が勝つかを調べる | 勝ち負けの表、CER が悪い画像の一覧と原因のメモ |
| 4 | 3 の結果を if 文にする（例：画面が暗い → invert、文字が小さい → up2、それ以外 → 単独ベスト） | ルール版オペレータ |
| 5 | 画像の特徴量から、各設定の CER を予想するモデルを学習し、予想が一番小さい設定を選ぶ。最初は決定木・ランダムフォレスト（scikit-learn） | 学習版オペレータ |
| 6 | 学習に使っていない画像で、単独ベスト・ルール版・学習版・オラクルを比べる | 評価の表 |

2 で単独ベストとオラクルの差が小さければ、振り分けで伸ばせる余地は小さい。その場合は先に候補の設定を増やす（`invert` ＋ `up2` のような組み合わせ）。

## 担当

| 担当 | 範囲 | 状態 |
|---|---|---|
| natsuko | オペレータAI全体のまとめ役。`image_dataset`（学習データ）、採点表・分析（進め方 1〜3）、学習版（5）、評価（6） | `image_dataset`・採点表（1）は済み。2〜3 はこれから |
| chihiro | ルールベース。構造化器 `image_rule_based`（済み）と、ルール版オペレータ（進め方 4）を担当するか | **要確認** |

- 進め方 3 の分析結果は、4（ルール版）の根拠にも、5（学習版）の材料にもなる。3 を先に共有すれば、ルール版と学習版の作業がかぶらない
- 他の人のフォルダ（`tesseract_ocr/`・`easyocr_ocr/`・`image_rule_based/`・`evaluation/`）のコードは書き換えない。このフォルダから呼ぶだけにする

## フォルダ構成（予定）

[ai/README.md](../README.md) の「新しいモデルを作るときの構成」に合わせる。

```
ai/operator_ai/
  README.md      # これ
  __init__.py
  score_all.py   # 進め方 1：全部の設定で読んで採点表を作る（済み）
  features.py    # 画像の特徴量を測る（明るさ・暗い画素の割合・文字の高さ・画像の大きさ・周りの灰色の帯・段組みの数など）
  analyze.py     # 進め方 2・3：単独ベスト・オラクル、条件ごとの勝ち負けの表
  rules.py       # 進め方 4：ルール版
  train.py       # 進め方 5：学習版
  predict.py     # 推論の入口。画像 → 設定を選ぶ → OCR → IR
  data/          # 採点表・OCRの結果・学習済みモデル（.gitignore）
  tests/
```

## 決めごと

- **params は分析だけに使う。** `image_dataset` の正解には崩し方（ダークモードか、など）が書いてあるが、本物のスクショには無い。振り分けには、画像から計算した特徴量だけを使う（そうしないと点数が実力より高く出る）
- **学習用と評価用は文書ごとに分ける。** `image_dataset.build` の `--seed` を変えて別々に作る（例：学習 seed 0、検証 seed 1、最終テスト seed 2）。同じ文書の別ページ・別の崩し方が両方に入らないようにする
- **「一番の設定」を当てる問題にしない。** 上位の設定がほぼ同点の画像が多いので、各設定の CER を予想して一番小さいものを選ぶ
- **本物のスクショでも確かめる。** 最終テストに `ocr/sample/` の正解付き画像を入れる。本物で負けたら、その特徴を `image_dataset` の崩し方に足す
- **振り分けの単位はページごとから始める。** 表の部分だけ別の設定、のような領域ごとの振り分けは、ページごとで結果が出てから

## 評価

学習に使っていない画像で、次の表を埋める。

| 方法 | CER（小さいほど良い） | f1 | 1枚の秒数 | 差の埋まり具合 |
|---|---|---|---|---|
| 単独ベスト | | | | 0% |
| ルール版 | | | | |
| 学習版 | | | | |
| オラクル（上限） | | | | 100% |

差の埋まり具合 ＝（単独ベストの CER − オペレータの CER）÷（単独ベストの CER − オラクルの CER）

- ダークモード／通常、日本語／英語、文字の大小、のような切り口ごとにも出す
- 採点は [evaluation/metrics.py](../evaluation/metrics.py) と同じ物差し（CER・f1）を使う

## 未決事項

- 「良い」の定義：CER だけか、f1（段落の切り方）や速さも入れるか
- 目標の数字（例：ルール版で単独ベストを上回る、学習版で差の半分以上を埋める）
- 進め方 4（ルール版オペレータ）の担当

## 関連

| | |
|---|---|
| [image_dataset](../image_dataset/README.md) | 学習データ（画像＋正解のラベル付きIR）の自動生成 |
| [tesseract_ocr](../tesseract_ocr/README.md) / [easyocr_ocr](../easyocr_ocr/README.md) | 振り分け先の OCR とプリセット |
| [image_common/preprocess.py](../image_common/preprocess.py) | 前処理の部品（Tesseract と EasyOCR で共有） |
| [image_rule_based](../image_rule_based/README.md) | OCR の後ろにつながる構造化器 |
| [evaluation](../evaluation/README.md) | 採点の物差し（metrics.py） |
