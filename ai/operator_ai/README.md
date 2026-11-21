# operator_ai — 画像ごとにOCRの設定を選ぶオペレータAI

**状態：進め方1〜3（採点表 `score_all.py`、分析 `analyze.py`）まで。val 全部（850枚）で採点・分析済み。** この README は、何を作るか・どう進めるか・誰が何を担当するかの共有用。

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
| `precision` / `recall` / `f1` / `miou` / `order_acc` | チームの評価と同じ値（段落の切り方・枠の位置・読み順）。EasyOCR は `easyocr_image_rule_based` と同じく段落にまとめてから比べる |
| `cer` | チームの評価と同じ値（対応が取れた箱の中だけの CER）。参考。取りこぼしても下がらないので、設定の比較には使わない |
| `sec_per_page` | 1枚を読むのにかかった秒数（OCRの結果を保存したときの値） |
| `n_pairs` `n_preds` `n_gts` `iou_sum` `order_concordant` `order_discordant` `text_errors` `text_chars` | チームの評価の値の分子と分母（[metrics.py](../evaluation/metrics.py) の `Counts`）。何枚かをまとめた値を、チームと同じく「足してから割る」で出すために使う |
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

## 分析する（進め方2・3）

```bash
python -m operator_ai.analyze operator_ai/data/val/scores.csv
```

`scores.csv` と同じ場所の `analysis/` に書く（git管理外）。

| 出力 | 内容 |
|---|---|
| `report.md` | まとめ（共有用）。下の表を全部まとめたもの |
| `settings.csv` | 進め方2。設定ごとの `cer_page`・画像ごとの CER の真ん中（`median`）・大きく崩れた画像の割合（`broken`、CER > 0.5）・チームの評価と同じ値（`precision`・`recall`・`f1`・`miou`・`order_acc`・`cer`・`sec_per_page`、足してから割る）・一番になった画像の数（`wins`） |
| `pairs.csv` | 2つの設定のどちらかを画像ごとに選べるときの上限。ルール版で「A か B か」を決めるときの目安 |
| `conditions.csv` | 進め方3。条件（params の値。dpi・縮小率・JPEG品質は区間にまとめる）ごとに、その条件での単独ベスト・オラクル・各設定の CER |
| `worst.csv` | 単独ベストで CER が悪い画像の上位30（正解が50字未満のページは除く）。`memo` 列に原因を書く。作り直しても `memo` は消えない |

CER が悪い画像を目で確かめるページも作れる（[review.py](review.py)）。画像に正解とOCRの枠を重ね、原因の下書き（自動の判定）を付ける。memo はページの上で直して `worst.csv` に書き出せる。

```bash
python -m operator_ai.review image_dataset/data/val operator_ai/results/val/analysis/worst.csv   # → operator_ai/data/val/review.html
```

- CER はすべて「誤りを足してから文字数の合計で割る」
- どれかの設定でエラーになった画像は、比べる条件を揃えるため集計から外す（`report.md` に一覧が出る）
- 「一番良い設定」が同点なら速いほうを選ぶ
- `effective_dpi` は dpi × 縮小率（画面上の文字の細かさ）。params には無い、分析用に足した切り口

### val 全部での結果（850枚、CPU で約5.5時間、2026-11-21）

全854枚のうち、EasyOCR が落ちた1枚と、正解の文字が無い3枚を除いた850枚。

| 方法 | cer_page | 差の埋まり具合 |
|---|---|---|
| 単独ベスト（`tesseract:up2`） | 0.119 | 0% |
| 2つから選ぶときの上限（`tesseract:up2` / `easyocr:up2_lowtext`） | 0.093 | 70% |
| オラクル（上限） | 0.081 | 100% |

- 振り分けで誤りを約3分の1減らせる余地がある。2つの設定から選ぶだけで、その7割が埋まる（200枚のときとほぼ同じ）
- `tesseract:psm6` は全体では最下位（0.349）だが、画像ごとの真ん中は 0.060 と2番目に良く、311枚で一番になる。崩れる画像（22%）を見分けられれば強い
- 文字が細かい画像（`effective_dpi` < 70、71枚）は、どの設定でも悪い（オラクルでも 0.31）
- `effective_dpi` が 140 以上の画像では、拡大しない `tesseract:invert`（≒ `base`）のほうが `up2` より良い（0.064 と 0.074）
- ダークモード（0.181）・2段組（0.185）・ビューアあり（0.157）は単独ベストで悪く、振り分けの余地も大きい
- CER が悪い画像の上位は、Tesseract がほとんど何も読めず（CER ≒ 1）、EasyOCR では読めているものが多い。例：`doc_000040_p2_v0`（ビューアの暗い帯があり、縮小率 0.556）
- 確認が要る画像：
  - `doc_000092_p3_v0`：全部の設定で正解のブロックの外に約320字読んでいる。正解の箱がずれている（`image_dataset` の不具合）かもしれない
  - `doc_000130_p1_v0`：`tesseract:up2` だけ CER 1.0。コードの行と隣の図を1行にまとめて読み、行の中心がコードのブロックから外れたため、行ごと誤りに数えている（`cer_page` の数え方の弱いところ）
- 英語の画像は73枚

## ルール版を作る（進め方4）— chihiro へ

OCR を回し直さなくても始められるように、val（854枚）の採点表と特徴量を `operator_ai/results/val/` に入れてある（git で入る）。

| ファイル | 内容 |
|---|---|
| `results/val/scores.csv` | 採点表（854枚 × 10設定）。列は上の「採点表の列」 |
| `results/val/features.csv` | 画像から測った特徴量（1行 = 画像1枚）。列は下の「特徴量」 |
| `results/val/analysis/report.md` | 分析のまとめ（進め方2・3）。ルールの根拠はここから探す |

**手順**

1. `results/val/analysis/report.md` を読む。どの条件でどの設定が勝つか、2つの設定から選ぶだけでどこまで届くか
2. [rule_example.py](rule_example.py) を写して `rules.py` を作り、`choose(f)` を書く。`f` は特徴量の dict、返すのは設定の名前
3. 評価する。採点表から点数を引くだけなので1秒で終わる。2〜3 を繰り返す

```bash
python -m operator_ai.evaluate operator_ai.rules:choose                                  # ai/ で
python -m operator_ai.evaluate operator_ai.rules:choose operator_ai.rule_example:choose   # 並べて比べる
```

出力は、単独ベスト・自分のルール・オラクルの `cer_page` と「差の埋まり具合」、チームの評価と同じ値（`precision`・`recall`・`f1`・`miou`・`order_acc`・`cer`・`sec_per_page`）、条件ごとの `cer_page`、オラクルとの差が大きい画像の一覧。

**守ること**

- ルールに使えるのは `f`（画像から測った値）だけ。正解の params（`dark_mode`・`dpi` など）は本物のスクショに無いので使えない。
  条件ごとの表で params を見るのは、ルールの手がかりを探すため
- 欲しい特徴量が足りなければ、`features.py` に足す（natsuko に相談）
- val で閾値を合わせると、val の点数は実力より良く出る。最後は別の画像（seed 2 で作る最終テスト用、natsuko が用意）で確かめる

**いまの手がかり**（val、`report.md` から）

- 単独ベストは `tesseract:up2`（0.119）、オラクルは 0.081。`tesseract:up2` と `easyocr:up2_lowtext` の2択で 0.093（差の70%）まで届く
- 見本の `rule_example.py`（文字が大きければ `invert`）は、単独ベストと同じ 0.119 で、まだ上回らない。文字の高さだけでは足りない
- `tesseract:psm6` は311枚で一番だが、22% の画像では大きく崩れる。崩れる画像を見分けられれば大きい
- 単独ベストで悪いのは、ダークモード・2段組・ビューアあり・文字がとても細かい画像

### 特徴量

[features.py](features.py) の `measure(画像のパス)` で測る。854枚で約10秒。

| 列 | 内容 | val での当たり具合 |
|---|---|---|
| `width` / `height` | 画像の大きさ（px） | |
| `is_jpeg` | 拡張子が .jpg / .jpeg | JPEG の劣化ありと 854枚すべて一致 |
| `brightness` | 画像全体の明るさ（0〜1） | |
| `page_dark` / `page_bg` | ページの地が暗いか / 地の明るさ（画像の真ん中の明るさの中央値） | ダークモードと 854枚すべて一致 |
| `band_ratio` | ビューアの帯・背景（ページと違う色で一様な行・列）の面積の割合 | ビューアありの中央値 0.22、なし 0 |
| `ink_ratio` | ページの中で文字（インク）の画素の割合 | |
| `char_height` / `char_height_rel` | 文字の高さ（px、連結成分の高さの中央値）/ 画像の高さに対する割合 | dpi × 縮小率との相関 0.85 |
| `n_chars` | 文字くらいの大きさの連結成分の数 | |
| `center_gap` | ページの真ん中に、左右に文字があってすき間が縦に通っている行の割合（2段組の手がかり） | **弱い**。0.5 で分けると段組みとの一致は約7割 |

各ファイルを作り直すとき（natsuko）:

```bash
python -m operator_ai.score_all image_dataset/data/val
python -m operator_ai.features image_dataset/data/val
cp operator_ai/data/val/scores.csv operator_ai/data/val/features.csv operator_ai/results/val/
python -m operator_ai.analyze operator_ai/results/val/scores.csv
```

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
| natsuko | オペレータAI全体のまとめ役。`image_dataset`（学習データ）、採点表・分析（進め方 1〜3）、学習版（5）、評価（6） | `image_dataset`・採点表（1）・分析と原因のメモ（2〜3）・特徴量・評価ツールは済み。最終テスト用の画像（seed 2）はこれから |
| chihiro | ルールベース。構造化器 `image_rule_based`（済み）と、ルール版オペレータ（進め方 4） | 4 はこれから。始め方は上の「ルール版を作る」 |

- 進め方 3 の分析結果は、4（ルール版）の根拠にも、5（学習版）の材料にもなる。3 を先に共有すれば、ルール版と学習版の作業がかぶらない
- 他の人のフォルダ（`tesseract_ocr/`・`easyocr_ocr/`・`image_rule_based/`・`evaluation/`）のコードは書き換えない。このフォルダから呼ぶだけにする

## フォルダ構成

[ai/README.md](../README.md) の「新しいモデルを作るときの構成」に合わせる。

```
ai/operator_ai/
  README.md      # これ
  __init__.py
  score_all.py   # 進め方 1：全部の設定で読んで採点表を作る（済み）
  features.py    # 画像の特徴量を測る（済み）
  evaluate.py    # オペレータを採点表だけで評価する（済み。進め方 4〜6 で使う）
  rule_example.py  # ルール版の見本（済み）
  review.py      # CER が悪い画像を目で確かめるページ（済み）
  analyze.py     # 進め方 2・3：単独ベスト・オラクル、条件ごとの勝ち負けの表（済み）
  rules.py       # 進め方 4：ルール版
  train.py       # 進め方 5：学習版
  predict.py     # 推論の入口。画像 → 設定を選ぶ → OCR → IR
  data/          # 採点表・OCRの結果・学習済みモデル（.gitignore）
  results/       # チームで共有する採点表・特徴量・分析（git に入れる）
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

**評価項目は CER・f1・速さの3つ**（決定、2026-11-21）。どの方法も3つを並べて出す。

| 項目 | 列 | 見るもの |
|---|---|---|
| CER | `cer_page` | 文字の読み取りの良さ（取りこぼし・余計な文字も誤り）。小さいほど良い |
| f1 | `f1` | 段落の切り方・枠の位置（チームの評価と同じ値）。大きいほど良い |
| 速さ | `sec_per_page` | 1枚を読む秒数（選んだ設定のOCRにかかった時間）。小さいほど良い |

- 単独ベスト・オラクル・差の埋まり具合は CER で決める。f1 と速さは、そのとき一緒に出す
- 3つは食い違うことがある。val では、EasyOCR（`up2_lowtext`）は CER で負けるが f1 は高く（0.59 と 0.57）、秒数は約10倍（9.1秒と 0.9秒）
- CER だけで選ぶと f1 が下がる。val のオラクル（CER が一番の設定を選ぶ）は、CER 0.081 と単独ベストの 0.119 より良いが、f1 は 0.43 と単独ベストの 0.57 より悪い。CER で勝ちやすい `psm6`・`psm11` は段落をうまく切れない（f1 0.16・0.15）ため
- 読み順（`order_acc`）は、Tesseract が 0.96〜1.00、EasyOCR が約 0.95。`psm6` は段落をほとんど切らない（recall 0.11）ので、読み順はほぼ崩れない（0.999）
- 値は「足してから割る」で、チームの評価と同じ計算（画像ごとの値の平均ではない）
- ダークモード／通常、日本語／英語、文字の大小、のような切り口ごとにも出す
- 採点は [evaluation/metrics.py](../evaluation/metrics.py) と同じ物差しを使う。3項目のほかに、PDF 側と同じ `precision`・`recall`・`miou`・`order_acc`（読み順）・`cer` も並べて出す（何枚かをまとめるときは、チームと同じく `Counts` を足してから `summarize` で割る）
- PDF 側の指標のうち、`label_acc`・`level_acc`・`deco_f1`・表・数式・コードの指標は出さない。ラベルや装飾は OCR の後ろの構造化器が付けるので、OCR の設定を選ぶオペレータでは変わらないため

## 未決事項

- 3つの評価項目の重み：CER が良くなる代わりに、f1 や速さがどこまで悪くなってよいか（例：1枚の秒数の上限）
- 目標の数字（例：ルール版で単独ベストを上回る、学習版で差の半分以上を埋める）

## 関連

| | |
|---|---|
| [image_dataset](../image_dataset/README.md) | 学習データ（画像＋正解のラベル付きIR）の自動生成 |
| [tesseract_ocr](../tesseract_ocr/README.md) / [easyocr_ocr](../easyocr_ocr/README.md) | 振り分け先の OCR とプリセット |
| [image_common/preprocess.py](../image_common/preprocess.py) | 前処理の部品（Tesseract と EasyOCR で共有） |
| [image_rule_based](../image_rule_based/README.md) | OCR の後ろにつながる構造化器 |
| [evaluation](../evaluation/README.md) | 採点の物差し（metrics.py） |
