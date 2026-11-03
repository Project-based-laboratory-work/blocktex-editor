# evaluation — モデルの評価（正解付きデータセットで採点し、サイトで見る）

PDF・画像を読み込むモデル（抽出器 + 構造化器）を、**正解が分かっている文書**で採点する。
どのモデルの出力も同じ形（[common/ir.py](../common/ir.py) のラベル付きIR）なので、同じ物差しで比べられる。
結果は評価サイト（[web/eval.html](../../web/eval.html)）で、一覧・モデル詳細・ケース詳細として見られる。

```
make_dataset.py                       run.py                                     web/eval.html
──────────────                        ──────                                     ─────────────
ランダムな文書(TeX)                    models.json に登録したモデルを              結果一覧（モデル × 指標）
  → PDF（input.pdf）          ──▶     全ケースで動かし、正解と比べて採点   ──▶   モデル詳細（強み・弱み、切り口ごと、混同行列）
  → 正解IR（gt.labeled.json）          → data/results/<run_id>/report.json          ケース詳細（ページに正解と予測の枠を重ねる）
  → ページ画像（page-<n>.png）          → data/results/<run_id>/preds/
```

## 使い方

`ai/` で実行する。データセットを作るには uplatex と dvipdfmx（TeX Live）が要る。

```bash
python -m evaluation.make_dataset --out evaluation/data/synth_all --docs 40 --features all   # 全部入りのデータセット（約1分半）
python -m evaluation.run --model rule_based --dataset evaluation/data/synth_all              # 採点する（成績と強み・弱みが出る）
python -m pytest evaluation                                                                  # テスト
```

### 機能ごとに評価する

`--features` で文書に入れるものを選び、`--rate` でそれぞれを入れる文書の割合を決める（既定 0.5）。

| feature | 入るもの | 正解に入るもの | tags |
|---|---|---|---|
| `title` | `\maketitle` のタイトル・著者・日付 | `title` / `author` / `date` | `has_title` |
| `decoration` | 段落・箇条書きの中の太字・斜体（英語だけ）・下線 | 要素の `runs` | `has_decoration` |
| `table` | 表（列の揃え・罫線の種類がランダム）とキャプション | `kind="table"`、`cells` / `aligns` / `col_rules` / `row_rules`、`caption` | `has_table` |
| `figure` | 図（画像のPNG、ときどき枠線だけの図）とキャプション | `kind="figure"`、`caption` | `has_figure` |
| `math` | 別行立ての数式（`align*`、1〜3行） | `label="math"`、正解のLaTeX `tex` | `has_math` |
| `code` | コード（`verbatim` / `lstlisting`、字下げあり） | `label="code"`、字下げを保った `text` | `has_code` |
| `two_column` | 2段組 | 段ごとに分けた要素 | `columns` |
| `page_number` | ページ番号 | `label="other"` | `has_page_number` |

```bash
# 1つの機能だけを、全文書に入れたデータセット（10件で約15秒）
python -m evaluation.make_dataset --out evaluation/data/synth_table --docs 10 --features table --rate 1
# 全部入り。各機能は半分くらいの文書に入るので、「表がある文書 / 無い文書」などで比べられる
python -m evaluation.make_dataset --out evaluation/data/synth_all --docs 40 --features all
```

`--features` を指定しなければ、以前と同じ文書（ときどきタイトル、文中の装飾あり、表などは無し）になる。

サイトで見る・サイトから評価を実行するときは、サーバーを `npm run dev` で起動してから `web/eval.html` を開く
（Dockerのサーバーにはイメージの中に `ai/` が入っていないので、評価の窓口は使えない）。

```bash
npm run build              # リポジトリ直下。web/src/eval.ts → web/js/eval.js
cd server && npm run dev   # listening on :8080 と出たら web/eval.html をブラウザで開く
```

`evaluation/data/`（データセットと結果）は `.gitignore` の `ai/*/data/` で git の管理外。コマンドで作り直せる。

## データセット（make_dataset.py）

```
<out>/manifest.json               全ケースの一覧と tags（lang / pages / has_list / has_title / has_decoration、--features で選んだものの tags）
<out>/cases/<id>/input.pdf        モデルに渡すPDF
<out>/cases/<id>/gt.labeled.json  正解（ラベル付きIR、label_source: "auto"）
<out>/cases/<id>/page-<n>.png     ページの画像（サイトの下絵。100dpi）
<out>/cases/<id>/source.tex       元のTeX（確認用）
```

- 文書は [synth_doc.py](synth_doc.py) がランダムに作る（日本語 / 英語を交互）。見出し・段落・箇条書きはいつも入り、ほかは `--features` で選ぶ
- **正解は目印で作る**（色は使わない、チームの方針）。各ブロックの始まりと終わりに `\special{pdf:dest}` の目印を置き（[marks.py](marks.py)）、目印の間にある文字の位置を包む枠を正解の bbox にする（[ground_truth.py](ground_truth.py)）。ラベル・見出しレベルは文書を作ったときに分かっている。文中の装飾も、装飾した語の前後の目印から `runs` にする
  - 表・図・コードは、目印をブロックの上と下の別の行に置き、その間だけを集める。表・図は範囲の中の罫線・画像も枠に含める
  - 表の `cells`・揃え・罫線、数式の `tex`、コードの文字（字下げごと）は、PDFからではなく文書を作ったときの値を使う
  - 数式は分数の分子・分母などがベースラインからずれるので、範囲を上下に 14pt 広げる
  - 2段組では、範囲を段ごとに分けてその段の文字だけを集める。段・ページをまたぐブロックは、段・ページごとの要素になる
  - ページ番号は、フッターに置いた目印（`pn<ページ>s` / `pn<ページ>e`）から `label="other"` にし、その文字はほかのブロックに入れない
- フォントや PyMuPDF のブロックの区切り方に頼らないので、特定の抽出器だけが有利になることがない
- [block_classifier/doc_gen.py](../block_classifier/doc_gen.py) とは別に作っている。学習データと同じ作り方の文書で評価すると、block_classifier の点数が実力より高く出るため
- 古いデータセットにページの画像を足すだけなら `python -m evaluation.images evaluation/data/synth_v1`

## モデルの登録簿（models.json）

サイトで選べるのはここに登録したモデルだけ（任意のPythonファイルは実行しない）。各モデルのコードは書き換えず、呼ぶ関数を書く。

| キー | 意味 |
|---|---|
| `id` / `name` | 識別子 / サイトに出す名前 |
| `entry` | `モジュール:関数`。ラベル付きIR（またはIR）を返す関数 |
| `input` | `"pdf"`：PDFのパスを渡す。`"image"`：各ページを画像（既定 200dpi、`params.dpi` で変更）にして1枚ずつ渡し、ページをまとめる |
| `outputs` | 出すもの（`bbox` / `text` / `label` / `runs`）。無いものの指標は 0点ではなく「対象外」になる |
| `params` | 関数に渡す引数（`dpi` だけは画像にする解像度に使う） |
| `note` | サイトに出す補足（動かすのに必要なものなど） |

画像のモデル（Tesseract・EasyOCR + [image_rule_based](../image_rule_based/README.md)）は [adapters.py](adapters.py) で「画像1枚 → そのページのIR」の形にしている。
OCRのライブラリが入っていない環境では、そのモデルの評価は全ケース失敗（エラーとして記録され、全部「見逃し」で採点）になる。

## 指標（metrics.py）

予測と正解の枠を IoU ≥ 0.5 で1対1に対応させ（大きい順に貪欲に）、対応した組・余った予測（誤検出）・余った正解（見逃し）から計算する。
どの指標も「数を足してから割る」ので、全体・切り口ごと・ケースごとが同じ仕組みで出せる。

| 指標 | 内容 |
|---|---|
| `precision` / `recall` / `f1` | ブロックの検出。誤検出が多いと precision、見逃しが多いと recall が下がる |
| `miou` | 対応した組の IoU の平均（枠の位置の正確さ） |
| `label_acc` | 対応した組のうちラベルが一致した割合。混同行列も出す |
| `level_acc` | 見出し同士の組のうち見出しレベルが一致した割合 |
| `cer` / `cer_ja` / `cer_en` | 文字の誤り率（編集距離 ÷ 正解の文字数、空白・改行は数えない）。言語別は文字の読み取りだけを見て、並び順の違いは数えない |
| `deco_precision` / `deco_recall` / `deco_f1` | 段落・箇条書き・キャプションの太字・斜体・下線が、正しい文字に付いたか（見出しは書体ごと太字なので採点しない） |
| `table_shape_acc` | 当たった表のうち、行数・列数が一致した割合 |
| `table_cell_acc` / `table_align_acc` / `table_rule_acc` | 同じ位置どうしで比べた、セルの文字・列の揃え・罫線の有無の正解率（表の文字は CER では数えない） |
| `math_ter` | 数式の LaTeX のトークン誤り率（`&` は数えない）。`tex` を出すモデルだけ |
| `code_indent_acc` | コードの各行の字下げ（行頭の空白の数）が一致した割合 |
| `sec_per_page` | 1ページにかかった秒数 |

`None`（サイトでは「—」）は対象外。比べるものが無かったか、モデルがそれを出さない。

**ブロックの種類ごとの成績**（`report.json` の `by_label`）も出す。正解のラベルごとに、

| 項目 | 内容 |
|---|---|
| `found_rate` | その種類の正解のうち、ラベルを問わず枠が対応した割合（見つけられたか） |
| `recall` / `precision` / `f1` | ラベルまで合っていたものだけで数えた分類の成績。precision の分母は、その種類と予測したブロック全部 |

## 強み・弱み（run.py）

`report.json` の `findings` に、次の2種類を文章で出す。

- **切り口ごと**：tags（`lang=en`、`has_table=True` など）ごとの成績を全体と比べ、**5件以上**の切り口で**0.05以上**の差があるものを、差の大きい順に12個まで
- **ブロックの種類ごと**：正解が5個以上ある種類で、分類の F1（ラベルを出さないモデルは見つけた割合）が 0.85 以上なら強み、0.5 以下なら弱み

サイトでは、同じデータセットで評価した各モデルの最新の実行の中での順位も出す。

## 未対応

- 表の列幅（`col_widths`。セル内で折り返す列）と、セルを結合した表
- 段をまたぐ表・図（`table*` / `figure*`）と、ヘッダー（柱）
- 数式の正解の `text`（PDFの文字を並べたもので、分数などは行が分かれる）。数式は `tex` の TER で見る
