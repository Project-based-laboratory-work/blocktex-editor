# hybrid_extractor — ルールベース＋自作AIのPDF抽出器

PDFから文字・表・図をブロックとして取り出し、ブロックの種類（見出し・段落・数式…）を当てる。
**ルールのオペレータ**が、小さい**専門家**（ルール版 → 1人ずつ自作AIに入れ替える）に聞いて決める。

```
PDF ─ extract.py ─▶ 行 ─ 区切りの専門家「次の行は今のブロックの続きか」─▶ ブロック
                                                                            │
    ┌───────────────────────────────────────────────────────────────────────┘
    └▶ operator_rules.py（ルールのオペレータ）
         1. 表・図の領域 → table / figure
         2. 確実な決まり → caption / title・author・date / other（ページ番号・ヘッダー）
         3. 手がかりで聞く専門家を選ぶ（等幅フォントがあればコードの専門家、など）
         4. ラベルの専門家（experts.py）に「自分のラベルである自信 0〜1」を聞き、一番自信のある答え。
            誰も 0.5 に届かなければ paragraph
       ─▶ ラベル付きIR
```

| 専門家 | 見るもの | 答え | 今 |
|---|---|---|---|
| 区切りの専門家 | 今のブロックと次の行 | 続きである自信 | ルール版（`extract.rule_continues`） |
| コード / 数式 / 箇条書き / 見出しの専門家 | 1ブロック | 自分のラベルである自信 | ルール版（`experts.RULE_EXPERTS`） |

専門家が全員ルール版のときは、`rules.py`（hybrid_rules）と同じ答えになる（テストで確かめている）。

入出力はチーム共通のIR（[common/README.md](../common/README.md)）。評価は [evaluation/](../evaluation/README.md) に `hybrid_rules` として登録している。

## 使い方

`ai/` で実行する。

```bash
# 抽出だけ（IR ①）
python -m hybrid_extractor.extract hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.ir.json
# 抽出＋ルール分類（ラベル付きIR ②）
python -m hybrid_extractor.rules hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.labeled.json
# オペレータ＋専門家（--trace で、ブロックごとにどの専門家がどう答えたかを表示）
python -m hybrid_extractor.operator_rules hybrid_extractor/sample_data/sample.pdf hybrid_extractor/data/sample.op.json --trace
# 評価用データセットで採点（既存の rule_based と比べる）
python -m evaluation.run --model hybrid_rules --dataset evaluation/data/synth_all
python -m evaluation.run --model hybrid_operator --dataset evaluation/data/synth_all
# 学習データを作る（12並列で 40文書 15秒くらい）
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/train --docs 1500 --seed 0
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/val --docs 300 --seed 1
python -m hybrid_extractor.build_dataset --out hybrid_extractor/data/test --docs 300 --seed 2
# テスト
python -m pytest hybrid_extractor
```

サンプルのPDFは [sample_data/sample.tex](sample_data/sample.tex) から作る（PDFは git に入れない）:

```bash
cd hybrid_extractor/sample_data && uplatex sample.tex && uplatex sample.tex && dvipdfmx sample.dvi
```

## ファイル

| ファイル | 内容 |
|---|---|
| [look.py](look.py) | PDFの中身をのぞく練習用（Step 1） |
| [extract.py](extract.py) | 抽出器。行 → 横につなぐ → 表・図 → 縦につないでブロックにする |
| [rules.py](rules.py) | ルール分類器（ベースライン）。オペレータも、確実な決まり（キャプション・タイトル部分・ページ番号）をここから使う |
| [experts.py](experts.py) | ラベルの専門家（ルール版）。AIの専門家も同じ形（`Expert`）で作る |
| [operator_rules.py](operator_rules.py) | ルールのオペレータ。聞く専門家を選び、答えをまとめる。最後に数式を読む係で数式を LaTeX にする |
| [math_reader.py](math_reader.py) | 数式を読む係（PDFの文字 / ルールで組み立て / math_ocr / ルール＋math_ocr）と、文中の数式の範囲を探す |
| [inline_math_eval.py](inline_math_eval.py) | 文中の数式の比較実験（読む係を、正確さと処理時間で比べる） |
| [doc_gen.py](doc_gen.py) | 学習用のランダムな文書（TeX）を作る。評価用とは別の作り方 |
| [build_dataset.py](build_dataset.py) | 学習データ（ブロック＋正解のラベル＋ルールの答え）を作る |
| [sample_data/sample.tex](sample_data/sample.tex) | 開発用の固定サンプル（IR版3の項目がひと通り入る） |
| [tests/](tests/) | pytest |

## 抽出器（extract.py）のしくみ

| 段階 | やること | 手がかり |
|---|---|---|
| 行 | PyMuPDF の line を1文字ずつの情報つきで取る | `get_text("rawdict")` |
| 横につなぐ | 同じ高さに並ぶ line をつなぐ（「1」＋「はじめに」） | 縦の範囲の重なり（`same_row`）。2段組では左右の段をつながない |
| 表 | 左右の端がそろった横線が2本以上 → 表。罫線の無い表は「表1」の下の文字の並び | 列の境目は縦線か「どの行にも文字の無いすき間」。空のセルがある行は上の行の折り返し |
| 図 | 画像と、ベクターの図形のまとまり | 直線だけで中に文字がある囲みは、図ではなく文字の枠（枠付きのコード） |
| 縦につなぐ | 段落の続きの行・数式の断片・コードの行をまとめる | 行の間のすき間、字下げ（段の左端と比べる）、文字の大きさ・書体の変化、数式・等幅フォントの割合 |
| 装飾 | 太字・斜体・下線 → `runs` | 書体名（本文が明朝ならゴシックは太字）、ベースラインのすぐ下の横線 |
| コードの字下げ | 行の左端のずれ ÷ 1文字の幅 → 空白の数 | 等幅フォント |
| 読み順 | 2段組は、段をまたぐもので帯に区切り、左の段 → 右の段 | |

## 数式を読む係（math_reader.py）

PyMuPDF の文字のままだと、数式は `∑ n / f(x) = aixi / i=1` のように崩れる。数式の範囲の文字を「数式を読む係」に渡して LaTeX にする。
対応させる数式は、math_ocr v2 が読める記号の一覧（ikumi さんのドキュメント「数式AI（math_ocr v2）が読める記号一覧」）。

| 読む係 | やり方 | 重み（best.pt） |
|---|---|---|
| `pdf_text` | PDFの文字を左から並べるだけ（基準） | 不要 |
| `rule` | 文字の位置・大きさ・フォントから組み立てる | 不要 |
| `ocr` | 行ごとに 200dpi で切り抜いて math_ocr で読む | 要る |
| `rule+ocr` | `rule` で読めなければ `ocr` | 要る |

`rule` の組み立て方（先に「1文字にまとめる」下ごしらえをしてから、左から添字を付ける）:

| 形 | PDF の中 | 読み方 |
|---|---|---|
| `x_{i}^{2}` | 小さくてベースラインが上（下）にずれた文字 | 上付き（下付き）。上下に重なる添字は `_{}^{}` の順 |
| `\sum_{i=1}^{n}`、`\lim_{x \to 0}` | 別行立ては記号の真上・真下、文中は右横 | 真上・真下なら範囲、右横なら添字。`lim` などは立体の文字の並びから1つにまとめる |
| `\frac{a}{b}`、`\sqrt[3]{x}` | 細い横線（図形）の上下・`√` の右 | 線に一番近い段の文字だけを分子・分母にする（入れ子も読める）。`√` の上半分の小さい文字は何乗根 |
| `\binom{n}{k}`、`\left( \right)` | CMEX の大きいかっこ | 中が横線なしの上下2段なら `\binom`、それ以外は `\left( \right)` |
| `\bar{x}` など | `¯ ˆ ⃗ ˜ ˙` が土台の文字に重なる別の文字 | 重なる文字と組にする（幅の無い `⃗` は 3pt ほど左に置かれる） |
| `\neq` | 斜線（U+0338）と `=` を重ねて描く | 1つにまとめる |
| `\mathbb{R}`、`\mathbf{x}` | 文字は同じで、フォントが MSBM・CMBX | フォント名で見分ける |
| `\dots`、`\cdots`、`f'` | `.` 3つ・`·` 3つ・小さい `′` | 3つ並んだらまとめる。`′` は上付きにしない |
| `\phi` / `\varphi`、`\Delta`、`\Omega` | `ϕ`/`φ`、見た目が同じ別の文字（U+2206・U+2126） | 対応表で区別する |
| 式番号 `(1)` | 右端に離れて置かれる | 除く |

- **大きい記号の枠**: PyMuPDF の普通の枠は、CMEX などのフォントの大きい記号（`∑ ∫ √`、大きいかっこ）では実際より 10pt ほど上にずれる。
  抽出器（`extract.page_lines`）で、この記号だけ文字の形から計算した正確な枠（`TEXT_ACCURATE_BBOXES`）に差し替えている
- **文中の数式の範囲**（`inline_math_spans`）: 数式のフォント（CMMI・CMSY・MSBM など）とアクセント記号の文字を種にして、間が `=` `+` `(` や数字・空白・関数名だけならつなぐ。後ろの添字・閉じかっこ・`!`、前の開きかっこ・関数名（`sin`・`log`・`lim`）も入れる
- **数式のかけらの行**: PyMuPDF は、文中の分数の分子や `∫` を別の行にすることがあるので、抽出器（`merge_rows`）で、数式のフォントだけの小さい行を近くの段落の行の横の位置に差し込む
- 別行立ての数式（label="math"）→ 要素の `tex`。文中の数式 → `runs` の `tex`（IR版4）。エディタ用JSONでは文中の数式になる
- オペレータでは `--math rule`（既定）で選ぶ。評価の登録簿では `hybrid_operator` の `params.math`
- 重みは math_ocr の担当者が更新することがあるので、コードに書かず引数（`--math-checkpoint`）で渡す

### 比較実験（inline_math_eval.py）

文中の数式をたくさん含む文書を作り（`$...$` の前後に目印）、目印の間の文字を正解の範囲、TeX に書いた数式を正解の LaTeX にする。
数式は記号の一覧の18項目（文字・ギリシャ文字・演算子・関係・集合・矢印・添字・大きな演算子・lim/max/min・関数・分数・根号・二項係数・アクセント・書体・かっこ・点々・その他）から作る。
比べ方はトークン誤り率（evaluation の `math_ter` と同じ区切り方。`\,` などの間隔だけの命令は除く）と完全一致率、1式あたりの時間。

```bash
python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math --docs 60
python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math --docs 60 --math-checkpoint <best.pt>  # ocr・rule+ocr も
```

結果（60文書、文中の数式 1,195個。CPU）:

| | 読む係 | トークン誤り率 | 完全一致率 | 1式あたり |
|---|---|---|---|---|
| 読む（正解の範囲を渡す） | pdf_text | 0.501 | 0.441 | 0.02ms |
| | **rule** | **0.008** | **0.985** | 0.16ms |
| 通し（見つけた範囲を読む） | pdf_text | 0.507 | 0.419 | 0.01ms |
| | **rule** | **0.095** | **0.872** | 0.08ms |
| | ocr・rule+ocr | （重みが届いたら） | | |

見つける段階（範囲の検出）: 適合率 0.843、再現率 0.902、F1 0.872。通しの成績が読む段階より低いのは、ここで落としているため。

評価用データセットの別行立ての数式（`rule`）: math_ter synth_all **0.026**、synth_math **0.017**（`\,` の分も数えている）。

## 結果

### 評価用データセット（evaluation/data/synth_all、40文書）

| 指標 | rule_based（既存） | hybrid_rules |
|---|---|---|
| ブロックの検出 F1 | 0.414 | **0.980** |
| ラベルの正解率 | 0.511 | **1.000** |
| 読み順 | 0.943 | **0.969** |
| 装飾の F1 | − | **1.000** |
| 表のセル / 揃え / 罫線 / 列幅 | 0.817 / 0.830 / 0.974 / 0.167 | **1.000** / 0.807 / **1.000** / **1.000** |
| コードの字下げ | 0.333 | **0.750** |

この評価用データセットは作りがそろっている（jsarticle、番号付きの見出し、明朝の本文）ので、ルールだけでラベルが全部当たる。
AIの効果は、作りをばらつかせた学習データの検証用・テスト用（下）と、本物のPDFで見る。

### 学習データ（doc_gen.py の文書）でのルールの成績

学習用 1500文書（28,550ブロック）・検証用 300文書（5,654ブロック）。正解は目印から作り、抽出器のブロックの面積の半分以上が重なる正解のラベルを付けた（どれとも重ならない 1% 弱のブロックは使わない）。

ルールの正解率: 学習用 **0.785**、検証用 **0.775**。検証用のラベルごと:

| ラベル | ブロック数 | ルールの正解率 | ルールの主な間違い |
|---|---|---|---|
| paragraph | 1270 | 0.998 | |
| heading | 1011 | 0.790 | 段落（番号の無い、本文と同じ大きさの太字の見出し） |
| list | 779 | 0.389 | 段落（★・―・(a) の項目、description） |
| code | 599 | 0.514 | 段落（行番号つき・小さい文字・枠つきのコード） |
| other | 446 | 0.774 | 段落（ページ番号・ヘッダー） |
| caption | 372 | 0.895 | 段落 |
| math | 340 | 0.818 | 段落（式番号など） |
| table | 269 | 0.680 | 段落（罫線が無く、キャプションが下の表） |
| title / author / date / figure | | 1.000 | |

ここが自作AIの目標: **ルールの 0.775 を超え、とくに list・code・heading・table を当てる**。

### 区切り（行と行をつなぐか）のルールの成績

学習データの各ページの `pairs`（隣り合う2行の組）で数えた。検証用 8,762組のうち、正解で「つなぐ」は 52.8%。

| | 学習用 | 検証用 |
|---|---|---|
| ルールの区切りの正解率 | 0.903 | 0.902 |
| 割りすぎ（本当は続きなのに分けた） | 3,825 | 782 |
| つなぎすぎ（本当は別なのにつないだ） | 411 | 78 |

割りすぎは、項目の間が広い箇条書き・行番号つきのコード・数式に多い。つなぎすぎは、字下げの無い段落（`\noindent`）どうし。

## 計画（自作AIの部分）

| Step | 内容 |
|---|---|
| 6 | データを知る・特徴づくり（専門家みんなで使う） |
| 7 | 区切りの専門家（AI）: 2行は続きか。二値分類。目標: 検証用でルールの 0.902 を超え、割りすぎを減らす |
| 8 | ラベルの専門家（AI）: 箇条書き → コード → 見出し → 表… を1人ずつ。目標: ルールのラベル正解率 0.775 を超える |
| 9 | オペレータの調整: 呼ぶ専門家の決め方、しきい値を検証用データで決める |
| 10 | 比べる: ルールだけ / 専門家＋オペレータ（テスト用データ・評価用データセット・本物のPDF） |

## 分かっている課題

- 文中の数式の見つけ方: 数式のフォントの文字の後ろの `= 99` のような数字だけの部分、太字の書体の1文字（`\mathbf{x}`。本文の太字と区別がつかない）は取りこぼす
- 数字だけの数式（`$3.14$`）は、本文の数字と区別がつかないので見つけない

- 罫線の無い表で、キャプションが表の下にあるものは見つけられない（ブロックごとに段落になる）
- 列の揃えは、セルの文字の幅が全部同じ列では余白で決めるので、ときどき外れる（評価の揃えの正解率 0.807）
- 数式の文字は、PDFの文字を読み順（上から）に並べただけで、LaTeX にはしていない
