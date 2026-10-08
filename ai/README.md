# ai

Phase 4「PDF読み込み→ブロック化＋AI（自作Transformer）分類」で使う、Python側の学習コードを置くディレクトリ。フロントエンド（`frontend/`）とは独立して動かす。

## 役割分担

| 場所 | 担当範囲 |
|---|---|
| `ai/`（ここ） | 学習データの前処理、自作Transformerの実装、学習、精度評価、ONNX形式へのエクスポート |
| `frontend/` | エクスポートされたONNXモデルを onnxruntime-web で読み込み、ブラウザ上で推論してブロックデータに変換 |

学習はIS計算機サーバ上で行う想定（アカウント登録は10/14または10/16の説明会以降）。学習済みモデルはONNXに変換して `frontend/` から読み込む。

## 環境構築（Phase 4着手時）

```bash
cd ai
python3 -m venv .venv
source .venv/bin/activate   # Windows(WSL2)でも同じ
pip install -r requirements.txt
```

`.venv/` と `__pycache__/` はリポジトリ管理外（`.gitignore` 済み）。

## Phase 4での実装予定

- 入力特徴量の設計（PDFから抽出したテキスト・フォントサイズ・座標など）
- 出力クラス: 見出し / 段落 / 表 / 図
- ルールベース分類（ベースライン）との精度比較
- 学習済みモデルのONNXエクスポート（`frontend/` 側で推論するため）

## 4-1-1 調査結果: PDF解析ライブラリの選定

テキスト抽出には **PyMuPDF (pymupdf)**、表検出には **pdfplumber** を使う。

| ライブラリ | 用途 | 選定理由 |
|---|---|---|
| PyMuPDF | テキスト・bbox・フォントサイズ・フォント名・太字判定 | `get_text("dict")` でブロック/行/span単位の詳細なメタデータが取れる。画像（`get_images`）やベクター図形（`get_drawings`）のbboxも同じAPI体系で取得できる |
| pdfplumber | 表領域（bbox）の検出 | `find_tables()` の罫線・列位置の検出精度がPyMuPDF単体より高い |

実装は [rule_based/pdf_extract.py](rule_based/pdf_extract.py) を参照。動作確認用サンプルPDFは [rule_based/sample_data/generate_sample_pdf.py](rule_based/sample_data/generate_sample_pdf.py) で生成する（見出し・段落・表・図を含む）。

```bash
cd ai
source .venv/bin/activate
python rule_based/sample_data/generate_sample_pdf.py   # rule_based/sample_data/sample.pdf を生成
python -m rule_based.pdf_extract rule_based/sample_data/sample.pdf
```

**既知の制限:** 表のヘッダー行の背景色塗りつぶしがPyMuPDFの`get_drawings()`でベクター図形として検出され、同じ領域が「表」と「図」の両方に該当することがある。4-1-2のルールベース分類では表bboxとの重なりを優先しているため実害はないが、4-1-4のラベリング方針やTransformer学習時の前処理では考慮が必要。

## 4-1-2: ルールベース分類（ベースライン）

[rule_based/rule_based_classifier.py](rule_based/rule_based_classifier.py) で、抽出したブロックを 見出し/段落/表/図 に分類する。

```bash
python -m rule_based.rule_based_classifier rule_based/sample_data/sample.pdf
```

ルール:

1. 表・図として検出された領域と重なるテキストは、そのままその種別として扱う（テキストを別途paragraphとして重複計上しない）
2. 残ったテキストのうち、ページ内の本文フォントサイズ（最頻値）の1.3倍以上、または太字のものを見出しと判定
3. それ以外は段落

Transformer（4-3）との精度比較の基準として使う。

## 数式画像 → LaTeX（math_ocr）

数式の画像をLaTeXに変換する、CNN＋Transformerのエンコーダ・デコーダモデル。学習データは自動生成する。
詳細・実行方法・読む順番は [math_ocr/README.md](math_ocr/README.md) を参照。

## モデルの棲み分けと構成ルール

比較検証のため、**1モデル = `ai/` 直下の1ディレクトリ**とし、そのモデルでしか使わないものは全部その中に置く。

| ディレクトリ | 内容 |
|---|---|
| [math_ocr/](math_ocr/README.md) | 数式画像 → LaTeX（自作CNN+Transformer） |
| [tesseract_ocr/](tesseract_ocr/README.md) | Tesseractによる文字認識・サイズ判定（学習なし） |
| [rule_based/](rule_based/) | PDFブロック分類のルールベースライン（PDF抽出・ラベル定義・サンプルPDF） |
| [models/](models/README.md) | アプリから選択できる、配布用モデルの置き場 |

### 新しいモデルを作るときの構成

```
ai/<モデル名>/
  README.md        # 課題（入力→出力）、実行方法、他モデルとの違い
  __init__.py
  predict.py       # 推論の入口（predict関数 + CLI）。アプリ・比較から呼ぶ
  train.py         # 学習があるモデルのみ
  sample_data/     # 動作確認用の少量データ（git管理してよい小さいもの）
  data/            # 生成・ダウンロードした学習データ（.gitignore）
  checkpoints/     # 学習中の保存物（.gitignore）
  tests/           # そのモデルのpytest（`ai/` で `pytest` を実行すると各モデルの tests/ を拾う）
```

- 実行は `ai/` で `python -m <モデル名>.predict ...` の形に統一する（モデル間のimportも `<モデル名>.xxx` で書く）。
- 他モデルのフォルダの中身には依存しない。複数モデルで使う処理が出てきたら `ai/common/` に切り出す。
- 学習済みで、アプリから選べるようにするモデルは [models/](models/README.md) に登録する。
