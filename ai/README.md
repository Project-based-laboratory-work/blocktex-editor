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

実装は [pdf_extract.py](pdf_extract.py) を参照。動作確認用サンプルPDFは [sample_data/generate_sample_pdf.py](sample_data/generate_sample_pdf.py) で生成する（見出し・段落・表・図を含む）。

```bash
cd ai
source .venv/bin/activate
python sample_data/generate_sample_pdf.py   # sample_data/sample.pdf を生成
python pdf_extract.py sample_data/sample.pdf
```

**既知の制限:** 表のヘッダー行の背景色塗りつぶしがPyMuPDFの`get_drawings()`でベクター図形として検出され、同じ領域が「表」と「図」の両方に該当することがある。4-1-2のルールベース分類では表bboxとの重なりを優先しているため実害はないが、4-1-4のラベリング方針やTransformer学習時の前処理では考慮が必要。

## 4-1-2: ルールベース分類（ベースライン）

[rule_based_classifier.py](rule_based_classifier.py) で、抽出したブロックを 見出し/段落/表/図 に分類する。

```bash
python rule_based_classifier.py sample_data/sample.pdf
```

ルール:

1. 表・図として検出された領域と重なるテキストは、そのままその種別として扱う（テキストを別途paragraphとして重複計上しない）
2. 残ったテキストのうち、ページ内の本文フォントサイズ（最頻値）の1.3倍以上、または太字のものを見出しと判定
3. それ以外は段落

Transformer（4-3）との精度比較の基準として使う。
