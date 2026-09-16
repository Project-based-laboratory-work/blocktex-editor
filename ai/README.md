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
