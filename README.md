# blocktex-editor

ブロックエディタでコンテンツを組み立て、コンパイル可能なLaTeXへ変換するツール（作業名。正式なプロジェクトタイトルは未決定 — 下記「未決事項」参照）。

## リポジトリ構成（モノレポ）

| ディレクトリ | 役割 |
|---|---|
| `frontend/` | Vite + React + TypeScript のWebアプリ本体（Phase 1〜3） |
| `ai/` | 自作TransformerによるPDFブロック分類の学習コード（Phase 4、Python） |
| `docker/` | TeX Live（uplatex）入りのコンパイル確認用Dockerイメージ |

チーム向けドキュメント: [CONTRIBUTING.md](CONTRIBUTING.md)（ブランチ運用・TeX環境）、[BACKEND.md](BACKEND.md)（Firebase設計）

## MVP範囲（対応ブロック種別）

データモデル（[frontend/src/model/block.ts](frontend/src/model/block.ts)）は以下の6種類を持つ:

- 見出し (heading)
- 段落 (paragraph)
- 箇条書き (list)
- 数式 (math)
- 表 (table)
- 画像 (image)

ただし、Phase 1のエディタUIで実装するのは **見出し・段落・箇条書き・数式** の4種類のみ。表・画像はPhase 2でUIを実装し、そこで「ブロックエディタ→TeX」のMVPが完成する（第2回中間発表の目標）。

## 対応文書クラス・LaTeXエンジン

**uplatex + jsarticle** を採用する。ソースコード・保存用JSONはいずれもUTF-8前提のため、UTF-8ネイティブなuplatexを使うことで文字コード変換が不要になる。コンパイルは `uplatex`（2回）→ `dvipdfmx` の流れ。必要なパッケージはブロック種別ごとに [frontend/src/tex/preamble.ts](frontend/src/tex/preamble.ts) にまとめている。

## 画面のワイヤーフレーム

```
+----------------------------------------------------+
|  [+見出し] [+段落] [+箇条書き] [+数式] ...  ブロック追加  |
+---------------------------+--------------------------+
|                           |                          |
|   ブロックエディタ         |    TeXプレビュー          |
|   （左ペイン）             |    （右ペイン）           |
|                           |                          |
+---------------------------+--------------------------+
```

左ペインでブロックを編集し、右ペインに変換後のTeXをリアルタイム表示する2ペイン構成。

## 未決事項

- 正式なプロジェクトタイトル: 未決定（"blocktex-editor" は作業名）。
- 表・画像の編集UX / ローカル保存のデータ形式 / アカウント機能の目的（個人保存か共同編集か）/ 学習データのラベリング方法: 要件定義書の「未決事項」を参照し、各フェーズ着手前にチームで確定する。

## frontend/ のディレクトリ構成

| ディレクトリ | 役割 |
|---|---|
| `src/model` | ブロック/文書の型定義、ID生成、スキーマバージョン、クラウド保存の型 |
| `src/tex` | TeX変換方針（プリアンブル定義）。変換ロジック本体はPhase 1 |
| `src/components` | ブロックエディタのUIコンポーネント（Phase 1〜） |
| `src/store` | zustandによる状態管理（Phase 1〜） |
| `src/fixtures` | サンプルJSON・手書き期待TeX（Phase 1のスナップショットテストで再利用） |

## 開発

```bash
cd frontend
npm install
npm run dev          # 開発サーバ起動
npm run lint         # oxlint
npm run format       # Prettier整形
npm run format:check # Prettierチェックのみ
npm run test         # Vitest（一回実行）
npm run test:watch   # Vitest（watchモード）
npm run build        # 型チェック + 本番ビルド
```
