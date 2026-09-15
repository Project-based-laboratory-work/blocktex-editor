# blocktex-editor

ブロックエディタでコンテンツを組み立て、コンパイル可能なLaTeXへ変換するツール（作業名。正式なプロジェクトタイトルは未決定 — 下記「未決事項」参照）。

## MVP範囲（対応ブロック種別）

データモデル（[src/model/block.ts](src/model/block.ts)）は以下の6種類を持つ:

- 見出し (heading)
- 段落 (paragraph)
- 箇条書き (list)
- 数式 (math)
- 表 (table)
- 画像 (image)

ただし、Phase1のエディタUIで実装するのは **見出し・段落・箇条書き・数式** の4種類のみ。表・画像はPhase2でUIを実装する（データモデル・TeX変換方針は先に6種類とも決めておく）。

## 対応文書クラス・LaTeXエンジン

**uplatex + jsarticle** を採用する。ソースコード・保存用JSONはいずれもUTF-8前提のため、UTF-8ネイティブなuplatexを使うことで文字コード変換が不要になる。コンパイルは `uplatex`（2回）→ `dvipdfmx` の流れ。必要なパッケージはブロック種別ごとに [src/tex/preamble.ts](src/tex/preamble.ts) にまとめている。

## 画面のワイヤーフレーム（0-1-3）

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
- 表・画像編集UX / 保存データのファイル形式 / 開発体制（役割分担）: チームの要件定義書側で別途決定。

## ディレクトリ構成

| ディレクトリ     | 役割                                                                  |
| ---------------- | --------------------------------------------------------------------- |
| `src/model`      | ブロック/文書の型定義、ID生成、スキーマバージョン                     |
| `src/tex`        | TeX変換方針（プリアンブル定義）。変換ロジック本体はPhase1             |
| `src/components` | ブロックエディタのUIコンポーネント（Phase1〜）                        |
| `src/store`      | zustandによる状態管理（Phase1〜）                                     |
| `src/fixtures`   | サンプルJSON・手書き期待TeX（Phase1のスナップショットテストで再利用） |

## 開発

```bash
npm install
npm run dev          # 開発サーバ起動
npm run lint         # ESLint
npm run format       # Prettier整形
npm run format:check # Prettierチェックのみ
npm run test         # Vitest（一回実行）
npm run test:watch   # Vitest（watchモード）
npm run build        # 型チェック + 本番ビルド
```

チームでの開発ルール・LaTeX環境構築手順は [CONTRIBUTING.md](CONTRIBUTING.md) を参照。
