# blocktex-editor

ブロックエディタでコンテンツを組み立て、コンパイル可能なLaTeXへ変換するツール（作業名。正式なプロジェクトタイトルは未決定 — 下記「未決事項」参照）。

## リポジトリ構成（モノレポ）

| ディレクトリ | 役割 |
|---|---|
| `web/` | HTML + CSS + TypeScript のWebアプリ本体（Phase 1〜3） |
| `ai/` | 自作TransformerによるPDFブロック分類の学習コード（Phase 4、Python） |
| `server/` | PDFプレビュー用のコンパイルサーバー（Node.js + TypeScript）とそのDockerイメージ |
| `docker/` | TeX Live（uplatex）入りのコンパイル確認用Dockerイメージ |

チーム向けドキュメント: [CONTRIBUTING.md](CONTRIBUTING.md)（ブランチ運用・TeX環境）、[BACKEND.md](BACKEND.md)（Firebase設計）

## MVP範囲（対応ブロック種別）

データモデル（[web/src/model.ts](web/src/model.ts)）は以下の6種類を持つ:

- 見出し (heading)
- 段落 (paragraph)
- 箇条書き (list)
- 数式 (math)
- 表 (table)
- 画像 (image)

ただし、Phase 1のエディタUIで実装するのは **見出し・段落・箇条書き・数式** の4種類のみ。表・画像はPhase 2でUIを実装し、そこで「ブロックエディタ→TeX」のMVPが完成する（第2回中間発表の目標）。

## 対応文書クラス・LaTeXエンジン

**uplatex + jsarticle** を採用する。ソースコード・保存用JSONはいずれもUTF-8前提のため、UTF-8ネイティブなuplatexを使うことで文字コード変換が不要になる。コンパイルは `uplatex`（2回）→ `dvipdfmx` の流れ。必要なパッケージはブロック種別ごとに [web/src/tex.ts](web/src/tex.ts) にまとめている。

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

左ペインでブロックを編集し、右ペインに変換後のTeXをリアルタイム表示する2ペイン構成。右ペインは「TeX」と「PDFプレビュー」をタブで切り替えられる（PDFプレビューには下記のコンパイルサーバーが必要）。

## 未決事項

- 正式なプロジェクトタイトル: 未決定（"blocktex-editor" は作業名）。
- 表・画像の編集UX / ローカル保存のデータ形式 / アカウント機能の目的（個人保存か共同編集か）/ 学習データのラベリング方法: 要件定義書の「未決事項」を参照し、各フェーズ着手前にチームで確定する。

## web/ のディレクトリ構成

| ディレクトリ | 役割 |
|---|---|
| `src/model.ts` | ブロック/文書の型定義、ID生成、スキーマバージョン、クラウド保存の型 |
| `src/tex.ts` | TeX変換方針（プリアンブル定義）。変換ロジック本体はPhase 1 |
| `src/script.ts` | 画面の動作（Phase 1〜） |
| `js/` | `src/` のコンパイル結果（`npm run build` で生成。gitignore済み） |
| `css/` | スタイル定義 |

## 開発

`web/src/*.ts` を tsc でコンパイルして `web/js/` に出力する。バンドラは使わず、`import`/`export` も使わないグローバルスクリプト方式のため、ビルド後は `web/index.html` をブラウザで直接開けば動作確認できる。

```bash
npm install
npm run build      # web/js/ を生成
npm run watch      # 変更を監視して自動ビルド
npm run typecheck  # 型チェックのみ
```

## PDFプレビュー（コンパイルサーバー）

ブラウザはTeXをコンパイルできないため、`server/` のコンパイルサーバーにTeXを送ってPDFにしてもらう。サーバーが動いていなくても、エディタとTeX表示はそのまま使える。

```
ブラウザ（web/）── POST /api/compile {"tex": "..."} ──▶ server/
                                                       一時ディレクトリで uplatex ×2 → dvipdfmx
PDFプレビュータブ ◀──────── PDF（失敗時はエラー文）──────┘
```

### 起動する（Dockerを使う方法・推奨）

本番と同じTeX Live（Debian bookworm = TeX Live 2022）で確認できる。初回のビルドは十数分かかる。

```bash
docker build -t retex-compile server/
docker run --rm --init -p 8080:8080 retex-compile
```

`listening on :8080` と出たら、`web/index.html` を開き、右ペインの「PDFプレビュー」タブを選ぶ。編集が止まってから1.5秒後に自動で更新される。止めるときは `Ctrl + C`。

### 起動する（Dockerを使わない方法）

手元にTeX Live（[CONTRIBUTING.md](CONTRIBUTING.md) の「Dockerを使わない方法」）とNode.js 22.18以上があれば、`.ts` をそのまま動かせる（ビルド不要）。

```bash
cd server
npm install
npm run dev        # ファイルを保存すると自動で再起動する
```

手元のTeX Liveのバージョンによって結果が変わることがあるため、最終確認はDocker側で行う。

### APIの仕様

`POST /api/compile` に `{"tex": "<TeXソース全体>"}` をJSONで送る。

| ステータス | 意味 | 本文 |
|---|---|---|
| 200 | 成功 | PDF（`application/pdf`） |
| 400 | JSONが壊れている／`tex` が文字列でない | `bad request` など |
| 413 | 本文が大きすぎる（約1MB超） | `too large` |
| 422 | TeXのエラー、またはコンパイルが20秒を超えた | ログのエラー部分（3行） |
| 500 | サーバー側の問題 | `server error` |

### 安全対策

誰でも任意のTeXを送れる前提で動かしている。

- `-no-shell-escape`（`\write18` で外部コマンドを実行させない）。そのため、shell-escapeが必要なパッケージ（minted など）は使えない
- コンパイルのタイムアウト（20秒）と、本文のサイズ上限（約1MB）
- リクエストごとに別の一時ディレクトリを作り、終わったら必ず消す
- Dockerイメージの中では一般ユーザー（`node`）で動かす

### 制限・未決事項

- **画像ブロックは未対応**: `\includegraphics` で参照する画像ファイルをサーバーに送っていないため、画像を含む文書はPDFプレビューでエラーになる
- **デプロイ先は未定**: いまはローカルで動かす前提。ブラウザから `file://` で開いたページが呼べるよう、CORSを全オリジンに許可している（`Access-Control-Allow-Origin: *`）。公開する場合はこの設定を見直す
- サーバーのURLは [web/src/script.ts](web/src/script.ts) の `COMPILE_URL`、ポートは環境変数 `PORT`（既定 8080）で変えられる
