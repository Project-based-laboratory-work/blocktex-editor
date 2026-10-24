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

ブラウザはTeXをコンパイルできないため、`server/` のコンパイルサーバーにTeXを送ってPDFにしてもらう。サーバーは各メンバーが自分のPCで動かす前提で、公開（デプロイ）はしない。サーバーが動いていなくても、エディタとTeX表示はそのまま使える。

```
ブラウザ（web/）── POST /api/compile {"tex": "...", "images": [...]} ──▶ server/
                                                     一時ディレクトリで uplatex ×2 → dvipdfmx
PDFプレビュータブ ◀──────────── PDF（失敗時はエラー文）────────────┘
```

### 起動する（Dockerを使う方法・推奨）

チーム共通のTeX Live（Debian bookworm = TeX Live 2022）で確認できる。初回のビルドは十数分かかる。`server/` を変更したら、もう一度 `docker build` する。

```bash
docker build -t retex-compile server/
docker run --rm --init -p 127.0.0.1:8080:8080 retex-compile
```

`listening on :8080` と出たら、`web/index.html` を開き、右ペインの「PDFプレビュー」タブを選ぶ。編集が止まってから1.5秒後に自動で更新される。止めるときは `Ctrl + C`。`-p 127.0.0.1:8080:8080` は、同じネットワークのほかのPCから使われないよう、自分のPCからだけ受け付ける指定。

### 起動する（Dockerを使わない方法）

手元にTeX Live（[CONTRIBUTING.md](CONTRIBUTING.md) の「Dockerを使わない方法」）とNode.js 22.18以上があれば、`.ts` をそのまま動かせる（ビルド不要）。

```bash
cd server
npm install
npm run dev        # ファイルを保存すると自動で再起動する
```

手元のTeX Liveのバージョンによって結果が変わることがあるため、最終確認はDocker側で行う。

### APIの仕様

`POST /api/compile` に、次のJSONを送る。`images` は省略できる。

```json
{
  "tex": "<TeXソース全体>",
  "images": [{ "name": "写真.png", "data": "<ファイルの中身を base64 にしたもの>" }]
}
```

画像は `name` の名前で `main.tex` と同じ一時ディレクトリに置かれるので、TeXからは `\includegraphics{写真.png}` で読める。画面からは、画像ブロックで選んだファイルが自動で送られる。

| ステータス | 意味 | 本文 |
|---|---|---|
| 200 | 成功 | PDF（`application/pdf`） |
| 400 | JSONが壊れている／`tex` が文字列でない／画像のファイル名が使えない | `bad request` など |
| 403 | 許可していない場所（オリジン）からのリクエスト | `forbidden` |
| 413 | 本文が大きすぎる（約20MB超） | `too large` |
| 422 | TeXのエラー、コンパイルが20秒を超えた、画像を読めない | ログのエラー部分（3行）など |
| 500 | サーバー側の問題 | `server error` |

画像のファイル名は、拡張子が png / jpg / jpeg / pdf で、フォルダの区切り（`/` `\`）を含まず、`.` や `main.` で始まらないもの。

### 安全対策

送られてきた任意のTeXを動かす前提で動かしている。

- `-no-shell-escape`（`\write18` で外部コマンドを実行させない）。そのため、shell-escapeが必要なパッケージ（minted など）は使えない。画像の大きさを調べる `extractbb` も TeX からは呼べないので、サーバーが先に実行している
- コンパイルのタイムアウト（20秒）と、本文のサイズ上限（約20MB）
- リクエストごとに別の一時ディレクトリを作り、終わったら必ず消す
- 受け付けるのは、`file://` で開いた画面（オリジン `null`）と `localhost` / `127.0.0.1` からのリクエストだけ。ほかのサイトを開いているときに、そのページから勝手にコンパイルさせられないようにするため（`Origin` を付けない curl などはそのまま使える）
- Dockerイメージの中では一般ユーザー（`node`）で動かす

### 補足

- 保存した文書を開き直した直後は、画像ファイルが手元に無い（ブラウザは選んだファイルを保存しない）。PDFプレビューにその旨が出るので、画像ブロックでファイルを選び直す
- サーバーのURLは [web/src/script.ts](web/src/script.ts) の `COMPILE_URL`、ポートは環境変数 `PORT`（既定 8080）で変えられる
- うまく動かないときの調べ方は [CONTRIBUTING.md](CONTRIBUTING.md) の「PDFプレビューのデバッグ」

## PDFの読み込み（抽出サーバー）

トップ画面の「PDF・画像を読み込む」でPDFを選ぶと、コンパイルサーバーの `/api/extract` にPDFを送り、ルールベースの抽出・分類（`ai/rule_based/`）で見出し・段落・表・図のブロックにしてからエディタで開く。画像からの変換は未対応（空の文書で開く）。

```
ブラウザ（web/）── POST /api/extract（PDFのバイト列）──▶ server/
                                     一時ディレクトリに保存 → ai/.venv の Python で
                                     python -m rule_based.ir_adapter <PDF> --stdout
エディタ ◀── {"labeled": ラベル付きIR, "document": エディタ用JSON} ──┘
```

JSONの形（IR・ラベル付きIR・エディタ用JSON）は [ai/common/README.md](ai/common/README.md) を参照。

### 起動する

**今はDockerでは動かない**（`server/Dockerfile` に Python が入っていない）ので、Dockerを使わない方法で起動する。PDFの読み込みにTeX Liveは要らない。

1. `ai/` に Python の仮想環境を作り、抽出に使うライブラリだけを入れる（`requirements.txt` には学習用の torch も入っていて重いため）

   ```bash
   cd ai
   python3 -m venv .venv
   source .venv/bin/activate
   pip install pymupdf pdfplumber reportlab pytest
   python rule_based/sample_data/generate_sample_pdf.py   # 試し用の rule_based/sample_data/sample.pdf を作る
   ```

2. サーバーを起動する（[PDFプレビューの「Dockerを使わない方法」](#起動するdockerを使わない方法) と同じ）

   ```bash
   cd server
   npm run dev
   ```

サーバーは `ai/.venv/bin/python` を直接呼ぶので、`activate` しておく必要はない。

### APIの仕様

`POST /api/extract` に、PDFファイルの中身を**そのまま**（JSONやbase64にせず）送る。

```bash
curl -X POST --data-binary @ai/rule_based/sample_data/sample.pdf http://localhost:8080/api/extract
```

| ステータス | 意味 | 本文 |
|---|---|---|
| 200 | 成功 | `{"labeled": ラベル付きIR, "document": エディタ用JSON}`（`application/json`） |
| 403 | 許可していない場所（オリジン）からのリクエスト | `forbidden` |
| 413 | 本文が大きすぎる（約20MB超） | `too large` |
| 422 | PDFではない（先頭が `%PDF-` でない）／壊れていて読めない／抽出が20秒を超えた | `PDFファイルではありません` など |
| 500 | `ai/.venv` が無い、サーバー側の問題 | `抽出機のPythonが見つかりません（ai/.venv を作ってください）` など |

ブラウザから送るときは `file` ではなく `await file.arrayBuffer()` を `body` にする。`file` のままだと `Content-Type: application/pdf` が付いてプリフライト（OPTIONS）が飛び、このサーバーはそれに答えないため止められる。

### デバッグ用ページ

[web/ir_debug.html](web/ir_debug.html) を開くと、抽出・分類の結果をエディタを通さずに確認できる。

- 「PDF」のボタン: `/api/extract` に送り、要素ごとに `[heading] 第1章 …` の形で1行ずつ並べる。その下に、ラベル付きIRとエディタ用JSONをそのまま表示する
- 「JSON」のボタン: `python -m rule_based.pdf_extract` などで保存したIRのJSONファイルを選んで、同じ形で表示する（サーバー不要）

### 補足

- 見出しのレベルは文字サイズの大きい順に決める。サンプルPDFは「第1章」も「1.1」も同じ18ptなので、全部 `\section` になる
- 図は画像ファイルを取り出していないので、画像ブロックで選び直す
- 8080番を別のアプリ（ほかのプロジェクトのDockerコンテナなど）が使っていると、`npm run dev` が `EADDRINUSE` で起動できない。`docker ps` で確認して止める
