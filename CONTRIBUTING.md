# 開発ルール

## ブランチ運用

- `main` への直接pushは禁止。
- 作業は `feature/xxx`（機能追加）や `phase0/xxx`（フェーズ単位の作業）のようなブランチを切って行い、GitHub上のPull Requestを経由して `main` にマージする。

## CI（自動チェック）

現在 `web/` はtscでコンパイルするだけの静的サイト（HTML/CSS/TypeScript）のため、専用のCIは設けていない。PR前に `npm run typecheck` が通ることを確認すること。

GitHubのリポジトリ設定で `main` ブランチに保護ルール（PR必須）を入れておくと、「直接pushは禁止」が運用ではなく仕組みで担保される。

## PR通知（Teams）

PRを作成（またはreopen、draftからReady for reviewに変更）すると、GitHub Actions（`.github/workflows/notify-pr.yml`）がTeamsのチャンネルに通知を投稿する。draftの間は通知されない。

通知にはリポジトリのSecret `TEAMS_WEBHOOK_URL` を使う。未設定の場合は通知をスキップするだけで、ジョブは失敗しない。設定手順:

1. Teamsで通知先チャンネルの「ワークフロー」から「Webhook要求を受信したらチャンネルに投稿する」を作成し、URLを控える
2. GitHubのリポジトリ設定（Settings → Secrets and variables → Actions）で `TEAMS_WEBHOOK_URL` にそのURLを登録する（Admin権限が必要）

フォークからのPRにはSecretが渡されないため通知されない。

## リポジトリ構成

モノレポ構成のため、作業するディレクトリに注意すること。

- `web/` — Webアプリ本体（TypeScript）。`npm install && npm run build` で `web/js/` を生成すれば、`web/index.html` をブラウザで直接開いて動作確認できる。
- `ai/` — Phase 4の学習コード（Python）。Pythonの仮想環境はこのディレクトリに作る。
- `server/` — PDFプレビュー用のコンパイルサーバー。起動方法は [README.md](README.md) の「PDFプレビュー（コンパイルサーバー）」を参照。
- `docker/` — TeXコンパイル確認用のDockerイメージ定義とスクリプト。

## TeX環境（Dockerを使う方法・推奨）

メンバーのOSがバラバラでもTeXの環境を揃えられるよう、コンパイル確認用のDockerイメージを用意してある。各自ローカルにTeX Liveを入れる必要はない。

### 1. イメージをビルドする（初回のみ）

```bash
docker build -t blocktex-tex -f docker/tex.Dockerfile docker/
```

TeX Liveのダウンロードがあるため、初回は数分〜十数分かかる。

### 2. .texをコンパイルする

```bash
docker/compile-tex.sh path/to/your.tex
```

指定した`.tex`と同じディレクトリにPDFが生成される。中では `uplatex`（2回）→ `dvipdfmx` を実行している。

生成物（`.aux` `.dvi` `.log` `.pdf`）はgitignore済み。

## TeX環境（Dockerを使わない方法）

Dockerを使いたくない場合は各自でTeX Liveを導入する。バージョン差で結果が変わる可能性があるため、その場合も最終確認はDocker側で行うこと。

- **Linux（Debian/Ubuntu系）**: `sudo apt install texlive-lang-japanese texlive-latex-extra texlive-plain-generic`（下線に使う `ulem` は `texlive-plain-generic` に入っている）
- **macOS**: [MacTeX](https://www.tug.org/mactex/) をインストール（BasicTeX + `tlmgr install collection-langjapanese` でも可）
- **Windows**: WSL2を導入してLinuxと同じ手順

動作確認:

```bash
uplatex --version
dvipdfmx --version
```

## PDFプレビューのデバッグ

PDFプレビュー（[README.md](README.md) の「PDFプレビュー（コンパイルサーバー）」）がうまく動かないときの調べ方。

### 作業を始めるとき

PCを再起動するとターミナルで動かしていたものは止まるので、毎回次の2つを立ち上げる。

| ターミナル | 場所 | コマンド | 動いている印 |
|---|---|---|---|
| 1 | リポジトリ直下 | `docker run --rm --init -p 127.0.0.1:8080:8080 retex-compile`（Docker Desktop を先に起動）または `cd server && npm run dev` | `listening on :8080` |
| 2 | リポジトリ直下 | `npm run watch` | `Found 0 errors. Watching for file changes.` |

そのうえで `web/index.html` をブラウザで開く（開いていれば F5）。

### 見る場所

| 場所 | 開き方 | 分かること |
|---|---|---|
| サーバーのターミナル | 上のターミナル1 | サーバーが動いているか、サーバー側のエラー |
| PDFプレビューの赤い文字 | 右ペイン | TeXのエラー（最初のエラーから3行）、画像のエラー |
| ブラウザの Console | F12 → Console | 画面側のエラー（つながらない、プログラムの間違い） |
| ブラウザの Network | F12 → Network | サーバーに何を送り、何が返ってきたか |

### 症状ごとの調べ方

**「サーバーに接続できません」と出る**

1. ターミナル1の最後が `listening on :8080` か確認する。止まっていれば起動し直す
2. Console のエラーを見る。`ERR_CONNECTION_REFUSED` などはサーバーが動いていないかポートが違う。`CORS` を含むなら、`index.html` を `file://` か `localhost` 以外から開いていないか確認する
3. サーバー単体で動くか curl で確かめる。`200` が返ればサーバーは正常で、原因は画面側にある

   ```bash
   cd server
   curl -i -X POST localhost:8080/api/compile --data-binary @sample.json --output out.pdf
   ```

**書き直したのに変わらない**

| 直したもの | 必要なこと |
|---|---|
| `web/src/*.ts` | `npm run watch` が動いているか確認して、ブラウザで F5 |
| `web/index.html`・`web/css/` | ブラウザで F5 |
| `server/`（Dockerで動かしている） | `docker build -t retex-compile server/` をやり直して、`docker run` し直す |
| `server/`（`npm run dev` で動かしている） | 保存すれば自動で再起動する |

ブラウザが新しいJSを読んでいるかは、F12 → Sources で `js/script.js` を開いて確かめられる。

**TeXのエラーが出る**

赤い文字はログの一部だけなので、分からないときはログ全体を見る。

1. 右ペインの「TeX」タブで「コピー」を押し、`debug.tex` として保存する（画像を使っているなら、その画像も同じフォルダに置く）
2. `docker/compile-tex.sh path/to/debug.tex` でコンパイルする（イメージは上の「TeX環境」の手順で作っておく）
3. 同じフォルダの `debug.log` で `!` から始まる行を探す。`l.12 \hogehoge` は「TeXの12行目の `\hogehoge` まで読んで止まった」という意味で、行番号はTeXタブの行と同じ

**自動で更新されない**

- PDFプレビュータブを開き、右ペインを開いた状態でないと送らない（わざとそうしている）
- 入力をやめてから1.5秒待つ
- Network を開いて文字を打ち、1.5秒後に `compile` が増えるか見る。増えなければ画面側の問題なので Console を見る。`(canceled)` は、次の更新が始まって前のリクエストを取り消しただけなので正常

**画像が出ない**

- 「画像ファイルが手元にありません」: ブラウザは選んだファイルを保存しないので、保存した文書を開き直したあとは、画像ブロックでファイルを選び直す
- 「この画像のファイル名は使えません」: png / jpg / pdf にして、`main.` で始まらない名前にする

### サーバーの中を見る

Dockerで動かしている場合:

```bash
docker ps                          # 動いているコンテナの一覧（CONTAINER ID が分かる）
docker logs <CONTAINER ID>         # サーバーの出力をあとから見る
docker exec -it <CONTAINER ID> sh  # コンテナの中でコマンドを打つ（exit で出る）
```

サーバーのコードを調べるときは `npm run dev` で動かし、気になるところに `console.log(...)` を書いて保存すると、ターミナル1に表示される。確認が終わったら消しておく。
