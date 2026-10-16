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

- **Linux（Debian/Ubuntu系）**: `sudo apt install texlive-lang-japanese texlive-latex-extra`
- **macOS**: [MacTeX](https://www.tug.org/mactex/) をインストール（BasicTeX + `tlmgr install collection-langjapanese` でも可）
- **Windows**: WSL2を導入してLinuxと同じ手順

動作確認:

```bash
uplatex --version
dvipdfmx --version
```
