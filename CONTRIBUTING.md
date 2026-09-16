# 開発ルール

## ブランチ運用

- `main` への直接pushは禁止。
- 作業は `feature/xxx`（機能追加）や `phase0/xxx`（フェーズ単位の作業）のようなブランチを切って行い、GitHub上のPull Requestを経由して `main` にマージする。

## CI（自動チェック）

PRを出すと [.github/workflows/frontend.yml](.github/workflows/frontend.yml) が動き、`frontend/` に対して lint → フォーマット確認 → テスト → ビルド を実行する。これが通らないPRはマージしない。

手元で同じ内容を先に確認できる:

```bash
cd frontend
npm run lint && npm run format:check && npm run test && npm run build
```

`format:check` で落ちた場合は `npm run format` で整形すれば直る。

GitHubのリポジトリ設定で `main` ブランチに保護ルール（PR必須・上記チェック必須）を入れておくと、「直接pushは禁止」が運用ではなく仕組みで担保される。

## リポジトリ構成

モノレポ構成のため、作業するディレクトリに注意すること。

- `frontend/` — Webアプリ本体。npmコマンドはこのディレクトリで実行する（`cd frontend && npm install`）。
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
docker/compile-tex.sh frontend/src/fixtures/sample-document.expected.tex
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
