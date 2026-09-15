# 開発ルール

## ブランチ運用

- `main` への直接pushは禁止。
- 作業は `feature/xxx`（機能追加）や `phase0/xxx`（フェーズ単位の作業）のようなブランチを切って行い、GitHub上のPull Requestを経由して `main` にマージする。

## LaTeX（uplatex）開発環境の構築手順

このプロジェクトは **uplatex + jsarticle** でTeXをコンパイルする（`uplatex` を2回実行後、`dvipdfmx` でPDF化）。以下はメンバー各自の環境構築手順。

### Linux（Debian/Ubuntu系）

```bash
sudo apt install texlive-lang-japanese texlive-latex-extra
```

### macOS

[MacTeX](https://www.tug.org/mactex/) をインストールする（フル版で `uplatex`/`dvipdfmx` が含まれる）。容量を抑えたい場合は BasicTeX + `tlmgr` で `collection-langjapanese` を追加インストールする。

### Windows

WSL2を導入し、Linux（Debian/Ubuntu）と同じ手順でTeX Liveを導入することを推奨する。

### 動作確認

```bash
uplatex --version
dvipdfmx --version
```

リポジトリのサンプルで実際にコンパイルできることを確認する:

```bash
cd src/fixtures
cp sample-document.expected.tex /tmp/sample.tex
cp sample-image.png /tmp/
cd /tmp
uplatex sample.tex && uplatex sample.tex && dvipdfmx sample.dvi
# sample.pdf が生成され、エラーが出なければOK
```
