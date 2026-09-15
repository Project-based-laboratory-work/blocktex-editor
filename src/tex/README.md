# src/tex

TeX変換方針（Phase0で決定した内容）を置く場所。

- `preamble.ts` — 採用したドキュメントクラス（`jsarticle`）と、ブロック種別ごとに必要なLaTeXパッケージの対応表。`src/fixtures/sample-document.expected.tex` のプリアンブルはここから導出したもので、実際に `uplatex` でコンパイル確認済み。
- ブロック配列をTeX文字列に変換する `texify()` 本体はPhase1で実装する。
