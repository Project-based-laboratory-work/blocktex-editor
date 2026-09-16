# src/fixtures

Phase0で作成したサンプルデータ。Phase1のTeX変換ロジック（`texify()`）のスナップショットテストにそのまま再利用する想定。

- `sample-document.json` — `BlockDocument` の6種類のブロック（見出し・段落・箇条書き・数式・表・画像）を1つずつ含むサンプル。段落には `% & _ #` などエスケープ確認用の文字を含めている。表の行が `{ "cells": [...] }` 形式なのはFirestoreが配列の入れ子を許さないため（BACKEND.md 0-5-3 を参照）。
- `sample-document.ts` — 上記JSONを `BlockDocument` 型で取り出すための入口。JSONを直接importすると文字列リテラルが `string` に広がって型が合わないため、ここで一度だけ型を確定させている。**利用側はこちらをimportすること。**
- `sample-document.expected.tex` — 上記JSONに対応する手書きの期待TeX出力。`uplatex`（2回）→ `dvipdfmx` で実際にコンパイルし、エラーなくPDFが生成できることを確認済み（Phase0タスク0-4-1）。プリアンブルは `src/tex/preamble.ts` の `PACKAGES_BY_BLOCK_TYPE` から実際に使うブロック分だけを列挙したもの。コンパイル確認は `docker/compile-tex.sh frontend/src/fixtures/sample-document.expected.tex` で再現できる。
- `sample-image.png` — `\includegraphics` が参照する画像ブロック用のプレースホルダ画像（ImageMagickで生成）。

Phase1でのスナップショットテストでの読み込み例:

```ts
import { sampleDocument } from '../fixtures/sample-document';
import expectedTex from '../fixtures/sample-document.expected.tex?raw';

expect(texify(sampleDocument).trim()).toBe(expectedTex.trim());
```
