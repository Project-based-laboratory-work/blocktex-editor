# image_table — 表の画像 → 表の中身（表AI）

表の部分の画像（[image_layout](../image_layout/README.md) が見つけた表の四角を切り抜いたもの）から、
**行・列の区切り**を AI で当て、チーム共通のIRの表の要素（`cells`・`aligns`・`row_rules`・`col_rules`）にする。

```
表の画像 ─ 表AI（行・列の区切り）─→ マス目 ─┬ EasyOCR でマスごとに読む → cells
                                          ├ 境目に線があるか          → row_rules / col_rules
                                          └ 文字が左・右・真ん中のどれに寄るか → aligns
```

## なぜAIか

ルールベース（縦・横の位置ごとに文字があるかを数え、空白で区切る）は、折り返しの無い表ならほぼ完璧に区切れる。
しかし**セルの中で折り返した表**では、折り返した1行1行の間も空白なので、1つの行を何行にも切ってしまう。
本当の行の境目かどうかは、「次の行で、他の列にも文字が始まるか」のように表全体を見ないと決まらない。

## しくみ

1. 表の画像を 768×768 に収め、CNN で文字・罫線・空白の形を見る（[model.py](model.py)）
2. 横方向にまとめて「縦の位置ごとの特徴」、縦方向にまとめて「横の位置ごとの特徴」を作る
3. それぞれを双方向GRU（並びを上下・左右の両方向から読む部品）に通し、位置ごとに「行（列）の外か」を当てる
4. 外ではない位置が続くところを1つの行（列）にする（[postprocess.py](postprocess.py)）

## 学習データ

[generate.py](generate.py) で、ランダムな表を TeX で組んで画像にする。罫線の種類・セルの語・折り返す列は
評価用データ（evaluation/synth_doc.py の table_blocks）に合わせ、行・列を多く（最大10行×7列）、中身も多様にした。
各セルを目印で挟み、PDFから目印の間の文字の位置を集めて、行・列の正確な範囲を正解にする。解像度は 130〜260dpi でばらつかせる。

## 使い方

`ai/` で実行する。

```bash
python -m image_table.generate --out image_table/data/train --tables 20000 --seed 0 --workers 10   # CPU 10並列で約4分
python -m image_table.generate --out image_table/data/val --tables 2000 --seed 1 --workers 10
python -m image_table.team_tables --out image_table/data/team --tables 500   # 評価用データと同じ作り方の表（試験用）
python -m image_table.train                                    # → checkpoints/split/best.pt（計算機サーバのGPUで約2.5時間）
python -m image_table.evaluate image_table/data/val            # 区切り方をルールベースと比べる
python -m image_table.evaluate image_table/data/team
python -m image_table.evaluate_contents image_table/data/team  # セルの文字・揃え・罫線まで、チームの物差しで（EasyOCRが要る）
python -m image_table.predict <表の画像>                        # 1つの表をIRの表の要素（JSON）にする
python -m pytest image_table/tests
```

`data/` と `checkpoints/` は git の管理外。

## 結果

RESULTS

## ファイル

| ファイル | 内容 |
|---|---|
| [generate.py](generate.py) | 学習データ（表の画像 ＋ 行・列の範囲）を作る |
| [team_tables.py](team_tables.py) | 評価用データと同じ作り方の表で、試験用のデータを作る（正解は cells・揃え・罫線） |
| [dataset.py](dataset.py) | 画像をキャンバスに収める・行（列）の外かどうかの正解を作る |
| [model.py](model.py) | CNN ＋ 縦・横それぞれの双方向GRU |
| [postprocess.py](postprocess.py) | 位置ごとの点数 → 行・列の範囲 |
| [train.py](train.py) | 学習。best.pt は検証データで行・列が全部正しく区切れた表の割合で選ぶ |
| [evaluate.py](evaluate.py) | 区切り方をルールベースと比べる |
| [evaluate_contents.py](evaluate_contents.py) | 区切り方だけ入れ替えて、セルの中身・揃え・罫線までチームの物差しで比べる |
| [predict.py](predict.py) | 表の画像 → IRの表の要素 |
