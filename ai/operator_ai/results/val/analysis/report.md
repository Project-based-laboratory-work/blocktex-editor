# オペレータAI 分析（進め方2・3）

採点表 `operator_ai/results/val/scores.csv`。画像 850 枚 × 設定 10 個。CER は小さいほど良い。

エラーがあった 1 枚は集計から外した：
- doc_000041_p3_v0 easyocr:up2: OcrCrashed: 前回この画像を読んでいる途中でOCRが落ちた（Segmentation fault など）。読み直すには --force

## 進め方2：単独ベストとオラクル

| 方法 | cer_page | 差の埋まり具合 |
|---|---|---|
| 単独ベスト（tesseract:up2） | 0.1191 | 0% |
| 2つから選ぶときの上限（tesseract:up2 / easyocr:up2_lowtext） | 0.0925 | 70% |
| オラクル（上限） | 0.0813 | 100% |

振り分けで伸ばせる余地：0.119 → 0.081（誤りを 32% 減らせる）

### 設定ごと

`median` は画像ごとの CER の真ん中、`broken` は CER が 0.5 を超えた画像の割合、`wins` はその設定が一番良かった画像の数（同点なら速いほう）。

| setting | cer_page | cer_page_ja | cer_page_en | median | broken | f1 | sec_per_page | wins |
|---|---|---|---|---|---|---|---|---|
| tesseract:up2 | 0.1191 | 0.1071 | 0.26 | 0.0581 | 0.0694 | 0.5664 | 0.81 | 323 |
| easyocr:up2_lowtext | 0.161 | 0.1516 | 0.2502 | 0.1006 | 0.0682 | 0.6114 | 9.54 | 30 |
| tesseract:psm11 | 0.1813 | 0.1762 | 0.4512 | 0.0833 | 0.1235 | 0.1908 | 0.52 | 74 |
| easyocr:up2 | 0.1819 | 0.1716 | 0.2847 | 0.1136 | 0.0835 | 0.6072 | 9.69 | 5 |
| easyocr:lowtext | 0.1887 | 0.1643 | 0.3377 | 0.131 | 0.0906 | 0.5611 | 2.13 | 11 |
| tesseract:invert | 0.1923 | 0.182 | 0.4461 | 0.0817 | 0.1494 | 0.5097 | 0.5 | 84 |
| tesseract:base | 0.1924 | 0.1822 | 0.4473 | 0.0819 | 0.1518 | 0.5113 | 0.6 | 7 |
| easyocr:invert | 0.2196 | 0.1882 | 0.3933 | 0.1569 | 0.1141 | 0.5491 | 2.06 | 4 |
| easyocr:base | 0.2237 | 0.1928 | 0.3966 | 0.1536 | 0.1224 | 0.5534 | 2.31 | 1 |
| tesseract:psm6 | 0.3486 | 0.316 | 0.6533 | 0.0601 | 0.2153 | 0.145 | 0.53 | 311 |

### 2つの設定から選ぶときの上限（上位5）

| pair | cer_page |
|---|---|
| tesseract:up2 / easyocr:up2_lowtext | 0.0925 |
| tesseract:up2 / easyocr:up2 | 0.096 |
| tesseract:up2 / easyocr:lowtext | 0.096 |
| tesseract:up2 / easyocr:invert | 0.0996 |
| tesseract:up2 / easyocr:base | 0.1005 |

## 進め方3：条件ごとの勝ち負け

`best_setting` はその条件の画像だけで見た単独ベスト。`single_best_cer` は全体の単独ベスト（tesseract:up2）のまま読んだときの CER。設定ごとの CER は `conditions.csv`。
`effective_dpi` は dpi × 縮小率（画面上の文字の細かさ）。

| condition | value | n | best_setting | best_cer | single_best_cer | oracle_cer | most_wins |
|---|---|---|---|---|---|---|---|
| style | clean | 174 | tesseract:up2 | 0.0674 | 0.0674 | 0.0392 | tesseract:psm6（86/174） |
| style | screenshot | 676 | tesseract:up2 | 0.1337 | 0.1337 | 0.0932 | tesseract:up2（279/676） |
| dpi | 100-140 | 321 | tesseract:up2 | 0.1603 | 0.1603 | 0.1136 | tesseract:up2（155/321） |
| dpi | <100 | 70 | tesseract:up2 | 0.2007 | 0.2007 | 0.1643 | tesseract:up2（50/70） |
| dpi | >=140 | 459 | tesseract:up2 | 0.0795 | 0.0795 | 0.0473 | tesseract:psm6（211/459） |
| scale | 0.75-1 | 335 | tesseract:up2 | 0.1033 | 0.1033 | 0.0606 | tesseract:psm6（131/335） |
| scale | 1（縮小なし） | 178 | tesseract:up2 | 0.0679 | 0.0679 | 0.0393 | tesseract:psm6（88/178） |
| scale | <0.75 | 337 | tesseract:up2 | 0.1631 | 0.1631 | 0.1247 | tesseract:up2（175/337） |
| jpeg_quality | <80 | 214 | tesseract:up2 | 0.1524 | 0.1524 | 0.1012 | tesseract:up2（73/214） |
| jpeg_quality | >=80 | 124 | tesseract:up2 | 0.1378 | 0.1378 | 0.097 | tesseract:up2（50/124） |
| jpeg_quality | なし | 512 | tesseract:up2 | 0.1011 | 0.1011 | 0.0696 | tesseract:up2（200/512） |
| dark_mode | False | 642 | tesseract:up2 | 0.1002 | 0.1002 | 0.0743 | tesseract:up2（253/642） |
| dark_mode | True | 208 | tesseract:up2 | 0.1807 | 0.1807 | 0.1041 | tesseract:up2（70/208） |
| crop | False | 527 | tesseract:up2 | 0.1029 | 0.1029 | 0.0688 | tesseract:psm6（220/527） |
| crop | True | 323 | tesseract:up2 | 0.1559 | 0.1559 | 0.1097 | tesseract:up2（140/323） |
| viewer | False | 447 | tesseract:up2 | 0.0872 | 0.0872 | 0.0632 | tesseract:up2（175/447） |
| viewer | True | 403 | tesseract:up2 | 0.1566 | 0.1566 | 0.1025 | tesseract:up2（148/403） |
| background | False | 651 | tesseract:up2 | 0.1198 | 0.1198 | 0.0784 | tesseract:up2（239/651） |
| background | True | 199 | tesseract:up2 | 0.1168 | 0.1168 | 0.0909 | tesseract:up2（84/199） |
| lang | en | 73 | tesseract:up2 | 0.1726 | 0.1726 | 0.1136 | tesseract:psm6（36/73） |
| lang | ja | 777 | tesseract:up2 | 0.1177 | 0.1177 | 0.0805 | tesseract:up2（303/777） |
| columns | 1 | 698 | tesseract:up2 | 0.0972 | 0.0972 | 0.0688 | tesseract:psm6（297/698） |
| columns | 2 | 152 | tesseract:up2 | 0.1845 | 0.1845 | 0.1187 | tesseract:up2（65/152） |
| effective_dpi | 100-140 | 311 | tesseract:up2 | 0.0891 | 0.0891 | 0.0527 | tesseract:psm6（143/311） |
| effective_dpi | 70-100 | 245 | tesseract:up2 | 0.1125 | 0.1125 | 0.0801 | tesseract:up2（150/245） |
| effective_dpi | <70 | 71 | tesseract:up2 | 0.3794 | 0.3794 | 0.3091 | tesseract:up2（55/71） |
| effective_dpi | >=140 | 223 | tesseract:invert | 0.0641 | 0.0737 | 0.0399 | tesseract:psm6（118/223） |

## CER が悪い画像（単独ベストで悪い順、上位 30。正解の文字が 50 字未満のページは除く）

`best_cer` も悪いなら OCR の限界か正解の問題、良くなっていれば振り分けで直せる。原因は `worst.csv` の memo 列に書く。

| case | chars | single_best_cer | best_setting | best_cer | style | dark_mode | dpi | scale | lang | columns |
|---|---|---|---|---|---|---|---|---|---|---|
| doc_000092_p3_v0 | 271 | 1.2472 | tesseract:up2 | 1.2472 | screenshot | True | 135.6 | 0.739 | ja | 2 |
| doc_000103_p1_v0 | 171 | 1.0 | easyocr:up2_lowtext | 0.6959 | screenshot | True | 105.4 | 0.621 | en | 1 |
| doc_000123_p3_v0 | 270 | 1.0 | easyocr:up2_lowtext | 0.6222 | screenshot | False | 107.1 | 0.533 | ja | 1 |
| doc_000130_p1_v0 | 439 | 1.0 | tesseract:invert | 0.0273 | clean | False | 126.4 | 1.0 | ja | 2 |
| doc_000255_p2_v0 | 333 | 1.0 | easyocr:up2_lowtext | 0.5435 | screenshot | True | 90.0 | 0.815 | ja | 1 |
| doc_000291_p2_v0 | 668 | 0.9925 | easyocr:up2_lowtext | 0.5689 | screenshot | True | 119.4 | 0.557 | ja | 2 |
| doc_000040_p2_v0 | 797 | 0.9799 | easyocr:up2_lowtext | 0.5006 | screenshot | False | 109.3 | 0.556 | ja | 1 |
| doc_000110_p1_v0 | 356 | 0.9663 | easyocr:up2_lowtext | 0.5056 | screenshot | True | 138.9 | 0.547 | ja | 2 |
| doc_000055_p1_v0 | 317 | 0.9621 | easyocr:lowtext | 0.4953 | screenshot | True | 126.0 | 0.556 | ja | 1 |
| doc_000068_p2_v0 | 275 | 0.96 | easyocr:up2_lowtext | 0.1745 | screenshot | True | 174.1 | 0.515 | ja | 1 |
| doc_000044_p1_v0 | 430 | 0.9558 | easyocr:lowtext | 0.5953 | screenshot | False | 94.4 | 0.629 | ja | 2 |
| doc_000057_p1_v0 | 255 | 0.9373 | tesseract:psm11 | 0.0863 | screenshot | False | 117.8 | 0.791 | ja | 2 |
| doc_000034_p3_v0 | 610 | 0.9115 | easyocr:lowtext | 0.3393 | screenshot | True | 109.5 | 0.916 | ja | 1 |
| doc_000093_p2_v0 | 372 | 0.8952 | easyocr:up2_lowtext | 0.1935 | screenshot | True | 113.9 | 0.822 | ja | 1 |
| doc_000180_p1_v0 | 317 | 0.8707 | easyocr:up2_lowtext | 0.2019 | screenshot | True | 162.8 | 0.704 | ja | 2 |
| doc_000009_p1_v0 | 352 | 0.8466 | easyocr:lowtext | 0.767 | screenshot | False | 102.3 | 0.511 | ja | 1 |
| doc_000189_p1_v0 | 760 | 0.8395 | easyocr:up2_lowtext | 0.3895 | screenshot | False | 101.0 | 0.665 | ja | 2 |
| doc_000134_p2_v0 | 537 | 0.8063 | tesseract:invert | 0.0149 | screenshot | True | 157.8 | 0.945 | ja | 2 |
| doc_000026_p2_v0 | 400 | 0.6875 | easyocr:up2_lowtext | 0.62 | screenshot | False | 108.8 | 0.506 | ja | 1 |
| doc_000132_p1_v0 | 736 | 0.6834 | easyocr:up2_lowtext | 0.341 | screenshot | True | 91.3 | 0.757 | ja | 2 |
| doc_000044_p2_v0 | 434 | 0.6452 | tesseract:up2 | 0.6452 | screenshot | False | 97.1 | 0.658 | ja | 2 |
| doc_000033_p2_v0 | 241 | 0.639 | easyocr:up2_lowtext | 0.5062 | screenshot | False | 104.5 | 0.642 | ja | 1 |
| doc_000267_p3_v0 | 315 | 0.6286 | easyocr:up2_lowtext | 0.5841 | screenshot | False | 104.1 | 0.512 | ja | 1 |
| doc_000121_p1_v0 | 207 | 0.6232 | easyocr:up2_lowtext | 0.1256 | screenshot | True | 126.7 | 0.879 | ja | 2 |
| doc_000069_p1_v0 | 164 | 0.622 | tesseract:psm11 | 0.1707 | screenshot | False | 98.4 | 0.826 | ja | 2 |
| doc_000079_p1_v0 | 435 | 0.5954 | tesseract:up2 | 0.5954 | screenshot | False | 105.8 | 0.598 | ja | 1 |
| doc_000155_p2_v0 | 284 | 0.5951 | easyocr:up2_lowtext | 0.5634 | screenshot | False | 106.5 | 0.58 | ja | 1 |
| doc_000108_p2_v0 | 201 | 0.592 | easyocr:lowtext | 0.2239 | screenshot | True | 125.5 | 0.591 | ja | 1 |
| doc_000153_p4_v0 | 195 | 0.5846 | easyocr:up2_lowtext | 0.4667 | screenshot | False | 101.0 | 0.691 | ja | 1 |
| doc_000092_p2_v0 | 558 | 0.5681 | tesseract:up2 | 0.5681 | screenshot | False | 97.2 | 0.647 | ja | 2 |
