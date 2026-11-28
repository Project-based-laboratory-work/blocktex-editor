# モデルの精度を上げる手法のまとめ（出典つき）

蒸留のような「モデルの精度を上げるための手法」を、出典の論文と一緒にまとめる。
それぞれ、この抽出器（手がかり＋MLP の専門家、文中の数式の小さい Transformer、合成データ）に当てはめるとどうか、
もう試したかも書く。試した結果は [FINDINGS.md](FINDINGS.md)。

**先に結論:** この抽出器で一番効いたのは、モデルの手法よりも**データと手がかり**だった（FINDINGS.md 1・4 節）。
下の手法は「データと手がかりを直したあとに、もう少し伸ばす」もの、と考えるとよい。順番の目安は最後の表。

---

## 1. 複数のモデルを使う

### 1.1 知識の蒸留（Knowledge Distillation）
- **何をするか:** 強いモデル（先生）の出す確率を、正解と一緒に目標にして、小さいモデル（生徒）を学習する。確率を温度 T でなだらかにして、
  「どれくらい迷ったか」を伝える
- **出典:** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015). [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
- **同じ形の生徒に蒸留しても良くなる（Born-Again Networks）:** Furlanello ら (ICML 2018). [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
- **この抽出器では:** 試した（文中の数式）。生徒は1人分の速さで、1文字ずつの F1 が先生1人より +0.006、学習も3エポックで頭打ちになり安定した。
  数式のまとまりでは効いたと言い切れない

### 1.2 アンサンブル（いくつかのモデルの平均）
- **何をするか:** 種や形を変えたモデルをいくつか作り、確率を平均する。間違いが打ち消し合う。計算は人数分かかる
- **出典:** Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017).
  [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)
- **1回の学習でいくつも作る（Snapshot Ensembles）:** Huang ら (ICLR 2017). [arXiv:1704.00109](https://arxiv.org/abs/1704.00109)
- **この抽出器では:** 試した（文中の数式、3人）。1文字ずつの F1 が先生1人の平均 0.9495 → 0.959。MLP は軽いので、区切り・ラベルの専門家にも
  使える（3人でも 1ページ 0.01秒くらいの増え方）。種によるぶれ（FINDINGS.md 4.7）も小さくなる

### 1.3 重みの平均（SWA・EMA）
- **何をするか:** 学習の終わりのほうの重みを平均して、1つのモデルにする。アンサンブルに近い効き目が、1人分の計算で得られる
- **出典:** Izmailov ら. *Averaging Weights Leads to Wider Optima and Better Generalization* (UAI 2018). [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
- **この抽出器では:** 未。早期終了で一番良かった重みを1つ選んでいるのを、最後の何エポックかの平均に変えるだけで試せる

---

## 2. 学習の目標（損失・ラベル）を工夫する

### 2.1 ラベルのなめらか化（Label Smoothing）
- **何をするか:** 正解を 1 / 0 ではなく 0.9 / 0.1 のようにして学習する。自信過剰を防ぎ、少し汎化が良くなる。蒸留の「先生の確率」を、
  先生なしで一様に入れたものと見ることもできる
- **出典:** Szegedy ら. *Rethinking the Inception Architecture for Computer Vision* (CVPR 2016). [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
  ／ 効く条件の分析: Müller, Kornblith, Hinton. *When Does Label Smoothing Help?* (NeurIPS 2019). [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
- **この抽出器では:** 未。蒸留より手軽（先生が要らない）なので、まず比べる相手としてよい

### 2.2 少ない種類を重く数える（クラスの重み・Focal Loss）
- **何をするか:** 「はい」が少ないときに見逃しを重く数える（クラスの重み）。Focal Loss は、もう当たっている簡単な例の重みを下げ、難しい例に集中させる
- **出典:** Lin ら. *Focal Loss for Dense Object Detection* (ICCV 2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **この抽出器では:** クラスの重み（pos_weight）は使っている。外しても点は変わらなかった（FINDINGS.md 3 節。区切りは半々、ラベルはオペレータが選ぶため）。
  Focal Loss は未。箇条書きのような、数が少なく難しいラベルで試す価値がある

### 2.3 確率の目盛り合わせ（Temperature Scaling）
- **何をするか:** 学習のあと、検証用データでロジットを1つの数 T で割り、確率の自信の度合いを本当の当たりやすさに合わせる。
  正解率は変わらないが、「自信 0.9 なら 9割当たる」ようになる
- **出典:** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks* (ICML 2017). [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- **この抽出器では:** 未。オペレータは専門家たちの自信を比べて一番を選ぶので、専門家ごとに目盛りがずれていると選び方を間違える。
  pos_weight で自信過剰になることも分かっている（文中の数式のしきい値を 0.9 にした理由）。**オペレータの方式と相性が良い**

---

## 3. データを工夫する

### 3.1 データの水増し・ばらつかせ（Data Augmentation・Domain Randomization）
- **何をするか:** 見た目をわざと大きくばらつかせて学習し、本番の見た目を「ばらつきの1つ」にする
- **出典:** Tobin ら. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (IROS 2017).
  [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
- **この抽出器では:** 試した（書体・文書クラス・材料のパック）。**一番効いた手法の1つ。** 書体を混ぜて AI の検出F1 0.970 → 0.981、
  材料を増やして（＋見た目の手がかり）本物の文のラベル 0.956 → 0.982

### 3.2 近道学習に気をつける（Shortcut Learning）
- **何をするか:** モデルが「本当の手がかり」ではなく、データの偏り（近道）で当てていないかを確かめ、近道を潰すデータと本当の手がかりを足す
- **出典:** Geirhos ら. *Shortcut Learning in Deep Neural Networks* (Nature Machine Intelligence 2020). [arXiv:2004.07780](https://arxiv.org/abs/2004.07780)
- **この抽出器では:** 何度も起きた（「表1 で始まればキャプション」、箇条書きの見出し語・記号）。手法というより、**いつも疑う姿勢**として一番大事

### 3.3 混ぜ合わせ（Mixup）
- **何をするか:** 2つの例の特徴と正解を、同じ割合で混ぜた例を作って学習する。境目がなめらかになり、覚えすぎを防ぐ
- **出典:** Zhang ら. *mixup: Beyond Empirical Risk Minimization* (ICLR 2018). [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
- **この抽出器では:** 未。手がかりは数の並びなので、MLP の専門家にはそのまま使える。ただ、データを増やす（3.1）方が直接的

### 3.4 自分で答えを付けて学習する（Self-training・Noisy Student）
- **何をするか:** 正解の無いデータ（本物の PDF など）にモデルが答えを付け、自信のあるものを正解として足して学習し直す。
  Noisy Student は、生徒に水増しやドロップアウトを入れ、先生より大きく育てる
- **出典:** Xie ら. *Self-training with Noisy Student improves ImageNet classification* (CVPR 2020). [arXiv:1911.04252](https://arxiv.org/abs/1911.04252)
  ／ 元になった考え: Lee. *Pseudo-Label* (ICML 2013 Workshop)
- **この抽出器では:** 未。TeX のソースが無い本物の PDF（kiso_jikken1 など）を学習に使える道。ただし、間違った答えも学習してしまうので、
  自信の目盛り合わせ（2.3）とセットにする

### 3.5 難しい例を集めて学習する（Hard Example Mining）
- **何をするか:** モデルが間違えた例・迷った例を多めに学習させる
- **出典:** Shrivastava, Gupta, Girshick. *Training Region-based Object Detectors with Online Hard Example Mining* (CVPR 2016).
  [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
- **この抽出器では:** 未。合成データなので、間違えやすい見た目（description の箇条書き、本文と同じ大きさの見出し）を多めに作る形で簡単にできる

---

## 4. 学習のしかたを工夫する

### 4.1 学習率の変え方（Cosine Annealing・Warmup）と AdamW
- **何をするか:** 学習率を一定にせず、だんだん下げる（コサイン）。AdamW は Adam の重みの減衰（weight decay）の入れ方を直したもの
- **出典:** Loshchilov, Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts* (ICLR 2017). [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
  ／ *Decoupled Weight Decay Regularization* (ICLR 2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- **この抽出器では:** 未。学習率は 3e-4〜1e-2 でほぼ同じだった（FINDINGS.md 3 節）ので、大きくは効かない見込み。
  文中の数式の Transformer（エポックごとに ±0.01 上下する）を安定させるのには効くかもしれない

### 4.2 早期終了の判定
- **何をするか:** 検証用データの成績が良くならなくなったら止める。何で判定するかが大事
- **出典:** Prechelt. *Early Stopping — But When?* (Neural Networks: Tricks of the Trade, 1998)
- **この抽出器では:** 試した。検証の「損失」で判定していて、正解率はまだ伸びていたのに止めていた（止めずに60エポックで区切り +0.003）。
  判定を見たい成績にそろえる

### 4.3 ハイパーパラメータの探し方（ランダムサーチ・ベイズ最適化）
- **何をするか:** 格子状に全部試す（グリッド）より、ランダムに選んで試す方が、同じ回数で良い設定に当たりやすい
  （効くハイパーパラメータは少数なので）。もっと賢く探す道具として Optuna がある
- **出典:** Bergstra, Bengio. *Random Search for Hyper-Parameter Optimization* (JMLR 13, 2012).
  [jmlr.org](https://jmlr.org/papers/v13/bergstra12a.html) ／ Akiba ら. *Optuna* (KDD 2019). [arXiv:1907.10902](https://arxiv.org/abs/1907.10902)
- **この抽出器では:** 1つずつ変える比べ方（sweep.py）をした。ラベルは ±0.001 で、ハイパーパラメータで上がる余地は小さい

### 4.4 ドロップアウト
- **何をするか:** 学習中に、ランダムに一部の値を 0 にする。覚えすぎを防ぐ
- **出典:** Srivastava ら. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR 15, 2014).
  [jmlr.org](https://jmlr.org/papers/v15/srivastava14a.html)
- **この抽出器では:** 使っている（MLP は 0.1）。強さは比べていない

---

## 5. モデルの形を変える

### 5.1 勾配ブースティングの決定木（LightGBM など）
- **何をするか:** 決定木をたくさん、前の木の間違いを直すように積み重ねる。表の形の手がかり（数の並び）では、ニューラルネットより強いことが多い
- **出典:** Ke ら. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* (NeurIPS 2017).
  [proceedings](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- **この抽出器では:** 未。区切り・ラベル・レベルの専門家は「手がかりの数の並び → はい/いいえ」なので、そのまま置き換えて比べられる。
  **MLP と比べる価値が一番高いモデル。** どの手がかりが効いているかも見やすい（近道探しにも使える）

### 5.2 並び全体を見るモデル（CRF・ブロック単位の Transformer）
- **何をするか:** ブロックを1つずつ判定するのではなく、ページのブロックの並び全体で、つじつまの合うラベルの並びを選ぶ
  （箇条書きの項目が続く、見出しのレベルが順に並ぶ）
- **出典:** Lafferty, McCallum, Pereira. *Conditional Random Fields* (ICML 2001) ／ 文書のレイアウトに Transformer を使った例:
  Xu ら. *LayoutLM: Pre-training of Text and Layout for Document Image Understanding* (KDD 2020). [arXiv:1912.13318](https://arxiv.org/abs/1912.13318)
- **この抽出器では:** 未。今のラベルの専門家は前後1つのブロックしか見ない。大きな作り直しになるが、箇条書き・見出しのレベルの残りの間違いに効く見込み

### 5.3 事前学習したモデルを使う（転移学習）
- **何をするか:** 大量の文書で先に学習したモデル（LayoutLM など）を、少しのデータで仕上げる
- **出典:** 上の LayoutLM
- **この抽出器では:** プロジェクトの方針（自作の AI）から外れるので、比べる相手として使う程度

---

## この抽出器での優先順位（目安）

| 順 | 手法 | 期待 | 手間 | 理由 |
|---|---|---|---|---|
| 1 | データと手がかり（3.1・3.2・3.5） | 大 | 中 | ここまで一番効いた。弱いラベルから、近道を潰して本当の手がかりを足す |
| 2 | 勾配ブースティング（5.1） | 中 | 小 | 専門家の形はそのままで、モデルだけ置き換えて比べられる |
| 3 | 確率の目盛り合わせ（2.3） | 中 | 小 | オペレータが専門家の自信を比べる方式なので、目盛りのずれがそのまま間違いになる |
| 4 | アンサンブル・重みの平均（1.2・1.3） | 小〜中 | 小 | MLP は軽い。種によるぶれも小さくなる |
| 5 | Focal Loss・ラベルのなめらか化（2.2・2.1） | 小 | 小 | 箇条書きのような難しいラベルで試す |
| 6 | 自分で答えを付けて学習（3.4） | 中 | 中 | 本物の PDF を学習に使える。目盛り合わせとセット |
| 7 | 並び全体を見るモデル（5.2） | 中〜大 | 大 | 残りのつじつまの間違いに効くが、作り直しになる |
| — | 学習率の変え方・ハイパーパラメータ探し（4.1・4.3） | 小 | 小 | 比べた範囲では、ほとんど効かなかった |
