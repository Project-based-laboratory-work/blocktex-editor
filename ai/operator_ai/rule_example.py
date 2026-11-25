"""
ルール版オペレータ（進め方4）の書き方の見本。rules.py を作るときに写して使う。

    python -m operator_ai.evaluate operator_ai.rule_example:choose

f は operator_ai/features.py で画像から測った値（列の意味は operator_ai/README.md の「特徴量」）。
正解の params（dark_mode・dpi など）は使えない。本物のスクショには無いため。

この見本は、val では単独ベスト（tesseract:up2）とほぼ同じ点数で、まだ上回らない。
文字の高さだけでは足りず、ほかの手がかりを組み合わせる必要がある（analysis/report.md を参照）。
"""


def choose(f: dict) -> str:
    if f["char_height"] >= 20:  # 文字が十分大きい → 拡大すると、かえって崩れる
        return "tesseract:invert"
    return "tesseract:up2"  # それ以外は単独ベスト
