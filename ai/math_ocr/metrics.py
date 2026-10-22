"""
予測したトークン列と正解のトークン列を比べる評価指標。

- exact_match: 数式が丸ごと一致した割合。1トークンでも違えば不正解なので厳しい
- token_error_rate (TER): 編集距離 ÷ 正解のトークン数。音声認識の「単語誤り率(WER)」と同じ考え方で、
  「人が何か所直せば正解になるか」に近い。ReTeXの「人が直すのは誤った部分だけ」という目標と相性が良い
"""

from common.text import edit_distance  # noqa: F401  math_ocr.evaluate などはここから import している


def evaluate(predictions: list[list[str]], references: list[list[str]]) -> dict[str, float]:
    exact = sum(p == r for p, r in zip(predictions, references))
    errors = sum(edit_distance(p, r) for p, r in zip(predictions, references))
    ref_tokens = sum(len(r) for r in references)
    return {
        "exact_match": exact / len(references),
        "token_error_rate": errors / max(ref_tokens, 1),
    }
