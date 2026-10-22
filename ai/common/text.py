"""
複数のモデル・評価で使う、文字列（トークン列）の比較。
"""
from collections.abc import Sequence


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """レーベンシュタイン距離: a を b にするのに必要な 挿入・削除・置換 の最小回数（動的計画法）。
    トークンのリストでも文字列（1文字ずつ）でも使える。"""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]
