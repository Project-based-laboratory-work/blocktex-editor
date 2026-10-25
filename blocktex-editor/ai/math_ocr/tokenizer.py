"""
LaTeXの数式文字列を「トークン」の列に分割し、モデルが扱える整数ID列と相互変換する。

モデルは文字列を直接扱えないので、
    r"\\frac{a}{2}"  →  ["\\frac", "{", "a", "}", "{", "2", "}"]  →  [1, 7, 4, 30, 5, 4, 12, 5, 2]
のように「トークン列 → ID列」に変換してから入力する（先頭の1と末尾の2は <bos>/<eos>）。

トークンの単位:
  - \\alpha や \\frac のようなコマンドは1トークン（文字単位に分けると "\\", "a", "l", ... となり、
    モデルが余計に長い系列を覚える必要がある）
  - \\{ のような「バックスラッシュ + 記号1文字」も1トークン
  - それ以外は1文字1トークン。空白は捨てる（数式モードでは空白が出力に影響しないため）
"""

import json
import re
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

_TOKEN_RE = re.compile(r"\\[a-zA-Z]+|\\.|\S")
_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")


def tokenize(latex: str) -> list[str]:
    return _TOKEN_RE.findall(latex)


def detokenize(tokens: list[str]) -> str:
    """トークン列をLaTeX文字列に戻す。\\sin と x の間のように、空白が無いと別のコマンド（\\sinx）に
    なってしまう箇所にだけ空白を入れる。"""
    out = []
    for i, token in enumerate(tokens):
        out.append(token)
        is_command = _COMMAND_RE.fullmatch(token) is not None
        next_is_letter = i + 1 < len(tokens) and tokens[i + 1][0].isalpha()
        if is_command and next_is_letter:
            out.append(" ")
    return "".join(out)


class Vocab:
    """トークン ⇔ ID の対応表。特殊トークンは常に先頭に置き、IDを固定する（PAD=0, BOS=1, EOS=2, UNK=3）。"""

    def __init__(self, itos: list[str]):
        self.itos = itos  # index → string
        self.stoi = {token: i for i, token in enumerate(itos)}  # string → index

    @classmethod
    def build(cls, formulas: list[str]) -> "Vocab":
        tokens = {token for formula in formulas for token in tokenize(formula)}
        return cls(SPECIAL_TOKENS + sorted(tokens - set(SPECIAL_TOKENS)))

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, latex: str) -> list[int]:
        """文字列 → [BOS, ...トークンID..., EOS]"""
        unk = self.stoi[UNK]
        return [self.bos_id] + [self.stoi.get(t, unk) for t in tokenize(latex)] + [self.eos_id]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """ID列 → トークン列。EOSで打ち切り、特殊トークンは捨てる（モデル出力の後処理用）。"""
        tokens = []
        for i in ids:
            if i == self.eos_id:
                break
            token = self.itos[i]
            if token not in SPECIAL_TOKENS:
                tokens.append(token)
        return tokens

    def decode(self, ids: list[int]) -> str:
        return detokenize(self.ids_to_tokens(ids))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
