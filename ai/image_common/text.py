"""
OCRが区切って返した単語（または同じ行の箱の文字列）を1行の文字列につなぐ。

EasyOCR（easyocr_ocr/grouping.py）と Tesseract（tesseract_ocr/ir_adapter.py）で同じつなぎ方にして、
空白の入れ方の違いで CER に差が出ないようにする。
"""


def join_words(words: list[str]) -> str:
    """単語をつなぐ。英数字・記号のあとに英数字が続くときだけ空白を入れる。
    日本語の間には入れない（「本 文」にしない）。句読点・記号の前にも入れない（"Hello ," にしない）。"""
    out = ""
    for w in words:
        if out and w and out[-1].isascii() and not out[-1].isspace() and w[0].isascii() and w[0].isalnum():
            out += " "
        out += w
    return out
