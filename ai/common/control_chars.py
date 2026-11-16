"""
PDFから取り出した文字に混ざる制御文字を、本来の文字に戻す（戻せないものは捨てる）。

Type3 の書体（TeX のビットマップの書体など）には、文字を Unicode に戻す対応表（ToUnicode）が無いことがある。
すると PyMuPDF は書体の中の番号をそのまま文字として返し、合字の ff（T1 の 0x1B）や \\textbullet（TS1 の 0x88）が
制御文字（U+001B・U+0088）になる。制御文字のまま TeX に渡すと「Unicode character ^^[ (U+001B) not set up」で止まる。

制御文字は本物の文章には出てこないので、TeX の書体の番号の決まりで置き換えても、正しい文字を壊さない。
番号の意味は書体の決まり（OT1・T1・TS1）で違うが、ぶつかる所は、取り出した文で見て多い方にした。
"""
import re

# 制御文字の番号 → 本来の文字
CONTROL_MAP = {
    # OT1（TeX の既定の欧文の書体）の合字
    "\x0b": "ff", "\x0c": "fi", "\x0d": "fl", "\x0e": "ffi", "\x0f": "ffl",
    # T1（\usepackage[T1]{fontenc}）の引用符・ダッシュ・合字
    "\x10": "“", "\x11": "”", "\x12": "„", "\x13": "«", "\x14": "»", "\x15": "–", "\x16": "—",
    "\x1b": "ff", "\x1c": "fi", "\x1d": "fl", "\x1e": "ffi", "\x1f": "ffl",
    # TS1（textcomp。\textbullet など）
    "\x88": "•",
}
# 戻せない制御文字（改行とタブは残す）
OTHER_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def fix_char(c: str) -> str:
    """1文字。制御文字なら本来の文字（2文字以上になることもある: ff）か ""（捨てる）。それ以外はそのまま"""
    if c in CONTROL_MAP:
        return CONTROL_MAP[c]
    if OTHER_CONTROL.fullmatch(c):
        return ""
    return c


def fix_text(text: str) -> str:
    """文字列の中の制御文字を、本来の文字に戻すか捨てる"""
    if not any(ord(c) < 0x20 or 0x7f <= ord(c) < 0xa0 for c in text):
        return text
    return "".join(fix_char(c) for c in text)
