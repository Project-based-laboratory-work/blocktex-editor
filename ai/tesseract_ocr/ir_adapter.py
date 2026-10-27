"""
Tesseractで画像の文字を読み取り、共通のIR形式（common/ir.py）で出力するアダプタ。
英語・日本語（横書き・縦書き）に対応する。要素は Tesseract の段落単位（段落内の行は "\\n" でつなぐ）。

使い方（ai/ で）:
    python -m tesseract_ocr.ir_adapter ../ocr/sample/jphorizon/jphorizontal_001.jpg -o out/
      → out/jphorizontal_001.base.ir.json
    python -m tesseract_ocr.ir_adapter ../ocr/sample/eng/eng_001.jpg --lang eng --preset otsu -o out/

--direction（文字の向き）:
    horizontal … 横書き（jpn+eng, psm 3）
    vertical   … 縦書き（jpn_vert+eng, psm 5）
    auto       … 横・縦の両方で読み、「文字数 × 信頼度」が大きい方を採用（既定）
    mixed      … 1枚に横書きと縦書きが混在（jpn+jpn_vert+eng, psm 3。段落ごとに形で縦横を判断）

事前に Tesseract 本体と言語データ eng / jpn / jpn_vert を入れておく（tesseract --list-langs で確認）。
言語データを別の場所に置いた場合は、環境変数 TESSDATA_PREFIX にそのフォルダを指定する。
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from easyocr_ocr.preprocess import apply_preprocess
from tesseract_ocr.presets import PRESETS

NAME = "tesseract"
DEFAULT_PRESET = "base"
DEFAULT_DIRECTION = "auto"
DEFAULT_OEM = 1  # LSTM

# 向きごとの既定の (言語, ページ分割モード)
DIRECTION_DEFAULTS: dict[str, tuple[str, int]] = {
    "horizontal": ("jpn+eng", 3),
    "vertical": ("jpn_vert+eng", 5),
    "mixed": ("jpn+jpn_vert+eng", 3),
}
DIRECTIONS = [*DIRECTION_DEFAULTS, "auto"]

# 単語ごとの conf（0〜100）がこれ未満なら捨てる。-1 は文字以外の枠（ブロック・行など）
MIN_WORD_CONF = 0.0

# mixed のとき、行の「高さ ÷ 幅」の中央値がこれより大きい段落を縦書きとみなす
VERTICAL_RATIO = 1.5


@dataclass
class OcrRun:
    """Tesseract 1回分の結果"""

    direction: str
    lang: str
    psm: int
    data: dict[str, list]

    def score(self) -> float:
        """読めた量と確からしさをまとめた点数（auto で向きを選ぶのに使う）。
        「文字数 × 信頼度」の合計。向きを間違えると文字化けして信頼度が下がるため、正しい向きが高くなる。"""
        total = 0.0
        for word, conf in zip(self.data["text"], self.data["conf"]):
            if word.strip() and float(conf) >= MIN_WORD_CONF:
                total += len(word.strip()) * float(conf) / 100
        return round(total, 2)


def run_tesseract(image: Image.Image, direction: str, lang: str, psm: int, oem: int) -> OcrRun:
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    return OcrRun(direction, lang, psm, data)


def join_words(words: list[str]) -> str:
    """単語をつなぐ。半角（英数字・記号）どうしの間だけ空白を入れ、日本語の間には入れない。
    （日本語は Tesseract が1〜数文字ずつに区切って返すため、空白でつなぐと「本 文」のようになる）"""
    out = ""
    for w in words:
        if out and out[-1].isascii() and w[0].isascii():
            out += " "
        out += w
    return out


def is_vertical(direction: str, line_boxes: list[tuple[float, float, float, float]]) -> bool:
    """段落が縦書きかどうか。vertical / horizontal は指定どおり、mixed は行の形から判断する。"""
    if direction != "mixed":
        return direction == "vertical"
    ratios = sorted((b[3] - b[1]) / max(b[2] - b[0], 1e-6) for b in line_boxes)
    return ratios[len(ratios) // 2] > VERTICAL_RATIO


def to_elements(run: OcrRun, width: int, height: int, scale: float = 1.0) -> list[Element]:
    """単語を Tesseract の段落（block_num, par_num）ごとにまとめ、IR の text 要素にする。

    width / height は元の画像の大きさ。前処理で拡大した場合、座標は拡大後の画像のものなので、
    scale（拡大率）で割って元の画像の座標に戻す。
    """
    data = run.data

    # (block, par) → line → [単語の番号]。Tesseract が返した順（＝読み順）を保つ
    paragraphs: dict[tuple[int, int], dict[int, list[int]]] = {}
    for i, word in enumerate(data["text"]):
        if not word.strip() or float(data["conf"][i]) < MIN_WORD_CONF:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        paragraphs.setdefault(key, {}).setdefault(data["line_num"][i], []).append(i)

    def box(idx: list[int]) -> tuple[float, float, float, float]:
        """単語群を囲む枠（元の画像の px）"""
        return (
            min(data["left"][i] for i in idx) / scale,
            min(data["top"][i] for i in idx) / scale,
            max(data["left"][i] + data["width"][i] for i in idx) / scale,
            max(data["top"][i] + data["height"][i] for i in idx) / scale,
        )

    elements = []
    for lines_by_num in paragraphs.values():
        lines = list(lines_by_num.values())
        line_boxes = [box(line) for line in lines]

        vertical = is_vertical(run.direction, line_boxes)
        if vertical:
            # 縦書き：行（列）は右から左、行の中は上から下
            order = sorted(range(len(lines)), key=lambda n: -(line_boxes[n][0] + line_boxes[n][2]))
            lines = [sorted(lines[n], key=lambda i: data["top"][i]) for n in order]
            line_boxes = [line_boxes[n] for n in order]

        idx = [i for line in lines for i in line]
        x0, y0, x1, y1 = box(idx)
        if x1 <= x0 or y1 <= y0:
            continue

        text = "\n".join(join_words([data["text"][i].strip() for i in line]) for line in lines)

        # 文字サイズ：横書きは行の高さ、縦書きは列の幅の中央値（元の画像の px）で推定する
        sizes = sorted((b[2] - b[0]) if vertical else (b[3] - b[1]) for b in line_boxes)

        confs = [float(data["conf"][i]) for i in idx]
        elements.append(
            Element(
                kind="text",
                bbox=normalize_bbox((x0, y0, x1, y1), width, height),
                text=text,
                font_size=round(sizes[len(sizes) // 2], 1),
                # bold は Tesseract では取れないので書かない
                confidence=round(sum(confs) / len(confs) / 100, 4),  # 0〜1 に直した参考値
            )
        )
    return elements


def required_langs(direction: str, lang: str | None) -> set[str]:
    if lang:
        return set(lang.split("+"))
    directions = ["horizontal", "vertical"] if direction == "auto" else [direction]
    return {code for d in directions for code in DIRECTION_DEFAULTS[d][0].split("+")}


def check_langs(direction: str, lang: str | None) -> None:
    """必要な言語データが入っているか、実行前に確かめる。"""
    missing = required_langs(direction, lang) - set(pytesseract.get_languages(config=""))
    if missing:
        raise SystemExit(
            f"Tesseract の言語データが足りません: {sorted(missing)}\n"
            "tesseract --list-langs で確認し、入れるか、置いたフォルダを TESSDATA_PREFIX に指定してください。"
        )


# 解析の流れ：画像を開く → 前処理 → Tesseractを実行（auto なら横・縦の2回） → IRに変換
def extract_ir(
    image_path: str, preset: str = DEFAULT_PRESET, direction: str = DEFAULT_DIRECTION, lang: str | None = None
) -> IRDocument:
    """画像1枚を、プリセットの設定でTesseractに読ませ、IRを返す。lang を省略すると向きごとの既定値。"""
    if preset not in PRESETS:
        raise ValueError(f"不明なプリセット {preset!r}（使えるのは {list(PRESETS)}）")
    if direction not in DIRECTIONS:
        raise ValueError(f"不明な向き {direction!r}（使えるのは {DIRECTIONS}）")
    settings = PRESETS[preset]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size

    processed, scale = apply_preprocess(rgb, settings["preprocess"])

    oem = settings["tesseract"].get("oem", DEFAULT_OEM)
    runs = []
    for d in ["horizontal", "vertical"] if direction == "auto" else [direction]:
        default_lang, default_psm = DIRECTION_DEFAULTS[d]
        psm = settings["tesseract"].get("psm", default_psm)
        runs.append(run_tesseract(processed, d, lang or default_lang, psm, oem))
    best = max(runs, key=lambda r: r.score())

    params = {
        "preset": preset,
        **settings,
        "direction": direction,
        "chosen_direction": best.direction,
        "lang": best.lang,
        "psm": best.psm,
        "oem": oem,
        "tesseract_version": str(pytesseract.get_tesseract_version()),
    }
    if len(runs) > 1:
        params["direction_scores"] = {r.direction: r.score() for r in runs}

    page = Page(page=1, width=width, height=height, unit="px", elements=to_elements(best, width, height, scale))
    return IRDocument(source=os.path.basename(image_path), extractor=NAME, pages=[page], params=params)


def main() -> None:
    parser = argparse.ArgumentParser(description="画像 → IR（Tesseract）")
    parser.add_argument("image", help="入力画像")
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS), help="設定の組（presets.py）")
    parser.add_argument("--direction", default=DEFAULT_DIRECTION, choices=DIRECTIONS, help="文字の向き")
    parser.add_argument("--lang", default=None, help="Tesseractの言語（例: eng, jpn+eng）。省略時は向きごとの既定値")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"エラー: {args.image} が見つかりません。")
    check_langs(args.direction, args.lang)

    doc = extract_ir(args.image, args.preset, args.direction, args.lang)
    problems = validate(doc)
    if problems:
        raise SystemExit("IR の形式に問題があります:\n" + "\n".join(problems))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{Path(args.image).stem}.{args.preset}.ir.json")
    save_ir(doc, out_path)
    print(f"{out_path}（要素 {len(doc.pages[0].elements)} 個, 向き {doc.params['chosen_direction']}）")


if __name__ == "__main__":
    main()
