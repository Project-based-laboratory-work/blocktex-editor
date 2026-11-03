"""
Tesseractで画像の文字を読み取り、共通のIR形式（common/ir.py）で出力するアダプタ。
英語・日本語の横書きに対応する（縦書きは対象外）。要素は Tesseract の段落単位（段落内の行は "\\n" でつなぐ）。

使い方（ai/ で）:
    python -m tesseract_ocr.ir_adapter ../ocr/sample/jphorizon/jphorizontal_001.jpg -o out/
      → out/jphorizontal_001.base.ir.json
    python -m tesseract_ocr.ir_adapter ../ocr/sample/eng/eng_001.jpg --lang eng --preset otsu -o out/

言語の既定は jpn+eng、ページ分割モード（psm）の既定は 3（自動でレイアウト解析）。

事前に Tesseract 本体と言語データ eng / jpn を入れておく（tesseract --list-langs で確認）。
言語データを別の場所に置いた場合は、環境変数 TESSDATA_PREFIX にそのフォルダを指定する。
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

from common.ir import Element, IRDocument, Page, normalize_bbox, save_ir, validate
from image_common.preprocess import apply_preprocess
from image_common.text import join_words
from tesseract_ocr.presets import PRESETS

NAME = "tesseract"
DEFAULT_PRESET = "base"
DEFAULT_LANG = "jpn+eng"
DEFAULT_PSM = 3  # 自動でレイアウト解析
DEFAULT_OEM = 1  # LSTM

# 単語ごとの conf（0〜100）がこれ未満なら捨てる。-1 は文字以外の枠（ブロック・行など）
MIN_WORD_CONF = 0.0


@dataclass
class OcrRun:
    """Tesseract 1回分の結果"""

    lang: str
    psm: int
    data: dict[str, list]


def run_tesseract(image: Image.Image, lang: str, psm: int, oem: int) -> OcrRun:
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    return OcrRun(lang, psm, data)


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

        idx = [i for line in lines for i in line]
        x0, y0, x1, y1 = box(idx)
        if x1 <= x0 or y1 <= y0:
            continue

        text = "\n".join(join_words([data["text"][i].strip() for i in line]) for line in lines)

        # 文字サイズ：行の高さの中央値（元の画像の px）で推定する
        sizes = sorted(b[3] - b[1] for b in line_boxes)

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


def check_langs(lang: str | None) -> None:
    """必要な言語データが入っているか、実行前に確かめる。"""
    missing = set((lang or DEFAULT_LANG).split("+")) - set(pytesseract.get_languages(config=""))
    if missing:
        raise SystemExit(
            f"Tesseract の言語データが足りません: {sorted(missing)}\n"
            "tesseract --list-langs で確認し、入れるか、置いたフォルダを TESSDATA_PREFIX に指定してください。"
        )


# 解析の流れ：画像を開く → 前処理 → Tesseractを実行 → IRに変換
def extract_ir(image_path: str, preset: str = DEFAULT_PRESET, lang: str | None = None) -> IRDocument:
    """画像1枚を、プリセットの設定でTesseractに読ませ、IRを返す。lang を省略すると DEFAULT_LANG。"""
    if preset not in PRESETS:
        raise ValueError(f"不明なプリセット {preset!r}（使えるのは {list(PRESETS)}）")
    settings = PRESETS[preset]

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    width, height = rgb.size

    processed, scale = apply_preprocess(rgb, settings["preprocess"])

    oem = settings["tesseract"].get("oem", DEFAULT_OEM)
    psm = settings["tesseract"].get("psm", DEFAULT_PSM)
    run = run_tesseract(processed, lang or DEFAULT_LANG, psm, oem)

    params = {
        "preset": preset,
        **settings,
        "lang": run.lang,
        "psm": run.psm,
        "oem": oem,
        "tesseract_version": str(pytesseract.get_tesseract_version()),
    }

    page = Page(page=1, width=width, height=height, unit="px", elements=to_elements(run, width, height, scale))
    return IRDocument(source=os.path.basename(image_path), extractor=NAME, pages=[page], params=params)


def main() -> None:
    parser = argparse.ArgumentParser(description="画像 → IR（Tesseract）")
    parser.add_argument("image", help="入力画像")
    parser.add_argument("-o", "--out-dir", default=".", help="出力先のディレクトリ")
    parser.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS), help="設定の組（presets.py）")
    parser.add_argument("--lang", default=None, help=f"Tesseractの言語（例: eng, jpn+eng）。省略時は {DEFAULT_LANG}")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise SystemExit(f"エラー: {args.image} が見つかりません。")
    check_langs(args.lang)

    doc = extract_ir(args.image, args.preset, args.lang)
    problems = validate(doc)
    if problems:
        raise SystemExit("IR の形式に問題があります:\n" + "\n".join(problems))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{Path(args.image).stem}.{args.preset}.ir.json")
    save_ir(doc, out_path)
    print(f"{out_path}（要素 {len(doc.pages[0].elements)} 個）")


if __name__ == "__main__":
    main()
