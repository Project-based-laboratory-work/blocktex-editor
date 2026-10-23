"""
Tesseractを使った一般テキストのOCR処理。

役割：
- 画像を読み込む
- 文字を認識する
- 文字の信頼度と位置を取得する
- 認識結果をPythonの辞書として返す

JSONへの保存は行わない。
"""

from pathlib import Path

import pytesseract
from PIL import Image


def recognize_text(
    image_path: str,
    language: str = "jpn+eng",
    psm: int = 3,
) -> dict:
    """
    画像をOCRで解析し、認識結果を返す。

    Parameters
    ----------
    image_path : str
        入力画像のパス。
    language : str
        Tesseractの言語設定。
    psm : int
        Tesseractのページ分割モード。

    Returns
    -------
    dict
        画像情報と認識した文字・信頼度・座標を含む辞書。
    """

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"画像ファイルが見つかりません: {path}"
        )

    # 1. 画像を読み込む
    with Image.open(path) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size

        # 2. Tesseractで文字・信頼度・座標を取得する
        config = f"--oem 3 --psm {psm}"

        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

    # 3. OCRの結果を整理する
    results = []

    for i, text in enumerate(data["text"]):
        text = text.strip()

        # 空文字や認識対象外の領域は除外する
        if not text:
            continue

        confidence = float(data["conf"][i])

        if confidence < 0:
            continue

        results.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": {
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                },
            }
        )

    # 4. Pythonの辞書として結果を返す
    return {
        "image": path.name,
        "image_size": {
            "width": image_width,
            "height": image_height,
        },
        "engine": "tesseract",
        "language": language,
        "settings": {
            "oem": 3,
            "psm": psm,
        },
        "results": results,
    }