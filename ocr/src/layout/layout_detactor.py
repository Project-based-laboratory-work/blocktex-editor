"""
OpenCVを使った文書内の領域検出。

役割：
- 画像を読み込む
- 画像を前処理する
- 文書内の領域を検出する
- 各領域の位置や大きさなどの特徴を取得する
- 検出結果をPythonの辞書として返す

JSONへの保存は行わない。
"""

from pathlib import Path

import cv2
import numpy as np


def detect_layout(image_path: str) -> dict:
    """
    画像内の領域を検出する。

    Parameters
    ----------
    image_path : str
        入力画像のファイルパス。

    Returns
    -------
    dict
        画像情報と検出した領域の情報。
    """

    # 1. 画像ファイルの存在を確認
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"画像ファイルが見つかりません: {image_path}"
        )

    # 2. 画像を読み込む
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"画像を読み込めませんでした: {image_path}"
        )

    # 画像の高さ・幅を取得
    image_height, image_width = image.shape[:2]

    # 3. グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. 二値化
    # 文字を白、背景を黒にする
    #THRESH_BINARY_INV:通常の二値化とは白黒を反転させる指定
    #THRESH_OTSU:画像の明るさの分布から二値化のしきい値を自動的に決める
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 5. 文字を横方向につなげて、領域の候補を作る
    # 横書きの文章や数式などを、まとまりとして検出するための処理
    #一旦横に繋げている
    kernel_width = max(15, image_width // 40)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 3)
    )

    connected = cv2.dilate(
        binary,
        kernel,
        iterations=1
    )

    # 6. 領域の輪郭を検出
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []

    # 7. 領域ごとの位置と特徴を取得
    for contour in contours:
        #検出した領域を囲む長方形の位置と大きさを取得
        x, y, width, height = cv2.boundingRect(contour)
        
        # 小さすぎる領域は除外
        if width < 10 or height < 5:
            continue

        # 画像全体に近い大きな領域は除外
        if (
            width >= image_width * 0.95
            and height >= image_height * 0.95
        ):
            continue

        # 領域内の文字の密集度を計算
        roi = binary[
            y:y + height,
            x:x + width
        ]

        total_pixels = width * height
        foreground_pixels = cv2.countNonZero(roi)

        density = foreground_pixels / total_pixels

        # 領域の特徴を計算
        aspect_ratio = width / height

        regions.append(
            {
                "region_id": len(regions) + 1,
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height)
                },
                "features": {
                    #領域の面積；領域の大きさを比較していく、、？
                    "area": int(width * height),
                    #幅➗高さ；横長の数式などを見つける手がかりにする
                    "aspect_ratio": round(
                        float(aspect_ratio), 4
                    ),
                    #領域内で白い画素が占める割合；文字や記号の密集度を調べる？
                    "density": round(
                        float(density), 4
                    )
                }
            }
        )

    # 8. 領域を上から下、左から右の順に並べる
    regions.sort(
        key=lambda region: (
            region["bbox"]["y"],
            region["bbox"]["x"]
        )
    )

    # 並べ替え後に領域番号を振り直す
    for index, region in enumerate(regions, start=1):
        region["region_id"] = index

    # 9. 検出結果を辞書として返す
    return {
        "image": path.name,
        "image_size": {
            "width": int(image_width),
            "height": int(image_height)
        },
        "detector": "opencv",
        "method": "threshold_and_dilation",
        "region_count": len(regions),
        "regions": regions
    }