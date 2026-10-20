import time
import easyocr
from PIL import Image
from src.ocr_result import save_ocr_result

# 日本語と英語を認識するOCRエンジン
reader = easyocr.Reader(["ja", "en"], gpu=False)


def run_easyocr(image_path, image_id):
    # 画像の幅と高さを取得
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    # OCRの実行時間を測定
    start = time.perf_counter()
    raw_results = reader.readtext(str(image_path))
    elapsed = time.perf_counter() - start

    results = []

    # OCRの結果を共通形式に変換
    for box, text, confidence in raw_results:
        results.append({
            "text": text,
            "confidence": float(confidence),
            "confidence_scale": "0-1",
            "box": [
                [round(float(x)), round(float(y))]
                for x, y in box
            ],
        })

    # 共通関数を使ってJSONに保存
    return save_ocr_result(
        image_id=image_id,
        engine_name="easyocr",
        engine_version=getattr(easyocr, "__version__", "unknown"),
        image_width=image_width,
        image_height=image_height,
        processing_time_sec=elapsed,
        results=results,
    )


if __name__ == "__main__":
    saved_path = run_easyocr(
        image_path="data/images/doc001.jpg",
        image_id="doc001",
    )
    print(f"保存先: {saved_path}")