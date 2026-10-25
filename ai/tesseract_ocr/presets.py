"""
Tesseractの設定の組（プリセット）。easyocr_ocr/presets.py と同じ考え方で、実験のたびにコードを書き換えず、ここに名前を付けて足していく。

- preprocess: easyocr_ocr/preprocess.py の STEPS の名前を、かける順に並べる（EasyOCRと同じ名前のプリセットは同じ前処理）
- tesseract:  Tesseract の設定。psm / oem を書くと、向き（--direction）ごとの既定値より優先する

比べやすいように、base から1か所だけ変えたものを並べている。
使った設定はIRの params にそのまま記録される。
"""

PRESETS: dict[str, dict] = {
    # 何もしない（基準）
    "base": {"preprocess": [], "tesseract": {}},
    # 前処理を1つだけ変える（EasyOCRと同じ名前・同じ中身）
    "up2": {"preprocess": ["upscale2"], "tesseract": {}},
    "up3": {"preprocess": ["upscale3"], "tesseract": {}},
    "gray": {"preprocess": ["grayscale"], "tesseract": {}},
    "contrast": {"preprocess": ["autocontrast"], "tesseract": {}},
    "invert": {"preprocess": ["invert_if_dark"], "tesseract": {}},
    "otsu": {"preprocess": ["otsu"], "tesseract": {}},
    "adaptive": {"preprocess": ["adaptive"], "tesseract": {}},
    "adaptive_mean": {"preprocess": ["adaptive_mean"], "tesseract": {}},
    "up2_adaptive": {"preprocess": ["upscale2", "adaptive"], "tesseract": {}},
    # Tesseract のページ分割モードを1つだけ変える
    #   3: 自動でレイアウト解析（横書きの既定） / 5: 縦書きの1ブロック（縦書きの既定）
    #   4: 1段組みの文章 / 6: 1つのブロック / 11: まばらな文字（スクショ向き）
    "psm3": {"preprocess": [], "tesseract": {"psm": 3}},
    "psm4": {"preprocess": [], "tesseract": {"psm": 4}},
    "psm5": {"preprocess": [], "tesseract": {"psm": 5}},
    "psm6": {"preprocess": [], "tesseract": {"psm": 6}},
    "psm11": {"preprocess": [], "tesseract": {"psm": 11}},
}
