"""
画像を入力にするモデルを、評価の登録簿（models.json）から呼べる形（画像1枚 → そのページのIR）にまとめる。

OCR（Tesseract / EasyOCR）は文字と位置だけのIR（①）を返し、画像の構造化器（image_rule_based）が
ラベルを付ける（②）。この組み合わせを1つの関数にして、登録簿の "entry" に書けるようにする。
OCRのライブラリは重いので、関数の中で読み込む（入っていない環境でも、ほかのモデルの評価は動くように）。
"""
from common.ir import IRDocument


def tesseract(image_path: str, preset: str = "base", lang: str | None = None) -> IRDocument:
    """Tesseract で文字を読むだけ（ラベルは付けない）"""
    from tesseract_ocr.ir_adapter import extract_ir

    return extract_ir(image_path, preset=preset, lang=lang)


def tesseract_image_rule_based(image_path: str, preset: str = "base", lang: str | None = None) -> IRDocument:
    """Tesseract で読み、image_rule_based でラベルを付ける"""
    from image_rule_based.ir_adapter import classify_ir

    return classify_ir(tesseract(image_path, preset=preset, lang=lang))


def easyocr_image_rule_based(image_path: str, preset: str = "base", langs: list[str] | None = None) -> IRDocument:
    """EasyOCR で読んで行を段落にまとめ、image_rule_based でラベルを付ける"""
    from easyocr_ocr.grouping import group_document
    from easyocr_ocr.ir_adapter import DEFAULT_LANGS, extract_ir
    from image_rule_based.ir_adapter import classify_ir

    doc = extract_ir(image_path, preset=preset, langs=tuple(langs) if langs else DEFAULT_LANGS)
    return classify_ir(group_document(doc))
