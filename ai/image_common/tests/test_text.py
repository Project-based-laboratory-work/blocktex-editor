from image_common.text import join_words


def test_spaces_only_between_latin_words():
    assert join_words(["Hello", "world"]) == "Hello world"
    assert join_words(["本", "文", "です"]) == "本文です"
    assert join_words(["Tesseract", "で", "OCR"]) == "TesseractでOCR"


def test_no_space_before_punctuation():
    assert join_words(["Hello", ","]) == "Hello,"
    assert join_words(["Hello", ",", "world", "."]) == "Hello, world."


def test_empty_pieces_are_ignored():
    assert join_words(["", "abc", "", "def"]) == "abc def"
