"""PDFの中身をのぞく練習用スクリプト。

使い方（ai/ で実行）:
    python -m hybrid_extractor.look rule_based/sample_data/sample.pdf
"""

import sys

import pymupdf


def main() -> None:
    pdf_path = sys.argv[1]
    doc = pymupdf.open(pdf_path)

    print("ページ数:", len(doc))

    page = doc[0]
    print("幅:", page.rect.width, "高さ:", page.rect.height)

    print("---- 1ページ目の文字 ----")
    print(page.get_text())

    print("---- span ごとの情報 ----")
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block["type"] != 0:           # 画像のブロックは飛ばす
            continue
        for line in block["lines"]:                 # ③ このブロックの行の一覧
            for span in line["spans"]:             # ④ この行の span の一覧
                print(span["text"], round(span["size"], 1), span["font"], span["bbox"])


if __name__ == "__main__":
    main()
