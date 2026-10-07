"""
ステップ0: 色付けによるラベル付けが成立するかを確認する。

確認する2つの仮定:
  1. PyMuPDF が span["color"] で文字色を読める
  2. 色を付けてもレイアウト（bbox）がずれない

使い方:
    python3 probe_color.py probe_colored.pdf probe_plain.pdf
"""

import sys
import pymupdf


def dump_spans(pdf_path: str) -> None:
    """各spanの色・サイズ・テキストを表示する。"""
    print(f"=== spanの色 ({pdf_path}) ===")
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        print(
                            f"  p{page_no} color=#{span['color']:06X} "
                            f"size={span['size']:.1f} text={text[:30]!r}"
                        )


def block_bboxes(pdf_path: str) -> list[tuple[int, tuple[float, ...]]]:
    """(ページ番号, bbox) の一覧を返す。"""
    result = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                result.append((page_no, tuple(round(v, 2) for v in block["bbox"])))
    return result


def compare(colored_pdf: str, plain_pdf: str, tol: float = 0.01) -> None:
    """色付き版と色なし版でブロックのbboxが一致するか比べる。"""
    print("=== bbox比較 ===")
    a = block_bboxes(colored_pdf)
    b = block_bboxes(plain_pdf)

    if len(a) != len(b):
        print(f"NG: ブロック数が違う（色付き={len(a)} 色なし={len(b)}）")
        return

    ok = True
    for i, ((page_a, bbox_a), (page_b, bbox_b)) in enumerate(zip(a, b)):
        if page_a != page_b or max(abs(x - y) for x, y in zip(bbox_a, bbox_b)) > tol:
            print(f"NG: ブロック{i} がずれている")
            print(f"    色付き={bbox_a}")
            print(f"    色なし={bbox_b}")
            ok = False

    if ok:
        print(f"OK: {len(a)}ブロックすべて一致（許容誤差 {tol}pt）")


def main() -> None:
    colored_pdf, plain_pdf = sys.argv[1], sys.argv[2]
    dump_spans(colored_pdf)
    print()
    compare(colored_pdf, plain_pdf)


if __name__ == "__main__":
    main()