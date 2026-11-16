"""
エディタの「PDF・画像を読み込む」で、登録簿（models.json）のモデルを使ってファイルを読み、エディタ用JSONにする。

評価サイトで選べるモデルと同じものを使う（server/ の /api/extract がこれを呼ぶ）。
{"labeled": ラベル付きIR, "document": エディタ用JSON} を標準出力に出す（rule_based.ir_adapter --stdout と同じ形）。

使い方:
    python -m evaluation.extract rule_based rule_based/sample_data/sample.pdf
    python -m evaluation.extract tesseract_image_rule_based page.png --title スライド
"""
import argparse
import json
from pathlib import Path

from common.blocks import ir_to_document
from common.ir import to_dict, validate
from evaluation.models import load_registry, predict, predict_image

# 画像として読む拡張子。それ以外はPDFとして読む
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="登録簿のモデルでPDF・画像を読み、エディタ用JSONを標準出力に出す")
    parser.add_argument("model", help="models.json の id（例: rule_based）")
    parser.add_argument("file", help="入力のPDFか画像（png / jpg）")
    parser.add_argument("--title", help="エディタ用JSONの文書名（省略時は入力ファイル名）")
    args = parser.parse_args(argv)

    registry = load_registry()
    if args.model not in registry:
        raise SystemExit(f"登録されていないモデルです: {args.model}")
    spec = registry[args.model]

    if Path(args.file).suffix.lower() in IMAGE_SUFFIXES:
        labeled = predict_image(spec, args.file)
    else:
        labeled = predict(spec, args.file)
    problems = validate(labeled)
    if problems:
        raise SystemExit("labeled の形式に問題があります:\n" + "\n".join(problems))
    print(json.dumps({"labeled": to_dict(labeled), "document": ir_to_document(labeled, args.title)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
