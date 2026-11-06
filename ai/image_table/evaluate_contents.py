"""
表の中身（セルの文字・列の揃え・罫線）まで含めて、チーム共通の物差し（evaluation/metrics.py の table_counts）で採点する。

    python -m image_table.evaluate_contents image_table/data/team --limit 150

区切り方だけを「ルールベース（空白で区切る）」と「表AI」で入れ替え、その後の処理（EasyOCRでセルを読む・罫線・揃え）は
同じにする（predict.element_from_spans）。違いが出れば、それは区切り方の違い。
EasyOCR はセル1つずつ読むので遅い（CPUで表1つ数秒）。--no-ocr ならセルの文字は採点しない（形・揃え・罫線だけ）。
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image

from common.ir import Element
from evaluation.metrics import Counts, summarize, table_counts
from image_table.evaluate import load_model, predict, rule_predict
from image_table.predict import DEFAULT_CHECKPOINT, element_from_spans, get_reader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="team_tables.py で作ったディレクトリ")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    reader = None if args.no_ocr else get_reader()
    splitters = {"ルールベース（空白で区切る）": rule_predict, "表AI": lambda im: predict(model, im)}
    counts = {n: Counts() for n in splitters}
    labels = sorted((args.dir / "labels").glob("*.json"))[: args.limit]
    for path in labels:
        label = json.loads(path.read_text(encoding="utf-8"))
        gt = Element(kind="table", bbox=(0, 0, 1, 1), cells=label["cells"], aligns=label["aligns"],
                     row_rules=label["row_rules"], col_rules=label["col_rules"])  # fmt: skip
        with Image.open(args.dir / "images" / f"{path.stem}.png") as image:
            image.load()
        for name, split in splitters.items():
            rows, cols = split(image)
            pred = element_from_spans(image, rows, cols, reader=reader)
            if pred.cells is None:
                pred.cells = []  # 区切れなかった表も、外れとして数える
            if args.no_ocr:
                pred.cells = [[""] * len(row) for row in pred.cells]
            table_counts(pred, gt, counts[name])

    print(f"{args.dir}: {len(labels)} 個の表" + ("（セルの文字は採点しない）" if args.no_ocr else ""))
    keys = ["table_shape_acc", "table_align_acc", "table_rule_acc"] + ([] if args.no_ocr else ["table_cell_acc"])
    print(f"{'':<26}" + "".join(f"{k.removeprefix('table_'):>12}" for k in keys))
    for name in splitters:
        s = summarize(counts[name])
        print(f"{name:<24}" + "".join(f"{s[k] or 0:>12.3f}" for k in keys))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
