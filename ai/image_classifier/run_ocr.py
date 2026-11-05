"""
image_dataset の画像を EasyOCR で読み、段落にまとめたIRを <dir>/ocr/ に保存する（GPUで実行する用）。

保存先とファイル名は image_rule_based/evaluate.py の load_or_run_ocr と同じなので、ルールベース側の採点も
この結果をそのまま使い回せる（同じOCR結果で、ルールベースとAIを比べられる）。

    python -m image_classifier.run_ocr image_classifier/data/train --shard 0/4   # 4分割の0番目だけ

すでに保存されている画像は飛ばすので、途中で止まっても続きから再開できる。
"""

import argparse
import glob
import os
from functools import lru_cache

import easyocr_ocr.ir_adapter as easyocr_adapter
from common.ir import load_ir, save_ir
from easyocr_ocr.grouping import group_document


@lru_cache(maxsize=None)
def _gpu_reader(langs: tuple[str, ...]):
    """easyocr_ocr の get_reader は gpu=False 固定なので、GPUがあれば使う版に差し替える（読み取り方は同じ）。"""
    import easyocr
    import torch

    return easyocr.Reader(list(langs), gpu=torch.cuda.is_available(), verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="image_dataset の画像をEasyOCRで読み、段落単位のIRを保存する")
    parser.add_argument("dir", help="image_dataset.build の出力先（images/ と truth/ があるディレクトリ）")
    parser.add_argument("--shard", default="0/1", help="i/n: 全体をn分割したi番目だけ処理する（ジョブを並べて速くする）")
    args = parser.parse_args()

    index, count = (int(x) for x in args.shard.split("/"))
    easyocr_adapter.get_reader = _gpu_reader
    truth_paths = sorted(glob.glob(os.path.join(args.dir, "truth", "*.ir.json")))[index::count]
    os.makedirs(os.path.join(args.dir, "ocr"), exist_ok=True)

    done = 0
    for n, truth_path in enumerate(truth_paths, 1):
        out = os.path.join(args.dir, "ocr", os.path.basename(truth_path))
        if os.path.exists(out):
            continue
        truth = load_ir(truth_path)
        doc = group_document(easyocr_adapter.extract_ir(os.path.join(args.dir, "images", truth.source)))
        save_ir(doc, out)
        done += 1
        if n % 50 == 0:
            print(f"{n}/{len(truth_paths)}", flush=True)
    print(f"完了: 新しく {done} 枚を読み取り（担当 {len(truth_paths)} 枚）", flush=True)


if __name__ == "__main__":
    main()
