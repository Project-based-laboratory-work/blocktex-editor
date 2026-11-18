"""
表AIの採点。行・列の区切り方を、ルールベース（空白で区切る）と比べる。

    python -m image_table.evaluate image_table/data/val --checkpoint image_table/checkpoints/split/best.pt

  - 行（列）の正解率: 正解の行（列）のうち、予測の行（列）と範囲が IoU 0.5 以上で1対1に重なった割合
  - 形が合った表: 行の数・列の数が両方とも正解と同じで、すべての行・列が重なった表の割合（マス目が全部正しい）。
    位置の正解が無いデータ（team_tables.py で作る、評価用データと同じ作り方の表）では、行の数・列の数が合った表の割合

ルールベース: 縦（横）の位置ごとに、文字の画素があるかを数える（射影）。罫線（端から端まで続く線）は文字から除く。
文字の無い位置が続くところで区切る。列は、単語の間の空白で切らないよう、文字の高さより狭い空白では区切らない。
"""

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from image_table.dataset import TableDataset, load_canvas
from image_table.model import ModelConfig, TableSplitter
from image_table.postprocess import spans

RULE_SHARE = 0.6  # 横（縦）の長さのこの割合以上が黒い列（行）は罫線とみなす
MIN_RUN = 3  # これより細い区間（画素）は、罫線の切れ端などとして捨てる


def iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def matched(pred: list, gold: list) -> int:
    """IoU 0.5 以上の1対1の対応の数。範囲は上（左）から並んでいて重ならないので、順に貪欲に取ればよい"""
    used, n = set(), 0
    for g in gold:
        for k, p in enumerate(pred):
            if k not in used and iou_1d(p, g) >= 0.5:
                used.add(k)
                n += 1
                break
    return n


def load_model(checkpoint: Path) -> TableSplitter:
    state = torch.load(checkpoint, map_location="cpu")
    model = TableSplitter(ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model: TableSplitter, image: Image.Image) -> tuple[list, list]:
    canvas, scale = load_canvas(image)
    rows, cols = model(torch.from_numpy(canvas)[None, None])
    return spans(rows[0].numpy(), scale), spans(cols[0].numpy(), scale)


def _ink_runs(profile: np.ndarray, min_gap: float) -> list[tuple[float, float]]:
    """文字がある位置の並び → 区間。min_gap より狭い空白はつなぐ"""
    on = np.flatnonzero(profile)
    if len(on) == 0:
        return []
    out, start, prev = [], on[0], on[0]
    for k in on[1:]:
        if k - prev - 1 >= min_gap:
            out.append((float(start), float(prev + 1)))
            start = k
        prev = k
    out.append((float(start), float(prev + 1)))
    return out


def rule_predict(image: Image.Image) -> tuple[list, list]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if np.median(gray) < 128:
        gray = 255 - gray
    ink = gray < 128
    faint = gray < 220  # 罫線は、ふちの薄い灰色まで含めて見つける
    for axis in (1, 0):  # 横の罫線、縦の罫線
        lines = np.flatnonzero(faint.mean(axis) > RULE_SHARE)
        near = np.unique(np.clip(np.concatenate([lines - 1, lines, lines + 1]), 0, ink.shape[1 - axis] - 1))
        if axis == 1:
            ink[near, :] = False
        else:
            ink[:, near] = False
    rows = [r for r in _ink_runs(ink.any(1), 1) if r[1] - r[0] >= MIN_RUN]
    heights = [b - a for a, b in rows]
    line = float(np.median(heights)) if heights else 10.0
    cols = [c for c in _ink_runs(ink.any(0), 0.8 * line) if c[1] - c[0] >= MIN_RUN]
    return rows, cols


def main() -> None:
    parser = argparse.ArgumentParser(description="表AIとルールベースを、行・列の区切り方で比べる")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("image_table/checkpoints/split/best.pt"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    names = ["ルールベース（空白で区切る）", "表AI"]
    stats = {n: collections.Counter() for n in names}
    by_kind = {n: collections.defaultdict(collections.Counter) for n in names}
    ds = TableDataset(args.dir, args.limit)
    for path in ds.label_paths:
        label = json.loads(Path(path).read_text(encoding="utf-8"))
        name = os.path.splitext(os.path.basename(path))[0]
        with Image.open(os.path.join(args.dir, "images", f"{name}.png")) as image:
            image.load()
        wrap = label["wrap_col"] is not None if "wrap_col" in label else label["wrap"]
        kinds = [label["style"], "折り返しあり" if wrap else "折り返しなし"]
        for n, fn in zip(names, [rule_predict, lambda im: predict(model, im)]):
            started = time.perf_counter()
            rows, cols = fn(image)
            elapsed = time.perf_counter() - started
            if "rows" in label:
                mr, mc = matched(rows, label["rows"]), matched(cols, label["cols"])
                ok = len(rows) == len(label["rows"]) and len(cols) == len(label["cols"]) and mr == len(rows) and mc == len(cols)
                more = {"rows": mr, "rows_total": len(label["rows"]), "cols": mc, "cols_total": len(label["cols"])}
            else:  # 位置の正解が無いデータ（team_tables.py）は、行の数・列の数だけ比べる
                ok, more = len(rows) == len(label["cells"]) and len(cols) == len(label["cells"][0]), {}
            for counter in [stats[n], *(by_kind[n][k] for k in kinds)]:
                counter.update(ok=ok, tables=1, **more)
            stats[n]["seconds"] += elapsed

    print(f"{args.dir}: {len(ds)} 個の表\n")
    print(f"{'':<24} {'行の正解率':>8} {'列の正解率':>8} {'形が合った表':>10} {'ms/表':>7}")
    for n in names:
        s = stats[n]
        rows, cols = ((f"{s['rows'] / s['rows_total']:.3f}", f"{s['cols'] / s['cols_total']:.3f}") if s["rows_total"] else ("-", "-"))
        print(f"{n:<20} {rows:>10} {cols:>10} {s['ok'] / s['tables']:>12.3f} {s['seconds'] / s['tables'] * 1000:>8.1f}")
    print("\n種類ごとの「形が合った表」の割合")
    kinds = sorted(by_kind[names[1]])
    print(f"{'':<14}" + "".join(f"{k:>12}" for k in kinds))
    for n in names:
        print(f"{n[:12]:<14}" + "".join(f"{by_kind[n][k]['ok'] / by_kind[n][k]['tables']:>12.3f}" for k in kinds))
    print("".ljust(14) + "".join(f"{by_kind[names[1]][k]['tables']:>11}個" for k in kinds))


if __name__ == "__main__":
    main()
