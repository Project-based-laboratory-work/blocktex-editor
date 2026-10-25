"""
登録したモデルを、データセットの全ケースで動かして採点する。
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from common.ir import IRDocument, load_ir, save_ir
from evaluation.metrics import Counts, count_document, summarize
from evaluation.models import ModelSpec, predict


def tag_keys(tags: dict) -> list[str]:
    """ケースの tags を、集計の切り口の名前にする。{"lang": "en", "pages": 1} → ["lang=en", "pages=1"]"""
    return [f"{k}={v}" for k, v in tags.items()]


@dataclass
class CaseResult:
    """1ケース分の結果（サイトの「ケース詳細」画面に出す）"""
    id: str
    tags: dict
    seconds: float              # モデルが読むのにかかった時間
    pages: int                  # 正解のページ数（秒/ページを出すのに使う）
    scores: dict                # このケースだけの summarize の結果
    error: str | None = None    # モデルが失敗したときのエラー


@dataclass
class DatasetResult:
    total: Counts = field(default_factory=Counts)                      # 全ケースを足したもの
    by_tag: dict[str, Counts] = field(default_factory=dict)            # "lang=en" などの切り口ごと
    cases: list[CaseResult] = field(default_factory=list)


def evaluate_dataset(spec: ModelSpec, dataset_dir: Path, pred_dir: Path | None = None) -> DatasetResult:
    """データセットの全ケースでモデルを動かして採点する。pred_dir を渡すと、モデルの出力をそこに保存する。
    モデルがエラーになったケースは、何も見つけられなかった（空の予測）として採点する。"""
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    result = DatasetResult()
    for case in manifest["cases"]:
        case_dir = dataset_dir / "cases" / case["id"]
        gt = load_ir(str(case_dir / "gt.labeled.json"))

        error = None
        started = time.perf_counter()
        try:
            pred = predict(spec, str(case_dir / "input.pdf"))
        except Exception as e:  # モデルが何のエラーを出すか分からないので、全部受け止める
            error = f"{type(e).__name__}: {e}"
            pred = IRDocument("input.pdf", spec.id, [])
        seconds = time.perf_counter() - started

        if pred_dir is not None:
            pred_dir.mkdir(parents=True, exist_ok=True)
            save_ir(pred, str(pred_dir / f"{case['id']}.labeled.json"))

        case_counts = Counts()
        count_document(pred, gt, case_counts)
        count_document(pred, gt, result.total)
        for key in tag_keys(case["tags"]):
            count_document(pred, gt, result.by_tag.setdefault(key, Counts()))
        result.cases.append(CaseResult(case["id"], case["tags"], seconds, len(gt.pages), summarize(case_counts), error))
    return result
