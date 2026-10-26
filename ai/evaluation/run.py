"""
登録したモデルを、データセットの全ケースで動かして採点する。
"""
import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from common.ir import LABELS, IRDocument, load_ir, save_ir
from evaluation.metrics import IOU_THRESHOLD, Counts, confusion_matrix, count_document, ratio_or_none, summarize
from evaluation.models import ModelSpec, load_registry, predict

RESULTS_DIR = Path(__file__).parent / "data" / "results"  # data/ は .gitignore 済み


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

def build_report(spec: ModelSpec, dataset_dir: Path, result: DatasetResult, run_id: str) -> dict:
    """サイトが読む成績表（JSONにできる辞書）を作る"""
    summary = summarize(result.total)
    total_seconds = sum(c.seconds for c in result.cases)
    total_pages = sum(c.pages for c in result.cases)
    summary["sec_per_page"] = ratio_or_none(total_seconds, total_pages)
    return {
        "run_id": run_id,
        "model": {"id": spec.id, "name": spec.name, "outputs": spec.outputs, "params": spec.params},
        "dataset": dataset_dir.name,
        "iou_threshold": IOU_THRESHOLD,
        "summary": summary,
        "by_tag": {key: summarize(counts) for key, counts in result.by_tag.items()},
        "confusion": {"labels": LABELS, "matrix": confusion_matrix(result.total.label_pairs, LABELS)},
        "cases": [asdict(c) | {"pred_path": f"preds/{c.id}.labeled.json"} for c in result.cases],
    }


def main(argv: list[str] | None = None) -> Path:
    """評価を実行して結果を保存し、結果のフォルダを返す"""
    parser = argparse.ArgumentParser(description="登録したモデルを、データセットで採点する")
    parser.add_argument("--model", required=True, help="models.json の id（例: rule_based）")
    parser.add_argument("--dataset", required=True, help="データセットのフォルダ（例: evaluation/data/synth_v1）")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="結果を置くフォルダ")
    args = parser.parse_args(argv)

    spec = load_registry()[args.model]
    dataset_dir = Path(args.dataset)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{spec.id}_{dataset_dir.name}"
    run_dir = Path(args.out) / run_id

    result = evaluate_dataset(spec, dataset_dir, pred_dir=run_dir / "preds")
    report = build_report(spec, dataset_dir, result, run_id)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{run_dir / 'report.json'} に保存した")
    for name, value in report["summary"].items():
        print(f"  {name:12} {'—' if value is None else f'{value:.3f}'}")  # None は「対象外」
    return run_dir


if __name__ == "__main__":
    main()
