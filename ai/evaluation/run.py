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
from evaluation.metrics import IOU_THRESHOLD, Counts, confusion_matrix, count_document, label_scores, ratio_or_none, summarize
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
    tag_cases: dict[str, int] = field(default_factory=dict)            # 切り口ごとのケースの数
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
            result.tag_cases[key] = result.tag_cases.get(key, 0) + 1
        result.cases.append(CaseResult(case["id"], case["tags"], seconds, len(gt.pages), summarize(case_counts), error))
    return result


# モデルが出さないもの（models.json の outputs に無いもの）で決まる指標は、0点ではなく「対象外」（None）にする
METRICS_BY_OUTPUT = {
    "label": ["label_acc", "level_acc"],
    "runs": ["deco_precision", "deco_recall", "deco_f1"],
}

# 強み・弱みの文章にする指標。(画面に出す名前, 大きいほど良いか)
FINDING_METRICS = {
    "f1": ("ブロックの検出（F1）", True),
    "miou": ("枠の位置の正確さ（mIoU）", True),
    "label_acc": ("ラベルの正解率", True),
    "level_acc": ("見出しレベルの正解率", True),
    "cer": ("文字の誤り率（CER）", False),
    "deco_f1": ("装飾（太字・斜体・下線）のF1", True),
    "order_acc": ("読み順の正しさ", True),
}
# 表・数式・コードの指標は、1文書に1つくらいしか無く切り口ごとだと数が少なすぎるので、切り口の比較には使わない
# （ブロックの種類ごとの成績 by_label と、全体の値で見る）
FINDING_MAX_SLICES = 12   # 切り口の文章は、差の大きい順にこの数まで
FINDING_LABEL_MIN = 5         # ブロックの種類ごとの文章にするのは、正解がこの数以上ある種類だけ
FINDING_LABEL_STRONG = 0.85   # 種類ごとの F1（ラベルを出さないモデルは見つけた割合）がこれ以上なら強み
FINDING_LABEL_WEAK = 0.5      # これ以下なら弱み

# ブロックの種類（ラベル）の読み方
LABEL_NAMES = {
    "heading": "見出し", "paragraph": "段落", "list": "箇条書き", "math": "数式", "caption": "キャプション",
    "table": "表", "figure": "図", "other": "ページ番号など", "code": "コード", "title": "タイトル",
    "author": "著者", "date": "日付",
}
FINDING_MIN_CASES = 5     # ケースがこれより少ない切り口は、偶然の差かもしれないので文章にしない
FINDING_MIN_DIFF = 0.05   # 全体との差がこれより小さいときは、強み・弱みとしない

# 切り口（tag_keys の名前）を、文章で読める言葉にする。ここに無いものは "lang=en" のまま出す
TAG_NAMES = {
    "lang=ja": "日本語の文書",
    "lang=en": "英語の文書",
    "has_list=True": "箇条書きがある文書",
    "has_list=False": "箇条書きが無い文書",
    "has_title=True": "タイトルがある文書",
    "has_title=False": "タイトルが無い文書",
    "has_decoration=True": "文中に装飾がある文書",
    "has_decoration=False": "文中に装飾が無い文書",
    "pages=1": "1ページの文書",
    "pages=2": "2ページの文書",
    "pages=3": "3ページの文書",
    "has_table=True": "表がある文書",
    "has_table=False": "表が無い文書",
    "has_figure=True": "図がある文書",
    "has_figure=False": "図が無い文書",
    "has_math=True": "数式がある文書",
    "has_math=False": "数式が無い文書",
    "has_code=True": "コードがある文書",
    "has_code=False": "コードが無い文書",
    "has_page_number=True": "ページ番号がある文書",
    "has_page_number=False": "ページ番号が無い文書",
    "columns=1": "1段組の文書",
    "columns=2": "2段組の文書",
    "has_header=True": "ヘッダーがある文書",
    "has_header=False": "ヘッダーが無い文書",
    "label_source=manual": "人が付けた正解の文書",
    "label_source=prelabel": "正解が下書きのままの文書",
    "label_source=auto": "正解を自動で作った文書",
}


def mask_unsupported(scores: dict, outputs: list[str]) -> dict:
    """モデルが出さないもので決まる指標を None（対象外）にした、新しい辞書を返す"""
    masked = dict(scores)
    for output, metrics in METRICS_BY_OUTPUT.items():
        if output not in outputs:
            for metric in metrics:
                masked[metric] = None
    return masked


def find_strengths(summary: dict, by_tag: dict[str, dict], tag_cases: dict[str, int]) -> list[dict]:
    """切り口ごとの成績を全体と比べて、目立って良い（強み）・悪い（弱み）ものを、差の大きい順に返す"""
    findings = []
    for key, scores in by_tag.items():
        n = tag_cases.get(key, 0)
        if n < FINDING_MIN_CASES:
            continue
        for metric, (name, higher_is_better) in FINDING_METRICS.items():
            overall, value = summary.get(metric), scores.get(metric)
            if overall is None or value is None:
                continue
            gain = value - overall if higher_is_better else overall - value  # プラスなら全体より良い
            if abs(gain) < FINDING_MIN_DIFF:
                continue
            findings.append({
                "kind": "strength" if gain > 0 else "weakness",
                "metric": metric,
                "slice": key,
                "value": value,
                "overall": overall,
                "n": n,
                "text": f"{TAG_NAMES.get(key, key)}（{n}件）では、{name}が {value:.3f} で、全体の {overall:.3f} より{'良い' if gain > 0 else '悪い'}",
            })
    findings.sort(key=lambda f: abs(f["value"] - f["overall"]), reverse=True)
    return findings[:FINDING_MAX_SLICES]


def find_label_strengths(by_label: dict[str, dict]) -> list[dict]:
    """ブロックの種類ごとに、よく当てられる（強み）・苦手（弱み）なものを文章にする。
    ラベルを出すモデルは F1、出さないモデルは「見つけた割合」で見る"""
    findings = []
    for label, scores in by_label.items():
        n = scores["n_gt"]
        value = scores["f1"] if scores["f1"] is not None else scores["found_rate"]
        measure = "分類のF1" if scores["f1"] is not None else "見つけた割合"
        if n < FINDING_LABEL_MIN or value is None:
            continue
        if value >= FINDING_LABEL_STRONG:
            kind = "strength"
        elif value <= FINDING_LABEL_WEAK:
            kind = "weakness"
        else:
            continue
        name = LABEL_NAMES.get(label, label)
        findings.append({
            "kind": kind,
            "metric": "label_f1" if scores["f1"] is not None else "found_rate",
            "slice": f"label={label}",
            "value": value,
            "overall": None,
            "n": n,
            "text": f"{name}（正解 {n}個）の{measure}が {value:.3f} で、{'よく当てられている' if kind == 'strength' else '苦手'}",
        })
    findings.sort(key=lambda f: f["value"], reverse=True)
    return findings


def build_report(spec: ModelSpec, dataset_dir: Path, result: DatasetResult, run_id: str) -> dict:
    """サイトが読む成績表（JSONにできる辞書）を作る"""
    summary = mask_unsupported(summarize(result.total), spec.outputs)
    total_seconds = sum(c.seconds for c in result.cases)
    total_pages = sum(c.pages for c in result.cases)
    summary["sec_per_page"] = ratio_or_none(total_seconds, total_pages)
    by_tag = {key: mask_unsupported(summarize(counts), spec.outputs) for key, counts in result.by_tag.items()}
    by_label = label_scores(result.total)
    has_labels = "label" in spec.outputs
    return {
        "run_id": run_id,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": {"id": spec.id, "name": spec.name, "input": spec.input, "outputs": spec.outputs, "params": spec.params},
        "dataset": dataset_dir.name,
        "iou_threshold": IOU_THRESHOLD,
        "summary": summary,
        "by_tag": by_tag,
        "tag_cases": result.tag_cases,
        "by_label": by_label,
        "findings": find_strengths(summary, by_tag, result.tag_cases) + find_label_strengths(by_label),
        "confusion": {"labels": LABELS, "matrix": confusion_matrix(result.total.label_pairs, LABELS)} if has_labels else None,
        "errors": sum(1 for c in result.cases if c.error is not None),
        "cases": [
            asdict(c) | {"scores": mask_unsupported(c.scores, spec.outputs), "pred_path": f"preds/{c.id}.labeled.json"}
            for c in result.cases
        ],
    }


def main(argv: list[str] | None = None) -> Path:
    """評価を実行して結果を保存し、結果のフォルダを返す"""
    parser = argparse.ArgumentParser(description="登録したモデルを、データセットで採点する")
    parser.add_argument("--model", required=True, help="models.json の id（例: rule_based）")
    parser.add_argument("--dataset", required=True, help="データセットのフォルダ（例: evaluation/data/synth_v1）")
    parser.add_argument("--out", default=str(RESULTS_DIR), help="結果を置くフォルダ")
    parser.add_argument("--quiet", action="store_true", help="成績表を出さず、run_id だけを出す（サーバーから呼ぶとき）")
    args = parser.parse_args(argv)

    spec = load_registry()[args.model]
    dataset_dir = Path(args.dataset)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{spec.id}_{dataset_dir.name}"
    run_dir = Path(args.out) / run_id

    result = evaluate_dataset(spec, dataset_dir, pred_dir=run_dir / "preds")
    report = build_report(spec, dataset_dir, result, run_id)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quiet:
        print(run_id)
        return run_dir
    print(f"{run_dir / 'report.json'} に保存した")
    for name, value in report["summary"].items():
        print(f"  {name:14} {'—' if value is None else f'{value:.3f}'}")  # None は「対象外」
    for finding in report["findings"]:
        print(f"  {'強み' if finding['kind'] == 'strength' else '弱み'}: {finding['text']}")
    return run_dir


if __name__ == "__main__":
    main()
