import json

from common.ir import LABELS
from evaluation.models import ModelSpec, load_registry
from evaluation.run import build_report, evaluate_dataset, find_label_strengths, find_strengths, main, mask_unsupported, tag_keys


def test_tag_keys():
    assert tag_keys({"lang": "en", "pages": 1}) == ["lang=en", "pages=1"]


def test_evaluate_dataset_with_rule_based(small_dataset, tmp_path):
    result = evaluate_dataset(load_registry()["rule_based"], small_dataset, pred_dir=tmp_path / "preds")
    assert [c.id for c in result.cases] == ["0000", "0001"]
    assert all(c.error is None and c.seconds > 0 for c in result.cases)
    # 日本語と英語の切り口を足すと、全体と同じになる（どのケースもどちらか1つに入るため）
    assert result.by_tag["lang=ja"].n_gts + result.by_tag["lang=en"].n_gts == result.total.n_gts
    assert sorted(p.name for p in (tmp_path / "preds").iterdir()) == ["0000.labeled.json", "0001.labeled.json"]


def test_broken_model_counts_as_missed(small_dataset):
    # json.loads に PDF のパスを渡すと必ずエラーになるので、「失敗するモデル」の代わりに使う
    broken = ModelSpec(id="broken", name="broken", entry="json:loads", input="pdf", outputs=["bbox"])
    result = evaluate_dataset(broken, small_dataset)
    assert result.total.n_preds == 0                  # 何も出していない
    assert result.total.n_gts > 0                     # 正解は全部数えられている → 全部見逃し
    assert result.cases[0].error.startswith("JSONDecodeError")
    assert result.cases[0].scores["recall"] == 0.0

def test_build_report_is_json_ready(small_dataset):
    spec = load_registry()["rule_based"]
    result = evaluate_dataset(spec, small_dataset)
    report = build_report(spec, small_dataset, result, run_id="test-run")
    json.dumps(report)  # JSON にできない値（dataclass など）が混ざっていたら、ここで TypeError になる
    assert report["model"]["id"] == "rule_based"
    assert report["summary"]["sec_per_page"] > 0
    assert {"lang=ja", "lang=en"} <= set(report["by_tag"])  # 左が右に全部含まれている
    matrix = report["confusion"]["matrix"]
    assert len(matrix) == len(LABELS)                                  # ラベルの数 × ラベルの数の表
    assert sum(sum(row) for row in matrix) == len(result.total.label_pairs)  # 表の合計 = ラベルを比べた組の数
    assert report["cases"][0]["pred_path"] == "preds/0000.labeled.json"


def test_main_writes_report_and_preds(small_dataset, tmp_path):
    run_dir = main(["--model", "rule_based", "--dataset", str(small_dataset), "--out", str(tmp_path)])
    assert run_dir.name.endswith("_rule_based_small")  # 日時_モデル_データセット
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["run_id"] == run_dir.name
    assert len(list((run_dir / "preds").iterdir())) == 2


def test_mask_unsupported_marks_missing_outputs_as_none():
    scores = {"f1": 0.5, "label_acc": 0.0, "level_acc": 0.0, "deco_f1": 0.0}
    masked = mask_unsupported(scores, outputs=["bbox", "text"])  # ラベルも装飾も出さない（OCRだけのモデル）
    assert masked["f1"] == 0.5
    assert masked["label_acc"] is None and masked["level_acc"] is None and masked["deco_f1"] is None
    assert scores["label_acc"] == 0.0  # 元の辞書は変えない


def test_find_strengths_needs_enough_cases_and_a_clear_difference():
    summary = {"f1": 0.5, "cer": 0.10}
    by_tag = {
        "lang=en": {"f1": 0.75, "cer": 0.02},  # F1 が 0.25 高く、CER が 0.08 低い → どちらも強み
        "lang=ja": {"f1": 0.52, "cer": 0.30},  # F1 の差は小さい。CER が 0.2 高い → 弱み
        "pages=2": {"f1": 0.0, "cer": 0.9},    # 差は大きいが2件しかないので、文章にしない
    }
    findings = find_strengths(summary, by_tag, {"lang=en": 10, "lang=ja": 10, "pages=2": 2})
    assert [(f["slice"], f["metric"], f["kind"]) for f in findings] == [
        ("lang=en", "f1", "strength"),
        ("lang=ja", "cer", "weakness"),
        ("lang=en", "cer", "strength"),
    ]
    assert findings[0]["text"].startswith("英語の文書（10件）では")


def test_main_quiet_prints_only_run_id(small_dataset, tmp_path, capsys):
    run_dir = main(["--model", "rule_based", "--dataset", str(small_dataset), "--out", str(tmp_path), "--quiet"])
    assert capsys.readouterr().out.strip() == run_dir.name  # サーバーが run_id を受け取るため


def test_find_label_strengths():
    by_label = {
        "table": {"n_gt": 10, "f1": 0.95, "found_rate": 1.0},
        "list": {"n_gt": 20, "f1": 0.1, "found_rate": 0.2},
        "code": {"n_gt": 2, "f1": 0.0, "found_rate": 0.0},        # 2個しかないので文章にしない
        "heading": {"n_gt": 30, "f1": 0.7, "found_rate": 1.0},    # 強みでも弱みでもない
        "other": {"n_gt": 8, "f1": None, "found_rate": 0.25},     # ラベルを出さないモデルは見つけた割合で見る
    }
    findings = find_label_strengths(by_label)
    assert [(f["slice"], f["kind"], f["metric"]) for f in findings] == [
        ("label=table", "strength", "label_f1"),
        ("label=other", "weakness", "found_rate"),
        ("label=list", "weakness", "label_f1"),
    ]
    assert findings[0]["text"].startswith("表（正解 10個）")


def test_report_has_per_label_scores(small_dataset):
    spec = load_registry()["rule_based"]
    report = build_report(spec, small_dataset, evaluate_dataset(spec, small_dataset), run_id="test-run")
    assert report["by_label"]["heading"]["n_gt"] > 0
    assert set(report["by_label"]["heading"]) == {"n_gt", "n_pred", "found_rate", "recall", "precision", "f1"}
