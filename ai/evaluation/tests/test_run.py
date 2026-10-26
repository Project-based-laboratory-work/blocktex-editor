import json

from common.ir import LABELS
from evaluation.models import ModelSpec, load_registry
from evaluation.run import build_report, evaluate_dataset, main, tag_keys


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
