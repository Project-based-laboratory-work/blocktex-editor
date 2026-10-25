from evaluation.models import ModelSpec, load_registry
from evaluation.run import evaluate_dataset, tag_keys


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
