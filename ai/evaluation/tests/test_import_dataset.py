import json
import shutil

from common.ir import load_ir
from evaluation.import_dataset import main


def test_import_with_ground_truth_and_with_draft(small_dataset, tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    # 1つ目: 正解付き（合成データセットの正解を、人が付けた正解の代わりに使う）
    shutil.copy(small_dataset / "cases" / "0000" / "input.pdf", pdfs / "a.pdf")
    shutil.copy(small_dataset / "cases" / "0000" / "gt.labeled.json", pdfs / "a.labeled.json")
    # 2つ目: 正解なし → rule_based の出力を下書きにする
    shutil.copy(small_dataset / "cases" / "0001" / "input.pdf", pdfs / "b.pdf")

    out = tmp_path / "real"
    main(["--out", str(out), "--draft", "rule_based", str(pdfs / "a.pdf"), str(pdfs / "b.pdf")])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [c["source"] for c in manifest["cases"]] == ["a.pdf", "b.pdf"]
    assert manifest["cases"][0]["tags"]["label_source"] == "auto"      # 合成データセットの正解は auto
    assert manifest["cases"][1]["tags"]["label_source"] == "prelabel"  # 下書き。人が直す必要がある
    assert manifest["cases"][0]["tags"]["lang"] == "ja" and manifest["cases"][1]["tags"]["lang"] == "en"
    draft = load_ir(str(out / "cases" / "0001" / "gt.labeled.json"), check=True)
    assert draft.structurer == "draft:rule_based"
    assert (out / "cases" / "0001" / "page-1.png").exists()


def test_pdf_without_ground_truth_is_skipped(small_dataset, tmp_path, capsys):
    shutil.copy(small_dataset / "cases" / "0000" / "input.pdf", tmp_path / "a.pdf")
    out = tmp_path / "real"
    main(["--out", str(out), str(tmp_path / "a.pdf")])  # --draft なし
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["cases"] == []
    assert "飛ばす" in capsys.readouterr().out
