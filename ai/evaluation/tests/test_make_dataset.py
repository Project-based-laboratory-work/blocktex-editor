import json
import random

from common.ir import load_ir
import pytest

from evaluation.make_dataset import main, make_case, parse_features


def test_make_case_writes_files(has_tex, tmp_path):
    case_dir = tmp_path / "0000"
    tags = make_case(random.Random(0), "en", case_dir)
    assert (case_dir / "input.pdf").exists()
    assert (case_dir / "source.tex").exists()
    assert (case_dir / "page-1.png").exists()  # サイトで枠を重ねる下絵
    load_ir(str(case_dir / "gt.labeled.json"), check=True)  # 形式チェックで問題があれば ValueError で止まる
    assert tags["lang"] == "en"
    assert tags["pages"] >= 1
    assert set(tags) == {"lang", "pages", "has_list", "has_title", "has_decoration"}


def test_main_writes_manifest(has_tex, tmp_path):
    out = tmp_path / "synth_test"
    main(["--out", str(out), "--docs", "3"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "synth_test"
    assert [c["id"] for c in manifest["cases"]] == ["0000", "0001", "0002"]
    assert [c["tags"]["lang"] for c in manifest["cases"]] == ["ja", "en", "ja"]  # 交互に使う


def test_parse_features():
    assert parse_features(None) is None
    assert parse_features("table,math") == {"table", "math"}
    assert "two_column" in parse_features("all")
    with pytest.raises(SystemExit):
        parse_features("tabel")


def test_features_become_tags(has_tex, tmp_path):
    out = tmp_path / "synth_table"
    main(["--out", str(out), "--docs", "2", "--features", "table,two_column", "--rate", "1"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"]["features"] == ["table", "two_column"]
    for case in manifest["cases"]:
        assert case["tags"]["has_table"] is True and case["tags"]["columns"] == 2
        gt = load_ir(str(out / "cases" / case["id"] / "gt.labeled.json"), check=True)
        assert any(el.label == "table" for page in gt.pages for el in page.elements)
