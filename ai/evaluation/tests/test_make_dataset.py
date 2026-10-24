import json
import random

from common.ir import load_ir
from evaluation.make_dataset import main, make_case


def test_make_case_writes_files(has_tex, tmp_path):
    case_dir = tmp_path / "0000"
    tags = make_case(random.Random(0), "en", case_dir)
    assert (case_dir / "input.pdf").exists()
    assert (case_dir / "source.tex").exists()
    load_ir(str(case_dir / "gt.labeled.json"), check=True)  # 形式チェックで問題があれば ValueError で止まる
    assert tags["lang"] == "en"
    assert tags["pages"] >= 1


def test_main_writes_manifest(has_tex, tmp_path):
    out = tmp_path / "synth_test"
    main(["--out", str(out), "--docs", "3"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "synth_test"
    assert [c["id"] for c in manifest["cases"]] == ["0000", "0001", "0002"]
    assert [c["tags"]["lang"] for c in manifest["cases"]] == ["ja", "en", "ja"]  # 交互に使う
