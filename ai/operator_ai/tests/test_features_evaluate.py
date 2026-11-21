import csv
import sys
import types

import cv2
import numpy as np

from operator_ai import evaluate
from operator_ai.features import FEATURES, measure
from operator_ai.tests.test_analyze import SCORES, write_scores


def page_image(path, dark=False, band=0, columns=1, char=12):
    """白い（ダークモードなら黒い）ページに、文字の代わりの小さな四角を並べた画像"""
    bg, fg = (30, 220) if dark else (255, 0)
    w, h = 600, 800
    img = np.full((h, w), bg, np.uint8)
    col_ranges = [(40, 560)] if columns == 1 else [(40, 280), (320, 560)]
    for y in range(60, 740, char * 2):
        for x0, x1 in col_ranges:
            for x in range(x0, x1 - char, char + 3):
                cv2.rectangle(img, (x, y), (x + char - 3, y + char - 1), fg, -1)
    if band:
        img = cv2.copyMakeBorder(img, 0, 0, band, band, cv2.BORDER_CONSTANT, value=60)
    cv2.imwrite(str(path), img)
    return str(path)


def test_measure_light_page(tmp_path):
    f = measure(page_image(tmp_path / "a.png"))
    assert list(f) == FEATURES
    assert f["page_dark"] is False and f["is_jpeg"] is False
    assert f["band_ratio"] == 0 and 10 <= f["char_height"] <= 13


def test_measure_dark_page_with_viewer_band(tmp_path):
    f = measure(page_image(tmp_path / "a.jpg", dark=True, band=150))
    assert f["page_dark"] is True and f["is_jpeg"] is True
    assert 0.3 < f["band_ratio"] < 0.4  # 左右に 150px ずつの帯（幅 900 のうち 300）


def test_center_gap_separates_columns(tmp_path):
    one = measure(page_image(tmp_path / "1.png"))["center_gap"]
    two = measure(page_image(tmp_path / "2.png", columns=2))["center_gap"]
    assert one < 0.2 < 0.8 < two


def write_features(path, values):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["case", *FEATURES])
        w.writeheader()
        for case, dark in values.items():
            w.writerow({"case": case, **dict.fromkeys(FEATURES, 0), "page_dark": dark})


def test_evaluate_operator_from_features(tmp_path, monkeypatch):
    write_scores(tmp_path / "scores.csv", SCORES)
    write_features(tmp_path / "features.csv", {"light1": False, "light2": False, "dark1": True})
    rules = types.ModuleType("my_rules")
    rules.choose = lambda f: "t:B" if f["page_dark"] else "t:A"  # page_dark は bool で渡る
    monkeypatch.setitem(sys.modules, "my_rules", rules)

    md = evaluate.main(["my_rules:choose", "--scores", str(tmp_path / "scores.csv"),
                        "--features", str(tmp_path / "features.csv")])
    row = next(line for line in md.splitlines() if line.startswith("| my_rules:choose"))
    assert "| 0.1167 | 0.6667 | 100% | 1.0 | 0.5 | 0.8 | 0.75 | 0.1 |" in row  # 暗い画像だけ B にすると、オラクルと同じ
