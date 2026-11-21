import csv

from operator_ai import analyze
from operator_ai.score_all import COLUMNS

TAGS = {"style": "screenshot", "dpi": "120.0", "scale": "0.5", "jpeg_quality": "", "dark_mode": "False",
        "crop": "False", "viewer": "False", "background": "False", "lang": "ja", "columns": "1"}


def row(case, setting, errors, chars=100, sec=1.0, error="", **tags):
    r = dict.fromkeys(COLUMNS, "")
    r.update(TAGS, case=case, setting=setting, errors=errors, chars=chars, ja_errors=errors, ja_chars=chars,
             en_errors=0, en_chars=0, f1=0.5, sec_per_page=sec, error=error,
             n_pairs=1, n_preds=1, n_gts=2, iou_sum=0.8, order_concordant=3, order_discordant=1,
             text_errors=1, text_chars=10, **tags)
    if error:
        r.update(errors="", chars="")
    return r


def write_scores(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


# A は明るい画像に強く、B は暗い画像に強い。全体では A が単独ベスト
SCORES = [
    row("light1", "t:A", 5), row("light1", "t:B", 30),
    row("light2", "t:A", 10), row("light2", "t:B", 40),
    row("dark1", "t:A", 50, dark_mode="True"), row("dark1", "t:B", 20, dark_mode="True"),
]


def test_single_best_and_oracle(tmp_path):
    write_scores(tmp_path / "scores.csv", SCORES)
    cases, settings, skipped = analyze.load(tmp_path / "scores.csv")
    table = analyze.setting_table(cases, settings)
    assert table[0]["setting"] == "t:A" and table[0]["cer_page"] == round(65 / 300, 4)  # 足してから割る
    assert {r["setting"]: r["wins"] for r in table} == {"t:A": 2, "t:B": 1}
    assert analyze.pooled(cases, lambda c: c.best(settings)) == 35 / 300


def test_team_scores_are_pooled_like_the_team_evaluation(tmp_path):
    write_scores(tmp_path / "scores.csv", SCORES)
    cases, _, _ = analyze.load(tmp_path / "scores.csv")
    s = analyze.team_scores([c.settings["t:A"] for c in cases])  # 3枚分の数を足してから割る
    assert s["precision"] == 1.0 and s["recall"] == 0.5 and s["f1"] == round(2 / 3, 4)
    assert s["miou"] == 0.8 and s["order_acc"] == 0.75 and s["cer"] == 0.1 and s["sec_per_page"] == 1.0


def test_condition_table_finds_the_winner_per_group(tmp_path):
    write_scores(tmp_path / "scores.csv", SCORES)
    cases, settings, _ = analyze.load(tmp_path / "scores.csv")
    rows = {(r["condition"], r["value"]): r for r in analyze.condition_table(cases, settings, "t:A")}
    assert rows[("dark_mode", "True")]["best_setting"] == "t:B"
    assert rows[("dark_mode", "True")]["single_best_cer"] == 0.5
    assert rows[("dark_mode", "False")]["best_setting"] == "t:A"
    assert rows[("effective_dpi", "<70")]["n"] == 3  # 120 × 0.5 = 60


def test_tie_goes_to_the_faster_setting(tmp_path):
    write_scores(tmp_path / "scores.csv", [row("x", "t:slow", 5, sec=9.0), row("x", "t:fast", 5, sec=0.5)])
    cases, settings, _ = analyze.load(tmp_path / "scores.csv")
    assert cases[0].best(settings) == "t:fast"


def test_cases_with_errors_are_left_out(tmp_path):
    write_scores(tmp_path / "scores.csv", SCORES + [row("crash", "t:A", 0), row("crash", "t:B", 0, error="OcrCrashed: x")])
    cases, _, skipped = analyze.load(tmp_path / "scores.csv")
    assert "crash" not in {c.name for c in cases} and len(cases) == 3
    assert [r["case"] for r in skipped] == ["crash"]


def test_main_writes_files_and_keeps_memos(tmp_path):
    write_scores(tmp_path / "scores.csv", SCORES)
    out = analyze.main([str(tmp_path / "scores.csv")])
    assert {p.name for p in out.iterdir()} == {"report.md", "settings.csv", "pairs.csv", "conditions.csv", "worst.csv"}
    assert "単独ベスト（t:A）" in (out / "report.md").read_text(encoding="utf-8")

    # worst.csv に書いた memo は、作り直しても残る
    with open(out / "worst.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows[0]["memo"] = "暗い画面で読めていない"
    analyze.write_csv(out / "worst.csv", rows)
    analyze.main([str(tmp_path / "scores.csv")])
    with open(out / "worst.csv", encoding="utf-8-sig") as f:
        assert next(csv.DictReader(f))["memo"] == "暗い画面で読めていない"
