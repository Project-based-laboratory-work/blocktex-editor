"""
オペレータ（画像の特徴量 → 設定を選ぶ関数）を、採点表だけで評価する。OCR は読み直さない。
評価項目は CER（cer_page）と f1 の2つ（operator_ai/README.md の「評価」）。
参考に、チームの評価と同じ precision・recall・miou・order_acc・cer も並べる（計算は evaluation/metrics.py の summarize）。

採点表（scores.csv）には全部の画像 × 全部の設定の点数があるので、
「この画像ではこの設定を選んだ」が決まれば、その点数を引いてくるだけで評価できる（1秒で終わる）。
ルール版（進め方4）を書き直しながら、何度でも試せる。

使い方（ai/ で）:
    python -m operator_ai.evaluate operator_ai.rule_example:choose
    python -m operator_ai.evaluate operator_ai.rules:choose operator_ai.rule_example:choose   # いくつも並べて比べる
      --scores  operator_ai/results/val/scores.csv    （既定）
      --features operator_ai/results/val/features.csv （既定）

オペレータは「特徴量の dict を受け取り、設定の名前（"tesseract:up2" など）を返す関数」:
    def choose(f: dict) -> str:
        return "easyocr:up2_lowtext" if f["char_height"] < 10 else "tesseract:up2"
f の中身は operator_ai/features.py の FEATURES（画像から測った値）だけ。正解の params（dark_mode など）は渡さない。
"""

import argparse
import csv
import importlib
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from operator_ai.analyze import Case, condition_value, load, pooled, single_best, team_scores

RESULTS = Path("operator_ai/results/val")  # ai/ で実行する
BREAKDOWN = ["dark_mode", "viewer", "lang", "columns", "effective_dpi"]
WORST_N = 15

Operator = Callable[[dict], str]


def parse_value(value: str):
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def load_features(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig") as f:
        return {r["case"]: {k: parse_value(v) for k, v in r.items() if k != "case"} for r in csv.DictReader(f)}


def load_operator(spec: str) -> tuple[str, Operator]:
    """"operator_ai.rules:choose" → ("rules:choose", 関数)"""
    module, _, name = spec.partition(":")
    return spec.rsplit(".", 1)[-1], getattr(importlib.import_module(module), name or "choose")


def choices(op: Operator, cases: list[Case], features: dict[str, dict], settings: list[str]) -> dict[str, str]:
    picked = {}
    for c in cases:
        s = op(dict(features[c.name]))
        if s not in settings:
            raise SystemExit(f"エラー: {c.name} で {s!r} が選ばれたが、採点表に無い設定（使えるのは {settings}）")
        picked[c.name] = s
    return picked


def summary(name: str, cases: list[Case], pick: dict[str, str], base: float, oracle: float) -> dict:
    cer = pooled(cases, lambda c: pick[c.name])
    team = team_scores([c.settings[pick[c.name]] for c in cases])
    gap = base - oracle
    return {
        "方法": name,
        "cer_page": round(cer, 4),
        "f1": "-" if team["f1"] is None else team["f1"],
        "差の埋まり具合": f"{(base - cer) / gap:.0%}" if gap > 0 else "-",
        **{f"参考 {k}": "-" if v is None else v for k, v in team.items() if k != "f1"},
        "選んだ設定": "、".join(f"{s} {n}" for s, n in Counter(pick.values()).most_common()),
    }


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def evaluate(specs: list[str], scores: Path, features_path: Path) -> str:
    cases, settings, skipped = load(scores)
    features = load_features(features_path)
    missing = [c.name for c in cases if c.name not in features]
    if missing:
        raise SystemExit(f"エラー: {features_path} に無い画像があります（{missing[:3]} など {len(missing)} 枚）")

    best = single_best(cases, settings)
    picks = {f"単独ベスト（{best}）": {c.name: best for c in cases}}
    for spec in specs:
        name, op = load_operator(spec)
        picks[name] = choices(op, cases, features, settings)
    picks["オラクル（上限）"] = {c.name: c.best(settings) for c in cases}

    base = pooled(cases, lambda c: best)
    oracle = pooled(cases, lambda c: c.best(settings))
    out = [
        f"# オペレータの評価（画像 {len(cases)} 枚、`{scores}`）",
        "",
        md_table([summary(n, cases, p, base, oracle) for n, p in picks.items()]),
        "",
        "差の埋まり具合 ＝（単独ベスト − オペレータ）÷（単独ベスト − オラクル）。0% なら単独ベストと同じ、100% なら上限。",
        "",
        "## 条件ごとの cer_page（分析用。条件は正解の params から）",
        "",
    ]
    rows = []
    for key in BREAKDOWN:
        groups: dict[str, list[Case]] = defaultdict(list)
        for c in cases:
            groups[condition_value(c, key)].append(c)
        for value, group in sorted(groups.items()):
            row = {"条件": key, "値": value, "枚数": len(group)}
            row.update({n: round(pooled(group, lambda c, p=p: p[c.name]), 4) for n, p in picks.items()})
            rows.append(row)
    out += [md_table(rows), ""]

    for spec in specs:
        name = spec.rsplit(".", 1)[-1]
        pick = picks[name]
        lost = sorted(cases, key=lambda c: c.chars * (c.cer(c.best(settings)) - c.cer(pick[c.name])))[:WORST_N]
        out += [
            f"## {name}：オラクルとの差が大きい画像（上位 {WORST_N}、誤りの文字数の差の順）",
            "",
            md_table([{
                "case": c.name,
                "選んだ設定": pick[c.name],
                "cer_page": round(c.cer(pick[c.name]), 4),
                "一番良い設定": c.best(settings),
                "その cer_page": round(c.cer(c.best(settings)), 4),
                **{k: features[c.name][k] for k in ("page_dark", "band_ratio", "char_height", "center_gap")},
            } for c in lost]),
            "",
        ]
    if skipped:
        out += [f"エラーがあった {len({r['case'] for r in skipped})} 枚は外した（`analyze.py` と同じ）。", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description="オペレータを採点表だけで評価する（OCR は読み直さない）")
    parser.add_argument("operators", nargs="*", help="<モジュール>:<関数>（例 operator_ai.rule_example:choose）")
    parser.add_argument("--scores", type=Path, default=RESULTS / "scores.csv")
    parser.add_argument("--features", type=Path, default=RESULTS / "features.csv")
    parser.add_argument("--out", type=Path, help="結果を Markdown で書き出す先（省略すると画面に出すだけ）")
    args = parser.parse_args(argv)
    md = evaluate(args.operators, args.scores, args.features)
    print(md)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    main()
