"""
オペレータAIの進め方2・3: 採点表（score_all.py の scores.csv）から、
  2. 単独ベスト（全画像を同じ設定で読む）とオラクル（画像ごとに一番良い設定を選べた場合＝上限）
  3. 条件（ダークモード・dpi・言語など）ごとに、どの設定が勝つか / CER が悪い画像の一覧
を出す。

使い方（ai/ で）:
    python -m operator_ai.analyze operator_ai/data/val/scores.csv
      → operator_ai/data/val/analysis/report.md       まとめ（共有用）
        operator_ai/data/val/analysis/settings.csv    設定ごとの点数（進め方2）
        operator_ai/data/val/analysis/pairs.csv       2つの設定から選ぶだけのときの上限
        operator_ai/data/val/analysis/conditions.csv  条件ごと × 設定ごとの CER（進め方3）
        operator_ai/data/val/analysis/worst.csv       CER が悪い画像の一覧（memo 列に原因を書き込む用）

CER はすべて「誤りの数を足してから、文字数の合計で割る」（画像ごとの CER の平均にしない）。
どれかの設定でエラーになった画像は、比べる条件を揃えるため全部の集計から外す（report.md に一覧を出す）。
"""

import argparse
import csv
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from operator_ai.score_all import TAG_KEYS

# 「大きく崩れた」とみなす cer_page。これを超える画像の割合を、設定の危うさとして出す
BROKEN = 0.5
WORST_N = 30
# CER が悪い画像の一覧に入れる最低の文字数。ページ番号だけのページなどは、1文字違うだけで CER 1.0 になるので外す
WORST_MIN_CHARS = 50


@dataclass
class Case:
    """画像1枚。settings[setting] = その設定で読んだときの行（scores.csv の1行）"""

    name: str
    tags: dict
    settings: dict[str, dict] = field(default_factory=dict)

    @property
    def chars(self) -> int:
        return int(next(iter(self.settings.values()))["chars"])

    def errors(self, setting: str) -> int:
        return int(self.settings[setting]["errors"])

    def cer(self, setting: str) -> float:
        return self.errors(setting) / self.chars

    def best(self, candidates: list[str]) -> str:
        """一番 CER が小さい設定。同点なら速いほう"""
        return min(candidates, key=lambda s: (self.errors(s), seconds(self.settings[s])))


def seconds(row: dict) -> float:
    return float(row["sec_per_page"]) if row.get("sec_per_page") else float("inf")


def load(csv_path: Path) -> tuple[list[Case], list[str], list[dict]]:
    """scores.csv を読み、全部の設定がそろった画像だけを返す。外した行（エラー）も返す"""
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    cases: dict[str, Case] = {}
    for r in rows:
        case = cases.setdefault(r["case"], Case(r["case"], {k: r[k] for k in TAG_KEYS}))
        case.settings[r["setting"]] = r
    skipped = [r for r in rows if r["error"]]
    bad = {r["case"] for r in skipped}
    complete = [
        c for c in cases.values()
        if c.name not in bad and set(c.settings) == set(settings) and c.chars > 0
    ]
    return complete, settings, skipped


def pooled(cases: list[Case], pick) -> float | None:
    """画像ごとに pick(case) の設定を選んだときの、まとめた CER"""
    chars = sum(c.chars for c in cases)
    return sum(c.errors(pick(c)) for c in cases) / chars if chars else None


# ---------------------------------------------------------------------------
# 進め方2: 単独ベストとオラクル
# ---------------------------------------------------------------------------

def setting_table(cases: list[Case], settings: list[str]) -> list[dict]:
    wins = Counter(c.best(settings) for c in cases)
    table = []
    for s in settings:
        cers = [c.cer(s) for c in cases]
        f1s = [float(c.settings[s]["f1"]) for c in cases if c.settings[s]["f1"]]
        table.append({
            "setting": s,
            "cer_page": round(pooled(cases, lambda c: s), 4),
            "cer_page_ja": rate(cases, s, "ja"),
            "cer_page_en": rate(cases, s, "en"),
            "median": round(statistics.median(cers), 4),
            "broken": round(sum(x > BROKEN for x in cers) / len(cers), 4),
            "f1": round(statistics.mean(f1s), 4) if f1s else None,
            "sec_per_page": round(statistics.median(seconds(c.settings[s]) for c in cases), 2),
            "wins": wins[s],
        })
    return sorted(table, key=lambda r: r["cer_page"])


def rate(cases: list[Case], setting: str, script: str) -> float | None:
    errors = sum(int(c.settings[setting][f"{script}_errors"]) for c in cases)
    chars = sum(int(c.settings[setting][f"{script}_chars"]) for c in cases)
    return round(errors / chars, 4) if chars else None


def single_best(cases: list[Case], settings: list[str]) -> str:
    return min(settings, key=lambda s: pooled(cases, lambda c: s))


def pair_table(cases: list[Case], settings: list[str]) -> list[dict]:
    """2つの設定のどちらかを画像ごとに選べるときの上限。ルール版で「A か B か」を決めるときの目安"""
    table = [
        {"pair": f"{a} / {b}", "cer_page": round(pooled(cases, lambda c, p=[a, b]: c.best(p)), 4)}
        for a, b in itertools.combinations(settings, 2)
    ]
    return sorted(table, key=lambda r: r["cer_page"])


# ---------------------------------------------------------------------------
# 進め方3: 条件ごとの勝ち負け
# ---------------------------------------------------------------------------

def bucket(key: str, value: str) -> str:
    """数の params をいくつかの区間にまとめる。それ以外はそのまま"""
    if key == "dpi":
        x = float(value)
        return "<100" if x < 100 else "100-140" if x < 140 else ">=140"
    if key == "scale":
        x = float(value)
        return "1（縮小なし）" if x >= 0.999 else "0.75-1" if x >= 0.75 else "<0.75"
    if key == "jpeg_quality":
        return "なし" if not value else ">=80" if int(value) >= 80 else "<80"
    return value


def effective_dpi(case: Case) -> str:
    """画面上の文字の細かさ＝ dpi × 縮小率。up2（拡大）が効くかの目安"""
    x = float(case.tags["dpi"]) * float(case.tags["scale"] or 1)
    return "<70" if x < 70 else "70-100" if x < 100 else "100-140" if x < 140 else ">=140"


CONDITIONS = [*TAG_KEYS, "effective_dpi"]


def condition_value(case: Case, key: str) -> str:
    return effective_dpi(case) if key == "effective_dpi" else bucket(key, case.tags[key])


def condition_table(cases: list[Case], settings: list[str], overall_best: str) -> list[dict]:
    table = []
    for key in CONDITIONS:
        groups: dict[str, list[Case]] = defaultdict(list)
        for c in cases:
            groups[condition_value(c, key)].append(c)
        for value, group in sorted(groups.items()):
            best = single_best(group, settings)
            wins = Counter(c.best(settings) for c in group)
            top, top_n = wins.most_common(1)[0]
            row = {
                "condition": key,
                "value": value,
                "n": len(group),
                "best_setting": best,
                "best_cer": round(pooled(group, lambda c: best), 4),
                "single_best_cer": round(pooled(group, lambda c: overall_best), 4),
                "oracle_cer": round(pooled(group, lambda c: c.best(settings)), 4),
                "most_wins": f"{top}（{top_n}/{len(group)}）",
            }
            row.update({s: round(pooled(group, lambda c, s=s: s), 4) for s in settings})
            table.append(row)
    return table


def worst_table(cases: list[Case], settings: list[str], overall_best: str) -> list[dict]:
    """単独ベストで CER が悪い画像。一番良い設定に替えても悪いなら OCR の限界か正解の問題、良くなるなら振り分けで直せる"""
    rows = []
    enough = [c for c in cases if c.chars >= WORST_MIN_CHARS]
    for c in sorted(enough, key=lambda c: -c.cer(overall_best))[:WORST_N]:
        best = c.best(settings)
        rows.append({
            "case": c.name,
            "single_best_cer": round(c.cer(overall_best), 4),
            "best_setting": best,
            "best_cer": round(c.cer(best), 4),
            "chars": c.chars,
            **c.tags,
            "memo": "",
        })
    return rows


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keep_memos(path: Path, rows: list[dict]) -> None:
    """worst.csv に書き込んだ memo を、作り直しても消さない"""
    if not path.exists():
        return
    with open(path, encoding="utf-8-sig") as f:
        memos = {r["case"]: r.get("memo", "") for r in csv.DictReader(f)}
    for r in rows:
        r["memo"] = memos.get(r["case"], "")


def md_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        lines.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in columns) + " |")
    return "\n".join(lines)


def report(csv_path, cases, settings, skipped, settings_t, pairs_t, cond_t, worst_t) -> str:
    sb = settings_t[0]
    oracle = pooled(cases, lambda c: c.best(settings))
    gap = sb["cer_page"] - oracle
    best_pair = pairs_t[0]
    closed = (sb["cer_page"] - best_pair["cer_page"]) / gap if gap > 0 else 0

    out = [
        "# オペレータAI 分析（進め方2・3）",
        "",
        f"採点表 `{csv_path}`。画像 {len(cases)} 枚 × 設定 {len(settings)} 個。CER は小さいほど良い。",
    ]
    if skipped:
        out += ["", f"エラーがあった {len({r['case'] for r in skipped})} 枚は集計から外した："]
        out += [f"- {r['case']} {r['setting']}: {r['error']}" for r in skipped]

    out += [
        "",
        "## 進め方2：単独ベストとオラクル",
        "",
        md_table([
            {"方法": f"単独ベスト（{sb['setting']}）", "cer_page": sb["cer_page"], "差の埋まり具合": "0%"},
            {"方法": f"2つから選ぶときの上限（{best_pair['pair']}）", "cer_page": best_pair["cer_page"],
             "差の埋まり具合": f"{closed:.0%}"},
            {"方法": "オラクル（上限）", "cer_page": round(oracle, 4), "差の埋まり具合": "100%"},
        ], ["方法", "cer_page", "差の埋まり具合"]),
        "",
        f"振り分けで伸ばせる余地：{sb['cer_page']:.3f} → {oracle:.3f}（誤りを {gap / sb['cer_page']:.0%} 減らせる）",
        "",
        "### 設定ごと",
        "",
        f"`median` は画像ごとの CER の真ん中、`broken` は CER が {BROKEN} を超えた画像の割合、"
        "`wins` はその設定が一番良かった画像の数（同点なら速いほう）。",
        "",
        md_table(settings_t, list(settings_t[0])),
        "",
        "### 2つの設定から選ぶときの上限（上位5）",
        "",
        md_table(pairs_t[:5], ["pair", "cer_page"]),
        "",
        "## 進め方3：条件ごとの勝ち負け",
        "",
        "`best_setting` はその条件の画像だけで見た単独ベスト。`single_best_cer` は全体の単独ベスト"
        f"（{sb['setting']}）のまま読んだときの CER。設定ごとの CER は `conditions.csv`。",
        "`effective_dpi` は dpi × 縮小率（画面上の文字の細かさ）。",
        "",
        md_table(cond_t, ["condition", "value", "n", "best_setting", "best_cer", "single_best_cer", "oracle_cer",
                          "most_wins"]),
        "",
        f"## CER が悪い画像（単独ベストで悪い順、上位 {len(worst_t)}。正解の文字が {WORST_MIN_CHARS} 字未満のページは除く）",
        "",
        "`best_cer` も悪いなら OCR の限界か正解の問題、良くなっていれば振り分けで直せる。原因は `worst.csv` の memo 列に書く。",
        "",
        md_table(worst_t, ["case", "chars", "single_best_cer", "best_setting", "best_cer", "style", "dark_mode", "dpi", "scale",
                           "lang", "columns"]),
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="採点表から、単独ベスト・オラクルと条件ごとの勝ち負けを出す")
    parser.add_argument("scores", type=Path, help="score_all.py の scores.csv")
    parser.add_argument("--out", type=Path, help="出力先（既定は scores.csv と同じ場所の analysis/）")
    args = parser.parse_args(argv)

    cases, settings, skipped = load(args.scores)
    if not cases:
        raise SystemExit(f"エラー: {args.scores} に、全部の設定がそろった画像がありません。")
    out = args.out or args.scores.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    settings_t = setting_table(cases, settings)
    overall_best = settings_t[0]["setting"]
    pairs_t = pair_table(cases, settings)
    cond_t = condition_table(cases, settings, overall_best)
    worst_t = worst_table(cases, settings, overall_best)
    keep_memos(out / "worst.csv", worst_t)

    write_csv(out / "settings.csv", settings_t)
    write_csv(out / "pairs.csv", pairs_t)
    write_csv(out / "conditions.csv", cond_t)
    write_csv(out / "worst.csv", worst_t)
    md = report(args.scores, cases, settings, skipped, settings_t, pairs_t, cond_t, worst_t)
    (out / "report.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"\n{out}/ に report.md・settings.csv・pairs.csv・conditions.csv・worst.csv を書きました")
    return out


if __name__ == "__main__":
    main()
