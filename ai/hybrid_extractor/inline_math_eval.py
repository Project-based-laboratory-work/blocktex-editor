"""文中の数式（$...$）の比較実験。数式を読む係（A〜D）を、正確さと処理時間で比べる。

    python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math --docs 60
    python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math --docs 60 \\
        --math-checkpoint math_ocr/checkpoints/math_ocr_v2/best.pt          # C・D（math_ocr）も比べる

流れ:
    1. 文中の数式をたくさん含む文書（TeX）を作る。数式の前後に目印（\\special{pdf:dest}）を置く
    2. PDF にして、目印の間の文字を「正解の数式の範囲」、TeX に書いた数式を「正解の LaTeX」にする
    3. 見つける: 抽出器のブロックから、文中の数式の専門家（--finder。既定はルール版）で数式の範囲を探し、
       正解の範囲と比べる（適合率・再現率）
    4. 読む: 正解の範囲の文字を、それぞれの読む係に渡して、LaTeX を比べる（トークン誤り率・完全一致率・1式あたりの時間）
       見つける段階の失敗と分けるため、ここでは正解の範囲を渡す
    5. 通し: 見つけた範囲を読んだ結果も比べる（見つける＋読む）

比べ方: LaTeX をトークンに分けて（evaluation/metrics.py の tex_tokens と同じ）、編集距離 ÷ 正解のトークン数。
\\, や \\quad などの間隔だけの命令は、PDF からは読み取れないので、正解・予測の両方から除く。

出力（<out>/）: pdfs/<id>.pdf、report.json（全部の数字と、数式ごとの結果）
"""

import argparse
import json
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

from common.text import edit_distance
from evaluation.ground_truth import compile_tex
from evaluation.marks import read_marks
from evaluation.metrics import tex_tokens
from hybrid_extractor.doc_gen import WORDS, preamble, random_style
from hybrid_extractor.extract import extract_blocks, page_lines, union_all
from hybrid_extractor.experts import INLINE_MATH_EXPERTS, INLINE_MATH_FINDERS, InlineMathExpert, get_inline_expert
from hybrid_extractor.math_reader import ACCENTS, make_reader, shown

# 正解の数式の文字を集める範囲（目印のベースラインからの上・下、pt）。文字の種類で変える:
#   ふつうの文字: 文字の中心が [-10, +5] にある。広すぎると、行の間の狭い文書（article の 10pt は 12pt 送り）で上下の行の文字が混ざる
#   小さい文字（添字・分数の中、8pt 未満）: 中心が [-14, +8]。入れ子の分数の分子は高い位置にある
#   大きい記号（∑ ∫ √ と CMEX のかっこ）: PyMuPDF の枠が実際より上にずれるので、枠の半分以上が [-25, +5] に入っていればよい
GT_NORMAL = (10.0, 5.0)
GT_SMALL = (14.0, 8.0)
GT_BIG = (25.0, 5.0)
SMALL_SIZE = 8.0
BIG_SYMBOLS = set("∑∏∫√")
BIG_TEX = re.compile(r"\\(sum|prod|int|sqrt|left|binom)")       # 大きい記号が出る形
SMALL_TEX = re.compile(r"[_^']|\\(frac|binom|sqrt\[)")          # 小さい文字が出る形
ACCENT_X_MARGIN = 4.0  # アクセントの記号（幅の無い ⃗ など）は、目印の左右からこの幅（pt）まではみ出していても入れる
SPACING = {r"\,", r"\;", r"\!", r"\:", r"\quad", r"\qquad", r"\ ", "~"}
MATCH_IOU = 0.5

# ---- 文中の数式を作る ----
# 対応させたい数式は、math_ocr v2 が読める記号の一覧（ikumi さんのドキュメント「数式AI（math_ocr v2）が読める記号一覧」）。
# 結果を一覧の項目ごとに見られるように、項目（kind）ごとに作る。一覧の「読めないもの」（行列・場合分けなど）は作らない
LOWER = list("abcdefghknmprstuvwxyz")
UPPER = list("ABCDEFGHKLMNPRSTXYZ")
GREEK_LOWER = [r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\zeta", r"\eta", r"\theta", r"\kappa",
               r"\lambda", r"\mu", r"\nu", r"\xi", r"\pi", r"\rho", r"\sigma", r"\tau", r"\phi", r"\varphi", r"\chi",
               r"\psi", r"\omega"]
GREEK_UPPER = [r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Pi", r"\Sigma", r"\Phi", r"\Psi", r"\Omega"]
OPERATORS = ["+", "-", "/", r"\times", r"\cdot", r"\div", r"\pm", r"\circ"]
RELATIONS = ["=", "<", ">", r"\leq", r"\geq", r"\neq", r"\approx", r"\equiv", r"\sim", r"\propto"]
FUNCTIONS = [r"\sin", r"\cos", r"\tan", r"\log", r"\ln", r"\exp"]
ACCENT_COMMANDS = [r"\bar", r"\hat", r"\vec", r"\tilde", r"\dot"]  # math_reader.ACCENTS（記号の対応表）とは別
BB = list("RNZQC")


def atom(rng: random.Random) -> str:
    return rng.choice(LOWER + LOWER + GREEK_LOWER)


def number(rng: random.Random) -> str:
    return rng.choice([str(rng.randint(0, 9)), str(rng.randint(10, 99)), f"{rng.randint(0, 9)}.{rng.randint(1, 99)}"])


def script(rng: random.Random, base: str) -> str:
    r = rng.random()
    if r < 0.4:
        return f"{base}_{{{rng.choice(['i', 'j', 'k', 'n', '0', '1', '2', 'ij'])}}}"
    if r < 0.8:
        return f"{base}^{{{rng.choice(['2', '3', 'n', '-1', 'k'])}}}"
    return f"{base}_{{{rng.choice(['i', '0'])}}}^{{{rng.choice(['2', 'n'])}}}"


INLINE_KINDS = {
    # 文字と数字
    "letters": lambda r: rng_join(r, [r.choice(LOWER + UPPER + [number(r)]) for _ in range(r.randint(1, 3))]),
    "greek": lambda r: rng_join(r, [r.choice(GREEK_LOWER + GREEK_UPPER + [r"\ell", r"\infty"]) for _ in range(r.randint(1, 2))]),
    # 演算子と関係
    "operators": lambda r: f"{atom(r)} {r.choice(OPERATORS)} {atom(r)}",
    "relations": lambda r: f"{atom(r)} {r.choice(RELATIONS)} {r.choice([atom(r), number(r)])}",
    "sets": lambda r: r.choice([f"{r.choice(LOWER)} \\in {r.choice(UPPER)}", f"{r.choice(UPPER)} \\subset {r.choice(UPPER)}",
                                f"{r.choice(UPPER)} \\cup {r.choice(UPPER)}", f"{r.choice(UPPER)} \\cap {r.choice(UPPER)}",
                                f"\\{{ {r.choice(LOWER)}, {r.choice(LOWER)} \\}}"]),
    "arrow": lambda r: r.choice([f"f : {r.choice(UPPER)} \\to {r.choice(UPPER)}", f"\\partial {atom(r)}", f"\\nabla {r.choice(LOWER)}"]),
    # 添字・指数
    "script": lambda r: rng_join(r, [script(r, atom(r)) if r.random() < 0.7 else atom(r) for _ in range(r.randint(1, 3))]),
    # 大きな演算子と関数
    "big_op": lambda r: r.choice([f"\\sum_{{i=1}}^{{n}} {script(r, atom(r))}", f"\\prod_{{k=1}}^{{m}} {r.choice(LOWER)}_{{k}}",
                                  f"\\int_{{0}}^{{1}} f(x) dx", "\\int f(x) dx"]),
    "limit": lambda r: r.choice([f"\\lim_{{x \\to 0}} f(x)", f"\\max_{{{r.choice('xk')}}} {r.choice(LOWER)}",
                                 f"\\min_{{{r.choice('xk')}}} {r.choice(LOWER)}"]),
    "function": lambda r: r.choice([f"{r.choice(FUNCTIONS)} {atom(r)}", f"\\log_{{2}} {r.choice(LOWER)}"]),
    # 形を作るもの
    "fraction": lambda r: r.choice([f"\\frac{{{atom(r)}}}{{{atom(r)}}}", f"\\frac{{{atom(r)}}}{{{atom(r)}}} + {atom(r)}",
                                    f"\\frac{{\\frac{{{atom(r)}}}{{{atom(r)}}}}}{{{atom(r)}}}"]),
    "sqrt": lambda r: r.choice([f"\\sqrt{{{atom(r)}}}", f"\\sqrt[3]{{{r.choice(LOWER)}}}"]),
    "binom": lambda r: f"\\binom{{{r.choice('nm')}}}{{{r.choice('kr')}}}",
    "accent": lambda r: f"{r.choice(ACCENT_COMMANDS)}{{{r.choice(LOWER)}}}",
    "font": lambda r: r.choice([f"\\mathbb{{{r.choice(BB)}}}", f"{r.choice(LOWER)} \\in \\mathbb{{{r.choice(BB)}}}",
                                f"\\mathbf{{{r.choice(LOWER)}}}"]),
    "paren": lambda r: r.choice([f"\\left( \\frac{{{atom(r)}}}{{{atom(r)}}} \\right)", f"({atom(r)} + {atom(r)})",
                                 f"[{r.choice(LOWER)}, {r.choice(LOWER)}]"]),
    "dots": lambda r: r.choice([f"{r.choice(LOWER)}_{{1}}, \\dots, {r.choice(LOWER)}_{{n}}", f"{r.choice(LOWER)} \\cdots {r.choice(LOWER)}"]),
    "misc": lambda r: r.choice(["n!", "f'", f"f'({r.choice(LOWER)})", f"{r.choice(LOWER)}!"]),
}


def rng_join(rng: random.Random, parts: list[str]) -> str:
    out = parts[0]
    for p in parts[1:]:
        out += f" {rng.choice(['+', '-', '=', '<'])} {p}"
    return out


def inline_document(rng: random.Random, lang: str) -> tuple[str, list[dict], object]:
    """文中の数式をたくさん含む文書。戻り値は (TeX, 数式の一覧 [{"id", "kind", "tex"}], 見た目の設定)"""
    style = random_style(rng, lang)
    style.centered_headings = False
    w = WORDS[lang]
    items: list[dict] = []
    paragraphs = []
    for _ in range(rng.randint(5, 9)):
        sentences = []
        for _ in range(rng.randint(2, 4)):
            kind = rng.choice(list(INLINE_KINDS))
            tex = INLINE_KINDS[kind](rng)
            k = len(items)
            items.append({"id": k, "kind": kind, "tex": tex})
            math = rf"\mk{{m{k}s}}${tex}$\mk{{m{k}e}}"
            noun = rng.choice(w["nouns"])
            if lang == "ja":
                sentences.append(f"{noun}について、{math} を{rng.choice(['考える', '用いる', '比べる'])}。")
            else:
                sentences.append(f"We {rng.choice(['use', 'compare', 'consider'])} {math} for {noun}.")
        paragraphs.append(w["sep"].join(sentences))
    tex = preamble(style) + "\n\n".join(paragraphs) + "\n\\end{document}\n"
    return tex, items, style


# ---- 正解の範囲 ----


def band_overlap(box: tuple, top: float, bottom: float) -> float:
    """文字の枠の高さのうち、縦の範囲 [top, bottom] に入っている割合"""
    height = box[3] - box[1]
    inside = min(box[3], bottom) - max(box[1], top)
    return inside / height if height > 0 else 0.0


def x_margin(ch: dict) -> float:
    """目印の左右からはみ出してよい幅。アクセントの記号だけ広く（⃗ は土台の文字より 3pt ほど左に置かれる）"""
    return ACCENT_X_MARGIN if ch["c"] in ACCENTS else 0.5


def in_math_band(ch: dict, y: float, tex: str) -> bool:
    """文字が、ベースライン y の文中の数式（正解の LaTeX が tex）の縦の範囲に入るか（文字の種類で範囲を変える。GT_* を参照）。
    正解の LaTeX に無い形の文字は拾わない（上下の行の文字が混ざらないように）:
    大きい記号は \\sum などがあるときだけ、小さい文字は添字・分数などがあるときだけ"""
    if ch["c"] in BIG_SYMBOLS or "CMEX" in ch["font"]:
        if not BIG_TEX.search(tex):
            return False
        above, below = GT_BIG
        return band_overlap(ch["bbox"], y - above, y + below) >= 0.5
    if ch["size"] < SMALL_SIZE and not SMALL_TEX.search(tex):
        return False
    above, below = GT_SMALL if ch["size"] < SMALL_SIZE else GT_NORMAL
    return y - above <= (ch["bbox"][1] + ch["bbox"][3]) / 2 <= y + below


def gt_glyphs(doc: pymupdf.Document, marks: dict, k: int, tex: str) -> tuple[int, list[dict]] | None:
    """目印 m{k}s と m{k}e の間の文字（ページ, 文字の一覧）。数式が行をまたいだら None"""
    s, e = marks.get(f"m{k}s"), marks.get(f"m{k}e")
    if s is None or e is None or s.page != e.page or abs(s.y - e.y) > 2:
        return None
    chars = [ch for line in page_lines(doc[s.page - 1]) for ch in line["chars"] if ch["bbox"] is not None]
    picked = [ch for ch in chars
              if s.x - x_margin(ch) <= (ch["bbox"][0] + ch["bbox"][2]) / 2 <= e.x + x_margin(ch) and in_math_band(ch, s.y, tex)]
    return (s.page, picked) if shown(picked) else None


def glyph_key(g: dict) -> tuple:
    """文字を見分ける印（同じPDFを読み直しても同じになる）"""
    return (g["c"], round(g["bbox"][0], 2), round(g["bbox"][1], 2))


def make_case(args: tuple[int, int, str, str]) -> dict | None:
    """1文書を作って PDF を保存し、正解（数式ごとの範囲・文字・LaTeX）を返す"""
    index, seed, lang, out = args
    rng = random.Random(seed * 1_000_003 + index)
    tex, items, style = inline_document(rng, lang)
    doc_id = f"{index:04d}"
    pdf_path = Path(out) / "pdfs" / f"{doc_id}.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            pdf = compile_tex(tex, Path(tmp))
        except subprocess.CalledProcessError:
            return None
        shutil.copy(pdf, pdf_path)
    marks = read_marks(str(pdf_path))
    gts = []
    with pymupdf.open(pdf_path) as doc:
        for item in items:
            found = gt_glyphs(doc, marks, item["id"], item["tex"])
            if found is None:
                continue
            page, glyphs = found
            gts.append({**item, "page": page, "bbox": union_all([g["bbox"] for g in shown(glyphs)]),
                        "glyph_keys": [glyph_key(g) for g in glyphs]})
    return {"id": doc_id, "lang": lang, "cls": style.cls, "latin": style.latin, "gts": gts, "n_items": len(items)}


def find_spans(pages: list, finder: InlineMathExpert) -> list[dict]:
    """文中の数式の専門家（finder）で、全部の文字のブロックから数式の範囲を探す"""
    preds = []
    for page in pages:
        for b in page.blocks:
            if b.kind != "text":
                continue
            for start, end in finder.find(b.chars, page.width, page.height):
                glyphs = shown(b.chars[start:end])
                if glyphs:
                    preds.append({"page": page.page, "bbox": union_all([g["bbox"] for g in glyphs]), "start": start,
                                  "end": end, "block_bbox": b.bbox})
    return preds


# ---- 採点 ----


def norm_tokens(tex: str | None) -> list[str]:
    return [t for t in tex_tokens(tex or "") if t not in SPACING]


def iou(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_spans(gts: list[dict], preds: list[dict]) -> list[tuple[int, int]]:
    """正解と見つけた範囲を、同じページで IoU の大きい順に1対1で対応させる"""
    cands = sorted(((iou(g["bbox"], p["bbox"]), i, j) for i, g in enumerate(gts) for j, p in enumerate(preds)
                    if g["page"] == p["page"]), reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for score, i, j in cands:
        if score < MATCH_IOU or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return pairs


class Score:
    """読む係1つ分の集計"""

    def __init__(self) -> None:
        self.errors = 0
        self.tokens = 0
        self.exact = 0
        self.n = 0
        self.unread = 0
        self.seconds: list[float] = []

    def add(self, pred: str | None, gt: str, seconds: float) -> None:
        p, g = norm_tokens(pred), norm_tokens(gt)
        self.errors += edit_distance(p, g)
        self.tokens += len(g)
        self.exact += p == g
        self.n += 1
        self.unread += pred is None
        self.seconds.append(seconds)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "ter": round(self.errors / self.tokens, 4) if self.tokens else None,
            "exact": round(self.exact / self.n, 4) if self.n else None,
            "unread": round(self.unread / self.n, 4) if self.n else None,
            "ms_per_math": round(statistics.mean(self.seconds) * 1000, 3) if self.seconds else None,
        }


def evaluate(cases: list[dict], out: Path, readers: dict, finder: InlineMathExpert | None = None) -> dict:
    """文中の数式の専門家（finder）の見つけ方と、読む係ごとに、正解の範囲を読む（読む）と、見つけた範囲を読む（通し）を採点する"""
    finder = finder or INLINE_MATH_EXPERTS["rule"]
    read_scores = {name: Score() for name in readers}
    by_kind = {name: defaultdict(Score) for name in readers}
    e2e_scores = {name: Score() for name in readers}
    n_gt = n_pred = n_pair = 0
    for case in cases:
        pdf = out / "pdfs" / f"{case['id']}.pdf"
        pages = extract_blocks(str(pdf))
        case["preds"] = find_spans(pages, finder)
        pairs = match_spans(case["gts"], case["preds"])
        n_gt += len(case["gts"])
        n_pred += len(case["preds"])
        n_pair += len(pairs)
        with pymupdf.open(pdf) as doc:
            blocks = {p.page: p.blocks for p in pages}
            for gt in case["gts"]:
                page = doc[gt["page"] - 1]
                keys = set(map(tuple, gt["glyph_keys"]))
                glyphs = [ch for line in page_lines(page) for ch in line["chars"]
                          if ch["bbox"] is not None and glyph_key(ch) in keys]
                for name, reader in readers.items():
                    t0 = time.perf_counter()
                    pred = reader.read(page, glyphs)
                    dt = time.perf_counter() - t0
                    read_scores[name].add(pred, gt["tex"], dt)
                    by_kind[name][gt["kind"]].add(pred, gt["tex"], dt)
            for i, j in pairs:
                gt, p = case["gts"][i], case["preds"][j]
                block = next(b for b in blocks[p["page"]] if b.bbox == tuple(p["block_bbox"]))
                page = doc[p["page"] - 1]
                for name, reader in readers.items():
                    t0 = time.perf_counter()
                    pred = reader.read(page, block.chars[p["start"]:p["end"]])
                    e2e_scores[name].add(pred, gt["tex"], time.perf_counter() - t0)
    precision = n_pair / n_pred if n_pred else 0.0
    recall = n_pair / n_gt if n_gt else 0.0
    return {
        "finder": finder.name,
        "detect": {"n_gt": n_gt, "n_pred": n_pred, "n_pair": n_pair, "precision": round(precision, 4),
                   "recall": round(recall, 4),
                   "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0},
        "read": {name: s.summary() for name, s in read_scores.items()},
        "read_by_kind": {name: {k: s.summary() for k, s in sorted(kinds.items())} for name, kinds in by_kind.items()},
        "end_to_end": {name: s.summary() for name, s in e2e_scores.items()},
    }


def print_report(report: dict) -> None:
    d = report["detect"]
    print(f"\n見つける（{report['finder']}）: 正解 {d['n_gt']} / 見つけた {d['n_pred']} / 対応 {d['n_pair']}  "
          f"適合率 {d['precision']:.3f}  再現率 {d['recall']:.3f}  F1 {d['f1']:.3f}")
    for section, title in (("read", "読む（正解の範囲を渡す）"), ("end_to_end", "通し（見つけた範囲を読む）")):
        print(f"\n{title}")
        print(f"  {'読む係':10} {'件数':>5} {'TER':>7} {'完全一致':>8} {'読めず':>7} {'ms/式':>8}")
        for name, s in report[section].items():
            print(f"  {name:10} {s['n']:5d} {s['ter']:7.3f} {s['exact']:8.3f} {s['unread']:7.3f} {s['ms_per_math']:8.3f}")
    print("\n種類ごとの完全一致率（読む）")
    names = list(report["read_by_kind"])
    kinds = sorted({k for v in report["read_by_kind"].values() for k in v})
    print(f"  {'種類':10} " + " ".join(f"{n:>10}" for n in names))
    for k in kinds:
        print(f"  {k:10} " + " ".join(f"{report['read_by_kind'][n][k]['exact']:10.3f}" for n in names))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="文中の数式の比較実験（数式を読む係を比べる）")
    parser.add_argument("--out", required=True, help="出力先（例: hybrid_extractor/data/inline_math）")
    parser.add_argument("--docs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7, help="学習データ（0〜2）と重ならない種にする")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--math-checkpoint", help="math_ocr の重み。あれば ocr と rule+ocr も比べる")
    parser.add_argument("--finder", default="rule", choices=INLINE_MATH_FINDERS,
                        help="文中の数式の専門家（見つける段階）。ai は train.py inline で学習した重みを使う")
    args = parser.parse_args(argv)

    out = Path(args.out)
    (out / "pdfs").mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.seed, ["ja", "en"][i % 2], str(out)) for i in range(args.docs)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cases = [c for c in pool.map(make_case, jobs) if c is not None]
    print(f"{len(cases)}文書、文中の数式 {sum(len(c['gts']) for c in cases)}個"
          f"（行をまたいだ {sum(c['n_items'] - len(c['gts']) for c in cases)}個は除いた）")

    readers = {"pdf_text": make_reader("pdf_text"), "rule": make_reader("rule")}
    if args.math_checkpoint:
        readers["ocr"] = make_reader("ocr", args.math_checkpoint)
        readers["rule+ocr"] = make_reader("rule+ocr", args.math_checkpoint)
    report = evaluate(cases, out, readers, get_inline_expert(args.finder))
    report["settings"] = {"docs": len(cases), "seed": args.seed, "readers": list(readers),
                          "math_checkpoint": args.math_checkpoint, "finder": args.finder}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()
