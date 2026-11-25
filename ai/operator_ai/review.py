"""
CER が悪い画像（analysis/worst.csv）を、目で確かめるためのページを作る。原因の下書きも自動で付ける。

使い方（ai/ で）:
    python -m operator_ai.review image_dataset/data/val operator_ai/results/val/analysis/worst.csv
      → operator_ai/data/val/review.html   ブラウザで開く（画像も中に入っているので、このファイルだけで見られる）

1枚ごとに出すもの:
  - 画像に、正解のブロックの枠（緑）と、OCR が読んだ段落の枠（赤＝単独ベスト、青＝一番良かった設定）を重ねたもの
  - 設定ごとの cer_page・正解の外に読んだ文字の数
  - 正解の文字と、OCR が読んだ文字
  - 原因の下書き（自動の判定 ＋ worst.csv の memo）。ページの上で直して、「worst.csv を保存」で書き出せる
"""

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

from PIL import Image

from common.ir import load_ir
from evaluation.metrics import normalize_text
from operator_ai.analyze import load
from operator_ai.features import measure
from operator_ai.score_all import PAGE_CER_EXCLUDED

SHOW_WIDTH = 900  # ページに入れる画像の幅（px）


def truth_text(truth) -> str:
    return "\n".join(
        e.text or "" for e in truth.pages[0].elements
        if e.kind not in PAGE_CER_EXCLUDED and (e.label or "") not in PAGE_CER_EXCLUDED
    )


def diagnose(case, settings: list[str], single_best: str, feats: dict, ocr_chars: dict[str, int]) -> list[str]:
    """数字から分かる原因の候補。当てはまるものを全部返す"""
    chars = case.chars
    extra = {s: int(case.settings[s]["extra_chars"]) for s in settings}
    cer = {s: case.cer(s) for s in settings}
    best = case.best(settings)
    out = []
    if min(extra.values()) > 0.3 * chars:
        out.append(f"どの設定でも正解のブロックの外に多く読んでいる（少ない設定でも {min(extra.values())} 字）。正解の箱がずれている疑い")
    if ocr_chars[single_best] < 0.2 * chars:
        readers = [s for s in settings if s.startswith("easyocr") and cer[s] < cer[single_best] - 0.2]
        msg = f"{single_best} がほとんど読めていない（{ocr_chars[single_best]} 字 / 正解 {chars} 字）"
        out.append(msg + ("。EasyOCR なら読める" if readers else ""))
    elif extra[single_best] > 0.3 * chars and min(extra.values()) <= 0.3 * chars:
        where = "2段組の左右の段" if case.tags["columns"] == "2" else "隣の図・表など"
        out.append(f"{single_best} だけ正解の外に多く読んでいる（{extra[single_best]} 字）。{where}を1行にまとめて読んだ疑い")
    elif cer[single_best] > 0.4 and extra[single_best] < 0.1 * chars:
        out.append(f"{single_best} は枠は見つけている（{ocr_chars[single_best]} 字）が、文字を読み違えている")
    if feats["page_dark"]:
        out.append("ダークモード")
    if feats["band_ratio"] > 0.05:
        out.append(f"ビューアの帯・背景あり（画像の {feats['band_ratio']:.0%}）")
    eff = float(case.tags["dpi"]) * float(case.tags["scale"] or 1)
    if eff < 70 or feats["char_height"] < 8:
        out.append(f"文字が細かい（文字の高さ {feats['char_height']:.0f}px、dpi × 縮小率 {eff:.0f}）")
    if feats["is_jpeg"]:
        out.append(f"JPEG（品質 {case.tags['jpeg_quality'] or '?'}）")
    if cer[best] > 0.4:
        out.append(f"一番良い設定（{best}）でも cer_page {cer[best]:.2f}。設定を選ぶだけでは直らない")
    elif cer[best] < cer[single_best] - 0.2:
        out.append(f"{best} に替えれば cer_page {cer[single_best]:.2f} → {cer[best]:.2f}。振り分けで直せる")
    return out


def image_data_url(path: Path) -> tuple[str, int, int]:
    img = Image.open(path).convert("RGB")
    if img.width > SHOW_WIDTH:
        img = img.resize((SHOW_WIDTH, round(img.height * SHOW_WIDTH / img.width)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.width, img.height


def boxes(doc, w: int, h: int, cls: str, show_label: bool = False) -> str:
    out = []
    for e in doc.pages[0].elements:
        x0, y0, x1, y1 = e.bbox
        title = html.escape(f"{e.label or e.kind}: {(e.text or '')[:80]}")
        out.append(f'<rect class="{cls}" x="{x0 * w:.1f}" y="{y0 * h:.1f}" width="{(x1 - x0) * w:.1f}" '
                   f'height="{(y1 - y0) * h:.1f}"><title>{title}</title></rect>')
        if show_label:
            out.append(f'<text class="{cls}" x="{x0 * w + 2:.1f}" y="{y0 * h - 2:.1f}">{html.escape(e.label or e.kind)}</text>')
    return "\n".join(out)


def ocr_text(doc) -> str:
    return "\n".join(e.text or "" for e in doc.pages[0].elements)


STYLE = """
:root { --bg:#fafafa; --fg:#1d1d1f; --muted:#6e6e73; --card:#fff; --line:#d2d2d7; --truth:#1a9a4a; --sb:#d62d20; --best:#1f6fd1; }
@media (prefers-color-scheme: dark) { :root { --bg:#161617; --fg:#f5f5f7; --muted:#a1a1a6; --card:#1f1f21; --line:#3a3a3c; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.6 -apple-system, "Hiragino Sans", sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0 auto 0 0; }
button { font:inherit; padding:6px 12px; border-radius:8px; border:1px solid var(--line); background:var(--card); color:var(--fg); cursor:pointer; }
button.primary { background:var(--best); color:#fff; border-color:var(--best); }
.legend span { margin-right:12px; } .legend i { display:inline-block; width:12px; height:12px; border:2px solid; vertical-align:-1px; margin-right:4px; }
main { max-width:1500px; margin:0 auto; padding:16px; }
nav { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }
nav a { font-size:12px; padding:2px 8px; border-radius:6px; border:1px solid var(--line); color:var(--fg); text-decoration:none; }
nav a.done { background:var(--truth); color:#fff; border-color:var(--truth); }
section { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:24px; }
section h2 { margin:0 0 4px; font-size:17px; }
.meta { color:var(--muted); font-size:12px; margin-bottom:12px; }
.grid { display:grid; grid-template-columns:minmax(0,1.3fr) minmax(0,1fr); gap:16px; }
@media (max-width:900px) { .grid { grid-template-columns:1fr; } }
svg { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; background:#fff; }
rect { fill:none; stroke-width:2; } rect.truth { stroke:var(--truth); stroke-dasharray:6 3; } rect.sb { stroke:var(--sb); } rect.best { stroke:var(--best); }
text { font-size:11px; } text.truth { fill:var(--truth); }
.hide-truth .truth, .hide-sb .sb, .hide-best .best { display:none; }
.toggles { margin:6px 0; display:flex; gap:6px; flex-wrap:wrap; }
.toggles button.off { opacity:.4; }
table { border-collapse:collapse; width:100%; font-size:12px; margin-bottom:12px; }
td, th { border-bottom:1px solid var(--line); padding:3px 6px; text-align:left; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
tr.sb td:first-child { color:var(--sb); font-weight:600; } tr.best td:first-child { color:var(--best); font-weight:600; }
ul.auto { margin:0 0 8px; padding-left:18px; }
textarea { width:100%; min-height:70px; font:inherit; padding:8px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--fg); }
details { margin-top:8px; } summary { cursor:pointer; color:var(--muted); }
pre { white-space:pre-wrap; font-size:12px; background:var(--bg); padding:8px; border-radius:6px; max-height:240px; overflow:auto; margin:4px 0; }
.texts { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:8px; } @media (max-width:900px) { .texts { grid-template-columns:1fr; } }
label.check { display:inline-flex; gap:6px; align-items:center; margin-top:6px; color:var(--muted); }
"""

SCRIPT = """
const rows = JSON.parse(document.getElementById('rows').textContent);
const columns = JSON.parse(document.getElementById('columns').textContent);
function toggle(btn, id, cls) { document.getElementById(id).classList.toggle('hide-' + cls); btn.classList.toggle('off'); }
function markDone(box) { document.querySelector('nav a[href="#' + box.dataset.case + '"]').classList.toggle('done', box.checked); }
function saveCsv() {
  const memo = {}; document.querySelectorAll('textarea[data-case]').forEach(t => memo[t.dataset.case] = t.value);
  const esc = v => { v = v == null ? '' : String(v); return /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  const lines = [columns.join(',')].concat(rows.map(r => columns.map(c => esc(c === 'memo' ? memo[r.case] : r[c])).join(',')));
  const blob = new Blob(['\\ufeff' + lines.join('\\r\\n') + '\\r\\n'], {type: 'text/csv'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'worst.csv'; a.click();
}
"""


def render(dataset: Path, worst_rows: list[dict], scores: Path, out: Path) -> Path:
    cases, settings, _ = load(scores)
    by_name = {c.name: c for c in cases}
    single_best = min(settings, key=lambda s: sum(c.errors(s) for c in cases))
    ocr_dir = scores.parent / "ocr" if (scores.parent / "ocr").exists() else Path("operator_ai/data") / dataset.name / "ocr"

    sections, nav = [], []
    for n, row in enumerate(worst_rows, 1):
        name = row["case"]
        case = by_name[name]
        truth = load_ir(str(dataset / "truth" / f"{name}.ir.json"))
        image_path = dataset / "images" / truth.source
        feats = measure(str(image_path))
        docs = {s: load_ir(str(ocr_dir / s.split(":")[0] / s.split(":")[1] / f"{name}.ir.json")) for s in settings}
        ocr_chars = {s: len(normalize_text(ocr_text(d))) for s, d in docs.items()}
        best = case.best(settings)
        auto = diagnose(case, settings, single_best, feats, ocr_chars)
        memo = row.get("memo") or "／".join(auto)

        url, w, h = image_data_url(image_path)
        svg_id = f"svg-{n}"
        svg = (f'<svg id="{svg_id}" viewBox="0 0 {w} {h}"><image href="{url}" width="{w}" height="{h}"/>'
               + boxes(truth, w, h, "truth", show_label=True) + boxes(docs[single_best], w, h, "sb")
               + (boxes(docs[best], w, h, "best") if best != single_best else "") + "</svg>")

        rows_html = []
        for s in sorted(settings, key=case.cer):
            cls = "sb" if s == single_best else "best" if s == best else ""
            r = case.settings[s]
            rows_html.append(f'<tr class="{cls}"><td>{s}</td><td class="num">{case.cer(s):.3f}</td>'
                             f'<td class="num">{r["extra_chars"]}</td><td class="num">{ocr_chars[s]}</td>'
                             f'<td class="num">{r["f1"]}</td></tr>')
        tags = "・".join(f"{k} {case.tags[k]}" for k in ("style", "dark_mode", "viewer", "crop", "background", "dpi", "scale",
                                                         "jpeg_quality", "lang", "columns") if case.tags[k] not in ("", "False"))
        feat_text = (f"page_dark {feats['page_dark']}・band_ratio {feats['band_ratio']}・char_height {feats['char_height']}・"
                     f"center_gap {feats['center_gap']}")
        sections.append(f"""
<section id="{name}">
  <h2>{n}. {name}</h2>
  <div class="meta">正解 {case.chars} 字 ／ params: {html.escape(tags)} ／ 特徴量: {feat_text}</div>
  <div class="grid">
    <div>
      <div class="toggles">
        <button onclick="toggle(this,'{svg_id}','truth')">正解の枠</button>
        <button onclick="toggle(this,'{svg_id}','sb')">{single_best}</button>
        {f'<button onclick="toggle(this,{chr(39)}{svg_id}{chr(39)},{chr(39)}best{chr(39)})">{best}</button>' if best != single_best else ''}
      </div>
      {svg}
    </div>
    <div>
      <table><tr><th>設定</th><th>cer_page</th><th>正解の外</th><th>読んだ字数</th><th>f1</th></tr>{''.join(rows_html)}</table>
      <b>自動の判定</b>
      <ul class="auto">{''.join(f'<li>{html.escape(a)}</li>' for a in auto) or '<li>当てはまるものなし</li>'}</ul>
      <b>memo（直してから「worst.csv を保存」）</b>
      <textarea data-case="{name}">{html.escape(memo)}</textarea>
      <label class="check"><input type="checkbox" data-case="{name}" onchange="markDone(this)">確認した</label>
    </div>
  </div>
  <details><summary>読んだ文字を比べる</summary>
    <div class="texts">
      <div><b>正解</b><pre>{html.escape(truth_text(truth))}</pre></div>
      <div><b style="color:var(--sb)">{single_best}</b><pre>{html.escape(ocr_text(docs[single_best]))}</pre></div>
      <div><b style="color:var(--best)">{best}</b><pre>{html.escape(ocr_text(docs[best]))}</pre></div>
    </div>
  </details>
</section>""")
        nav.append(f'<a href="#{name}">{n}</a>')

    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CERが悪い画像の確認</title><style>{STYLE}</style></head>
<body>
<header>
  <h1>CER が悪い画像の確認（{len(worst_rows)} 枚）</h1>
  <span class="legend"><span><i style="border-color:var(--truth)"></i>正解</span><span><i style="border-color:var(--sb)"></i>{single_best}（単独ベスト）</span><span><i style="border-color:var(--best)"></i>一番良かった設定</span></span>
  <button class="primary" onclick="saveCsv()">worst.csv を保存</button>
</header>
<main>
<p>memo は自動の下書き。画像と見比べて直し、「確認した」に印を付ける。全部終わったら「worst.csv を保存」で書き出し、<code>operator_ai/results/val/analysis/worst.csv</code> と置き換える。枠にカーソルを乗せると、その枠の文字が出る。</p>
<nav>{''.join(nav)}</nav>
{''.join(sections)}
</main>
<script type="application/json" id="rows">{json.dumps(worst_rows, ensure_ascii=False).replace('</', '<\\/')}</script>
<script type="application/json" id="columns">{json.dumps(list(worst_rows[0]), ensure_ascii=False)}</script>
<script>{SCRIPT}</script>
</body></html>"""
    out.write_text(page, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="CER が悪い画像を目で確かめるページを作る")
    parser.add_argument("dataset", type=Path, help="image_dataset の出力先（images/・truth/ がある）")
    parser.add_argument("worst", type=Path, help="analyze.py の worst.csv")
    parser.add_argument("--scores", type=Path, help="scores.csv（既定は worst.csv の1つ上の階層）")
    parser.add_argument("--out", type=Path, help="出力先（既定は operator_ai/data/<データセット名>/review.html）")
    args = parser.parse_args(argv)

    with open(args.worst, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    scores = args.scores or args.worst.parent.parent / "scores.csv"
    out = args.out or Path("operator_ai/data") / args.dataset.name / "review.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(args.dataset, rows, scores, out)
    print(f"{out} を作りました（{len(rows)} 枚）")
    return out


if __name__ == "__main__":
    main()
