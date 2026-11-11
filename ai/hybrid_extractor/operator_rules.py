"""ルールのオペレータ: ブロックごとに、どの専門家に聞くかを決め、専門家の答えをまとめてラベルを決める。

    PDF ─ 抽出器（区切りの専門家で行をブロックにつなぐ）─▶ ブロック
        ─ オペレータ ─┬─ 1. 表・図の領域なら、そのまま table / figure
                       ├─ 2. 確実な決まり（キャプション・タイトル部分・ページ番号とヘッダー）に当てはまれば、それ
                       ├─ 3. 手がかりを見て、聞く専門家を選ぶ（等幅フォントがあればコードの専門家、など）
                       └─ 4. 一番自信のある専門家の答え。誰も THRESHOLD に届かなければ paragraph
        ─▶ ラベル付きIR

専門家が全員ルール版のときは、rules.py（hybrid_rules）と同じ答えになる。
専門家をAIに入れ替えるときは、LABEL_EXPERTS（ラベル）や continues（区切り）を差し替える。

数式は、最後に「数式を読む係」（math_reader.py）で LaTeX にする:
    - label="math" のブロック → 要素の tex
    - 段落・箇条書き・キャプションの中の数式 → runs の tex（IR版4）

使い方（ai/ で実行）:
    python -m hybrid_extractor.operator_rules hybrid_extractor/sample_data/sample.pdf out.labeled.json --trace
    python -m hybrid_extractor.operator_rules in.pdf out.json --math rule+ocr --math-checkpoint math_ocr/checkpoints/math_ocr_v2/best.pt
"""

import argparse
import os
from dataclasses import dataclass

from common.ir import IRDocument, save_ir, validate
from hybrid_extractor.experts import RULE_EXPERTS, DocInfo, Expert, text_of
import pymupdf

from hybrid_extractor.extract import Block, Continues, PageBlocks, extract_blocks, rule_continues, runs_with_math
from hybrid_extractor.math_reader import MathReader, inline_math_spans, make_reader
from hybrid_extractor.rules import (
    body_font,
    body_size,
    find_captions,
    is_edge_block,
    repeated_headers,
    title_labels,
    to_labeled_ir,
)

NAME = "hybrid_operator"
INLINE_MATH_LABELS = ("paragraph", "list", "caption")  # 文中の数式を探すブロック
THRESHOLD = 0.5  # 専門家の自信がこれ以上なら、その答えを採る

# 聞く順番。自信が同じときは前の専門家を優先する
LABEL_EXPERTS: list[Expert] = [RULE_EXPERTS["code"], RULE_EXPERTS["math"], RULE_EXPERTS["list"], RULE_EXPERTS["heading"]]


@dataclass
class Decision:
    """1ブロックの決め方の記録（デバッグ用）"""
    label: str
    by: str                          # 決めたもの（"kind" / "caption_rule" / 専門家の名前 / "default"）
    opinions: dict[str, float]       # 聞いた専門家ごとの自信


def choose_experts(block: Block) -> list[Expert]:
    """手がかりを見て、聞く専門家を選ぶ（安く分かることだけで絞る）"""
    chosen = []
    for expert in LABEL_EXPERTS:
        if expert.label == "code" and block.mono_ratio == 0:
            continue  # 等幅フォントの文字が1つも無い
        if expert.label == "math" and block.math_ratio == 0:
            continue  # 数式のフォントの文字が1つも無い
        if expert.label == "heading" and (block.n_rows > 3 or len(text_of(block)) > 120):
            continue  # 長すぎる
        chosen.append(expert)
    return chosen


def decide_block(block: Block, info: DocInfo) -> Decision:
    opinions = {expert.name: expert.judge(block, info) for expert in choose_experts(block)}
    best_name, best = None, 0.0
    for name, confidence in opinions.items():  # 同じ自信なら先に聞いた専門家
        if confidence > best:
            best_name, best = name, confidence
    if best_name is not None and best >= THRESHOLD:
        label = next(e.label for e in LABEL_EXPERTS if e.name == best_name)
        return Decision(label, best_name, opinions)
    return Decision("paragraph", "default", opinions)


def decide_pages(pages: list[PageBlocks]) -> list[list[tuple[Block, Decision]]]:
    """ページごとに (ブロック, 決め方) の並び（読み順）"""
    info = DocInfo(body_size(pages), body_font(pages))
    headers = repeated_headers(pages)
    result = []
    for page in pages:
        captions = find_captions(page, info.body_size)
        titles = title_labels(page, info.body_size, headers)
        items = []
        for i, b in enumerate(page.blocks):
            if b.kind in ("table", "figure"):
                decision = Decision(b.kind, "kind", {})
            elif i in captions:
                decision = Decision("caption", "caption_rule", {})
            elif i in titles:
                decision = Decision(titles[i], "title_rule", {})
            elif is_edge_block(page, b, headers):
                decision = Decision("other", "edge_rule", {})
            else:
                decision = decide_block(b, info)
            items.append((b, decision))
        result.append(items)
    return result


def read_math(pdf_path: str, decided: list[list[tuple[Block, Decision]]], reader: MathReader) -> None:
    """数式を読む係で、数式のブロックの tex と、文中の数式の runs を埋める（ブロックを書き換える）"""
    with pymupdf.open(pdf_path) as doc:
        for items in decided:
            for block, decision in items:
                if block.kind != "text" or not block.chars:
                    continue
                page = doc[block.page - 1]
                if decision.label == "math":
                    block.tex = reader.read(page, block.chars)
                elif decision.label in INLINE_MATH_LABELS:
                    spans = inline_math_spans(block.chars)
                    if spans:
                        read = {span: reader.read(page, block.chars[span[0]:span[1]]) for span in spans}
                        block.runs = runs_with_math(block.chars, block.decorations, read)


def predict(pdf_path: str, continues: Continues = rule_continues, math: str = "rule",
            math_checkpoint: str | None = None) -> IRDocument:
    """PDF → ラベル付きIR（オペレータ＋専門家）。評価の登録簿（evaluation/models.json）から呼ぶ。
    math は数式を読む係（"none" / "pdf_text" / "rule" / "ocr" / "rule+ocr"。OCR を使う係は math_checkpoint が要る）"""
    pages = extract_blocks(pdf_path, continues)
    decided = decide_pages(pages)
    reader = make_reader(math, math_checkpoint)
    if reader is not None:
        read_math(pdf_path, decided, reader)
    labeled = [[(b, d.label) for b, d in items] for items in decided]
    return to_labeled_ir(pages, labeled, os.path.basename(pdf_path), NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → ラベル付きIR（ルールのオペレータ＋専門家）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("out", help="出力するラベル付きIRのJSON")
    parser.add_argument("--trace", action="store_true", help="ブロックごとに、どの専門家がどう答えたかを表示する")
    parser.add_argument("--math", default="rule", help="数式を読む係（none / pdf_text / rule / ocr / rule+ocr）")
    parser.add_argument("--math-checkpoint", help="math_ocr の重み（best.pt）。ocr・rule+ocr のときに要る")
    args = parser.parse_args()
    if args.trace:
        for items in decide_pages(extract_blocks(args.pdf)):
            for b, d in items:
                ops = " ".join(f"{name}={c:.2f}" for name, c in d.opinions.items())
                print(f"p{b.page} {d.label:9} by={d.by:13} {ops:45} {text_of(b)[:30]!r}")
    doc = predict(args.pdf, math=args.math, math_checkpoint=args.math_checkpoint)
    problems = validate(doc)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(doc, args.out)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()
