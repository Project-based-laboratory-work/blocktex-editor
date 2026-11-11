"""ルールのオペレータ: ブロックごとに、どの専門家に聞くかを決め、専門家の答えをまとめてラベルを決める。

    PDF ─ 抽出器（区切りの専門家で行をブロックにつなぐ）─▶ ブロック
        ─ オペレータ ─┬─ 1. 表・図の領域なら、そのまま table / figure
                       ├─ 2. 確実な決まり（キャプション・タイトル部分・ページ番号とヘッダー）に当てはまれば、それ
                       ├─ 3. 手がかりを見て、聞く専門家を選ぶ（等幅フォントがあればコードの専門家、など）
                       └─ 4. 一番自信のある専門家の答え。誰も THRESHOLD に届かなければ paragraph
        ─▶ ラベル付きIR

専門家が全員ルール版のときは、rules.py（hybrid_rules）と同じ答えになる。
専門家をAIに入れ替えるときは、LABEL_EXPERTS（ラベル）や continues（区切り）を差し替える。

使い方（ai/ で実行）:
    python -m hybrid_extractor.operator_rules hybrid_extractor/sample_data/sample.pdf out.labeled.json --trace
"""

import argparse
import os
from dataclasses import dataclass

from common.ir import IRDocument, save_ir, validate
from hybrid_extractor.experts import RULE_EXPERTS, DocInfo, Expert, text_of
from hybrid_extractor.extract import Block, Continues, PageBlocks, extract_blocks, rule_continues
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


def predict(pdf_path: str, continues: Continues = rule_continues) -> IRDocument:
    """PDF → ラベル付きIR（オペレータ＋専門家）。評価の登録簿（evaluation/models.json）から呼ぶ"""
    pages = extract_blocks(pdf_path, continues)
    decided = decide_pages(pages)
    labeled = [[(b, d.label) for b, d in items] for items in decided]
    return to_labeled_ir(pages, labeled, os.path.basename(pdf_path), NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → ラベル付きIR（ルールのオペレータ＋専門家）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("out", help="出力するラベル付きIRのJSON")
    parser.add_argument("--trace", action="store_true", help="ブロックごとに、どの専門家がどう答えたかを表示する")
    args = parser.parse_args()
    if args.trace:
        for items in decide_pages(extract_blocks(args.pdf)):
            for b, d in items:
                ops = " ".join(f"{name}={c:.2f}" for name, c in d.opinions.items())
                print(f"p{b.page} {d.label:9} by={d.by:13} {ops:45} {text_of(b)[:30]!r}")
    doc = predict(args.pdf)
    problems = validate(doc)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(doc, args.out)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()
