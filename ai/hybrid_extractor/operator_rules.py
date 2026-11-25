"""ルールのオペレータ: ブロックごとに、どの専門家に聞くかを決め、専門家の答えをまとめてラベルを決める。

    ルールモード（predict）: 専門家は全員ルール版
    AIモード（predict_ai） : 区切り・ラベル・文中の数式の専門家を、学習した AI に入れ替える（ai_experts.py）

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
from pathlib import Path

from common.ir import IRDocument, save_ir, validate
import pymupdf

from hybrid_extractor.experts import INLINE_MATH_EXPERTS, RULE_EXPERTS, DocInfo, Expert, InlineMathExpert, text_of
from hybrid_extractor.extract import (
    Block,
    Continues,
    PageBlocks,
    extract_blocks,
    reindent_code,
    rule_continues,
    runs_with_math,
)
from hybrid_extractor.features import block_features, block_record, doc_info
from hybrid_extractor.math_reader import MathReader, make_reader
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
NAME_AI = "hybrid_ai"
INLINE_MATH_LABELS = ("paragraph", "list", "caption")  # 文中の数式を探すブロック
# 文中の数式の専門家（段落の中の $...$ の範囲を見つける）。AI版に入れ替えるときはここを差し替える
INLINE_MATH_EXPERT: InlineMathExpert = INLINE_MATH_EXPERTS["rule"]
THRESHOLD = 0.5  # 専門家の自信がこれ以上なら、その答えを採る
CHECKPOINTS_INLINE = Path(__file__).parent / "checkpoints" / "inline_math.pt"

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


def decide_block(block: Block, info: DocInfo, experts: list[Expert] | None = None) -> Decision:
    """専門家に聞いて、一番自信のある答えを採る。誰も THRESHOLD に届かなければ段落。
    experts を渡さなければ、ルールの専門家（LABEL_EXPERTS）のうち、手がかりで選んだ人だけに聞く"""
    asked = choose_experts(block) if experts is None else experts
    opinions = {expert.name: expert.judge(block, info) for expert in asked}
    best_name, best = None, 0.0
    for name, confidence in opinions.items():  # 同じ自信なら先に聞いた専門家
        if confidence > best:
            best_name, best = name, confidence
    if best_name is not None and best >= THRESHOLD:
        label = next(e.label for e in asked if e.name == best_name)
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


def block_feature_map(pages: list[PageBlocks]) -> dict[int, list[float]]:
    """AI のラベルの専門家が読む特徴を、文書全体で前もって作る（ブロックの id → 特徴）。
    学習データ（build_dataset.py）と同じ block_record → block_features を通す"""
    records = [{"page": p.page, "width": p.width, "height": p.height,
                "blocks": [block_record(b, p.width, p.height) for b in p.blocks]} for p in pages]
    info = doc_info(records)
    features = {}
    for page, rec in zip(pages, records):
        for i, b in enumerate(page.blocks):
            features[id(b)] = block_features(rec["blocks"], i, rec, info)
    return features


def decide_pages_ai(pages: list[PageBlocks], experts: list[Expert]) -> list[list[tuple[Block, Decision]]]:
    """AI モードの決め方（train.operator_decide と同じ順番）:
    表・図の領域 → そのまま、1ページ目のタイトル部分 → ルール、それ以外 → AI の専門家全員に聞く"""
    info = DocInfo(body_size(pages), body_font(pages), block_feature_map(pages))
    headers = repeated_headers(pages)
    result = []
    for page in pages:
        titles = title_labels(page, info.body_size, headers)
        items = []
        for i, b in enumerate(page.blocks):
            if b.kind in ("table", "figure"):
                decision = Decision(b.kind, "kind", {})
            elif i in titles:
                decision = Decision(titles[i], "title_rule", {})
            else:
                decision = decide_block(b, info, experts)
            items.append((b, decision))
        result.append(items)
    return result


def fix_code(decided: list[list[tuple[Block, Decision]]]) -> None:
    """コードと決まったブロックの字下げを戻し、行番号を取り除く（行の書体が等幅と判定できなかったコードのため）"""
    for items in decided:
        for block, decision in items:
            if decision.label == "code" and block.kind == "text" and block.chars:
                reindent_code(block)


def read_math(pdf_path: str, decided: list[list[tuple[Block, Decision]]], reader: MathReader,
              finder: InlineMathExpert | None = None) -> None:
    """数式を読む係で、数式のブロックの tex と、文中の数式の runs を埋める（ブロックを書き換える）。
    文中の数式の範囲は finder（文中の数式の専門家。省略すると INLINE_MATH_EXPERT）が見つける"""
    finder = finder or INLINE_MATH_EXPERT
    with pymupdf.open(pdf_path) as doc:
        for items in decided:
            for block, decision in items:
                if block.kind != "text" or not block.chars:
                    continue
                page = doc[block.page - 1]
                if decision.label == "math":
                    block.tex = reader.read(page, block.chars)
                elif decision.label in INLINE_MATH_LABELS:
                    spans = finder.find(block.chars, page.rect.width, page.rect.height)
                    if spans:
                        read = {span: reader.read(page, block.chars[span[0]:span[1]]) for span in spans}
                        block.runs = runs_with_math(block.chars, block.decorations, read)


def predict(pdf_path: str, continues: Continues = rule_continues, math: str = "rule",
            math_checkpoint: str | None = None) -> IRDocument:
    """PDF → ラベル付きIR（オペレータ＋専門家）。評価の登録簿（evaluation/models.json）から呼ぶ。
    math は数式を読む係（"none" / "pdf_text" / "rule" / "ocr" / "rule+ocr"。OCR を使う係は math_checkpoint が要る）"""
    pages = extract_blocks(pdf_path, continues)
    decided = decide_pages(pages)
    fix_code(decided)
    reader = make_reader(math, math_checkpoint)
    if reader is not None:
        read_math(pdf_path, decided, reader)
    labeled = [[(b, d.label) for b, d in items] for items in decided]
    return to_labeled_ir(pages, labeled, os.path.basename(pdf_path), NAME)


def predict_ai(pdf_path: str, math: str = "rule", math_checkpoint: str | None = None,
               inline_finder: str = "auto", checkpoints: str | None = None) -> IRDocument:
    """PDF → ラベル付きIR（AI モード）。区切り・ラベル・文中の数式の専門家を、学習した AI にする。
    重みは checkpoints/（train.py で作る）。inline_finder: "ai" / "rule" / "auto"（AI の重みがあれば AI）。
    checkpoints を渡すと checkpoints/<名前>/ の重みを使う（見た目を抜いて学習した実験の重み。train.py --checkpoints）。
    そこに文中の数式の重み（inline_math.pt）が無ければ、いつもの重みを使う"""
    from hybrid_extractor.ai_experts import CHECKPOINTS, boundary_expert, inline_expert, label_experts

    base = CHECKPOINTS / checkpoints if checkpoints else CHECKPOINTS
    inline_path = base / "inline_math.pt" if (base / "inline_math.pt").exists() else CHECKPOINTS_INLINE
    if inline_finder == "auto":
        inline_finder = "ai" if inline_path.exists() else "rule"

    pages = extract_blocks(pdf_path, boundary_expert(base / "boundary.pt"))
    decided = decide_pages_ai(pages, label_experts(base / "labels"))
    fix_code(decided)
    reader = make_reader(math, math_checkpoint)
    if reader is not None:
        finder = inline_expert(inline_path) if inline_finder == "ai" else INLINE_MATH_EXPERTS["rule"]
        read_math(pdf_path, decided, reader, finder)
    labeled = [[(b, d.label) for b, d in items] for items in decided]
    return to_labeled_ir(pages, labeled, os.path.basename(pdf_path), NAME_AI)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → ラベル付きIR（ルールのオペレータ＋専門家）")
    parser.add_argument("pdf", help="入力PDF")
    parser.add_argument("out", help="出力するラベル付きIRのJSON")
    parser.add_argument("--trace", action="store_true", help="ブロックごとに、どの専門家がどう答えたかを表示する")
    parser.add_argument("--math", default="rule", help="数式を読む係（none / pdf_text / rule / ocr / rule+ocr）")
    parser.add_argument("--math-checkpoint", help="math_ocr の重み（best.pt）。ocr・rule+ocr のときに要る")
    parser.add_argument("--ai", action="store_true", help="AI モード（学習した専門家を使う。checkpoints/ が要る）")
    args = parser.parse_args()
    if args.trace:
        if args.ai:
            from hybrid_extractor.ai_experts import boundary_expert, label_experts
            traced = decide_pages_ai(extract_blocks(args.pdf, boundary_expert()), label_experts())
        else:
            traced = decide_pages(extract_blocks(args.pdf))
        for items in traced:
            for b, d in items:
                ops = " ".join(f"{name}={c:.2f}" for name, c in d.opinions.items())
                print(f"p{b.page} {d.label:9} by={d.by:13} {ops:45} {text_of(b)[:30]!r}")
    if args.ai:
        doc = predict_ai(args.pdf, math=args.math, math_checkpoint=args.math_checkpoint)
    else:
        doc = predict(args.pdf, math=args.math, math_checkpoint=args.math_checkpoint)
    problems = validate(doc)
    if problems:
        raise SystemExit("IRの形式に問題があります:\n" + "\n".join(problems))
    save_ir(doc, args.out)
    print("保存しました:", args.out)


if __name__ == "__main__":
    main()
