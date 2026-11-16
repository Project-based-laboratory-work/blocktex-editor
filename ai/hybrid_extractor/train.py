"""自作AIの専門家を学習させ、ルール版と比べる。

    python -m hybrid_extractor.train boundary      # 区切りの専門家（2行は続きか）
    python -m hybrid_extractor.train labels        # ラベルの専門家（箇条書き・コード・見出し…を1人ずつ）
    python -m hybrid_extractor.train inline        # 文中の数式の専門家（1文字ずつ数式か）

どれも:
    1. 学習用データ（data/train など）から特徴（features.py）と正解を作る
    2. 学習し、検証用データ（data/val など）の損失が一番小さいときの重みを残す（nn.py）
    3. テスト用データ（data/test など。学習にも重みの選択にも使っていない）で、ルール版と同じ物差しで比べる
    4. 重みを checkpoints/ に保存する（git の管理外）。オペレータ（operator_rules.py の predict_ai）がここから読む
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from hybrid_extractor.features import (
    BLOCK_FEATURE_NAMES,
    CHAR_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    block_char_size,
    block_features,
    char_features,
    doc_info,
    pair_features,
)
from hybrid_extractor.nn import (
    Sequences,
    binary_scores,
    predict_binary,
    predict_tagger,
    save_checkpoint,
    train_binary,
    train_tagger,
)

DATA = Path(__file__).parent / "data"
CHECKPOINTS = Path(__file__).parent / "checkpoints"
# ラベルの専門家を作るラベル。title / author / date はルール、表・図の領域（kind が table / figure）はそのまま。
# 段落にも専門家を作る（作らないと、適合率の低い専門家（箇条書きなど）に段落が負けてしまう。テストで 0.959 → 0.979）。
# どの専門家も 0.5 に届かなければ段落
LABEL_TARGETS = ["paragraph", "list", "code", "heading", "math", "table", "caption", "other"]
SURE_LABELS = ("title", "author", "date")   # ルールで決める（1ページ目のタイトル部分。ルールで正解率 1.0）
THRESHOLD = 0.5
MAX_LEN = 512                                # 文中の数式の専門家が一度に見る文字の数（長いブロックは区切る）


def load_docs(split: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DATA / split / "docs").glob("*.json"))]


def show(title: str, scores: dict) -> str:
    return (f"{title:22} 正解率 {scores['acc']:.4f}  適合率 {scores['precision']:.4f}  "
            f"再現率 {scores['recall']:.4f}  F1 {scores['f1']:.4f}  ({scores['n']}件)")


# =====================================================================
# 区切りの専門家
# =====================================================================


def boundary_examples(docs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """隣り合う2行の組ごとに、特徴・正解（続きか）・ルールの答え。
    「今のブロックの1行目」は正解のつながり方でたどる（使うときは、専門家自身がつないだブロックの1行目になる）"""
    x, y, rule = [], [], []
    for doc in docs:
        for page in doc["pages"]:
            rows, ctx = page["rows"], page["ctx"]
            first = 0
            for pair in page["pairs"]:
                if pair["same"] is not None:
                    x.append(pair_features(rows[first], rows[pair["prev"]], rows[pair["next"]], ctx))
                    y.append(float(pair["same"]))
                    rule.append(float(pair["rule_same"]))
                if not pair["same"]:
                    first = pair["next"]  # 続きでなければ、次の行から新しいブロック
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(rule, dtype=np.float32)


def cmd_boundary(args: argparse.Namespace) -> None:
    print("区切りの専門家: 特徴を作る")
    x, y, _ = boundary_examples(load_docs("train"))
    xv, yv, _ = boundary_examples(load_docs("val"))
    xt, yt, rt = boundary_examples(load_docs("test"))
    print(f"  学習 {len(y)}組（続き {y.mean():.1%}）、検証 {len(yv)}組、テスト {len(yt)}組、特徴 {x.shape[1]}個")
    model, std, history = train_binary(x, y, xv, yv, hidden=args.hidden, epochs=args.epochs)
    ai = binary_scores(predict_binary(model, std, xt), yt)
    rule = binary_scores(rt, yt)
    print("テスト用データでの比較（はい = 続き）")
    print("  " + show("ルール版", rule))
    print("  " + show("AI版", ai))
    path = CHECKPOINTS / "boundary.pt"
    save_checkpoint(path, model, std, {"n_in": x.shape[1], "hidden": args.hidden, "features": PAIR_FEATURE_NAMES},
                    {"test": {"ai": ai, "rule": rule}, "history": history})
    print(f"保存した: {path}")


# =====================================================================
# ラベルの専門家
# =====================================================================


def label_examples(docs: list[dict]) -> dict[str, np.ndarray]:
    """正解の付いたブロックごとに、特徴・正解のラベル・ルールの答え・領域の種類（kind）"""
    x, labels, rules, kinds = [], [], [], []
    for doc in docs:
        pages = doc["pages"]
        info = doc_info(pages)
        for page in pages:
            blocks = page["blocks"]
            for i, b in enumerate(blocks):
                if b["label"] is None:
                    continue
                x.append(block_features(blocks, i, page, info))
                labels.append(b["label"])
                rules.append(b["rule_label"])
                kinds.append(b["kind"])
    return {"x": np.array(x, dtype=np.float32), "labels": np.array(labels), "rules": np.array(rules),
            "kinds": np.array(kinds)}


def operator_decide(probs: dict[str, np.ndarray], data: dict[str, np.ndarray]) -> np.ndarray:
    """AIのオペレータの決め方を、学習データの上でまねる（operator_rules.decide_pages の AI モードと同じ順番）:
    表・図の領域 → そのまま、タイトル部分 → ルール、それ以外 → 一番自信のある専門家（0.5 未満なら段落）"""
    names = list(probs)
    matrix = np.stack([probs[n] for n in names], axis=1)
    out = []
    for k in range(len(data["labels"])):
        if data["kinds"][k] in ("table", "figure"):
            out.append(data["kinds"][k])
        elif data["rules"][k] in SURE_LABELS:
            out.append(data["rules"][k])
        else:
            best = int(matrix[k].argmax())
            out.append(names[best] if matrix[k, best] >= THRESHOLD else "paragraph")
    return np.array(out)


def cmd_labels(args: argparse.Namespace) -> None:
    print("ラベルの専門家: 特徴を作る")
    train, val, test = (label_examples(load_docs(s)) for s in ("train", "val", "test"))
    print(f"  学習 {len(train['labels'])}、検証 {len(val['labels'])}、テスト {len(test['labels'])}ブロック、"
          f"特徴 {train['x'].shape[1]}個")
    probs = {}
    summary = {}
    for target in LABEL_TARGETS:
        print(f"\n[{target}] の専門家（学習用の「はい」{(train['labels'] == target).sum()}件）")
        y = (train["labels"] == target).astype(np.float32)
        yv = (val["labels"] == target).astype(np.float32)
        model, std, history = train_binary(train["x"], y, val["x"], yv, hidden=args.hidden, epochs=args.epochs,
                                           log=args.verbose)
        yt = (test["labels"] == target).astype(np.float32)
        probs[target] = predict_binary(model, std, test["x"])
        ai = binary_scores(probs[target], yt)
        rule = binary_scores((test["rules"] == target).astype(np.float32), yt)
        print("  " + show("ルール版", rule))
        print("  " + show("AI版", ai))
        summary[target] = {"ai": ai, "rule": rule}
        save_checkpoint(CHECKPOINTS / "labels" / f"{target}.pt", model, std,
                        {"n_in": train["x"].shape[1], "hidden": args.hidden, "label": target,
                         "features": BLOCK_FEATURE_NAMES}, {"test": summary[target], "epochs": len(history)})

    decided = operator_decide(probs, test)
    labels = test["labels"]
    ai_acc = float((decided == labels).mean())
    rule_acc = float((test["rules"] == labels).mean())
    print(f"\nオペレータで決めたラベルの正解率（テスト用データ）: ルール版 {rule_acc:.4f} → AI版 {ai_acc:.4f}")
    print(f"  {'ラベル':10} {'件数':>6} {'ルール':>8} {'AI':>8}")
    for label, n in Counter(labels.tolist()).most_common():
        mask = labels == label
        print(f"  {label:10} {n:6d} {float((test['rules'][mask] == label).mean()):8.3f} "
              f"{float((decided[mask] == label).mean()):8.3f}")
    (CHECKPOINTS / "labels" / "summary.json").write_text(json.dumps(
        {"experts": summary, "operator": {"ai_acc": ai_acc, "rule_acc": rule_acc}}, ensure_ascii=False, indent=2),
        encoding="utf-8")


# =====================================================================
# 文中の数式の専門家
# =====================================================================


def char_vocab(docs: list[dict], min_count: int) -> dict[str, int]:
    """文字 → ID。0 は埋め草、1 は「知らない文字」。学習用データで min_count 回以上出た文字だけに ID を付ける"""
    counts = Counter(ch["c"] for doc in docs for b in doc["blocks"] for ch in b["chars"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for c, n in counts.most_common():
        if n >= min_count:
            vocab[c] = len(vocab)
    return vocab


def inline_sequences(docs: list[dict], vocab: dict[str, int]) -> tuple[Sequences, np.ndarray]:
    """ブロックの文字の並びを MAX_LEN 文字ずつに区切った並びと、ルール版の答え（1文字ずつ、全部つなげたもの）"""
    ids, feats, labels, rules = [], [], [], []
    for doc in docs:
        for b in doc["blocks"]:
            chars = b["chars"]
            f = np.array(char_features(chars, block_char_size(chars), b["page_width"], b["page_height"]),
                         dtype=np.float32)
            i = np.array([vocab.get(ch["c"], 1) for ch in chars])
            y = np.array([ch["math"] for ch in chars], dtype=np.float32)
            for k in range(0, len(chars), MAX_LEN):
                ids.append(i[k:k + MAX_LEN])
                feats.append(f[k:k + MAX_LEN])
                labels.append(y[k:k + MAX_LEN])
            rules += [ch["rule"] for ch in chars]
    return Sequences(ids, feats, labels), np.array(rules, dtype=np.float32)


def cmd_inline(args: argparse.Namespace) -> None:
    print("文中の数式の専門家: 特徴を作る")
    train_docs = load_docs("inline_train")
    vocab = char_vocab(train_docs, args.min_count)
    train, _ = inline_sequences(train_docs, vocab)
    val, _ = inline_sequences(load_docs("inline_val"), vocab)
    test, rule = inline_sequences(load_docs("inline_test"), vocab)
    n_chars = sum(len(s) for s in train.ids)
    print(f"  学習 {len(train.ids)}並び・{n_chars}文字、語彙 {len(vocab)}、特徴 {train.feats[0].shape[1]}個")
    model_args = {"d_model": args.d_model, "layers": args.layers, "heads": 4, "max_len": MAX_LEN}
    model, std, history = train_tagger(train, val, len(vocab), epochs=args.epochs, **model_args)
    prob = np.concatenate(predict_tagger(model, std, test))
    yt = np.concatenate(test.labels)
    ai = binary_scores(prob, yt)
    rule_scores = binary_scores(rule, yt)
    print("テスト用データでの比較（1文字ずつ。はい = 数式の文字）")
    print("  " + show("ルール版", rule_scores))
    print("  " + show("AI版", ai))
    path = CHECKPOINTS / "inline_math.pt"
    save_checkpoint(path, model, std, {"vocab_size": len(vocab), "n_feat": train.feats[0].shape[1], **model_args,
                                       "features": CHAR_FEATURE_NAMES},
                    {"vocab": vocab, "test": {"ai": ai, "rule": rule_scores}, "history": history})
    print(f"保存した: {path}")
    print("見つける段階の比較は: python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math "
          "--docs 60 --finder ai")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="自作AIの専門家を学習させ、ルール版と比べる")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("boundary", help="区切りの専門家")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.set_defaults(func=cmd_boundary)
    p = sub.add_parser("labels", help="ラベルの専門家")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--verbose", action="store_true", help="エポックごとの損失を表示する")
    p.set_defaults(func=cmd_labels)
    p = sub.add_parser("inline", help="文中の数式の専門家")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-count", type=int, default=5)
    p.set_defaults(func=cmd_inline)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
