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


# どのデータで学習し、どこに保存するか（--data-prefix / --checkpoints。見た目を抜いた実験で、今の重みを上書きしないため）
SETTINGS = {"prefix": "", "checkpoints": ""}


def load_docs(split: str) -> list[dict]:
    """data/<prefix><split>/docs の文書。prefix は --data-prefix（例: xtimes_ → data/xtimes_train）"""
    folder = DATA / (SETTINGS["prefix"] + split) / "docs"
    if not folder.exists():
        raise SystemExit(f"学習データが無い: {folder}")
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]


def ckpt_dir() -> Path:
    """重みの保存先。--checkpoints を指定すると checkpoints/<名前>/（今の重みとは別の場所）"""
    return CHECKPOINTS / SETTINGS["checkpoints"] if SETTINGS["checkpoints"] else CHECKPOINTS


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
    path = ckpt_dir() / "boundary.pt"
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
        save_checkpoint(ckpt_dir() / "labels" / f"{target}.pt", model, std,
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
    (ckpt_dir() / "labels" / "summary.json").write_text(json.dumps(
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
    model, std, history = train_tagger(train, val, len(vocab), epochs=args.epochs, seed=args.seed, **model_args)
    prob = np.concatenate(predict_tagger(model, std, test))
    yt = np.concatenate(test.labels)
    ai = binary_scores(prob, yt)
    rule_scores = binary_scores(rule, yt)
    print("テスト用データでの比較（1文字ずつ。はい = 数式の文字）")
    print("  " + show("ルール版", rule_scores))
    print("  " + show("AI版", ai))
    path = ckpt_dir() / args.out
    save_checkpoint(path, model, std, {"vocab_size": len(vocab), "n_feat": train.feats[0].shape[1], **model_args,
                                       "features": CHAR_FEATURE_NAMES, "seed": args.seed},
                    {"vocab": vocab, "test": {"ai": ai, "rule": rule_scores}, "history": history})
    print(f"保存した: {path}")
    print("見つける段階の比較は: python -m hybrid_extractor.inline_math_eval --out hybrid_extractor/data/inline_math "
          "--docs 60 --finder ai")


# =====================================================================
# 蒸留（文中の数式の専門家）
# =====================================================================

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def block_probs(docs: list[dict], seq_values: list[np.ndarray]) -> list[np.ndarray]:
    """inline_sequences で MAX_LEN 文字ずつに区切った並びの値を、ブロックごとにつなぎ直す（同じ順に並んでいる）"""
    out, k = [], 0
    for doc in docs:
        for b in doc["blocks"]:
            n = (len(b["chars"]) + MAX_LEN - 1) // MAX_LEN
            out.append(np.concatenate(seq_values[k:k + n]) if n else np.zeros(0))
            k += n
    return out


def gt_spans(chars: list[dict]) -> list[tuple[int, int]]:
    """正解の数式の範囲: 同じ math_id が続くところ"""
    spans, k = [], 0
    while k < len(chars):
        mid = chars[k].get("math_id")
        if mid is None:
            k += 1
            continue
        start = k
        while k < len(chars) and chars[k].get("math_id") == mid:
            k += 1
        spans.append((start, k))
    return spans


def span_scores(docs: list[dict], probs: list[np.ndarray], threshold: float) -> dict:
    """数式のまとまりの成績。予測の範囲（spans_from_probs、実際の専門家と同じ後処理）と正解の範囲を、
    文字の重なり（共通 ÷ 合わせた範囲）が 0.5 以上なら1対1で対応させる"""
    from hybrid_extractor.ai_experts import spans_from_probs

    n_gt = n_pred = n_pair = 0
    k = 0
    for doc in docs:
        for b in doc["blocks"]:
            chars = b["chars"]
            gts = gt_spans(chars)
            preds = spans_from_probs(chars, probs[k], threshold) if len(chars) else []
            k += 1
            used: set[int] = set()
            for gs, ge in gts:
                for j, (ps, pe) in enumerate(preds):
                    inter = max(0, min(ge, pe) - max(gs, ps))
                    if j not in used and inter / (max(ge, pe) - min(gs, ps)) >= 0.5:
                        used.add(j)
                        n_pair += 1
                        break
            n_gt += len(gts)
            n_pred += len(preds)
    precision = n_pair / n_pred if n_pred else 0.0
    recall = n_pair / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def evaluate_tagger(name: str, val_docs, val_logits, test_docs, test_logits, test_labels) -> dict:
    """1文字ずつの成績（しきい値 0.5）と、数式のまとまりの成績（しきい値は検証用データで一番良いもの）"""
    val_probs = block_probs(val_docs, [sigmoid(z) for z in val_logits])
    best = max(THRESHOLDS, key=lambda t: span_scores(val_docs, val_probs, t)["f1"])
    char = binary_scores(sigmoid(np.concatenate(test_logits)), test_labels)
    span = span_scores(test_docs, block_probs(test_docs, [sigmoid(z) for z in test_logits]), best)
    print(f"  {name:22} 1文字 F1 {char['f1']:.4f}   まとまり F1 {span['f1']:.4f}"
          f"（適合率 {span['precision']:.3f} 再現率 {span['recall']:.3f}、しきい値 {best}）")
    return {"char": char, "span": span, "threshold": best}


def cmd_distill(args: argparse.Namespace) -> None:
    """先生（同じ形のモデルを何人か。平均のロジット = アンサンブル）の答えを使って、生徒を学習する（蒸留）"""
    from hybrid_extractor.ai_experts import load_tagger_with_extra

    teachers = [load_tagger_with_extra(ckpt_dir() / name) for name in args.teachers]
    vocab = teachers[0][2]["vocab"]
    if any(t[2]["vocab"] != vocab for t in teachers):
        raise SystemExit("先生どうしで語彙（文字の ID）が違う。同じ学習データで作った先生を使う")
    print(f"蒸留: 先生 {len(teachers)}人（{', '.join(args.teachers)}）、温度 T={args.temperature}、alpha={args.alpha}")
    docs = {split: load_docs(split) for split in ("inline_train", "inline_val", "inline_test")}
    seqs = {split: inline_sequences(d, vocab)[0] for split, d in docs.items()}
    _, rule = inline_sequences(docs["inline_test"], vocab)

    def teacher_logits(split: str) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        each = [predict_tagger(model, std, seqs[split], logits=True) for model, std, _ in teachers]
        mean = [np.mean([e[k] for e in each], axis=0) for k in range(len(seqs[split].ids))]
        return each, mean

    print("  先生の答えを計算する")
    _, train_teacher = teacher_logits("inline_train")
    val_each, val_mean = teacher_logits("inline_val")
    test_each, test_mean = teacher_logits("inline_test")

    model_args = {"d_model": args.d_model, "layers": args.layers, "heads": 4, "max_len": MAX_LEN}
    print(f"  生徒を学習する（d_model {args.d_model}、{args.layers}層、seed {args.seed}）")
    model, std, history = train_tagger(seqs["inline_train"], seqs["inline_val"], len(vocab), epochs=args.epochs,
                                       seed=args.seed, teacher=train_teacher, temperature=args.temperature,
                                       alpha=args.alpha, **model_args)
    student_val = predict_tagger(model, std, seqs["inline_val"], logits=True)
    student_test = predict_tagger(model, std, seqs["inline_test"], logits=True)

    yt = np.concatenate(seqs["inline_test"].labels)
    print("テスト用データでの比較")
    rule_char = binary_scores(rule, yt)
    print(f"  {'ルール版':22} 1文字 F1 {rule_char['f1']:.4f}")
    results = {}
    for name, v, t in zip(args.teachers, val_each, test_each):
        results[name] = evaluate_tagger(f"先生 {name}", docs["inline_val"], v, docs["inline_test"], t, yt)
    results["ensemble"] = evaluate_tagger("先生の平均（アンサンブル）", docs["inline_val"], val_mean,
                                          docs["inline_test"], test_mean, yt)
    results["student"] = evaluate_tagger("生徒（蒸留）", docs["inline_val"], student_val,
                                         docs["inline_test"], student_test, yt)
    path = ckpt_dir() / args.out
    save_checkpoint(path, model, std, {"vocab_size": len(vocab), "n_feat": seqs["inline_train"].feats[0].shape[1],
                                       **model_args, "features": CHAR_FEATURE_NAMES, "seed": args.seed,
                                       "distill": {"teachers": args.teachers, "temperature": args.temperature,
                                                   "alpha": args.alpha}},
                    {"vocab": vocab, "threshold": results["student"]["threshold"], "test": results, "history": history})
    print(f"保存した: {path}（しきい値 {results['student']['threshold']} も一緒に保存）")
    (ckpt_dir() / (Path(args.out).stem + "_report.json")).write_text(
        json.dumps({"args": vars(args) | {"func": None}, "rule": rule_char, "results": results}, ensure_ascii=False,
                   indent=2, default=float), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="自作AIの専門家を学習させ、ルール版と比べる")
    parser.add_argument("--data-prefix", default="", help="学習データのフォルダ名の頭（例: xtimes_ → data/xtimes_train など）")
    parser.add_argument("--checkpoints", default="", help="重みを checkpoints/<名前>/ に保存する（今の重みを上書きしない）")
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
    p.add_argument("--seed", type=int, default=0, help="乱数の種（蒸留の先生を何人か作るときに変える）")
    p.add_argument("--out", default="inline_math.pt", help="checkpoints/ の中の保存先のファイル名")
    p.set_defaults(func=cmd_inline)
    p = sub.add_parser("distill", help="文中の数式の専門家を蒸留する（先生の答えで生徒を学習する）")
    p.add_argument("--teachers", nargs="+", required=True, help="checkpoints/ の中の先生の重み（同じ学習データで作ったもの）")
    p.add_argument("--temperature", type=float, default=2.0, help="温度 T。大きいほど先生の確率をなだらかにする")
    p.add_argument("--alpha", type=float, default=0.5, help="正解との損失の割合（残りが先生との損失）")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=3, help="先生と違う種にする")
    p.add_argument("--out", default="inline_student.pt")
    p.set_defaults(func=cmd_distill)
    args = parser.parse_args(argv)
    SETTINGS.update(prefix=args.data_prefix, checkpoints=args.checkpoints)
    args.func(args)


if __name__ == "__main__":
    main()
