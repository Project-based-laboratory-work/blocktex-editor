"""
学習スクリプト。

    python -m math_ocr.train --data data/math_synth --epochs 10
    python -m math_ocr.train --data data/math_synth --overfit-batch   # 学習ループが正しいかの確認
    python -m math_ocr.train --data data/math_synth --epochs 10 --resume   # 中断した学習を last.pt から再開

学習1ステップの流れ（PyTorchの学習ループはほぼ必ずこの形になる）:
  1. 順伝播 (forward):   logits = model(入力)
  2. 損失の計算:          loss = criterion(logits, 正解)
  3. 勾配を初期化:        optimizer.zero_grad()   ← 忘れると前のステップの勾配に足し込まれる
  4. 逆伝播 (backward):   loss.backward()         ← 全パラメータの .grad に勾配が入る
  5. パラメータ更新:      optimizer.step()

teacher forcing:
  正解の列 [<bos>, \\frac, {, a, }, ..., <eos>] を1つずらして入力と目標に分ける。
    tgt_in  = [<bos>, \\frac, {,  a, ...]    ← デコーダに入力する
    tgt_out = [\\frac, {,     a,  }, ...]    ← 各位置で予測してほしい「次のトークン」
  学習時は、モデル自身の（間違っているかもしれない）出力ではなく正解を入力に使う。こうすると全位置を
  並列に一度に学習できる。
"""

import argparse
import itertools
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from math_ocr.dataset import Collator, MathFormulaDataset, read_labels
from math_ocr.metrics import evaluate
from math_ocr.model import MathOCRModel, ModelConfig, count_parameters
from math_ocr.tokenizer import Vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/math_synth"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/math_ocr"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=None, help="学習データの件数を絞る（お試し用）")
    parser.add_argument("--eval-samples", type=int, default=200, help="毎エポック生成して評価する検証データの件数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overfit-batch", action="store_true", help="1バッチだけを繰り返し学習する")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="CPUで使うスレッド数（多すぎると逆に遅くなることがある）")
    parser.add_argument("--save-every", type=int, default=200, help="何ステップごとに last.pt を保存するか")
    parser.add_argument("--resume", action="store_true", help="out/last.pt から学習を再開する")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine(total_steps: int, warmup_steps: int):
    """学習率の倍率を返す関数。最初は0から徐々に上げ（warmup）、その後cosのカーブで0に向けて下げる。

    Transformerは学習の序盤に大きな学習率を当てると発散しやすいので、warmupを入れるのが定番。
    """

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def compute_loss(model: MathOCRModel, criterion: nn.Module, images: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    tgt_in, tgt_out = seqs[:, :-1], seqs[:, 1:]  # teacher forcing のための1つずらし
    logits = model(images, tgt_in)  # (B, T, V)
    # CrossEntropyLoss は (N, V) と (N,) を受け取るので、バッチと時刻の次元をまとめて平らにする
    return criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))


@torch.no_grad()
def validate(
    model: MathOCRModel, criterion: nn.Module, loader: DataLoader, vocab: Vocab, device: torch.device, eval_samples: int
) -> dict[str, float]:
    # eval(): Dropoutを無効にし、BatchNormを学習中に溜めた統計値で動かすモード。推論・評価の前に必ず切り替える
    model.eval()
    total_loss, batches = 0.0, 0
    predictions, references = [], []
    for images, seqs in loader:
        images, seqs = images.to(device), seqs.to(device)
        total_loss += compute_loss(model, criterion, images, seqs).item()
        batches += 1
        if len(references) < eval_samples:
            generated = model.generate(images)
            predictions += [vocab.ids_to_tokens(row.tolist()[1:]) for row in generated]
            references += [vocab.ids_to_tokens(row.tolist()[1:]) for row in seqs]
    model.train()
    return {"val_loss": total_loss / batches, **evaluate(predictions, references)}


def show_examples(model: MathOCRModel, loader: DataLoader, vocab: Vocab, device: torch.device, n: int = 3) -> None:
    model.eval()
    images, seqs = next(iter(loader))
    generated = model.generate(images[:n].to(device))
    for pred, ref in zip(generated, seqs[:n]):
        print(f"  正解: {vocab.decode(ref.tolist()[1:])}")
        print(f"  予測: {vocab.decode(pred.tolist()[1:])}")
    model.train()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device()
    args.out.mkdir(parents=True, exist_ok=True)

    # 語彙は学習データだけから作る（検証データを覗かないため）
    vocab = Vocab.build([latex for _, latex in read_labels(args.data, "train")])
    vocab.save(args.out / "vocab.json")

    train_ds = MathFormulaDataset(args.data, "train", vocab, limit=args.limit_train)
    val_ds = MathFormulaDataset(args.data, "val", vocab)
    collate = Collator(vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers)

    config = ModelConfig(
        vocab_size=len(vocab), pad_id=vocab.pad_id, bos_id=vocab.bos_id, eos_id=vocab.eos_id,
        d_model=args.d_model, dim_feedforward=4 * args.d_model,  # FFNの中間層は d_model の4倍にするのが定番
    )  # fmt: skip
    model = MathOCRModel(config).to(device)
    print(f"device={device}  語彙数={len(vocab)}  パラメータ数={count_parameters(model):,}  学習データ={len(train_ds)}件")

    # ignore_index: <pad> の位置は損失に含めない（埋め草を当てても意味が無い）
    # label_smoothing: 正解に100%ではなく90%の確率を割り当てて学習させ、自信過剰になるのを防ぐ
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if args.overfit_batch:
        overfit_one_batch(model, criterion, optimizer, train_loader, vocab, device)
        return

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, warmup_steps=min(1000, total_steps // 10))
    )

    step, best_ter = 0, float("inf")
    if args.resume:
        # モデルだけでなく、optimizer（AdamWが溜めている勾配の移動平均）とschedulerの状態も戻さないと、
        # 再開した直後に学習率や更新の大きさが変わってしまう
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_ter = state["step"], state["best_ter"]
        print(f"last.pt から再開: step {step}/{total_steps}")

    def save_last() -> None:
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "best_ter": best_ter, "config": asdict(config), "vocab": vocab.itos,
            },
            args.out / "last.pt",
        )  # fmt: skip

    for epoch in range(step // steps_per_epoch + 1, args.epochs + 1):
        model.train()  # Dropout・BatchNormを学習モードにする
        started = time.time()
        # 途中から再開したエポックは、残りのステップ数だけ回す
        remaining = epoch * steps_per_epoch - step
        for images, seqs in itertools.islice(train_loader, remaining):
            images, seqs = images.to(device), seqs.to(device)
            loss = compute_loss(model, criterion, images, seqs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # 勾配の大きさ（ノルム）が1を超えたら縮める。たまに来る巨大な勾配で学習が壊れるのを防ぐ
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e}")
            if step % args.save_every == 0:
                save_last()

        metrics = validate(model, criterion, val_loader, vocab, device, args.eval_samples)
        elapsed = time.time() - started
        print(
            f"[epoch {epoch}] {elapsed:.0f}s  val_loss {metrics['val_loss']:.4f}  "
            f"exact_match {metrics['exact_match']:.3f}  TER {metrics['token_error_rate']:.3f}"
        )
        show_examples(model, val_loader, vocab, device)

        if metrics["token_error_rate"] < best_ter:
            best_ter = metrics["token_error_rate"]
            torch.save(
                {"model": model.state_dict(), "config": asdict(config), "vocab": vocab.itos, "metrics": metrics},
                args.out / "best.pt",
            )
            print(f"  → best.pt を保存（TER {best_ter:.3f}）")
        save_last()


def overfit_one_batch(model, criterion, optimizer, loader, vocab, device, steps: int = 300) -> None:
    """1バッチだけを繰り返し学習させ、ほぼ完璧に暗記できるかを確認する。

    モデルの表現力は十分なはずなので、最後の exact_match が1.0近くにならなければ、学習ループのどこか
    （1つずらし、マスク、ignore_index など）にバグがある。新しいモデルを書いたら最初にやるべき確認。
    （label_smoothing を入れているので、暗記できても損失は0にはならず0.5〜1程度で止まる）
    """
    images, seqs = next(iter(loader))
    images, seqs = images[:16].to(device), seqs[:16].to(device)
    model.train()
    for step in range(1, steps + 1):
        loss = compute_loss(model, criterion, images, seqs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    model.eval()
    generated = model.generate(images)
    preds = [vocab.ids_to_tokens(row.tolist()[1:]) for row in generated]
    refs = [vocab.ids_to_tokens(row.tolist()[1:]) for row in seqs]
    print(evaluate(preds, refs))


if __name__ == "__main__":
    main()
