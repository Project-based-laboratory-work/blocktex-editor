"""蒸留（nn.distill_loss・train_tagger の teacher、train.py の数式のまとまりの成績）のテスト"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hybrid_extractor.nn import Sequences, distill_loss, predict_tagger, train_tagger  # noqa: E402
from hybrid_extractor.train import block_probs, gt_spans, span_scores  # noqa: E402


def test_distill_loss_is_smallest_when_student_matches_teacher():
    teacher = torch.tensor([3.0, -2.0, 0.5])
    same = distill_loss(teacher.clone(), teacher, temperature=2.0).sum()
    other = distill_loss(torch.tensor([-3.0, 2.0, -0.5]), teacher, temperature=2.0).sum()
    assert same < other
    # 生徒のロジットで微分すると、先生と同じ所で勾配が 0 になる
    student = teacher.clone().requires_grad_(True)
    distill_loss(student, teacher, temperature=2.0).sum().backward()
    assert torch.allclose(student.grad, torch.zeros(3), atol=1e-6)


def test_temperature_softens_the_teacher():
    # 先生のロジット 4（確率 0.98）は、温度 4 で割ると σ(1) = 0.73 になだらかになる
    teacher = torch.tensor([4.0])
    soft = torch.sigmoid(teacher / 4.0)
    assert abs(float(soft) - 0.731) < 1e-3


def tiny_data(n: int, rng: np.random.Generator) -> Sequences:
    """「1つ目の特徴が正なら数式」を当てる、小さな並び"""
    ids, feats, labels = [], [], []
    for _ in range(n):
        t = int(rng.integers(5, 12))
        f = rng.normal(size=(t, 3)).astype(np.float32)
        ids.append(rng.integers(2, 6, size=t))
        feats.append(f)
        labels.append((f[:, 0] > 0).astype(np.float32))
    return Sequences(ids, feats, labels)


def test_train_tagger_with_teacher_learns():
    rng = np.random.default_rng(0)
    train, val = tiny_data(400, rng), tiny_data(60, rng)
    # 先生のロジット: 正解の向きに ±3
    teacher = [np.where(y > 0, 3.0, -3.0).astype(np.float32) for y in train.labels]
    model, std, history = train_tagger(train, val, vocab_size=8, epochs=40, log=False, teacher=teacher,
                                       temperature=2.0, alpha=0.5, d_model=16, layers=1, heads=2, max_len=16)
    prob = np.concatenate(predict_tagger(model, std, val))
    acc = ((prob >= 0.5) == (np.concatenate(val.labels) > 0)).mean()
    assert acc >= 0.9
    logits = predict_tagger(model, std, val, logits=True)
    assert all(len(z) == len(i) for z, i in zip(logits, val.ids))


def test_span_scores_and_block_probs():
    def ch(c, mid=None):
        return {"c": c, "bbox": None if c == " " else (0, 0, 1, 1), "size": 10.0, "math_id": mid}

    # 「a x = y b」の x = y が1つの数式
    chars = [ch("a"), ch(" "), ch("x", 0), ch(" ", 0), ch("=", 0), ch(" ", 0), ch("y", 0), ch(" "), ch("b")]
    docs = [{"blocks": [{"chars": chars}]}]
    assert gt_spans(chars) == [(2, 7)]
    good = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)
    assert block_probs(docs, [good]) [0].tolist() == good.tolist()
    assert span_scores(docs, [good], 0.5)["f1"] == 1.0
    none = np.zeros(9)
    assert span_scores(docs, [none], 0.5)["recall"] == 0.0
