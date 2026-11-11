"""オペレータ（operator_rules.py）と専門家（experts.py）のテスト"""
from hybrid_extractor import operator_rules
from hybrid_extractor.experts import RULE_EXPERTS, DocInfo, Expert
from hybrid_extractor.extract import Block
from hybrid_extractor.operator_rules import choose_experts, decide_block

INFO = DocInfo(body_size=10.0, body_font="Mincho")


def block(text: str, **kw) -> Block:
    return Block(page=1, kind="text", bbox=(0, 0, 100, 10), text=text, font_size=kw.pop("font_size", 10.0),
                 font=kw.pop("font", "Mincho"), n_rows=kw.pop("n_rows", 1), **kw)


def test_rule_experts():
    assert RULE_EXPERTS["code"].judge(block("x = 1", mono_ratio=0.9), INFO) == 1.0
    assert RULE_EXPERTS["math"].judge(block("a + b", math_ratio=0.5), INFO) == 1.0
    assert RULE_EXPERTS["list"].judge(block("• 項目", list_head=True), INFO) == 1.0
    assert RULE_EXPERTS["heading"].judge(block("はじめに", font_size=14.0), INFO) == 1.0     # 本文より大きい
    assert RULE_EXPERTS["heading"].judge(block("1.2 方法", font="Gothic"), INFO) == 1.0     # 番号＋本文と違う書体
    assert RULE_EXPERTS["heading"].judge(block("本文の文です。"), INFO) == 0.0


def test_choose_experts_skips_by_cheap_cues():
    # 等幅・数式のフォントが1文字も無ければ、コード・数式の専門家には聞かない
    names = [e.name for e in choose_experts(block("ふつうの文"))]
    assert names == ["list_rule", "heading_rule"]
    # 長いブロックには見出しの専門家を呼ばない
    names = [e.name for e in choose_experts(block("あ" * 200, n_rows=5, mono_ratio=0.1))]
    assert names == ["code_rule", "list_rule"]


def test_decide_block_default_paragraph():
    d = decide_block(block("本文の文です。"), INFO)
    assert (d.label, d.by) == ("paragraph", "default")


def test_decide_block_most_confident_wins(monkeypatch):
    # 2人の専門家が手を挙げたら、自信の高い方
    weak = Expert("weak", "heading", lambda b, i: 0.6)
    strong = Expert("strong", "list", lambda b, i: 0.9)
    monkeypatch.setattr(operator_rules, "LABEL_EXPERTS", [weak, strong])
    d = decide_block(block("★ 項目"), INFO)
    assert (d.label, d.by) == ("list", "strong")
    assert d.opinions == {"weak": 0.6, "strong": 0.9}


def test_decide_block_below_threshold(monkeypatch):
    # 誰も THRESHOLD（0.5）に届かなければ段落
    monkeypatch.setattr(operator_rules, "LABEL_EXPERTS", [Expert("unsure", "list", lambda b, i: 0.3)])
    assert decide_block(block("★ 項目"), INFO).label == "paragraph"
