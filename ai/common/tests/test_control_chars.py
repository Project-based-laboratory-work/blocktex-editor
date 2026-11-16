"""PDFから取り出した文字に混ざる制御文字（common/control_chars.py）のテスト"""
from common.blocks import ir_to_document
from common.control_chars import fix_char, fix_text
from common.ir import Element, IRDocument, Page


def test_fix_text_ligatures_bullet_and_others():
    # T1 の書体（Type3）の合字 ff は 0x1B、TS1 の \textbullet は 0x88 で出てくる
    assert fix_text("関数didi\x1b である") == "関数didiff である"
    assert fix_text("\x88 入力：関数f") == "• 入力：関数f"
    assert fix_text("e\x0ecient") == "efficient"      # OT1 の合字 ffi
    assert fix_text("a\x01b\x7fc") == "abc"            # 戻せない制御文字は捨てる
    assert fix_text("1行目\n\t2行目") == "1行目\n\t2行目"  # 改行とタブは残す
    assert fix_char("x") == "x"


def test_editor_document_has_no_control_chars():
    page = Page(page=1, width=595, height=842, unit="pt", elements=[
        Element(kind="text", bbox=(0.1, 0.1, 0.9, 0.12), text="二回微分を求める関数didi\x1b", label="paragraph"),
        Element(kind="text", bbox=(0.1, 0.2, 0.9, 0.22), text="x = di\x1b(f)\x02", label="code"),
    ])
    document = ir_to_document(IRDocument(source="in.pdf", extractor="t", pages=[page], structurer="t"))
    paragraph, code = document["blocks"]
    assert paragraph["content"][0]["text"] == "二回微分を求める関数didiff"
    assert code["code"] == "x = diff(f)"
