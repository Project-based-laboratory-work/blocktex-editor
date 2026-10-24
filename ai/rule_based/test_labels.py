from rule_based.labels import LabeledBlock, load_labels, save_labels, to_coarse, validate

# --- 正しいデータ ---
blocks = [
    LabeledBlock(page=0, bbox=(72.0, 100.0, 520.0, 130.0), label="heading", text="1. はじめに", source="manual"),
    LabeledBlock(page=0, bbox=(72.0, 140.0, 520.0, 300.0), label="paragraph", text="本研究では…", source="auto"),
    LabeledBlock(page=0, bbox=(100.0, 320.0, 400.0, 500.0), label="table", text=None, source="auto"),
]

print("--- 1. validate（正しいデータ）---")
print(validate(blocks))  # [] になるはず

print("--- 2. 保存と読み込み ---")
save_labels("sample.pdf", blocks, "test.json")
name, loaded = load_labels("test.json", check=True)
print(name)              # sample.pdf
print(loaded == blocks)  # True になるはず

print("--- 3. to_coarse ---")
print(to_coarse("heading"))  # heading
print(to_coarse("list"))     # paragraph

print("--- 4. validate（壊れたデータ）---")
broken = [
    LabeledBlock(page=0, bbox=(520.0, 100.0, 72.0, 130.0), label="headding", text="x", source="manual"),
    LabeledBlock(page=1, bbox=(72.0, 100.0, 520.0), label="caption", text="y", source="maual"),
]
for p in validate(broken):
    print(p)

print("--- 5. to_coarse（不明なラベル）---")
try:
    to_coarse("headding")
except ValueError as e:
    print("期待どおりエラー:", e)