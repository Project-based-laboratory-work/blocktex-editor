import json
from pathlib import Path
from jiwer import cer

# 正解テキスト
ground_truth_path = Path("data/ground_truth/doc001.txt")
ground_truth = ground_truth_path.read_text(encoding="utf-8")

# OCRの結果
result_path = Path("results/doc001_easyocr.json")

with result_path.open(encoding="utf-8") as f:
    data = json.load(f)

recognized_text = "".join(
    item["text"] for item in data["results"]
)

error_rate = cer(ground_truth, recognized_text)

print(f"CER: {error_rate:.4f}")