import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
from PIL import Image
import os

# 読み取らせたいファイル（画像またはPDF）のパスをここで指定します
file_path = "sampleopen.png"  # ※用意したファイル名に書き換えてください

if not os.path.exists(file_path):
    print(f"エラー: {file_path} が見つかりません。同じフォルダにファイルがあるか確認してください。")
    exit()

print(f"\n▼ {file_path} の読み取り結果（サイズ感付き） ▼\n")

_, ext = os.path.splitext(file_path)

# 画像リストの作成
images = []
if ext.lower() == '.pdf':
    images = convert_from_path(file_path)
else:
    try:
        images = [Image.open(file_path)]
    except Exception as e:
        print(f"エラー: 読み込めない形式のファイルです。({e})")
        exit()

# 読み込みとサイズ判別OCRの実行
for i, img in enumerate(images):
    if len(images) > 1:
        print(f"\n--- {i+1}ページ目 ---")
        
    data = pytesseract.image_to_data(img, lang='jpn+eng', output_type=Output.DICT)
    
    words_info = []
    
    n_boxes = len(data['text'])
    for j in range(n_boxes):
        text = data['text'][j].strip()
        if text:
            height = data['height'][j]
            words_info.append({"text": text, "height": height})
            
    if not words_info:
        print("文字が検出されませんでした。")
        continue
        
    total_height = sum(item["height"] for item in words_info)
    avg_height = total_height / len(words_info)
    
    print(f"【判定基準】このページの平均文字サイズ: 約{avg_height:.1f}ピクセル\n")
    
    for item in words_info:
        text = item["text"]
        height = item["height"]
        
        if height > avg_height * 1.5:
            size_label = "🔴 大 (見出しクラス)"
        elif height < avg_height * 0.7:
            size_label = "🔵 小 (補足・ルビなど)"
        else:
            size_label = "🟢 中 (本文クラス)"
            
        print(f"{size_label}\t高さ: {height}px\t文字: {text}")

print("\n▲ ここまで ▲")