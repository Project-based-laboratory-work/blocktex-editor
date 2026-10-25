#このコードは、OCRを使う前に、ざっくり文字の領域を分けて、文字の識別、タイトル・本文の見分けがしやすくなるもの（希望）


import cv2
import matplotlib.pyplot as plt


# =========================
# 画像を読み込む
# =========================

#画像のパス
image_path = "blocktex-editor/ai/sampleopen.png"

img = cv2.imread(image_path)

if img is None:
    print("❌ 画像を読み込めませんでした")
    exit()

print("✅ 画像を読み込みました")
print("画像サイズ:", img.shape)


# =========================
# 読み込んだ画像を表示
# =========================

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()


# =========================
# グレースケール化
# =========================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =========================
# 白黒画像の作成。背景を黒、文字・線を白に変換
#普通に二値化すると、背景白、文字黒になるが、THRESH_BINARY_INVを使うことによって、二値化と反転を行う。輪郭検出、モルフォロジー処理で「文字を前景」として扱いやすくなる。
#モルフォロジー処理：構造要素（カーネル）と呼ばれる小さなマスクを用いて画像内の物体の形状を変化させ、ノイズ除去や輪郭抽出を行う画像処理手法
# =========================

_, binary = cv2.threshold(
    gray,
    200,
    255,
    cv2.THRESH_BINARY_INV
)

plt.figure(figsize=(12, 8))
plt.imshow(binary, cmap="gray")
plt.axis("off")
plt.show()


# =========================
# 文字をまとまりにする。近くにある文字同士をくっつけることによって、塊を判別しやすくする。
# =========================

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (15, 5)
)

dilated = cv2.dilate(
    binary,
    kernel,
    iterations=1
)

plt.figure(figsize=(12, 8))
plt.imshow(dilated, cmap="gray")
plt.axis("off")
plt.show()


# =========================
# 輪郭を検出　contouesによって、まとまり（領域）の数を数える
# =========================

contours, _ = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("検出した領域数:", len(contours))


# =========================
#小さすぎる領域は無視する。このサイズ感は仮
# =========================

result = img.copy()

for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    # 小さすぎる領域を無視
    if w > 20 and h > 10:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            2
        )


# =========================
# 判別した塊を分かりやすいように、赤枠で囲った
# =========================

result_rgb = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))
plt.imshow(result_rgb)
plt.axis("off")
plt.show()