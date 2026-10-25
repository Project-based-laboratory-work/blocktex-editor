"""
EasyOCRの設定の組（プリセット）。実験のたびにコードを書き換えず、ここに名前を付けて足していく。

- preprocess: preprocess.py の STEPS の名前を、かける順に並べる
- readtext:   reader.readtext() にそのまま渡す引数（EasyOCRのドキュメントを参照）

比べやすいように、base から1か所だけ変えたものを並べている。
使った設定はIRの params にそのまま記録される。
"""

PRESETS: dict[str, dict] = {
    # 何もしない（基準）
    "base": {"preprocess": [], "readtext": {}},
    # 前処理を1つのみ変更
    "up2": {"preprocess": ["upscale2"], "readtext": {}},
    "up3": {"preprocess": ["upscale3"], "readtext": {}},
    "gray": {"preprocess": ["grayscale"], "readtext": {}},
    "contrast": {"preprocess": ["autocontrast"], "readtext": {}},
    "invert": {"preprocess": ["invert_if_dark"], "readtext": {}},
    "otsu": {"preprocess": ["otsu"], "readtext": {}},
    "adaptive": {"preprocess": ["adaptive"], "readtext": {}},
    "adaptive_mean": {"preprocess": ["adaptive_mean"], "readtext": {}},
    # 拡大してから二値化する
    "up2_adaptive": {"preprocess": ["upscale2", "adaptive"], "readtext": {}},
    # up2_adaptive と比べる：周りの範囲も2倍（61px）にして、文字の線が細くならないようにする
    "up2_adaptive61": {"preprocess": ["upscale2", "adaptive61"], "readtext": {}},
    # 検出（どこに文字があるか）の設定を1つだけ変える
    "lowtext": {"preprocess": [], "readtext": {"low_text": 0.3}},
    "mag15": {"preprocess": [], "readtext": {"mag_ratio": 1.5}},
    # up2 との比較
    "mag2": {"preprocess": [], "readtext": {"mag_ratio": 2.0}},
    # 近くの箱を段落にまとめる、信頼度は返らない
    "par": {"preprocess": [], "readtext": {"paragraph": True}},
}
