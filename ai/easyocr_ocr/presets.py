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
    # 前処理を1つだけ変える
    "up2": {"preprocess": ["upscale2"], "readtext": {}},
    "up3": {"preprocess": ["upscale3"], "readtext": {}},
    "gray": {"preprocess": ["grayscale"], "readtext": {}},
    "contrast": {"preprocess": ["autocontrast"], "readtext": {}},
    "invert": {"preprocess": ["invert_if_dark"], "readtext": {}},
    "otsu": {"preprocess": ["otsu"], "readtext": {}},
    "adaptive": {"preprocess": ["adaptive"], "readtext": {}},
    "adaptive_mean": {"preprocess": ["adaptive_mean"], "readtext": {}},
    # 拡大してから二値化する（順番で結果が変わる。小さい文字が潰れにくい順）
    "up2_adaptive": {"preprocess": ["upscale2", "adaptive"], "readtext": {}},
    # 検出（どこに文字があるか）の設定を1つだけ変える
    "lowtext": {"preprocess": [], "readtext": {"low_text": 0.3}},
    "mag15": {"preprocess": [], "readtext": {"mag_ratio": 1.5}},
    # 近くの箱を段落にまとめる（信頼度は返らなくなる）
    "par": {"preprocess": [], "readtext": {"paragraph": True}},
}
