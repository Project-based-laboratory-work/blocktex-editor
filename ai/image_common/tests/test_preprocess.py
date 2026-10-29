import numpy as np
import pytest
from PIL import Image

from image_common.preprocess import apply_preprocess


def test_no_steps_keeps_image():
    image = Image.new("RGB", (40, 20), "white")
    out, scale = apply_preprocess(image, [])
    assert out.size == (40, 20) and scale == 1.0


def test_upscale_returns_scale():
    image = Image.new("RGB", (40, 20), "white")
    out, scale = apply_preprocess(image, ["upscale2", "grayscale"])
    assert out.size == (80, 40) and scale == 2.0
    assert out.mode == "RGB"


def test_invert_only_dark_images():
    dark, _ = apply_preprocess(Image.new("RGB", (10, 10), (20, 20, 20)), ["invert_if_dark"])
    light, _ = apply_preprocess(Image.new("RGB", (10, 10), (240, 240, 240)), ["invert_if_dark"])
    assert np.asarray(dark).mean() > 200
    assert np.asarray(light).mean() > 200


def test_otsu_makes_two_values():
    pytest.importorskip("cv2")
    gradient = np.tile(np.arange(0, 256, dtype=np.uint8), (10, 1))
    out, _ = apply_preprocess(Image.fromarray(gradient).convert("RGB"), ["otsu"])
    assert set(np.unique(np.asarray(out))) <= {0, 255}


def test_unknown_step_is_rejected():
    with pytest.raises(ValueError):
        apply_preprocess(Image.new("RGB", (10, 10)), ["nope"])


@pytest.mark.parametrize("step", ["adaptive", "adaptive_mean"])
def test_adaptive_keeps_text_on_colored_band(step):
    # 左半分は白い背景、右半分は暗い帯。どちらにも同じ濃さの差で「文字」（四角）を置く
    pytest.importorskip("cv2")
    array = np.full((60, 120), 230, dtype=np.uint8)
    array[:, 60:] = 90
    array[25:35, 20:40] = 150  # 白い背景の上の文字（背景より暗い）
    array[25:35, 80:100] = 10  # 暗い帯の上の文字（帯より暗い）
    out, scale = apply_preprocess(Image.fromarray(array).convert("RGB"), [step])
    result = np.asarray(out.convert("L"))

    assert scale == 1.0
    assert set(np.unique(result)) <= {0, 255}
    # 大津（全体で1つの境目）だと暗い帯ごと黒くなるが、適応的なら両方の文字が黒・周りは白で残る
    assert result[30, 30] == 0 and result[30, 90] == 0
    assert result[5, 5] == 255 and result[5, 110] == 255
    # ただし帯の端（背景の色が変わる境目）には、黒い線が出る（mean の方が太い）
    assert result[5, 61] == 0


def test_wider_block_keeps_thick_strokes_filled():
    # 背景の上に、範囲31pxより太い線（40×40の黒い四角）がある。拡大後の太い文字を想定
    pytest.importorskip("cv2")
    array = np.full((120, 120), 230, dtype=np.uint8)
    array[40:80, 40:80] = 50
    image = Image.fromarray(array).convert("RGB")

    narrow = np.asarray(apply_preprocess(image, ["adaptive"])[0].convert("L"))
    wide = np.asarray(apply_preprocess(image, ["adaptive61"])[0].convert("L"))

    # 31pxだと四角の中心では「周り」が全部黒くなり、内側が白く抜ける。61pxなら背景が入るので黒のまま
    assert narrow[60, 60] == 255
    assert wide[60, 60] == 0


@pytest.mark.parametrize("step", ["adaptive", "adaptive_mean", "adaptive61"])
def test_adaptive_keeps_text_on_dark_background(step):
    # ダークモード：濃い灰色の背景（30）に薄い灰色の文字（220）。反転しないと文字も背景も白になって消える
    pytest.importorskip("cv2")
    array = np.full((80, 120), 30, dtype=np.uint8)
    array[35:45, 20:100] = 220
    out, _ = apply_preprocess(Image.fromarray(array).convert("RGB"), [step])
    result = np.asarray(out.convert("L"))

    assert result[40, 60] == 0  # 文字は黒で残る
    assert result[5, 5] == 255 and result[70, 110] == 255  # 背景は白
