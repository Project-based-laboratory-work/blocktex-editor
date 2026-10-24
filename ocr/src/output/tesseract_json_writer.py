"""
解析結果をJSONファイルに保存する処理。

役割：
- Pythonの辞書をJSON形式で保存する
- 保存先のフォルダがなければ作成する

OCRやレイアウト解析は行わない。
"""

import json
from pathlib import Path


def save_json(data: dict, output_path: str) -> Path:
    """
    Pythonの辞書をJSONファイルに保存する。

    Parameters
    ----------
    data : dict
        保存するデータ。
    output_path : str
        JSONファイルの保存先。

    Returns
    -------
    Path
        保存したファイルのパス。
    """

    path = Path(output_path)

    # 保存先のフォルダを作成する
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSONファイルに保存する
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return path

if __name__ == "__main__":
    from src.layout_detector import detect_layout

    # リポジトリのルートディレクトリを取得
    project_root = Path(__file__).resolve().parents[2]

    # 入力画像
    image_path = project_root / "sample" / "jphorizontal_001.jpg"

    # JSONの保存先
    output_path = (
        project_root
        / "data"
        / "layout"
        / "layout_results.json"
    )

    # レイアウト検出

    layout_result = detect_layout(str(image_path))

    # JSON保存

    save_json(layout_result, str(output_path))

    print(f"レイアウト検出結果を保存しました: {output_path}")