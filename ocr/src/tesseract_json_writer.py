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