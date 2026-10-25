"""
評価できるモデルの登録簿（models.json）を読み、登録したモデルを呼び出す。

サイトで選べるのは、ここに登録したモデルだけ（任意のPythonファイルは実行しない）。
各モデルのコードは書き換えずに、「どのモジュールのどの関数を呼ぶか」を登録簿に書く。
"""
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from common.ir import IRDocument

REGISTRY_PATH = Path(__file__).parent / "models.json"


@dataclass
class ModelSpec:
    id: str
    name: str
    entry: str          # "モジュール名:関数名"。ラベル付きIR（またはIR）を返す関数
    input: str          # "pdf" / "image"
    outputs: list[str]  # "bbox" / "text" / "label" など
    params: dict = field(default_factory=dict)  # 関数に渡す追加の引数


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, ModelSpec]:
    """登録簿を {id: ModelSpec} で返す"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: ModelSpec(**item) for item in items}


def load_entry(entry: str) -> Callable:
    """'モジュール名:関数名' から関数を取り出す"""
    module_name, func_name = entry.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def predict(spec: ModelSpec, pdf_path: str) -> IRDocument:
    """登録したモデルで、PDFを1つ読む"""
    fn = load_entry(spec.entry)
    if spec.input == "pdf":
        return fn(pdf_path, **spec.params)
    raise ValueError(f"input={spec.input!r} にはまだ対応していない（画像を入力にするモデルを足すときに作る）")
