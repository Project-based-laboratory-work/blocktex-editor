"""
評価できるモデルの登録簿（models.json）を読み、登録したモデルを呼び出す。

サイトで選べるのは、ここに登録したモデルだけ（任意のPythonファイルは実行しない）。
各モデルのコードは書き換えずに、「どのモジュールのどの関数を呼ぶか」を登録簿に書く。

input が "image" のモデルには、PDFの各ページを画像にして1枚ずつ渡し、返ってきたページをまとめて1つのIRにする。
"""
import importlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from common.ir import IRDocument
from evaluation.images import MODEL_DPI, render_pages

REGISTRY_PATH = Path(__file__).parent / "models.json"


@dataclass
class ModelSpec:
    id: str
    name: str
    entry: str          # "モジュール名:関数名"。ラベル付きIR（またはIR）を返す関数
    input: str          # "pdf" / "image"
    outputs: list[str]  # "bbox" / "text" / "label" など
    params: dict = field(default_factory=dict)  # 関数に渡す追加の引数。image のモデルの "dpi" だけは画像にする解像度に使う
    note: str = ""      # サイトに出す補足（動かすのに必要なものなど）


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


def merge_pages(docs: list[IRDocument], source: str) -> IRDocument:
    """1ページずつのIR（画像1枚 = 1ページ）を、ページ番号を付け直して1つの文書にまとめる"""
    pages = []
    for page_no, doc in enumerate(docs, start=1):
        for page in doc.pages:
            page.page = page_no
            pages.append(page)
    first = docs[0] if docs else IRDocument(source=source, extractor="", pages=[])
    return IRDocument(source=source, extractor=first.extractor, structurer=first.structurer, params=first.params, pages=pages)


def predict(spec: ModelSpec, pdf_path: str) -> IRDocument:
    """登録したモデルで、PDFを1つ読む"""
    fn = load_entry(spec.entry)
    if spec.input == "pdf":
        return fn(pdf_path, **spec.params)
    if spec.input == "image":
        params = dict(spec.params)
        dpi = params.pop("dpi", MODEL_DPI)
        with tempfile.TemporaryDirectory() as tmp:
            images = render_pages(pdf_path, Path(tmp), dpi)
            docs = [fn(str(image), **params) for image in images]
        return merge_pages(docs, source=Path(pdf_path).name)
    raise ValueError(f"input={spec.input!r} には対応していない（使えるのは \"pdf\" と \"image\"）")
