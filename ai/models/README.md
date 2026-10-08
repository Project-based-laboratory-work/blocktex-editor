# models — アプリから選択できるモデルの置き場

学習・評価が済み、アプリ（`frontend/`）のモデル選択に出すモデルをここに置く。
学習中のチェックポイントは各モデルの `checkpoints/` に置き、ここには**配布するもの**だけを入れる。

```
ai/models/
  <モデル名>/
    manifest.json     # アプリがモデル一覧を作るための情報
    model.onnx        # 推論用の重み（ブラウザで読む場合はONNX）
```

## manifest.json

```json
{
  "id": "math_ocr_d192",
  "name": "数式OCR (CNN+Transformer, d=192)",
  "task": "math_to_latex",
  "source": "math_ocr",
  "format": "onnx",
  "file": "model.onnx"
}
```

| キー | 意味 |
|---|---|
| `id` | モデルの識別子（ディレクトリ名と同じ） |
| `name` | アプリの選択肢に表示する名前 |
| `task` | 課題の種類。同じ `task` のモデルだけをアプリで入れ替えられる |
| `source` | 元になった `ai/` 直下のモデルディレクトリ |
| `format` / `file` | 重みの形式とファイル名 |

## 未決事項

- ONNXの置き場所（`frontend/` から読めるように配信する方法）はONNXエクスポート実装時に決める。
- `.pt` は容量が大きいため `.gitignore` 済み。ONNXをgit管理するかは容量を見て決める。
