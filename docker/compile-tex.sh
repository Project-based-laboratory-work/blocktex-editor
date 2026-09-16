#!/usr/bin/env bash
# TeX Live入りのDockerコンテナで .tex をコンパイルしてPDFを作る。
# 使い方: docker/compile-tex.sh frontend/src/fixtures/sample-document.expected.tex
set -euo pipefail

image=blocktex-tex

if [ $# -ne 1 ]; then
  echo "usage: $0 <path/to/file.tex>" >&2
  exit 1
fi

if [ ! -f "$1" ]; then
  echo "error: ファイルが見つかりません: $1" >&2
  exit 1
fi

if [ "${1%.tex}" = "$1" ]; then
  echo "error: .tex ファイルを指定してください: $1" >&2
  exit 1
fi

# イメージ未ビルドのまま docker run するとDocker Hubへの取得を試みて分かりにくく失敗するため、
# 先にローカルの有無を確認してビルド手順を案内する。
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "error: Dockerイメージ '$image' がありません。先に次を実行してください:" >&2
  echo "  docker build -t $image -f docker/tex.Dockerfile docker/" >&2
  exit 1
fi

tex_dir=$(cd "$(dirname "$1")" && pwd)
tex_file=$(basename "$1")
base="${tex_file%.tex}"

# --user でホスト側のユーザー権限に合わせる（生成物がroot所有になるのを防ぐ）
# HOME はコンテナ内に該当ユーザーのホームが無いため、書き込み可能な場所を渡す
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$tex_dir":/work \
  -w /work \
  "$image" \
  bash -c "uplatex -halt-on-error '$tex_file' && uplatex -halt-on-error '$tex_file' && dvipdfmx '$base.dvi'"

echo "generated: $tex_dir/$base.pdf"
