#!/usr/bin/env bash
# TeX Live入りのDockerコンテナで .tex をコンパイルしてPDFを作る。
# 使い方: docker/compile-tex.sh frontend/src/fixtures/sample-document.expected.tex
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <path/to/file.tex>" >&2
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
  blocktex-tex \
  bash -c "uplatex -halt-on-error '$tex_file' && uplatex -halt-on-error '$tex_file' && dvipdfmx '$base.dvi'"

echo "generated: $tex_dir/$base.pdf"
