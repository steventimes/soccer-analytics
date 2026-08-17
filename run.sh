#!/usr/bin/env bash

set -euo pipefail

mkdir -p ./models ./docs/data

# 当前 compose 文件使用 v2 的可选 env_file 语法，优先选择 Docker 插件。
if docker compose version >/dev/null 2>&1; then
  compose=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose=(docker-compose)
else
  echo "需要 Docker Compose v2 或 docker-compose" >&2
  exit 127
fi

"${compose[@]}" build app
"${compose[@]}" run --rm app all "$@"
