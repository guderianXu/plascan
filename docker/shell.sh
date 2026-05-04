#!/bin/bash
# 进入 PlaScan Docker 构建环境 (交互式, 支持 GPU)
# 用法: ./docker/shell.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="plascan-build"
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image $IMAGE ..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.ubuntu2404" "$PROJECT_DIR"
fi

docker run --rm -it \
    --gpus all \
    -v "$PROJECT_DIR:/src" \
    -w /src \
    "$IMAGE" \
    bash
