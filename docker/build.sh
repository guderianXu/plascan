#!/bin/bash
# Docker 内构建 PlaScan 并运行测试
# 用法: ./docker/build.sh [cmake-args...]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="plascan-build"
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image $IMAGE ..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.ubuntu2404" "$PROJECT_DIR"
fi

mkdir -p "$PROJECT_DIR/build"

docker run --rm \
    --runtime=nvidia --gpus all \
    -v "$PROJECT_DIR:/src" \
    -w /src/build \
    "$IMAGE" \
    bash -c "cmake .. $* -DBUILD_TESTS=ON && make -j\$(nproc) && ctest --output-on-failure"
