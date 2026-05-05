#!/bin/bash
# PlaScan Docker 交互式 shell (GPU 优先, 回退 CPU)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="plascan-build"

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building $IMAGE ..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.ubuntu2404" "$PROJECT_DIR"
fi

# 优先 nvidia runtime (需 nvidia-container-toolkit), 回退 runc
RUNTIME=""
if docker info 2>/dev/null | grep -q "nvidia"; then
    RUNTIME="--runtime=nvidia"
fi

docker run --rm -it $RUNTIME \
    -v "$PROJECT_DIR:/src" \
    -w /src \
    "$IMAGE" \
    bash
