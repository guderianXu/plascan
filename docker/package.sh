#!/bin/bash
# Docker 内构建并打包 .deb
# 用法: ./docker/package.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE="plascan-build"
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image $IMAGE ..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.ubuntu2404" "$PROJECT_DIR"
fi

rm -rf "$PROJECT_DIR/build"
mkdir -p "$PROJECT_DIR/build"

docker run --rm \
    --runtime=nvidia --gpus all \
    -v "$PROJECT_DIR:/src" \
    -w /src/build \
    "$IMAGE" \
    bash -c "
        cmake .. -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release &&
        make -j\$(nproc) &&
        ctest --output-on-failure &&
        cpack -G DEB
    "

echo ""
echo "Package:"
ls -lh "$PROJECT_DIR/build"/*.deb 2>/dev/null || echo "(check build directory)"
