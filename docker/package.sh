#!/bin/bash
# 在 Docker 中构建并打包为 .deb
# 用法: ./docker/package.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"

IMAGE="plascan-build"
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image $IMAGE ..."
    docker build -t "$IMAGE" -f "$SCRIPT_DIR/Dockerfile.ubuntu2404" "$PROJECT_DIR"
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

docker run --rm \
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
echo "Package built:"
ls -lh "$BUILD_DIR"/*.deb 2>/dev/null || echo "(check build directory)"
