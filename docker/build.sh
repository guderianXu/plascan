#!/bin/bash
# build.sh — 打包 PlaScan Docker 镜像（离线部署用）
#
# 用法：
#   cd /home/guderian/code/plascan
#   bash docker/build.sh
#
# 产物：
#   docker/plascan-docker.tar.gz  — 可直接 scp 到目标机器的镜像包

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
DOCKER_DIR="$SCRIPT_DIR"

echo "=== [1/5] 打包 conda 环境 ==="
echo "    源: $CONDA_PREFIX"
echo "    目标: $DOCKER_DIR/plascan-env.tar.gz"
echo "    （约 13G，需要几分钟...）"
tar -czf "$DOCKER_DIR/plascan-env.tar.gz" \
    -C "$(dirname "$CONDA_PREFIX")" \
    "$(basename "$CONDA_PREFIX")"

echo "=== [2/5] 整理应用文件 ==="
APP_DIR="$DOCKER_DIR/app"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/bin" "$APP_DIR/lib" "$APP_DIR/models" "$APP_DIR/scripts"

# 主程序
cp "$BUILD_DIR/bin/plascan_gui.bin" "$APP_DIR/bin/"

# 项目内的 .so（superpoint、superglue）
find "$BUILD_DIR/src" -name "*.so" -exec cp {} "$APP_DIR/lib/" \;

# 模型文件
cp "$PROJECT_DIR/resources/models/"*.pt "$APP_DIR/models/"

# Python 脚本（DISK/ALIKED/RoMa 特征提取）
cp "$PROJECT_DIR/scripts/extract_features.py" "$APP_DIR/scripts/"
cp "$PROJECT_DIR/scripts/match_roma.py"        "$APP_DIR/scripts/"

# LightGlue Python 包（DISK/ALIKED 提取依赖）
if [ -d "$PROJECT_DIR/3rdparty/LightGlue-main" ]; then
    cp -r "$PROJECT_DIR/3rdparty/LightGlue-main" "$APP_DIR/LightGlue"
fi

echo "=== [3/5] 复制启动脚本 ==="
cp "$DOCKER_DIR/entrypoint.sh" "$DOCKER_DIR/entrypoint.sh"

echo "=== [4/5] 构建 Docker 镜像 ==="
docker build -t plascan:latest "$DOCKER_DIR"

echo "=== [5/5] 导出镜像为 tar.gz ==="
OUTPUT="$DOCKER_DIR/plascan-docker.tar.gz"
docker save plascan:latest | gzip > "$OUTPUT"
echo ""
echo "✓ 完成！镜像已保存到: $OUTPUT"
echo "  大小: $(du -sh "$OUTPUT" | cut -f1)"
echo ""
echo "目标机器部署步骤："
echo "  1. scp $OUTPUT user@target:/path/"
echo "  2. ssh user@target"
echo "  3. docker load < plascan-docker.tar.gz"
echo "  4. bash run.sh   # 或见下方命令"
echo ""
echo "运行命令："
echo "  docker run --rm --gpus all \\"
echo "    -e DISPLAY=\$DISPLAY \\"
echo "    -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "    -v /path/to/data:/data \\"
echo "    plascan:latest"
