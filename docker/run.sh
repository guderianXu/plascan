#!/bin/bash
# run.sh — 在目标机器上启动 PlaScan 容器
#
# 用法：
#   bash run.sh [数据目录]
#
# 示例：
#   bash run.sh /home/user/survey_data

set -e

DATA_DIR="${1:-$HOME/plascan_data}"
mkdir -p "$DATA_DIR"

# 允许本地 X11 连接
xhost +local:docker 2>/dev/null || true

docker run --rm \
    --gpus all \
    --ipc=host \
    -e DISPLAY="$DISPLAY" \
    -e QT_QPA_PLATFORM=xcb \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$DATA_DIR":/data \
    --name plascan \
    plascan:latest

# 运行结束后撤销 X11 授权
xhost -local:docker 2>/dev/null || true
