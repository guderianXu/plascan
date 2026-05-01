#!/bin/bash
# PlaScan 容器入口点

set -e

CONDA_ENV=/opt/conda/envs/plascan
APP_DIR=/opt/plascan

# 覆盖 RPATH（二进制编译时硬编码了构建机器的 conda 路径）
export LD_LIBRARY_PATH=\
${CONDA_ENV}/lib:\
${CONDA_ENV}/targets/x86_64-linux/lib:\
${CONDA_ENV}/lib/python3.12/site-packages/torch/lib:\
${CONDA_ENV}/lib/python3.12/site-packages/nvidia/cu13/lib:\
${APP_DIR}/lib:\
${LD_LIBRARY_PATH}

export PATH=${CONDA_ENV}/bin:${PATH}
export CONDA_PREFIX=${CONDA_ENV}
export PLASCAN_MODEL_DIR=${APP_DIR}/models

# Python 脚本路径（DISK/ALIKED/RoMa 特征提取）
export PYTHONPATH=${APP_DIR}:${APP_DIR}/LightGlue:${PYTHONPATH}

# 如果传入了参数，直接执行（方便调试：docker run plascan bash）
if [ "$1" != "" ] && [ "$1" != "plascan" ]; then
    exec "$@"
fi

# 检查 DISPLAY
if [ -z "$DISPLAY" ]; then
    echo "[plascan] 警告: DISPLAY 未设置，尝试使用 :0"
    export DISPLAY=:0
fi

# 检查 GPU
if command -v nvidia-smi &>/dev/null; then
    echo "[plascan] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
    echo "[plascan] 警告: 未检测到 GPU，将使用 CPU 模式"
fi

exec ${APP_DIR}/bin/plascan_gui.bin "$@"
