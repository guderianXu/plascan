#!/bin/bash
# PlaScan Docker 运行器 — CLI 工具 (GPU 加速)
# GUI 请直接在宿主运行: build/bin/plascan_gui.bin
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]; then
    echo "用法: $0 <command> [args...]"
    echo ""
    echo "  CLI 工具 (Docker, GPU):"
    echo "    feature_extract_cli   特征提取"
    echo "    feature_match_cli     特征匹配"
    echo "    dense_match_cli       密集匹配"
    echo "    rectify_cli           极线校正"
    echo "    triangulate_cli       视差三角化"
    echo ""
    echo "  交互:"
    echo "    bash                  容器内 shell"
    echo ""
    echo "  GUI 请用宿主环境: build/bin/plascan_gui.bin"
    exit 1
fi

CMD="$1"; shift
BASE_FLAGS="--rm --runtime=nvidia --gpus all -v $PROJECT_DIR:/src -w /src"

case "$CMD" in
    feature_extract_cli|feature_match_cli|dense_match_cli|rectify_cli|triangulate_cli)
        sudo docker run $BASE_FLAGS plascan-build bash -c "/src/build/bin/$CMD $*"
        ;;
    bash|shell)
        sudo docker run -it $BASE_FLAGS plascan-build bash
        ;;
    *)
        echo "未知命令: $CMD"
        exit 1
        ;;
esac
