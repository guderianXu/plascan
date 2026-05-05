#!/bin/bash
# PlaScan Docker 运行器 (CLI + GUI, GPU 支持)
# 用法:
#   ./docker/run.sh feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda
#   ./docker/run.sh gui          # 启动 GUI
#   ./docker/run.sh bash         # 交互式 shell

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]; then
    echo "用法: $0 <command> [args...]"
    echo ""
    echo "CLI 工具:"
    echo "  feature_extract_cli   特征提取"
    echo "  feature_match_cli     特征匹配"
    echo "  dense_match_cli       密集匹配"
    echo "  rectify_cli           极线校正"
    echo "  triangulate_cli       视差三角化"
    echo ""
    echo "其他:"
    echo "  gui                   启动图形界面"
    echo "  bash                  交互式 shell"
    echo ""
    echo "示例:"
    echo "  $0 feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda"
    echo "  $0 gui"
    exit 1
fi

CMD="$1"
shift

DOCKER_FLAGS="--rm --runtime=nvidia --gpus all -v $PROJECT_DIR:/src -w /src"

case "$CMD" in
    gui)
        # X11 授权 (允许容器连接宿主 X 服务器)
        xhost +local:docker >/dev/null 2>&1 || true
        DOCKER_FLAGS="$DOCKER_FLAGS --network host -e DISPLAY=$DISPLAY"
        if [ -f "$HOME/.Xauthority" ]; then
            DOCKER_FLAGS="$DOCKER_FLAGS -e XAUTHORITY=/tmp/.Xauthority -v $HOME/.Xauthority:/tmp/.Xauthority:ro"
        fi
        sudo docker run $DOCKER_FLAGS plascan-build bash -c "build/bin/plascan_gui.bin"
        ;;
    feature_extract_cli|feature_match_cli|dense_match_cli|rectify_cli|triangulate_cli)
        sudo docker run $DOCKER_FLAGS plascan-build bash -c "/src/build/bin/$CMD $*"
        ;;
    bash|shell)
        sudo docker run -it $DOCKER_FLAGS plascan-build bash
        ;;
    *)
        echo "未知命令: $CMD"
        exit 1
        ;;
esac
