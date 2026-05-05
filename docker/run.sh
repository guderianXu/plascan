#!/bin/bash
# PlaScan Docker 运行器 — CLI + GUI, GPU 支持
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ $# -eq 0 ]; then
    echo "用法: $0 <command> [args...]"
    echo "  CLI:  feature_extract_cli | feature_match_cli | dense_match_cli"
    echo "        rectify_cli | triangulate_cli"
    echo "  GUI:  gui"
    echo "  Shell: bash"
    exit 1
fi

CMD="$1"; shift
BASE="--rm --runtime=nvidia --gpus all -v $PROJECT_DIR:/src -w /src"

case "$CMD" in
    gui)
        xhost +local:docker >/dev/null 2>&1 || true
        sudo docker run $BASE --network host \
            -e DISPLAY=$DISPLAY \
            -e QT_QPA_PLATFORM=xcb \
            -e QT_OPENGL=desktop \
            -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
            -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
            plascan-build bash -c "
                mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root && \
                build/bin/plascan_gui.bin \$@
            "
        ;;
    feature_extract_cli|feature_match_cli|dense_match_cli|rectify_cli|triangulate_cli)
        sudo docker run $BASE plascan-build bash -c "/src/build/bin/$CMD $*"
        ;;
    bash|shell)
        sudo docker run -it $BASE plascan-build bash
        ;;
    *) echo "未知: $CMD"; exit 1 ;;
esac
