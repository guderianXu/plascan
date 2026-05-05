#!/bin/bash
# 在 Docker 中运行 PlaScan CLI 工具 (带 GPU 支持)
# 用法:
#   ./docker/run.sh feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda
#   ./docker/run.sh feature_match_cli --sp1 a.sp --sp2 b.sp -o out.match --cuda
#   ./docker/run.sh dense_match_cli -L a.tif -R b.tif -o disp.tif --cuda
#   ./docker/run.sh bash   # 交互式 shell

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 确定要运行的程序
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
    echo "  bash                  交互式 shell"
    echo ""
    echo "示例:"
    echo "  $0 feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda"
    exit 1
fi

CMD="$1"
shift

case "$CMD" in
    feature_extract_cli|feature_match_cli|dense_match_cli|rectify_cli|triangulate_cli)
        BIN="build/bin/$CMD"
        ;;
    bash|shell)
        BIN="bash"
        ;;
    *)
        echo "未知命令: $CMD"
        exit 1
        ;;
esac

sudo docker run --rm --runtime=nvidia --gpus all \
    -v "$PROJECT_DIR:/src" \
    -w /src \
    plascan-build \
    bash -c "/src/$BIN $*"
