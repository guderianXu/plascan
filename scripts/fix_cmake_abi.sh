#!/bin/bash
# 修复 cmake 生成的 flags.make 中的 C++ ABI 冲突
# conda gcc 默认 -D_GLIBCXX_USE_CXX11_ABI=0 与 OpenCV/GTest 的新 ABI 冲突
BUILD_DIR=${1:-build}
find "$BUILD_DIR" -name "flags.make" -exec sed -i 's/-D_GLIBCXX_USE_CXX11_ABI=0//g' {} \;
echo "ABI fix applied to $BUILD_DIR"
