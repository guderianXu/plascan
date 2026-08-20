# macOS Apple Silicon 构建

PlaScan 在 Apple Silicon macOS 上使用原生 arm64 工具链构建。当前 C++ 后端不支持 Apple MPS，
macOS preset 会显式关闭 CUDA、TensorRT 和 OpenCL，保留 CPU 算法、CLI、Qt6 GUI 和测试。

## 准备工具

安装 Xcode Command Line Tools、Homebrew、Ninja、pkg-config 和 vcpkg port 所需的 autotools。
当前构建不要求安装完整 Xcode：

```bash
xcode-select --install
brew install ninja pkg-config autoconf autoconf-archive automake libtool libomp python@3.12
```

项目使用 vcpkg manifest 管理 Qt6、OpenCV、GDAL、Ceres、libtiff、libzip 和 GTest 等 C++ 依赖。
首次使用时在源码树外初始化 vcpkg：

```bash
git clone https://github.com/microsoft/vcpkg.git "$HOME/code/vcpkg"
"$HOME/code/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
export VCPKG_ROOT="$HOME/code/vcpkg"
```

如果 vcpkg 已安装在其它位置，只需把 `VCPKG_ROOT` 指向该目录。

测试及部分工作流脚本使用仓库统一的 Python 环境。用 Python 3.10 或更高版本初始化 CPU 运行时：

```bash
python3.12 scripts/env/setup_python_runtime.py --device cpu
```

## 配置与编译

在 PlaScan 仓库根目录运行：

```bash
export VCPKG_ROOT="$HOME/code/vcpkg"
env -u CONDA_PREFIX cmake --preset macos-vcpkg-release
cmake --build --preset macos-vcpkg-release
```

首次配置会从源码构建完整依赖树，之后会复用
`build/macos-vcpkg-release/vcpkg_installed` 和 vcpkg 二进制缓存。
配置命令只在 CMake 子进程中移除继承的 `CONDA_PREFIX`，避免 Conda 环境改变 GDAL 等 vcpkg port
的依赖探测；无需退出当前终端中已激活的 Python Conda 环境。

## 测试

```bash
ctest --preset macos-vcpkg-release
```

也可以使用仓库测试脚本：

```bash
.venv/bin/python scripts/env/run_tests.py \
  --test-dir build/macos-vcpkg-release \
  --output-on-failure
```

需要 offscreen 后端的 GUI 测试会自行选择并静态导入对应 Qt 插件，无需设置全局
`QT_QPA_PLATFORM`。全局强制 offscreen 会影响只使用原生 Cocoa 后端的程序。

## 打包

当前 macOS 产物使用 TGZ，不生成签名或公证的 `.app`/DMG：

```bash
cpack --preset macos-vcpkg-release
```

输出目录为 `dist/packages/macos-vcpkg-release/`。

## 当前限制

- Apple Silicon 没有 NVIDIA CUDA 和 TensorRT；依赖它们的 LightGlue、LoMa-R 等路径不可用。
  BiRefNet Dynamic 使用随程序部署的 ONNX Runtime CPU 后端，可以在 macOS 上直接处理蒙版。
- 新版 BA 生产路径使用 PlaMatrix CPU 后端；Ceres 只作为可选对照后端，macOS preset 默认不启用。
- macOS preset 当前使用 CPU 后端；项目尚未把 Metal/MPS 接入 C++ 核心计算路径。
- macOS 的 Qt6 不启用需要外部 MoltenVK SDK 的 Vulkan feature，GUI 使用 Qt 的平台 QRhi fallback。
- U2Net 可使用 OpenCV DNN CPU 后端，BiRefNet Dynamic 使用 ONNX Runtime CPU 后端。
- TGZ 主要用于开发构建验证，尚未覆盖 macOS 应用签名、公证和 DMG 分发流程。
