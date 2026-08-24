# PlaScan

[![CI](https://github.com/guderianXu/plascan/actions/workflows/ci.yml/badge.svg)](https://github.com/guderianXu/plascan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**行星表面摄影测量处理系统** — 从多视角影像生成高精度三维模型。

## 使用入口

- GUI：适合交互式建工程、检查影像、调整参数和查看阶段报告。
- CLI：适合批处理、自动化、服务器运行和可复现诊断；构建产物位于
  `build/<preset>/bin/`，可先运行 `<command> --help` 查看参数。
- [GUI 与 CLI 使用指南](docs/GUI_CLI_GUIDE.md)：工作流对应关系、统一参数语义、退出码和快速示例。
- [项目架构](docs/PROJECT_ARCHITECTURE.md)：模块职责和数据流。
- [模型资源](docs/models/README.md)：模型下载、校验与安装布局。
- [发版流程](docs/releases/RELEASE_PROCESS.md)：版本、打包和 Release 检查清单。

## 快速开始

### 依赖

- C++20 编译器：MSVC 2022、GCC 11+ 或 Clang 15+。
- CMake 3.25+ 和 Ninja。
- vcpkg；manifest 提供底层编解码、测试和系统库，OpenCV 仅支持仓库锁定的 5.0.0 源码构建。Qt 6.11.2、GDAL 3.12.4 与 AprilTag 3.4.5 也可由同一源码依赖入口构建。OpenMP 由编译器工具链提供，PoissonRecon 始终使用仓库固定源码，CPU 稠密线性代数由 PlaMatrix 原生实现，不依赖 BLAS/LAPACK。
- TensorRT 可选；用于 GPU 匹配与 AI 蒙版，并与 CUDA Toolkit 一样作为外部 SDK 提供。
- CUDA Toolkit 可选；启用后用于深度学习特征、匹配、MVS、点云处理和 dense match 加速。
- OpenCL 1.2 SDK/loader 可选；启用后可由 AMD、Intel 或 NVIDIA GPU 加速 MVS 与点云预处理。
- Python 3.10+ 可选；用于模型导出、数据准备和脚本化验证。

### 克隆并构建

普通开发需要拉取 `plapoint`、`plamatrix` 和 PoissonRecon。不要对整个仓库运行递归 submodule
初始化，因为 Qt 超级仓库还登记了许多 PlaScan 不使用的模块：

```bash
git clone https://github.com/guderianXu/plascan.git
cd plascan
git submodule update --init 3rdparty/plamatrix 3rdparty/plapoint 3rdparty/qt 3rdparty/opencv 3rdparty/gdal 3rdparty/apriltag 3rdparty/PoissonRecon
git -C 3rdparty/qt submodule update --init qtbase qtshadertools
```

设置 vcpkg 根目录后，使用锁定的 OpenCV 5 源码依赖入口配置、构建和测试：

```bash
export VCPKG_ROOT=/path/to/vcpkg
python scripts/env/configure_with_env.py --source-deps --build --test
```

该入口先在 `build/<platform>-source-deps-release/` 编译并安装 Qt 6.11.2、OpenCV 5.0.0、
GDAL 3.12.4 和 AprilTag 3.4.5；PoissonRecon 直接从固定 submodule 提供源码。PROJ、libgeotiff、libtiff、
zlib 等底层依赖仍由同一 vcpkg installed tree 提供，随后在 `build/<platform>-source-release/` 配置、编译和测试 PlaScan。
CMake 会校验所有源码 checkout 的固定 commit、安装包版本及加载路径，避免静默回退到 vcpkg。

Windows 和 Linux 构建默认优先启用 CUDA 与 TensorRT。CUDA 编译器不可用时自动回退到 CPU；CUDA
可用但完整 TensorRT C++ SDK 不可用时保留原生 CUDA 后端，并把神经网络推理回退到 ONNX Runtime
CPU。Windows 首次自动下载 TensorRT 10.15.1.29（CUDA 13.1）前，需要显式接受 NVIDIA 许可：

```powershell
python scripts\env\configure_with_env.py --source-deps --build --test --accept-tensorrt-license
```

SDK ZIP 会校验大小和 SHA-256，缓存在 `build/env/downloads/tensorrt/10.15.1.29/`，解压后的 SDK
位于 `build/env/sdk/tensorrt/10.15.1.29/`。后续普通编译会自动复用，无需再次传入许可参数。也可通过
`--tensorrt-root` 使用已有 SDK、通过 `--tensorrt-archive` 使用已下载的官方 ZIP，或通过
`--no-tensorrt-auto-install` 禁止自动安装。上述 SDK 和缓存均不提交到 Git。
配置脚本还会通过 `nvidia-smi` 自动读取本机 GPU 计算能力，例如 RTX 5080 只编译 `sm_120`，避免开发
构建为所有发布架构重复生成 CUDA 代码；需要制作通用安装包时可显式传入
`-DPLASCAN_CUDA_ARCHITECTURES=75;86;89;120`。编译和测试默认使用全部逻辑线程，可分别通过
`--build-jobs <n>` 和 `--test-jobs <n>` 覆盖。

可通过 `--build-dir` 把主构建树和源码第三方依赖放到指定目录，不需要修改或创建
`CMakeUserPresets.json`。例如：

```powershell
python scripts\env\configure_with_env.py --source-deps --build --test `
  --build-dir E:\code\plascan\build\8_23build
```

此时 PlaScan 输出位于 `build/8_23build/`，Qt、OpenCV、GDAL 等第三方库的构建、安装和 vcpkg
依赖位于 `build/8_23build/source-deps/`。

ONNX Runtime CPU 运行时固定为 1.29.0。官方发布包会校验 SHA-256 并缓存到
`build/env/downloads/onnxruntime/1.29.0/`，不同构建 preset 可以复用；离线机器也可通过
`PLASCAN_ONNXRUNTIME_ARCHIVE` 指定已下载的官方压缩包。

GUI 需要 Qt 6.7 或更高版本（使用 `QRhiWidget`）。只构建核心库、CLI 和非 GUI 测试时，使用相同的
OpenCV 5 源码依赖入口并显式关闭 GUI：

```bash
python scripts/env/configure_with_env.py --source-deps --build --test -- \
  -DPLASCAN_BUILD_GUI=OFF -DPLASCAN_BUILD_GUI_TESTS=OFF
```

项目通过 git submodule 引用自研点云库 [plapoint](https://github.com/guderianXu/plapoint) 和矩阵库 [plamatrix](https://github.com/guderianXu/plamatrix)，无需额外安装。

### vcpkg / CPack 跨平台构建

PlaScan 的 C++ 构建统一使用 `vcpkg.json` 和 `CMakePresets.json`，配置时必须启用 vcpkg manifest
toolchain。CMake 不读取 Conda 环境作为 C++ 依赖来源；CUDA 与 TensorRT 是可选的外部 SDK，通过其
标准 CMake 路径显式提供。根 manifest 不再安装 OpenCV；标准入口使用锁定的 OpenCV 5.0.0 源码包，
vcpkg 负责其余通用依赖。PlaMatrix 提供自包含的 CPU 稠密线性代数。

Linux:

```bash
export VCPKG_ROOT=/path/to/vcpkg
python scripts/env/configure_with_env.py --source-deps --build --test
```

本机 Ubuntu 的 CUDA 13.1 + NVIDIA OpenCL 开发构建使用独立 preset，不与 CPU/OpenCL 或正式 CUDA 13.1
打包目录混用。该 preset 使用 `/usr/local/cuda-13.1/bin/nvcc`、GCC 13 host compiler 和 RTX 40 系列的 `sm_89`，同时
启用 PlaScan/PlaMatrix/PlaPoint 的 CUDA 与 OpenCL 后端；TensorRT 不属于
vcpkg，当前开发 preset 默认关闭：

```bash
sudo apt install gcc-13 g++-13 cuda-compiler-13-1 cuda-libraries-dev-13-1 cuda-opencl-13-1 clinfo
export VCPKG_ROOT="$PWD/build/vcpkg"
export OpenCV_ROOT="$PWD/build/linux-source-deps-release/install"
cmake --preset linux-vcpkg-cuda-opencl-release
cmake --build --preset linux-vcpkg-cuda-opencl-release
QT_QPA_PLATFORM=offscreen python scripts/env/run_tests.py \
  --preset linux-vcpkg-cuda-opencl-release --output-on-failure
```

首次配置会在 `build/linux-vcpkg-cuda-opencl-release/vcpkg_installed` 安装独立的 manifest
依赖树。其它 CUDA Toolkit 路径、host compiler 或 GPU 架构应在命令行覆盖
`CMAKE_CUDA_COMPILER`、`CMAKE_CUDA_HOST_COMPILER` 和 `PLASCAN_CUDA_ARCHITECTURES`；不要修改或复用
CPU 构建的 vcpkg installed tree。可先用 `clinfo -l` 确认 OpenCL ICD 能枚举目标设备。
仓库的 `cuda` overlay 会让 vcpkg port 优先遵循 preset 的 `CUDACXX`。
Linux manifest 同时显式启用 `vulkan-loader[xcb]`，保证 Qt Vulkan RHI 能为 XCB 窗口创建 surface；
显式 OpenCL 模式允许使用 NVIDIA OpenCL，Auto/混合模式仍会与同一物理 GPU 的 CUDA 接口去重。

Linux 正式交付建议使用 Ubuntu 24.04 x86_64 基线的一键 DEB 工作流。首次配置会由 vcpkg 构建带
XCB/OpenSSL 的 Qt 运行时；构建机还需安装 GCC/G++、Ninja、`pkg-config` 和 `patchelf`。
打包前还需按下文准备对应变体的模型资产。日常修改代码后运行：

```bash
sudo apt install build-essential ninja-build pkg-config patchelf
```

然后运行增量打包检查：

```bash
export VCPKG_ROOT=/path/to/vcpkg
export OpenCV_ROOT="$PWD/build/linux-source-deps-release/install"
cmake --workflow --preset linux-package-smoke
build/linux-vcpkg-package-release/package-smoke/rootfs/opt/plascan/bin/plascan
```

该流程只增量编译并更新未压缩 rootfs，同时强制检查动态库闭包、Qt XCB/offscreen 插件、GDAL/PROJ
数据、相对 RUNPATH 和模型哈希。正式生成通用 CPU/OpenCL 包时运行：

```bash
cmake --workflow --preset linux-package-deb
sudo apt install ./build/linux-vcpkg-package-release/packages/release/plascan_1.1.7_amd64.deb
```

通用包可在没有 CUDA 的 Ubuntu 24.04 x86_64 电脑直接启动，并使用 OpenCV CPU 运行 U2Net、使用
ONNX Runtime CPU 运行 BiRefNet Dynamic；当前生产深度学习影像匹配算法仍需要 TensorRT/CUDA，
因此需要这些功能时使用 CUDA 变体：

```bash
# 构建机需要 CUDA 13.1、TensorRT 10.15.1（CUDA 13.1 变体）及 ONNX parser 开发库
cmake --workflow --preset linux-package-cuda-smoke
cmake --workflow --preset linux-package-cuda-deb
sudo apt install ./build/linux-vcpkg-cuda-package-release/packages/release/plascan-cuda_1.1.7_amd64.deb
```

CUDA DEB 不捆绑 NVIDIA 驱动和受系统 ABI 约束的 CUDA/TensorRT 库，而是在包元数据中声明 CUDA
13.1 与 TensorRT 10.15.1 CUDA 13.1 变体的运行时依赖；目标电脑需预先启用 NVIDIA Ubuntu
仓库并安装兼容驱动，随后 `apt install ./plascan-cuda_*.deb` 会补齐运行时。Qt、源码构建的 OpenCV 5、GDAL 等非系统 `.so`、Qt
plugins、GDAL/PROJ 数据和所选模型资产会装到 `/opt/plascan`，桌面入口和图标装到标准 `/usr` 路径。
两个包互斥，不能同时安装。DEB 与 `.sha256` 位于各自构建目录的 `packages/release`；依赖或安装布局
变化后应清理精确的 `package-smoke/rootfs` 再重新运行，普通源码修改无需反复做完整压缩。

Windows 通用 Release/ZIP（不等同于完整 CUDA/TensorRT 发布环境）：

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"
$env:OpenCV_ROOT = "$PWD\build\windows-source-deps-release\install"
cmake --preset windows-vcpkg-release
cmake --build --preset windows-vcpkg-release
python scripts\env\run_tests.py --preset windows-vcpkg-release
cpack --preset windows-vcpkg-release
```

CPack 默认启用 `PLASCAN_BUNDLE_ONNX_MODELS=ON`，安装 U2Net 与 LightGlue。Windows/Linux CUDA 打包
preset 还会启用 `PLASCAN_BUNDLE_BIREFNET_DYNAMIC=ON`；Windows CUDA 同时启用
`PLASCAN_BUNDLE_LOMA_R_MODELS=ON`。这些便携模型资源只读安装：

- `resources/models/U2Net_v1.onnx`：图像掩模；
- `resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx`：CUDA SIFT + LightGlue 匹配。
- `resources/models/loma_r_tensorrt/`：LoMa-R 共享特征/动态匹配 ONNX 和 K1024/K2048/K3840 清单。
- `resources/models/birefnet_dynamic/BiRefNet_dynamic_1024.onnx` 及 provenance：推荐的 TensorRT
  高分辨率蒙版模型。

模型默认从源码树同名路径读取，也可在配置时通过 `PLASCAN_U2NET_ONNX_PATH`、
`PLASCAN_LIGHTGLUE_ONNX_PATH`、`PLASCAN_LOMA_R_MODEL_DIR` 和 `PLASCAN_BIREFNET_DYNAMIC_MODEL_DIR`
指向外部缓存。安装/CPack 阶段会校验
固定字节数和 SHA-256；BiRefNet 两项资产来自 `models-v1.2.0`，原 U2Net、LightGlue、LoMa-R 仍来自
`models-v1.1.0`。缺失、损坏或拿错版本都会停止打包，且任何本机 `.engine` 都禁止进入安装包。
只需开发用轻量安装树时可显式关闭对应 bundle 开关；CUDA preset 需要同时设置
`PLASCAN_BUNDLE_ONNX_MODELS=OFF`、`PLASCAN_BUNDLE_BIREFNET_DYNAMIC=OFF` 和
`PLASCAN_BUNDLE_LOMA_R_MODELS=OFF`。干净 clone 不包含大模型，打包前按
[模型文档](docs/models/README.md#cpack-内置模型) 下载 Release 资产到默认路径。

Windows CUDA 开发机首次使用时，先用标准脚本初始化发布构建树和依赖，再进入同一套开发环境：

```powershell
# 仅全新环境或缺少依赖时需要 -InstallDeps
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -Jobs 8
. scripts\build_win\enter_plascan_dev_shell.ps1 -NoLaunch
```

以后每次修改代码，日常验证包只需运行：

```powershell
cmake --workflow --preset windows-package-smoke
& .\build\windows-vcpkg-cuda-release\package-smoke\PlaScan\bin\plascan.exe
```

该流程只增量构建 `plascan_gui` 及其依赖，并更新未压缩的 Runtime 安装树；不会重复执行数 GiB 的
Inno Setup 压缩。安装阶段仍会校验 U2Net、LightGlue、LoMa-R 五文件包以及 BiRefNet ONNX/provenance
的长度与 SHA-256，因此这个目录可以直接验证安装后的两种 AI 蒙版和两种匹配算法。依赖被删除或
安装布局发生变化时，先删除
`build/windows-vcpkg-cuda-release/package-smoke/PlaScan`，再运行一次 smoke 流程，避免保留旧 DLL。
升级 vcpkg、CUDA、TensorRT、Qt 或编译工具链后，应先重跑 `build_windows_cuda.ps1`（需要补依赖时加
`-InstallDeps`），让脚本重新同步运行时，再使用上述代码增量工作流。

即使开发机已安装 CUDA/OpenCL，也可用
`-DPLASCAN_ENABLE_CUDA=OFF -DPLASCAN_ENABLE_OPENCL=OFF -DPLASCAN_ENABLE_TENSORRT=OFF`
配置可重复的 CPU-only 构建。该配置会关闭 PlaMatrix/PlaPoint 的 CUDA 后端和 PlaPoint/MVS 的
OpenCL 后端，同时关闭 PlaMatrix 的 OpenCL 基础设施；MVS 仍编译并运行真实 CPU PatchMatch。

`cpack --preset windows-vcpkg-release` 生成 ZIP 离线包，归档第一层即为 `PlaScan/`。全架构 CUDA/TensorRT
ZIP 可能超过 GitHub Release 的 2 GiB 单文件上限，适合本地分发或内部制品库。正式发布时需使用
CMake/CPack 3.27+ 并安装 Inno Setup 6，然后运行完整工作流：

```powershell
cmake --workflow --preset windows-package-release
```

正式 Inno 包继续使用 `lzma2/ultra64` 与 solid compression，但默认通过
`PLASCAN_INNO_LZMA_BLOCK_THREADS=4` 并行压缩大文件；内存较小的发布机可降为 `2`，高核心数且内存充足
时可在配置 preset 中提高。日常开发仍优先使用不压缩的 `windows-package-smoke`。

工作流先增量构建同一个 CUDA Release 目录，再生成带开始菜单、桌面快捷方式、卸载入口和 `.plascan`
文件关联的安装程序；制品位于 `build/windows-vcpkg-cuda-release/packages/release`。只有准备正式交付时
才需要运行它，日常改动使用 `windows-package-smoke` 即可。

为完整保留各代 NVIDIA GPU 首次构建 TensorRT engine 所需的 builder resource，Inno Setup 默认使用
1,900,000,000 字节分卷。CPack 会输出同名 `.exe`、一个或多个 `-N.bin` 以及
`-INNOSETUP.sha256`；发布和安装时必须让 `.exe` 与全部 `.bin` 位于同一目录。每个 `.exe/.bin`
资产都会在打包时强制校验为小于 2 GiB，适合分别上传到同一个 GitHub Release。

Windows 构建使用原生 MSVC/Ninja/PowerShell，不需要 WSL。GUI 链接后会把当前 vcpkg triplet 和源码依赖的运行时 DLL 增量同步到 `build/bin`；PlaMatrix CPU 稠密线性代数不再带入 LAPACK/OpenBLAS DLL。打包后的 GUI 还需要 Qt platform plugins、GDAL/PROJ 数据目录和 TensorRT/CUDA 运行时 DLL；`PLASCAN_BUNDLE_RUNTIME=ON` 时 CMake install/CPack 会按主程序和 Qt 插件的传递依赖闭包收集 DLL，把 `share/gdal` 与 `share/proj` 一并安装，并补充 Vulkan、TensorRT Builder、NVRTC 和 nvFatbin 等动态加载运行时。要让内置 U2Net、BiRefNet、LightGlue 和 LoMa-R ONNX 安装后直接使用 GPU，发布构建必须启用 CUDA/TensorRT，并携带 `nvinfer`、`nvonnxparser`、plugin 和对应架构的 builder resource；OpenCV 始终保持 CPU-only，安装包不得携带 cuDNN 或开发机生成的 `.engine`。

PlaScan 只接受 OpenCV 5，并直接包含 `features`、`geometry` 与 `stereo` 模块头；不再保留 OpenCV 4
转发头或版本分支。通用 `*-vcpkg-*` preset 仅面向显式提供外部 OpenCV 5 的高级构建，推荐使用上述
锁定源码依赖入口。

Windows CUDA 开发机推荐固定使用 `scripts/build_win/build_windows_cuda.ps1`。首次运行前先用源码依赖
入口构建 OpenCV 5；脚本默认从 `build/windows-source-deps-release/install` 加载，也可通过
`-OpenCvRoot` 指定其它 OpenCV 5 安装前缀。脚本会把主构建目录收敛到
`build/windows-vcpkg-cuda-release`，并使用该目录自己的 `vcpkg_installed`、CUDA 13.1 和
构建目录自己的 TensorRT/CUDA 配置，避免其它 build cache 混入运行时 PATH。

正式 SfM/光束法平差使用 PlaMatrix；Windows CUDA 构建脚本不再安装额外的 BA 对照求解器。

标准 Windows CUDA 构建由 TensorRT 加速 U2Net 和 BiRefNet ONNX；U2Net 使用 OpenCV CPU 回退，
BiRefNet 使用 ONNX Runtime CPU 回退，不安装 `opencv-dnn-cuda`，也不链接或分发 cuDNN。
BiRefNet 自动模式优先使用所选 CUDA 设备。
首次准备依赖时运行：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps
pwsh scripts\build_win\build_windows_cuda.ps1
```

脚本通过 `-TensorRtRoot` 或 `TENSORRT_ROOT` 查找完整 TensorRT SDK，收集 runtime、ONNX parser、plugin、
全部 GPU 架构 builder resource 和所需 CUDA DLL。它会校验源码包确为 CPU-only OpenCV 5，清除
旧运行目录残留的 `cudnn*.dll`。发布前可在移除 CUDA/TensorRT/vcpkg 外部 PATH 的子进程中，从 ONNX
首次构建本机 engine 并执行真实 TensorRT 推理：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -Target test_mask_generation -RunU2NetTensorRtDeploymentTest
pwsh scripts\build_win\build_windows_cuda.ps1 -Target test_mask_generation -RunBiRefNetTensorRtDeploymentTest
```

两个部署门禁都会隔离用户缓存并清理外部 SDK PATH。BiRefNet 门禁从 `package-smoke` 读取已安装运行时
和模型，测试程序及 GTest DLL 只放在包外临时夹具；它要求首次运行新建 engine、第二个进程复用同一
engine，并确认安装树未产生 `.engine`。
RTX 4060 Laptop 8 GiB / TensorRT 10.15 的 FP16 实测中，首次 engine 构建加推理为 `2631483 ms`
（43 分 51 秒），engine `540031644` bytes；第二进程复用并推理为 `33573 ms`。首次使用应预留约
45 分钟，具体时间和缓存大小随 GPU、TensorRT 版本及磁盘性能变化。
通过该验证的发布目录不要求目标电脑安装 CUDA Toolkit、TensorRT SDK、cuDNN、vcpkg、PyTorch 或
Python，但仍需要受支持的 NVIDIA GPU 以及与所打包 CUDA/TensorRT 运行库兼容的 NVIDIA 驱动。

### Python 环境脚本

`scripts/env/` 集中管理 Python 本机环境准备脚本。Python 开发环境默认创建在仓库根目录 `.venv/`，用于模型导出和验证；生产 C++ 不链接 LibTorch。

安装版首次启动时会自动检查 Python 环境。若未找到，PlaScan 会询问是否下载并安装用户级专用环境；可以暂不处理，
也可以勾选“下次启动时不再提醒”。之后可随时通过 `帮助 > 更新 Python 环境...` 补装或更新。Windows 在系统没有
Python 时会从 python.org 下载经过数字签名校验的 Python 安装器，运行时安装到当前用户的应用数据目录，不要求管理员权限。
自动安装需要访问 python.org、PyTorch 软件源、PyPI 和 GitHub。

准备 vcpkg：

```bash
python scripts/env/setup_vcpkg.py --root /path/to/vcpkg --install
```

准备 Python 环境：

```bash
python scripts/env/setup_python_runtime.py --device cuda --cuda-wheel cu130
```

用生成的 `build/env/plascan-env.json` 配置、构建、测试和打包：

```bash
python scripts/env/configure_with_env.py --source-deps --build --test
```

Windows PowerShell 使用同一套脚本：

```powershell
python scripts\env\setup_vcpkg.py --root C:\src\vcpkg --clone --install --triplet x64-windows
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
python scripts\env\configure_with_env.py --source-deps --build
```

`.venv/` 已加入 git 忽略列表。后续需要运行 Python 模型导出、测试或辅助脚本时，优先复用这个环境；只有 CI、打包或特殊隔离场景才通过 `--runtime-dir` 指定其它虚拟环境位置。

### 工程文件

PlaScan 使用与 Metashape 相同的双实体工程结构：

```text
name.plascan
name.files/
├─ project.zip
├─ 1/
│  └─ chunk.zip
└─ 2/
   └─ chunk.zip
```

  `.plascan` 是轻量项目描述；`name.files/project.zip` 保存 Chunk 索引和项目 UI 状态，
  `1/`、`2/` 等只增不复用的数字目录分别保存各 Chunk 的元数据、影像和工作流产物。
  目录号单调递增且不复用，例如已经创建 `1/2/3`，删除 `2` 后再次新建会分配 `4`。
  复制、移动、重命名或备份时必须成对处理 `.plascan` 与 `.files`。资源使用项目 URI
  和 SHA-256 索引，不依赖原电脑盘符。旧版根级 `workspace/` 分体工程和旧版单体
  `.plascan` ZIP 均不再支持，打开时只报告格式错误，不修改旧工程内容。

  Windows 下 PlaScan 启动时会为当前用户注册 `.plascan` 文件关联。首次运行一次 PlaScan
  后，可在资源管理器中双击项目描述文件，软件将自动启动并通过异步加载流程打开该项目。

应用窗口和最近项目使用系统 `QSettings`；处理参数保存在 Chunk `doc.json` 的
`project_config` 字段；项目显示状态保存在根 `doc.json` 的 `ui_state` 字段。旧工程
不会自动迁移。格式细节见
[`docs/project/PLASCAN_PROJECT_FORMAT.md`](docs/project/PLASCAN_PROJECT_FORMAT.md)。

`文件 -> 导入` 提供两个工程级入口：`导入点云...` 接受 Metashape/通用 OBJ、PLY、XYZ，
`导入模型...` 接受 OBJ、PLY，并自动复制 OBJ 的 MTL 与纹理依赖。导入成果进入当前 Chunk 的
`assets/imported` 并登记到工作区，可直接参与模型查看、DEM 和点云 DOM 流程；Metashape
专有 `.psx/.oc3` 不直接读取，应先在 Metashape 中导出标准文件。

### GUI 工作流

GUI 的 `工作流程` 菜单按处理阶段提供互相独立的工程入口：

| 入口 | 输出 | 说明 |
|------|------|------|
| `空中三角测量` | 相机模型、连接点和正式稀疏点云 | 针孔影像执行位姿/内参 BA；全 RPC00B 影像固定传感器模型并执行 RPC 地面点空三，不自动进入密集重建 |
| `创建点云` | 密集点云 | 从当前项目的深度图融合生成密集点云 |
| `生成模型` | PLY/OBJ 三维模型 | 从当前项目已有的连接点、深度图或点云生成模型 |
| `生成纹理` | OBJ/MTL/PNG 纹理模型 | 将项目影像投影到已有模型，不重建几何 |
| `创建 DEM` | 局部 DEM、RPC 立体地理 DEM，或全球径向 DEM/DOM | 局部模式使用点云；RPC 模式使用带 RPC00B 的 8 位 GeoTIFF 立体像对；小天体模式使用体固连闭合三角网 |
| `生成正射影像` | 带覆盖 Alpha 的 DOM GeoTIFF/PNG | 常规模式支持 DEM/彩色点云；RPC 模式按地理 DEM 网格反投影 RPC GeoTIFF 并输出 GeoTIFF |

旧版 `工作流程 -> 三维重建` 一键对话框已移除；空三、密集处理、模型和地形产品由各自入口显式启动。

当参与空三的影像全部带有有效 RPC00B 时，工作流会自动进入 RPC 空三分支，不再将 RPC 静默降级为
估算焦距的针孔相机。无地面控制点时厂商 RPC 保持固定，连接点通过非线性 RPC 前方交会得到 WGS84
经纬高，并以带 WGS84 原点记录的局部 ENU 米制稀疏云写入工程；RPC/针孔混合批次会明确拒绝。

“生成纹理”的“色彩校正”默认关闭。显式启用后，v4 相机纹理只比较多个视图中通过深度、掩膜和
最终网格可见性检查的同一 3D 面中心，在 linear-sRGB 亮度上以中位数/MAD 鲁棒解算每视图标量
曝光增益，并把增益硬限制在 `0.90–1.10`。共同样本不足、成对亮度差离散度过高或视图重叠图不
连通时保持全部单位增益，不再根据整幅影像平均亮度强制校正。单视图、v3 兼容纹理和无相机的
顶点色回退路径不受影响。

“创建正射影像”对话框以产品模式区分常规摄影测量 DOM 和 RPC 地理正射 DOM。常规模式下，DEM 表面使用项目相机影像反投影，提供马赛克、加权
平均、首个有效影像、颜色校正、锐度权重、重影过滤、蒙版与孔洞处理；彩色点云表面直接使用
点的 RGB，可选择保留 XY 的局部平面投影，或按体固连经纬度生成小天体全球等距圆柱投影。
全球模式可自动估算点云中心和平均参考半径，也可手动指定中心、半径与中央经线，最终生效值
会回显并写入项目结果。RPC 模式直接跟随 DEM 投影、网格和范围，运行时逐张校验 GeoTIFF 的 RPC00B，
并登记覆盖率、预览及 JSON 报告。对话框采用双栏参数布局，切换 RPC 模式时会收起不生效的投影和区域参数；所有路径均支持后台进度与取消。

GeoTIFF 按 R/G/B/Alpha 波段写出，以 Alpha 区分无覆盖区和真实黑色。DEM 路径继承最终网格的
地理变换和 DEM WKT；点云平面路径写入本地米制 `LOCAL_CS`；全球路径写入以参考半径定义的
自定义小天体 `Equirectangular` WKT 和北向上仿射变换。当前全球体固连轴方向沿输入点云 XYZ，
因此跨批次拼接前应保证点云已统一到稳定的天体坐标框架。

“创建 DEM”中的“小天体全球径向 DEM + DOM”由 C++ 核心直接完成，不调用 Python/Matplotlib
脚本。核心从体心向行星中心经纬网的每个像元发射射线，以 BVH 加速的精确射线—三角形求交
获得半径，重心插值 PLY 顶点色或单图集 OBJ 纹理，并在同一 0–360° 正东经网格输出 `radial_dem.tif`、
`elevation_dem.tif`、带 Alpha 的 `dom.tif`、可靠性/覆盖栅格和四联图 PNG。可靠性表示面法向与
径向夹角的几何代理，不等同于多视支持度或测量精度；没有外部参考 DEM 时，报告右下角显示
hillshade，不会生成虚假的“官方误差”。模型必须已经位于体固连坐标系；`J2000`、`ICRF` 等
惯性系会被明确拒绝。所有坐标框架名称均标记为“用户声明、未验证”；若暂时只能确认轴向稳定，
可使用 `MODEL_LOCAL_BODY_FIXED`，不能冒充 IAU/SPICE 官方经纬网。由于 PLY/OBJ 不自带可靠的长度单位，GUI 和 CLI
都要求明确按 `m` 或 `km` 解释顶点坐标，核心统一换算为米并写入报告；参考半径和手动体心始终以米
输入。DOM 只接受 OBJ 纹理+UV 或 PLY/OBJ 的 RGB 顶点颜色，无真实颜色来源时会明确失败，不会输出
固定灰色占位图；多材质或多纹理 OBJ 会明确失败，避免把第一张纹理错误套到全部面。

同一能力也提供正式原生 CLI：

```bash
small_body_terrain_cli \
  --surface path/to/body_fixed_model.ply \
  --output-dir path/to/global_terrain \
  --target Ryugu \
  --body-fixed-frame RYUGU_FIXED \
  --surface-unit m \
  --reference-radius-m 448 \
  --angular-resolution-deg 0.1 \
  --manual-center --center-x 0 --center-y 0 --center-z 0
```

GUI 的“创建 DEM → RPC 立体 DEM”和“生成正射影像 → RPC 地理正射 DOM”已接入项目工作流；
带内嵌 RPC00B 的卫星 TIFF 立体像对可直接生成 WGS84 UTM DEM，并继续以该 DEM 对原始影像执行
RPC 正射投影生成 RGB+Alpha DOM：

```bash
rpc_stereo_products_cli \
  --left testData/rpc_stereo_pair/Images/img_01.tif \
  --right testData/rpc_stereo_pair/Images/img_02.tif \
  --output testData/rpc_stereo_pair/Products \
  --resolution 2.0
```

该链路输出 `dem.tif`、DEM 误差/点数/置信度/覆盖栅格、`stereo_points.ply`、`dom.tif` 以及两阶段
JSON 报告。高程为 WGS84 椭球高；水平投影按有效交会点中心自动选择 WGS84 UTM 分区。

### 重建链路状态

当前重建链路按四个阶段维护：

- MVS 稳定性：`MvsWorkspaceManifest` 记录每帧深度图状态、输入/输出路径、device、耗时、错误和配置 hash。深度图完成后写入项目 metadata，GUI 目录树按文件名自然排序刷新。
- MVS 质量：`MvsSourcePlanner` 基于 shared tracks、几何内点、三角角、覆盖率、baseline 和序列距离规划 source view。深度图同时输出 preview、raw depth、confidence 和 valid mask，融合阶段使用同一份 source plan。
- 模型生成：任意三维的深度源默认执行 `raw depth + confidence + valid/support mask + camera -> TSDF -> Marching Cubes`，不生成或消费密集点云中间产物。Visual Hull 与 Poisson 只保留为显式 legacy/诊断模式，TSDF 失败不会静默换算法。
- 模型显示：未计算顶点颜色且没有纹理时按真实面法线显示三角面；存在照片派生顶点颜色或完整纹理时才显示颜色。顶点颜色、纹理和网格几何是独立产品，不会因关闭颜色而改变模型拓扑。
- 深度检查：工作区只显示一个不可打开的聚合“深度图”节点，右键可删除整批最终层和金字塔层数据；照片工具栏的“显示深度信息”按当前照片叠加最终层或可用的金字塔诊断层，单个级别缺失不会禁用整个按钮。GUI 默认保存 Level 2/3 可视化栅格，关闭叠加后恢复原有特征点/残差显示偏好。
- Terrain 产品：`TerrainProductManifest` 记录 DEM/DOM、error、count、confidence 和 coverage 栅格。DEM/DOM 不再只是临时图，而是带质量 artifact 的 terrain product chain。
- 参考地形/QC：`ReconstructionQualityReport`、`PointCloudAlignment`、`DemDifference` 和 `ReferenceTerrainPrior` 支持外部 DEM/LiDAR 后验检查、点云/DEM 误差报告，以及 BA soft prior。

### CLI 一键重建

输入 `.lis` 文件每行是一组影像和相机文件，支持空格或逗号分隔：

```bash
path/to/image_001.png path/to/image_001.tsai
path/to/image_002.png path/to/image_002.tsai
```

外部相机文件可先用通用转换工具生成 PlaScan 输入。当前支持自动识别、Middlebury `*_par.txt`、
EPFL/Strecha `.camera`、COLMAP text sparse (`cameras.txt` / `images.txt`) 和 Metashape
`doc.xml` / `Project.files/0/chunk.zip`。Metashape adjusted calibration 中的 `k1/k2/k3/p1/p2`
会写入 PlaScan `.tsai`；暂不支持的 `k4/b1/b2` 会在 `summary.json` 中记录 warning：

```bash
cmake --build build --target camera_convert_cli -j$(nproc)
build/bin/camera_convert_cli --format auto \
  --input testData/photogrammetry_benchmarks/middlebury_dino_sparse_ring/extracted/dinoSparseRing \
  --output-dir build/camera_inputs/dino \
  --overwrite
```

三维建模专用 CLI 仍提供无 GUI 的批处理链路，只生成稀疏点云、密集点云和三维模型，不生成 DEM/DOM：

```bash
cmake --build build --target three_d_reconstruction_cli -j$(nproc)
build/bin/three_d_reconstruction_cli path/to/input.lis \
  --output-dir build/测试用临时文件/three_d_reconstruction \
  --device auto \
  --mvs-backend auto \
  --point-cloud-backend auto \
  --quality 3 \
  --threads 8 \
  --feature-max-image-dim 0
```

三个设备参数默认都是 `auto`。`--device` 控制稀疏前端；`--mvs-backend` 控制 PatchMatch；
`--point-cloud-backend` 控制 PlaPoint 过滤、降采样、法向估计和网格预处理。MVS 自动按
CUDA → OpenCL → CPU 选择；PlaPoint 的 SOR、RadiusOR、VoxelGrid 和法向估计会把低于传输收益门槛的
`Auto` 小任务留在 CPU，较大任务再尝试加速器；HeightGrid 与 Poisson 仍按各自的
CUDA → OpenCL → CPU 策略选择。显式指定的后端不可用时会明确失败，不会伪装成其他设备。
MVS 会在生成 workspace hash 和启动首帧前逐卡取得租约，并完成 OpenCL context/kernel 编译预检；
部分 CUDA 卡忙时继续使用其余 CUDA 卡，全部 CUDA 不可用时才进入 OpenCL，再失败才使用 CPU。
PlaMatrix 统一提供 OpenCL 1.2 设备枚举、context/command queue、program cache、device buffer 和执行封装；
PlaPoint 在这些公共设施之上只保留点云领域 kernel。第一阶段的 PlaPoint 高层接口仍是 CPU-owned：
SOR/Radius、Voxel、Normals 和 HeightGrid 的输入输出驻留主机，并仍包含主机建索引、排序、属性聚合或
协方差/SVD 等阶段。它可让非 NVIDIA GPU 参与计算，但是否快于原生 CPU 取决于点数、属性和驱动，
需以真实数据 benchmark 为准；超大近二维地表云或病态分布触发工作量保护时，Auto 会记录原因并回退
CPU，显式 OpenCL 则明确报错。显式 OpenCL 允许使用 NVIDIA 的 OpenCL 接口；Auto/混合调度仍会把
同一块 NVIDIA 物理 GPU 的 CUDA 与 OpenCL 接口去重，避免重复占用。
PlaMatrix 还提供 CPU-owned CSR 系统的 OpenCL Jacobi-PCG：矩阵和向量一次上传，稀疏乘法、预条件、
向量更新和分层归约在 GPU 执行，仅回读每轮收敛标量和最终解。Poisson 求解后端与点云预处理独立，
自动按 CUDA → OpenCL → CPU 选择；显式 OpenCL 会严格使用选中的 OpenCL 设备，设备或求解
失败时明确报错。
这层基础设施尚不代表 PlaMatrix 已提供通用 OpenCL GEMM 或 SVD。
多块 OpenCL GPU 并存时，可用 `PLAMATRIX_OPENCL_DEVICE_INDEX` 指定 PlaMatrix 枚举出的稳定设备索引；
未设置时会优先选择独立显卡和计算单元较多的设备。兼容期仍接受旧的
`PLAPOINT_OPENCL_DEVICE_INDEX`，但仅在对应 PlaMatrix 环境变量未设置时作为回退。
当前稀疏前端仍要求 CUDA/TensorRT，显式 `--device cpu` 会返回不支持错误，不会切换算法。
`--feature-max-image-dim 0` 表示使用质量档位的默认设置；最高质量档不会自动缩小 SIFT 输入。
显存紧张时可手动调小，
例如 `--feature-max-image-dim 1600`；传负数也会关闭缩放保护。

`bundle_adjust_cli` 默认请求 `--ba-backend auto`。BA 会先统计相机数、track 数和观测数：
point-only BA 使用 legacy/OpenMP；联合相机、共享内参或物方软约束的小问题使用 PlaMatrix CPU，达到阈值后
Auto 优先选择 PlaMatrix CUDA，其次选择 PlaMatrix OpenCL。
联合相机/三维点问题还可显式传 `--ba-backend plamatrix_cpu`、`plamatrix_cuda` 或
`plamatrix_opencl`，使用同一套 PlaMatrix 块法方程、Schur 消元和 LM。CPU 会按规模自动选择 PlaMatrix 原生
稠密 Cholesky 或矩阵自由块 Jacobi-PCG；CUDA/OpenCL 使用 CSR 块 Jacobi-PCG。线性容差会随 LM 进展动态收紧，
CPU 上限为 `2e-3`，CUDA/OpenCL 上限为 `1e-3`，以保持跨后端结果一致。法方程和目标代价只遍历一次，
LM 拒绝步复用当前线性化，track/Jacobian 采用确定性分片并行装配。设备路径会记录实际设备名且不隐式回退 CPU，
并复用 Schur CSR pattern、设备缓冲和固定拓扑；CUDA 的 Schur 数值由装配 kernel 直接交给 PCG，不再经过
device-host-device 往返。OpenCL 在 NVIDIA 595.84 驱动上保留稳定的主机 handoff，但同样复用装配缓冲和拓扑。
这些路径已进入 Auto 默认选择，并支持分组共享焦距、完整 Brown 内参以及
GCP/LiDAR/比例尺/位姿/相机平面/激光测距约束。GPU 不可用或求解失败时只回退 PlaMatrix CPU。
`ba_run_summary.json` 会写入 `ba_requested_backend`、`ba_used_backend`、`ba_used_gpu`、
`ba_valid_track_ratio`、setup/solve/total 耗时、
PlaMatrix 初始/最终代价、线性化/目标遍历次数、LM 接受/拒绝步数、线性求解器、设备名、PCG 迭代、
Schur 数值装配位置与耗时、混合精度实际使用状态、
质量门控和回退原因。Auto 后端会优先保证 RMS 和有效
track 比例；CUDA 候选若比 legacy 明显变差，
会自动回退而不是强行使用 GPU。
需要复现 point-only 旧路径时可传
`--ba-backend legacy_cpu`；联合 BA 可传 `--ba-backend plamatrix_cpu`、
`--ba-backend plamatrix_cuda` 或 `--ba-backend plamatrix_opencl`。
`--ba-plamatrix-device` 指定 PlaMatrix CUDA/OpenCL 设备索引；OpenCL 进程级选择仍由
`PLAMATRIX_OPENCL_DEVICE_INDEX` 初始化，二者不一致时明确报错。
`--ba-min-cuda-cameras` 和
`--ba-min-cuda-observations` 用于 PlaMatrix GPU 自动选择阈值。当前默认交叉阈值为 128 台相机且
30000 条观测；更小问题优先使用 CPU 稠密 Cholesky。

BA 后端基准可单独运行：

```bash
cmake --build build/windows-vcpkg-cuda-release --target ba_backend_benchmark -j32
python scripts/bench/run_ba_backend_benchmark.py \
  --exe build/windows-vcpkg-cuda-release/bin/ba_backend_benchmark.exe \
  --out build/ba_benchmarks/ba_backend_benchmark.csv \
  --summary-json build/ba_benchmarks/ba_backend_benchmark.json \
  --cases small,medium,large \
  --backends legacy_cpu,plamatrix_cpu,plamatrix_cuda,plamatrix_opencl,auto \
  --repeat 3 \
  --iterations 8 \
  --threads 32
```

也可以直接重放正式 SfM 输出的真实 BA 拓扑。`--dataset-json` 读取
`sfm_sparse_points.json` 中的三维点和像点观测，`--camera-list` 按顺序加载对应 TSAI 相机；
真实模式会关闭后端回退、质量门控对照和迭代日志，避免把第二次求解混入计时：

```powershell
build/windows-vcpkg-cuda-release/bin/ba_backend_benchmark.exe `
  --dataset-json build/benchmark_runs/<run>/sfm_sparse_points.json `
  --camera-list testData/photogrammetry_benchmarks/<dataset>/prepared/plascan/image_camera.lis `
  --backend plamatrix_cpu --iterations 20 --threads 32 --repetitions 5 --refine-pose
```

PlaMatrix CPU 按问题规模选择稠密 Cholesky 或矩阵自由块 Jacobi-PCG。基准调优时可用
`--max-dense-schur-cameras` 强制跨过稠密求解阈值；输出同时保留
`seconds` 兼容字段，并报告 API 墙钟、setup/solve/total、实际后端、实际线性求解器、PlaMatrix
LM/代价统计和 RMS。

调试和 benchmark 时可分阶段运行：`--stop-after-sfm` 只生成稀疏结果，`--skip-mvs` 在 SfM 后写报告并跳过后续阶段，
`--mvs-depth-only` 只生成 MVS 深度图、raw depth、confidence、valid mask 和 manifest，并在融合、网格和 terrain 前停止；
`--skip-mesh` 则保留 MVS 稠密点云但不生成网格。

批量测试 `testData/photogrammetry_benchmarks` 中已转换为 PlaScan 输入的数据：

```bash
python scripts/bench/run_photogrammetry_benchmarks.py \
  --root testData/photogrammetry_benchmarks \
  --output-dir build/benchmark_runs/photogrammetry_benchmarks \
  --stage sfm \
  --device cpu \
  --dry-run
```

完整地形产品流水线仍使用 `reconstruct_pipeline_cli` 或脚本封装，流程为 `SfM -> MVS 密集点云 -> 网格模型 -> DEM/DOM`：

```bash
cmake --build build --target reconstruct_pipeline_cli -j$(nproc)
python scripts/workflows/run_full_pipeline.py path/to/input.lis \
  --build-dir build \
  --output-dir build/测试用临时文件/full_pipeline \
  --device auto \
  --quality 3 \
  --feature-max-image-dim 0 \
  --dem-resolution 0
```

### Docker 构建

```bash
sudo docker build -t plascan-build -f docker/Dockerfile.ubuntu2404 .
./docker/shell.sh                    # 进入容器
./docker/build.sh                    # 一键构建+测试
./docker/package.sh                  # 打包 .deb
```

## 模块架构

```
src/
├── core/
│   ├── camera/                # 相机模型与外部相机格式转换
│   ├── image_matching/        # CUDA SIFT + TensorRT LightGlue、几何验证与 .pimatch I/O
│   ├── matchphototask/        # 候选对、任务内特征缓存、匹配及连接点编排
│   ├── aerial_triangulation/  # 对齐照片/空中三角测量工作流
│   ├── sfm/                   # 增量式 SfM + 光束法平差, ReferenceTerrainPrior
│   ├── mvs/                   # PatchMatch 深度图, MvsWorkspaceManifest, MvsSourcePlanner, 融合
│   ├── dense_match/           # MGM/SGM 密集立体匹配 (自研 CUDA)
│   ├── mesh/                  # Poisson 表面重建 + 纹理映射
│   ├── terrain/               # DEM/DOM, OrthoProjector, TerrainProductManifest, DEM 聚合与 mosaic
│   ├── stereo_dem/            # RPC TIFF 立体交会生成 DEM，并基于 DEM 生成 RPC DOM
│   ├── qc/                    # ReconstructionQualityReport, PointCloudAlignment, DemDifference
│   ├── overlap/               # 影像重叠度分析
│   ├── intersection/          # 前方交汇精度检验
│   └── pipeline/              # SfM 服务层
├── gui/                       # Qt6 图形界面
│   ├── dialogs/               # 参数配置对话框
│   ├── widgets/               # 3D 画布 + 影像查看器
│   └── project/               # 项目管理 (.plascan 归档)
├── cli/                       # 模块化命令行工具
│   ├── camera/                # 相机格式转换
│   ├── control_points/        # 标靶检测与打印
│   ├── features/              # 特征与连接点
│   ├── dense/                 # 密集重建阶段
│   ├── reconstruction/        # 可独立执行的重建阶段与诊断工具
│   ├── workflows/             # GUI“工作流程”菜单对应的 CLI
│   ├── quality/               # 质量验收
│   └── common/                # CLI 共享基础设施
└── common/                    # 通用工具 (日志, 数学, 结果包装)
3rdparty/
├── plapoint/   (submodule)    # 自研 GPU 点云库
└── plamatrix/  (submodule)    # 自研矩阵运算后端
```

## CLI 工具

CLI 在源码和 CMake 中按领域拆分。影像匹配接口已收敛为原始影像输入和逐影像 `.pimatch` 输出；
旧特征文件与成对 `.match` 参数不再兼容，调用脚本必须使用当前接口。
每个 CLI 模块也在自己的 `tests/` 中维护测试。模块职责和扩展规则见
[`src/cli/README.md`](src/cli/README.md)。

公共路径、token、UTF-8 控制台、JSON 和输出覆盖策略由 `plascan_cli_support` 统一提供。
一键重建进一步拆为 Options、Runner、Progress 和 Report；密集点云细化、流式深度融合及点云
PLY 写出由 `core/mvs` 服务承担，避免 CLI 与核心流程维护两套实现。

### 相机格式转换 (`camera_convert_cli`)

```bash
camera_convert_cli --list-formats
camera_convert_cli --format middlebury-par -i ./dinoSparseRing -o ./plascan_cameras --overwrite
camera_convert_cli --format epfl-camera -i ./epfl_scene -o ./plascan_cameras --overwrite
camera_convert_cli --format colmap-text -i ./south-building/sparse -o ./plascan_cameras --overwrite
camera_convert_cli --format metashape-xml -i ./depth_images -o ./plascan_cameras --overwrite
```

输出目录包含 `image_camera.lis`、`cameras/*.tsai` 和 `summary.json`，可直接传给重建类 CLI。
Metashape adjusted calibration 中的 `k1/k2/k3/p1/p2` 会写入 `.tsai`。

### 双影像匹配 (`feature_match_cli`)

```powershell
# 默认 Auto SIFT；无需下载模型，为 A、B 分别写一个 .pimatch 分片
feature_match_cli -L A.tif -R B.tif `
  -o E:\project\assets\image_matches `
  -a auto_sift --max-keypoints 40000 --guided-image-matching
```

SIFT 或 LoMa-R 描述子只存在于本次任务的内存缓存。Auto SIFT 按 CUDA、Metal、OpenCL、CPU 的顺序
选择可用后端且不依赖模型；可用 `--device cpu|cuda|opencl|metal` 显式固定设备，设备不可用时明确失败。
除快速档外，Auto SIFT 会保留原图特征，并只在 8x8 空间网格中覆盖不足的有效区域追加稳健灰度拉伸与
CLAHE 局部对比度增强特征；两个通道独立执行双向 ratio 互检后再合并，补点仍需通过 USAC 几何验证。
LightGlue 和 LoMa-R 都只使用 TensorRT；Release
分发 ONNX，目标机器首次使用时由 C++ TensorRT Builder 生成并缓存本机 engine。最终分片保存关键点观测、
相邻影像、置信度、几何内点和残差，不生成独立特征文件或 JSON sidecar。ONNX 导出、固定容量和
精度策略见 [docs/models/README.md](docs/models/README.md#sift--lightglue-tensorrt)。

### 密集重建流水线

```bash
feature_match_cli   -L A.tif -R B.tif -o ./assets/image_matches -a auto_sift
rectify_cli         -L A.tif -R B.tif --camL A.txt --camR B.txt -o rect
dense_match_cli     -L rect_L.tif -R rect_R.tif -o disp.tif --cuda --algorithm mgm
triangulate_cli     -d disp.tif --rect rect.xml --camL A.txt --camR B.txt -o cloud.ply
```

## 模型文件

工作流程设置从 [`models-v1.1.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.1.0)
下载 U2Net、LightGlue 和 LoMa-R，从
[`models-v1.2.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.2.0) 下载 BiRefNet Dynamic
ONNX/provenance，并按源码运行或安装版自动选择可写模型目录。TensorRT engine 不再作为跨机器资产发布。

标准 CPack 包已经内置 U2Net 和 LightGlue ONNX，CUDA 变体另内置 BiRefNet ONNX/provenance；程序会
直接从安装根下的 `resources/models` 发现它们。
安装目录中的 ONNX 只读使用。U2Net 首次构建的本机 engine 写入用户本地应用数据目录下
`models/u2net/engines/<fingerprint>`，BiRefNet、LightGlue 和 LoMa-R 使用各自的 engine 缓存；程序不会尝试修改
`Program Files`、`/opt/plascan` 或便携包目录，也不会把绑定 GPU/TensorRT 版本的 engine 打进安装包。

“生成蒙版 → AI: U2Net ONNX”会自动检测 `U2Net_v1.onnx`；缺失时可直接在对话框中下载并校验。
源码构建写入仓库 `resources/models/`，安装包运行则写入用户应用数据目录，避免修改只读安装目录。
NVIDIA GPU 默认使用 TensorRT FP16/FP32，TensorRT 不可用时按用户设置明确报错或回退 OpenCV CPU；
OpenCV 不使用 CUDA/cuDNN。

“生成蒙版 → AI: BiRefNet Dynamic（推荐）”使用固定 `1×3×1024×1024` RGB float32 模型：原图按宽高比
letterbox 后执行 ImageNet 归一化，C++ 端对 `output_image` 原始前景 logits 做 sigmoid，再裁掉 padding 并恢复
原尺寸。自动模式优先 TensorRT GPU，不可用时回退 ONNX Runtime CPU；最终用户部署不需要 Python 或
PyTorch。BiRefNet CPU 推理固定使用 ONNX Runtime，不经过 OpenCV DNN。

或通过导出脚本生成：

```bash
python scripts/models/export_lightglue_tensorrt.py
python scripts/models/export_loma_r_tensorrt.py --help
python scripts/models/export_birefnet_dynamic_onnx.py --help
```

## 平台支持

| 功能 | Windows (NVIDIA) | Linux (NVIDIA) | macOS (Apple Silicon) |
|------|:---:|:---:|:---:|
| CUDA 加速 | ✅ | ✅ | ❌ (MPS via PyTorch) |
| MVS PatchMatch | CUDA/OpenCL/CPU | CUDA/OpenCL/CPU | CPU |
| 点云预处理 | CUDA/OpenCL/CPU | CUDA/OpenCL/CPU | CPU |
| Poisson 稀疏 PCG | CUDA/OpenCL/CPU | CUDA/OpenCL/CPU | CPU |
| dense_match MGM/SGM | CUDA + CPU | CUDA + CPU | CPU only |
| CUDA SIFT + TensorRT LightGlue | CUDA | CUDA | 不支持 |
| TensorRT LoMa-R | CUDA | CUDA | 不支持 |
| BiRefNet Dynamic 蒙版 | TensorRT / ONNX Runtime CPU | TensorRT / ONNX Runtime CPU | ONNX Runtime CPU |
| U2Net 蒙版 | TensorRT/CPU | TensorRT/CPU | CPU |
| 全部 CLI 工具 | ✅ | ✅ | ✅ |
| Qt6 GUI | ✅ | ✅ | ✅ |
| CPack 打包 | ZIP/INNOSETUP | TGZ/DEB | TGZ |
| Docker 构建 | — | ✅ | — |

Windows/Linux 上的 AMD 与 Intel GPU 可通过 OpenCL 运行 MVS PatchMatch、点云阶段和 Poisson 稀疏求解；
标准影像到模型的稀疏特征前端目前仍依赖 NVIDIA CUDA/TensorRT，因此尚不是全链路无 NVIDIA 方案。

## 开发

```bash
git checkout -b feat/<name>    # 从 main 创建特性分支
# ... TDD: 红 → 绿 → 重构 ...
git checkout main && git merge feat/<name> --no-ff
git push origin main
```

代码规范见 `CLAUDE.md`（单文件 ≤ 400 行、嵌套 ≤ 4 层、Allman 花括号）。

### 回归测试建议

面向工作流和地形产品的改动至少运行：

```bash
cmake --build build --target test_ortho_generation test_map_project_dialog test_gui_project_utils test_mesh_reconstructor test_terrain_dem_dom plascan_gui -j$(nproc)
python scripts/env/run_tests.py --test-dir build \
  -R "OrthoGeneration|OrthoGridPlanner|OrthoProjector|MapProjectDialog" \
  --output-on-failure
QT_QPA_PLATFORM=offscreen ./build/tests/test_map_project_dialog
QT_QPA_PLATFORM=offscreen ./build/tests/test_gui_project_utils
./build/tests/test_mesh_reconstructor
./build/src/core/terrain/test_terrain_dem_dom
```

这些测试分别覆盖：

- 正射核心：参数解析、X/Y 像元与最大尺寸网格规划、融合、项目蒙版、孔洞、零覆盖和取消。
- 正射 GUI：仅开放真实支持项、DEM 元数据估算、设置往返、后台进度和取消。
- GUI 工作流边界：空三、模型、DEM 和 DOM 使用独立入口。
- 模型生成降级路径：点云缺少法向量时 Poisson 重建能回退到可用网格。
- DEM/DOM 工程质量：DEM 栅格、点云颜色/强度保留、DOM 锐度融合、OBJ/MTL 纹理、目录瓦片拼接输出。

## 文档

| 文档 | 内容 |
|------|------|
| `docs/PROJECT_ARCHITECTURE.md` | 完整目录树、模块职责、数据流、技术债务 |
| `docs/plans/2026-08-10-high-priority-code-optimization.md` | 数据安全、密集匹配、并发、MVS 与构建门禁优化计划 |
| `CONTEXT.md` | 当前环境、构建状态、系统依赖 |
| `docs/BUILD_MACOS.md` | macOS Apple Silicon 构建指南 |
| `docs/superpowers/specs/` | 功能规格说明 |

## 许可

MIT License，详见 [LICENSE](LICENSE)。
