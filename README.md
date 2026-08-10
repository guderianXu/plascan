# PlaScan

[![CI](https://github.com/guderianXu/plascan/actions/workflows/ci.yml/badge.svg)](https://github.com/guderianXu/plascan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**行星表面摄影测量处理系统** — 从多视角影像生成高精度三维模型。

## 快速开始

### 依赖

- C++20 编译器：MSVC 2022、GCC 11+ 或 Clang 15+。
- CMake 3.25+ 和 Ninja。
- Qt6、OpenCV 4、GDAL、libtiff、libzip、OpenMP、GTest 和 TensorRT（GPU 匹配）。
- CUDA Toolkit 可选；启用后用于深度学习特征、匹配、MVS、点云处理和 dense match 加速。
- OpenCL 1.2 SDK/loader 可选；启用后可由 AMD、Intel 或 NVIDIA GPU 加速 MVS 与点云预处理。
- Python 3.10+ 可选；用于模型导出、数据准备和脚本化验证。

### 克隆并构建

源码需要递归拉取 `plapoint` 和 `plamatrix` 两个 submodule：

```bash
git clone --recurse-submodules https://github.com/guderianXu/plascan.git
git submodule update --init --recursive
cd plascan
```

系统已经安装好依赖时，可直接配置本机构建目录：

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
python scripts/env/run_tests.py --test-dir build --output-on-failure
```

GUI 需要 Qt 6.7 或更高版本（使用 `QRhiWidget`）。Ubuntu 24.04 等只提供较旧 Qt 的环境仍可构建核心库、
CLI 和非 GUI 测试：

```bash
cmake -S . -B build -DBUILD_TESTS=ON -DPLASCAN_BUILD_GUI=OFF
cmake --build build -j$(nproc)
python scripts/env/run_tests.py --test-dir build --output-on-failure
```

项目通过 git submodule 引用自研点云库 [plapoint](https://github.com/guderianXu/plapoint) 和矩阵库 [plamatrix](https://github.com/guderianXu/plamatrix)，无需额外安装。

### vcpkg / CPack 跨平台构建

推荐新环境优先使用 `vcpkg.json` 和 `CMakePresets.json`。vcpkg 负责 Qt6、OpenCV 4、GDAL、libtiff、libzip、GTest 等通用依赖；CUDA 与 TensorRT 通过外部安装路径提供。

Linux:

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset linux-vcpkg-release
cmake --build --preset linux-vcpkg-release
python scripts/env/run_tests.py --preset linux-vcpkg-release
cpack --preset linux-vcpkg-release
```

Linux 正式交付建议使用 Ubuntu 24.04 x86_64 基线的一键 DEB 工作流。首次配置会由 vcpkg 构建带
XCB/OpenSSL 的 Qt 运行时；构建机还需安装 GCC/G++、`gfortran`、Ninja、`pkg-config` 和 `patchelf`。
打包前还需按下文准备两份 ONNX。日常修改代码后运行：

```bash
sudo apt install build-essential gfortran ninja-build pkg-config patchelf
```

然后运行增量打包检查：

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --workflow --preset linux-package-smoke
build/linux-vcpkg-package-release/package-smoke/rootfs/opt/plascan/bin/plascan
```

该流程只增量编译并更新未压缩 rootfs，同时强制检查动态库闭包、Qt XCB/offscreen 插件、GDAL/PROJ
数据、相对 RUNPATH 和模型哈希。正式生成通用 CPU/OpenCL 包时运行：

```bash
cmake --workflow --preset linux-package-deb
sudo apt install ./build/linux-vcpkg-package-release/packages/release/plascan_1.1.7_amd64.deb
```

通用包可在没有 CUDA 的 Ubuntu 24.04 x86_64 电脑直接启动并运行 U2Net CPU 掩模；当前生产影像匹配
算法为 CUDA SIFT + TensorRT LightGlue，因此需要完整匹配功能时使用 CUDA 变体：

```bash
# 构建机需要 CUDA 13.1、cuDNN 9、TensorRT 10.15.1（CUDA 13.1 变体）及 ONNX parser 开发库
cmake --workflow --preset linux-package-cuda-smoke
cmake --workflow --preset linux-package-cuda-deb
sudo apt install ./build/linux-vcpkg-cuda-package-release/packages/release/plascan-cuda_1.1.7_amd64.deb
```

CUDA DEB 不捆绑 NVIDIA 驱动和受系统 ABI 约束的 CUDA/TensorRT 库，而是在包元数据中声明 CUDA
13.1、cuDNN 9 与 TensorRT 10.15.1 CUDA 13.1 变体的运行时依赖；目标电脑需预先启用 NVIDIA Ubuntu
仓库并安装兼容驱动，随后 `apt install ./plascan-cuda_*.deb` 会补齐运行时。vcpkg 的 Qt、OpenCV、GDAL 等非系统 `.so`、Qt
plugins、GDAL/PROJ 数据和两份 ONNX 会装到 `/opt/plascan`，桌面入口和图标装到标准 `/usr` 路径。
两个包互斥，不能同时安装。DEB 与 `.sha256` 位于各自构建目录的 `packages/release`；依赖或安装布局
变化后应清理精确的 `package-smoke/rootfs` 再重新运行，普通源码修改无需反复做完整压缩。

Windows 通用 Release/ZIP（不等同于完整 CUDA/TensorRT 发布环境）：

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"
cmake --preset windows-vcpkg-release
cmake --build --preset windows-vcpkg-release
python scripts\env\run_tests.py --preset windows-vcpkg-release
cpack --preset windows-vcpkg-release
```

CPack 默认启用 `PLASCAN_BUNDLE_ONNX_MODELS=ON`，安装 U2Net 与 LightGlue。Windows CUDA 打包 preset
还会启用 `PLASCAN_BUNDLE_LOMA_R_MODELS=ON`，把 LoMa-R 完整便携模型包一并装入只读安装树：

- `resources/models/U2Net_v1.onnx`：图像掩模；
- `resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx`：CUDA SIFT + LightGlue 匹配。
- `resources/models/loma_r_tensorrt/`：LoMa-R 共享特征/动态匹配 ONNX 和 K1024/K2048/K3840 清单。

模型默认从源码树同名路径读取，也可在配置时通过 `PLASCAN_U2NET_ONNX_PATH`、
`PLASCAN_LIGHTGLUE_ONNX_PATH` 和 `PLASCAN_LOMA_R_MODEL_DIR` 指向外部缓存。安装/CPack 阶段会校验
固定字节数和 SHA-256；缺失、损坏或拿错版本都会停止打包，且任何本机 `.engine` 都禁止进入安装包。
只需开发用轻量安装树时可显式设置
`-DPLASCAN_BUNDLE_ONNX_MODELS=OFF`。干净 clone 不包含大模型，打包前按
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
Inno Setup 压缩。安装阶段仍会校验 U2Net、LightGlue 和 LoMa-R 五文件模型包的长度与 SHA-256，
因此这个目录可以直接验证安装后的图像掩模和两种匹配算法。依赖被删除或安装布局发生变化时，先删除
`build/windows-vcpkg-cuda-release/package-smoke/PlaScan`，再运行一次 smoke 流程，避免保留旧 DLL。
升级 vcpkg、CUDA、cuDNN、Qt 或编译工具链后，应先重跑 `build_windows_cuda.ps1`（需要补依赖时加
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

Windows 构建使用原生 MSVC/Ninja/PowerShell，不需要 WSL。GUI 链接后会把当前 vcpkg triplet 的运行时 DLL 增量同步到 `build/bin`，保证直接启动时包含 LAPACK/OpenBLAS 等传递依赖。打包后的 GUI 还需要 Qt platform plugins 和 TensorRT/CUDA 运行时 DLL；`PLASCAN_BUNDLE_RUNTIME=ON` 时 CMake install/CPack 会按主程序和 Qt 插件的传递依赖闭包收集 DLL，并补充 Vulkan、cuDNN 和 NVRTC 等动态加载运行时。要让内置 LightGlue ONNX 安装后直接匹配，发布构建必须启用 CUDA/TensorRT，并携带 `nvinfer`、`nvonnxparser` 和对应架构的 builder resource；CPU-only 包即使含有 ONNX 也不具备该匹配后端。

当前 manifest 使用 vcpkg 中可用的 OpenCV 4.x port。后续 vcpkg 正式提供 OpenCV 5 后，优先通过更新 `builtin-baseline`、OpenCV feature 列表和现有 `OpenCvCompat` 兼容测试切换。

Windows CUDA 开发机推荐固定使用 `scripts/build_win/build_windows_cuda.ps1`。脚本会把主构建目录收敛到
`build/windows-vcpkg-cuda-release`，并使用该目录自己的 `vcpkg_installed`、CUDA 13.1 和
构建目录自己的 TensorRT/CUDA 配置，避免其它 build cache 混入运行时 PATH。

同一脚本默认启用 `ceres-cuda` manifest feature，用于 SfM/光束法平差中的 Ceres CUDA 后端；
基础 Ceres 依赖同时启用 LAPACK 和 SuiteSparse，使 GPU 回退及 Linux/CPU 构建可使用稀疏 Schur 求解器。
如果只想构建 CPU/legacy BA，可传 `-EnableCeresCudaBa:$false`。已有 `vcpkg_installed` 若仍是
CPU 版 Ceres，脚本会提示重新运行 `-InstallDeps`，避免界面显示 CUDA 但实际只跑 CPU。

标准 Windows CUDA 构建默认同时安装 `opencv-dnn-cuda` manifest feature 和 cuDNN，使 U2Net ONNX
蒙版真正使用 OpenCV DNN CUDA 后端。首次准备依赖时运行：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps
pwsh scripts\build_win\build_windows_cuda.ps1
```

脚本需要可发现的 cuDNN 开发包（`include/cudnn.h` 和 `lib/x64/cudnn.lib`），可通过 `-CudnnRoot`
显式指定。只有明确不需要 U2Net CUDA 时才传 `-EnableOpenCvDnnCuda:$false`；GUI 默认禁止静默回退，
选择 CUDA 但后端不可用会给出明确错误，用户勾选“CUDA 不可用时回退 CPU”后才允许回退。

脚本会从 cuDNN 的 `bin` 和官方归档常用的 `bin/x64` 布局收集全部 `cudnn*.dll`，并校验部署目录
中的 OpenCV DNN、CUDA、cuBLAS 和 cuDNN 运行库。发布前可在移除 CUDA/cuDNN/vcpkg 外部 PATH 的
子进程中执行真实 U2Net CUDA 推理：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -Target test_mask_generation -RunU2NetCudaDeploymentTest
```

通过该验证的发布目录不要求目标电脑另行安装 CUDA Toolkit 或 cuDNN，但仍需要受支持的 NVIDIA GPU
以及与所打包 CUDA 运行库兼容的 NVIDIA 驱动。

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
python scripts/env/configure_with_env.py --build-type release --build --test --package
```

Windows PowerShell 使用同一套脚本：

```powershell
python scripts\env\setup_vcpkg.py --root C:\src\vcpkg --clone --install --triplet x64-windows
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
python scripts\env\configure_with_env.py --build-type release --build
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

### GUI 一键工作流

GUI 的 `工作流程` 菜单按处理阶段提供互相独立的工程入口：

| 入口 | 输出 | 说明 |
|------|------|------|
| `空中三角测量` | 相机外参、连接点和正式稀疏点云 | 只负责影像对齐和 BA，不自动进入密集重建 |
| `生成模型` | PLY/OBJ 三维模型 | 从当前项目已有的连接点、深度图或点云生成模型 |
| `创建 DEM` | 局部 DEM，或全球径向 DEM、高程 DEM、DOM、可靠性与四联图报告 | 局部模式使用点云；小天体模式使用已处于体固连坐标的闭合 PLY/OBJ 三角网 |
| `生成正射影像` | 带覆盖 Alpha 的 DOM GeoTIFF/PNG | 可按 DEM+影像反投影，或从彩色点云生成局部平面/小天体全球 DOM |

旧版 `工作流程 -> 三维重建` 一键对话框已移除；空三、密集处理、模型和地形产品由各自入口显式启动。

“创建正射影像”对话框有两类生产路径：DEM 表面使用项目相机影像反投影，提供马赛克、加权
平均、首个有效影像、颜色校正、锐度权重、重影过滤、蒙版与孔洞处理；彩色点云表面直接使用
点的 RGB，可选择保留 XY 的局部平面投影，或按体固连经纬度生成小天体全球等距圆柱投影。
全球模式可自动估算点云中心和平均参考半径，也可手动指定中心、半径与中央经线，最终生效值
会回显并写入项目结果。两条路径均支持独立像元/最大尺寸、范围裁剪、后台进度与取消。

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
`--point-cloud-backend` 控制 PlaPoint 过滤、降采样、法向估计和网格预处理。后两者自动按
CUDA → OpenCL → CPU 选择；显式指定的后端不可用时会明确失败，不会伪装成其他设备。
MVS 会在生成 workspace hash 和启动首帧前逐卡取得租约，并完成 OpenCL context/kernel 编译预检；
部分 CUDA 卡忙时继续使用其余 CUDA 卡，全部 CUDA 不可用时才进入 OpenCL，再失败才使用 CPU。
PlaMatrix 统一提供 OpenCL 1.2 设备枚举、context/command queue、program cache、device buffer 和执行封装；
PlaPoint 在这些公共设施之上只保留点云领域 kernel。第一阶段的 PlaPoint 高层接口仍是 CPU-owned：
SOR/Radius、Voxel、Normals 和 HeightGrid 的输入输出驻留主机，并仍包含主机建索引、排序、属性聚合或
协方差/SVD 等阶段。它可让非 NVIDIA GPU 参与计算，但是否快于原生 CPU 取决于点数、属性和驱动，
需以真实数据 benchmark 为准；超大近二维地表云或病态分布触发工作量保护时，Auto 会记录原因并回退
CPU，显式 OpenCL 则明确报错。
这层基础设施不代表 PlaMatrix 已提供 OpenCL GEMM、SVD、CSR 或稀疏 PCG。Poisson 求解后端与点云
预处理独立，当前自动在 CUDA 和 CPU 之间选择；因此显式 OpenCL 仍可用于前处理，而不会被误传给 Poisson。
多块 OpenCL GPU 并存时，可用 `PLAMATRIX_OPENCL_DEVICE_INDEX` 指定 PlaMatrix 枚举出的稳定设备索引；
未设置时会优先选择独立显卡和计算单元较多的设备。兼容期仍接受旧的
`PLAPOINT_OPENCL_DEVICE_INDEX`，但仅在对应 PlaMatrix 环境变量未设置时作为回退。
当前稀疏前端仍要求 CUDA/TensorRT，显式 `--device cpu` 会返回不支持错误，不会切换算法。
`--feature-max-image-dim 0` 表示使用质量档位的默认设置；最高质量档不会自动缩小 SIFT 输入。
显存紧张时可手动调小，
例如 `--feature-max-image-dim 1600`；传负数也会关闭缩放保护。

`bundle_adjust_cli` 默认请求 `--ba-backend auto`。BA 会先统计相机数、track 数和观测数：
point-only BA 和小规模局部 BA 优先使用 legacy/OpenMP 或 Ceres CPU；需要相机位姿优化且问题规模足够大时，
Auto 会尝试 Ceres CUDA dense Schur；固定相机的显式 point-only CUDA 请求才使用 PlaScan 自研
`native_cuda` 点块后端。
`ba_run_summary.json` 会写入 `ba_requested_backend`、`ba_used_backend`、`ba_used_gpu`、
`ba_ceres_linear_solver`、`ba_valid_track_ratio`、setup/solve/total 耗时、native CUDA 活动工作集统计、
质量门控和回退原因。Auto 后端会优先保证 RMS 和有效 track 比例；CUDA 候选若比 legacy 明显变差，
会自动回退而不是强行使用 GPU。显式请求大规模 point-only Ceres 时也会按安全阈值回退，
避免 dense QR 大矩阵不稳定。
需要复现旧路径时可传
`--ba-backend legacy_cpu`；需要强制 Ceres CPU 或 CUDA 时分别传 `--ba-backend ceres_cpu` /
`--ba-backend ceres_cuda`；需要强制自研 CUDA 路径时传 `--ba-backend native_cuda`。
可用 `--ba-native-cuda-device` 和 `--ba-native-cuda-max-point-step` 调整显式
native CUDA point-only 求解的设备与点块步长；`--ba-min-cuda-cameras` 和
`--ba-min-cuda-observations` 用于 Ceres CUDA 自动选择阈值。
Ceres CPU 按相机规模自动选择 Dense/Sparse/Iterative Schur；CUDA 在求解前按
`--ba-max-cuda-memory-fraction` 检查 dense 工作集显存预算。Ceres CUDA 当前加速的是
Ceres dense Schur 线性求解环节，不加速 residual/Jacobian 构建和 BA 输入构建。
native CUDA 当前首期接入的是固定相机投影下的 GPU 三维点块求解，并接入 Auto 质量门控；
相机 Schur/PCG 更新尚未实现，因此不再暴露伪 PCG 参数或统计。

BA 后端基准可单独运行：

```bash
cmake --build build/windows-vcpkg-cuda-release --target ba_backend_benchmark -j32
python scripts/bench/run_ba_backend_benchmark.py \
  --exe build/windows-vcpkg-cuda-release/bin/ba_backend_benchmark.exe \
  --out build/ba_benchmarks/ba_backend_benchmark.csv \
  --summary-json build/ba_benchmarks/ba_backend_benchmark.json \
  --cases small,medium,large \
  --backends legacy_cpu,ceres_cpu,ceres_cuda,native_cuda,auto \
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
  --backend ceres_cpu --iterations 20 --threads 32 --repetitions 5 --refine-pose
```

默认按生产阈值选择 Dense/Sparse/Iterative Schur。基准调优时可用
`--max-dense-schur-cameras` 和 `--max-sparse-schur-cameras` 强制跨过规划阈值；输出同时保留
`seconds` 兼容字段，并报告 API 墙钟、setup/solve/total、实际后端、实际线性求解器和 RMS。

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
# CUDA SIFT + TensorRT LightGlue；为 A、B 分别写一个 .pimatch 分片
feature_match_cli -L A.tif -R B.tif `
  -o E:\project\assets\image_matches `
  -m resources\models\lightglue_tensorrt\lightglue_sift_bucket4096.onnx `
  -a sift_lightglue --max-keypoints 40000
```

SIFT 或 LoMa-R 描述子只存在于本次任务的内存缓存。LightGlue 和 LoMa-R 都只使用 TensorRT；Release
分发 ONNX，目标机器首次使用时由 C++ TensorRT Builder 生成并缓存本机 engine。最终分片保存关键点观测、
相邻影像、置信度、几何内点和残差，不生成独立特征文件或 JSON sidecar。ONNX 导出、固定容量和
精度策略见 [docs/models/README.md](docs/models/README.md#sift--lightglue-tensorrt)。

### 密集重建流水线

```bash
feature_match_cli   -L A.tif -R B.tif -o ./assets/image_matches -m lightglue_sift_bucket4096.onnx
rectify_cli         -L A.tif -R B.tif --camL A.txt --camR B.txt -o rect
dense_match_cli     -L rect_L.tif -R rect_R.tif -o disp.tif --cuda --algorithm mgm
triangulate_cli     -d disp.tif --rect rect.xml --camL A.txt --camR B.txt -o cloud.ply
```

## 模型文件

工作流程设置从 [`models-v1.1.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.1.0)
下载便携 ONNX，并按源码运行或安装版自动选择可写模型目录。TensorRT engine 不再作为跨机器资产发布。

标准 CPack 包已经内置 U2Net 和 LightGlue ONNX；程序会直接从安装根下的 `resources/models` 发现它们。
安装目录中的 ONNX 只读使用，LightGlue 首次构建的本机 engine 写入用户模型目录下的
`lightglue_tensorrt/engines`，不会尝试修改 `Program Files`、`/opt/plascan` 或便携包目录。

“生成蒙版 → AI: U2Net ONNX”会自动检测 `U2Net_v1.onnx`；缺失时可直接在对话框中下载并校验。
源码构建写入仓库 `resources/models/`，安装包运行则写入用户应用数据目录，避免修改只读安装目录。

或通过导出脚本生成：

```bash
python scripts/models/export_lightglue_tensorrt.py
python scripts/models/export_loma_r_tensorrt.py --help
```

## 平台支持

| 功能 | Windows (NVIDIA) | Linux (NVIDIA) | macOS (Apple Silicon) |
|------|:---:|:---:|:---:|
| CUDA 加速 | ✅ | ✅ | ❌ (MPS via PyTorch) |
| MVS PatchMatch | CUDA/OpenCL/CPU | CUDA/OpenCL/CPU | CPU |
| 点云预处理 | CUDA/OpenCL/CPU | CUDA/OpenCL/CPU | CPU |
| dense_match MGM/SGM | CUDA + CPU | CUDA + CPU | CPU only |
| CUDA SIFT + TensorRT LightGlue | CUDA | CUDA | 不支持 |
| TensorRT LoMa-R | CUDA | CUDA | 不支持 |
| 全部 CLI 工具 | ✅ | ✅ | ✅ |
| Qt6 GUI | ✅ | ✅ | ✅ |
| CPack 打包 | ZIP/INNOSETUP | TGZ/DEB | TGZ |
| Docker 构建 | — | ✅ | — |

Windows/Linux 上的 AMD 与 Intel GPU 可通过 OpenCL 运行 MVS PatchMatch 和上述 PlaPoint 点云阶段；
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
| `CONTEXT.md` | 当前环境、构建状态、系统依赖 |
| `docs/BUILD_MACOS.md` | macOS Apple Silicon 构建指南 |
| `docs/superpowers/specs/` | 功能规格说明 |

## 许可

MIT License，详见 [LICENSE](LICENSE)。
