# PlaScan

[![CI](https://github.com/guderianXu/plascan/actions/workflows/ci.yml/badge.svg)](https://github.com/guderianXu/plascan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**行星表面摄影测量处理系统** — 从多视角影像生成高精度三维模型。

## 快速开始

### 依赖

- C++20 编译器：MSVC 2022、GCC 11+ 或 Clang 15+。
- CMake 3.25+ 和 Ninja。
- Qt6、OpenCV 4、GDAL、libtiff、libzip、OpenMP、GTest 和 TensorRT（GPU 匹配）。
- CUDA Toolkit 可选；启用后用于深度学习特征、匹配、MVS 和 dense match 加速。
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
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

GUI 需要 Qt 6.7 或更高版本（使用 `QRhiWidget`）。Ubuntu 24.04 等只提供较旧 Qt 的环境仍可构建核心库、
CLI 和非 GUI 测试：

```bash
cmake .. -DBUILD_TESTS=ON -DPLASCAN_BUILD_GUI=OFF
cmake --build . -j$(nproc)
ctest --output-on-failure
```

项目通过 git submodule 引用自研点云库 [plapoint](https://github.com/guderianXu/plapoint) 和矩阵库 [plamatrix](https://github.com/guderianXu/plamatrix)，无需额外安装。

### vcpkg / CPack 跨平台构建

推荐新环境优先使用 `vcpkg.json` 和 `CMakePresets.json`。vcpkg 负责 Qt6、OpenCV 4、GDAL、libtiff、libzip、GTest 等通用依赖；CUDA 与 TensorRT 通过外部安装路径提供。

Linux:

```bash
export VCPKG_ROOT=/path/to/vcpkg
cmake --preset linux-vcpkg-release
cmake --build --preset linux-vcpkg-release
ctest --preset linux-vcpkg-release
cpack --preset linux-vcpkg-release
```

Windows PowerShell:

```powershell
$env:VCPKG_ROOT = "C:\src\vcpkg"
cmake --preset windows-vcpkg-release
cmake --build --preset windows-vcpkg-release
ctest --preset windows-vcpkg-release
cpack --preset windows-vcpkg-release
```

`cpack --preset windows-vcpkg-release` 生成 ZIP 离线包。安装 Inno Setup 6 后，可从已配置的
Windows Release 构建生成带开始菜单、桌面快捷方式、卸载入口和 `.plascan` 文件关联的安装程序：

```powershell
cpack `
  --config build/windows-vcpkg-cuda-release/CPackConfig.cmake `
  -G INNOSETUP `
  -C Release `
  -D "CPACK_PACKAGING_INSTALL_PREFIX=/" `
  -D "CPACK_PACKAGE_DIRECTORY=$PWD/dist/packages/windows-vcpkg-release"
```

Windows 构建使用原生 MSVC/Ninja/PowerShell，不需要 WSL。打包后的 GUI 需要 Qt platform plugins、vcpkg 和 TensorRT/CUDA 运行时 DLL；`PLASCAN_BUNDLE_RUNTIME=ON` 时 CMake install/CPack 会按主程序和 Qt 插件的传递依赖闭包收集 DLL，并补充 Vulkan、cuDNN 和 NVRTC 等动态加载运行时。Windows 包同时内置 `U2Net_v1.onnx`。

当前 manifest 使用 vcpkg 中可用的 OpenCV 4.x port。后续 vcpkg 正式提供 OpenCV 5 后，优先通过更新 `builtin-baseline`、OpenCV feature 列表和现有 `OpenCvCompat` 兼容测试切换。

Windows CUDA 开发机推荐固定使用 `scripts/build_win/build_windows_cuda.ps1`。脚本会把主构建目录收敛到
`build/windows-vcpkg-cuda-release`，并使用该目录自己的 `vcpkg_installed`、CUDA 13.1 和
构建目录自己的 TensorRT/CUDA 配置，避免其它 build cache 混入运行时 PATH。

同一脚本默认启用 `ceres-cuda` manifest feature，用于 SfM/光束法平差中的 Ceres CUDA 后端。
如果只想构建 CPU/legacy BA，可传 `-EnableCeresCudaBa:$false`。已有 `vcpkg_installed` 若仍是
CPU 版 Ceres，脚本会提示重新运行 `-InstallDeps`，避免界面显示 CUDA 但实际只跑 CPU。

U2Net ONNX 蒙版的 CPU 推理只需要标准 OpenCV DNN。若要启用 OpenCV DNN CUDA 后端，需要让 vcpkg
额外安装 `opencv-dnn-cuda` manifest feature：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -EnableOpenCvDnnCuda
pwsh scripts\build_win\build_windows_cuda.ps1 -EnableOpenCvDnnCuda
```

不带 `-EnableOpenCvDnnCuda` 时，匹配、MVS 等 CUDA 路径不受影响，但 U2Net ONNX 会使用
OpenCV DNN CPU 或在 GUI 中从 CUDA 自动回退到 CPU。

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

### GUI 一键工作流

GUI 的 `工作流程` 菜单按处理阶段提供互相独立的工程入口：

| 入口 | 输出 | 说明 |
|------|------|------|
| `空中三角测量` | 相机外参、连接点和正式稀疏点云 | 只负责影像对齐和 BA，不自动进入密集重建 |
| `生成模型` | PLY/OBJ 三维模型 | 从当前项目已有的连接点、深度图或点云生成模型 |
| `创建 DEM` | `dem.tif`、`depth_map.png`、可选 DEM 网格模型 | 自动模式从立体影像开始，手动模式可直接使用已有密集点云 |
| `生成正射影像` | 带覆盖 Alpha 的 DOM GeoTIFF/PNG | 选择项目影像和 DEM，按相机反投影并登记有效参数与覆盖统计 |

旧版 `工作流程 -> 三维重建` 一键对话框已移除；空三、密集处理、模型和地形产品由各自入口显式启动。

“创建正射影像”对话框提供马赛克、加权平均和首个有效影像三种融合模式，可按独立 X/Y
像元或最大尺寸生成，并支持 DEM 范围裁剪、颜色校正、锐度权重、重影过滤、小孔洞填充和
项目蒙版。对话框会读取 DEM 坐标系及真实像元，实时显示输出宽高和预计内存；生成任务在
后台运行，阶段进度和取消状态保留在同一窗口。GeoTIFF 按 R/G/B/Alpha 波段写出，以 Alpha
区分无覆盖区和真实黑色，并继承最终网格地理变换及可用的 DEM 投影 WKT。有效 DEM 表面零影像
覆盖时会明确失败，不会保存全黑成果。

当前只支持跟随 DEM 网格的地理/本地坐标投影、DEM 表面和影像颜色源。平面投影、圆柱投影及
全局接缝线优化尚未实现，对应 GUI 选项明确禁用。当前版本也尚未建立逐相机地形遮挡深度
缓冲，陡峭地形应结合输出 Alpha 和质量检查复核。

### 重建链路状态

当前重建链路按四个阶段维护：

- MVS 稳定性：`MvsWorkspaceManifest` 记录每帧深度图状态、输入/输出路径、device、耗时、错误和配置 hash。深度图完成后写入项目 metadata，GUI 目录树按文件名自然排序刷新。
- MVS 质量：`MvsSourcePlanner` 基于 shared tracks、几何内点、三角角、覆盖率、baseline 和序列距离规划 source view。深度图同时输出 preview、raw depth、confidence 和 valid mask，融合阶段使用同一份 source plan。
- 模型生成：任意三维的深度源默认执行 `raw depth + confidence + valid/support mask + camera -> TSDF -> Marching Cubes`，不生成或消费密集点云中间产物。Visual Hull 与 Poisson 只保留为显式 legacy/诊断模式，TSDF 失败不会静默换算法。
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
  --quality 3 \
  --threads 8 \
  --feature-max-image-dim 0
```

`--device auto` 是默认值：CUDA 可用时 SIFT 提取、LightGlue 匹配和 MVS PatchMatch 会使用 GPU。
当前稀疏前端要求 CUDA/TensorRT，显式 `--device cpu` 会返回不支持错误，不会切换算法。
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
  -m lightglue_sift_fp32.engine `
  -a sift_lightglue --max-keypoints 40000
```

SIFT 或 LoMa-R 描述子只存在于本次任务的内存缓存。LightGlue 和 LoMa-R 都只使用 TensorRT；最终分片保存关键点观测、
相邻影像、置信度、几何内点和残差，不生成独立特征文件或 JSON sidecar。engine 导出、固定容量和
精度策略见 [docs/models/README.md](docs/models/README.md#sift--lightglue-tensorrt)。

### 密集重建流水线

```bash
feature_match_cli   -L A.tif -R B.tif -o ./assets/image_matches -m lightglue_sift_fp32.engine
rectify_cli         -L A.tif -R B.tif --camL A.txt --camR B.txt -o rect
dense_match_cli     -L rect_L.tif -R rect_R.tif -o disp.tif --cuda --algorithm mgm
triangulate_cli     -d disp.tif --rect rect.xml --camL A.txt --camR B.txt -o cloud.ply
```

## 模型文件

从 [Releases](https://github.com/guderianXu/plascan/releases) 下载预训练模型，放置到 `resources/models/`。

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
| dense_match MGM/SGM | CUDA + CPU | CUDA + CPU | CPU only |
| CUDA SIFT + TensorRT LightGlue | CUDA | CUDA | 不支持 |
| TensorRT LoMa-R | CUDA | CUDA | 不支持 |
| 全部 CLI 工具 | ✅ | ✅ | ✅ |
| Qt6 GUI | ✅ | ✅ | ✅ |
| CPack 打包 | ZIP/INNOSETUP | TGZ/DEB | TGZ |
| Docker 构建 | — | ✅ | — |

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
ctest --test-dir build -R "OrthoGeneration|OrthoGridPlanner|OrthoProjector|MapProjectDialog" --output-on-failure
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
