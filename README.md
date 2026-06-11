# PlaScan

[![CI](https://github.com/guderianXu/plascan/actions/workflows/ci.yml/badge.svg)](https://github.com/guderianXu/plascan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**行星表面摄影测量处理系统** — 从多视角影像生成高精度三维模型。

## 实测性能

*4608×3456 纳卫星全色影像, NVIDIA RTX 4060 (8GB)*

| 提取器 | 匹配器 | 匹配点数 | 提取 | 匹配 | 总计 | 方式 |
|--------|--------|---------|------|------|------|------|
| — | **LoFTR** (端到端) | **10,902** | — | 0.8s | 0.8s | GPU |
| DISK | NN (mutual) | 1,308 | 0.3s | 0.1s | 0.4s | GPU |
| SuperPoint | SuperGlue | 508 | 1.9s | 0.9s | 2.8s | GPU |
| SuperPoint | LightGlue | 463 | 1.9s | 0.8s | 2.7s | GPU |
| SuperPoint | BF (L2) | 267 | 1.9s | 0.3s | 2.2s | CPU |
| SIFT | BF (L2) | 230 | 2.7s | <0.1s | 2.7s | CPU |
| ORB | BF (Hamming) | 135 | 0.1s | <0.1s | 0.1s | CPU |

## 快速开始

### 依赖

- C++17 编译器 (GCC 11+ / Clang 15+)
- CMake ≥ 3.18
- CUDA Toolkit (可选，GPU 加速)
- Qt6, OpenCV, LibTorch, GDAL, libtiff, libzip, OpenMP, GTest

### 克隆并构建

```bash
git clone --recurse-submodules https://github.com/guderianXu/plascan.git
cd plascan
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

项目通过 git submodule 引用自研点云库 [plapoint](https://github.com/guderianXu/plapoint) 和矩阵库 [plamatrix](https://github.com/guderianXu/plamatrix)，无需额外安装。

### GUI 一键工作流

GUI 的 `工作流程` 菜单提供三个互相独立的工程入口：

| 入口 | 输出 | 说明 |
|------|------|------|
| `三维重建` | 稀疏点云、密集点云、PLY/OBJ 三维模型 | 不生成 DEM/DOM，适合检查建模与点云质量 |
| `创建 DEM` | `dem.tif`、`depth_map.png`、可选 DEM 网格模型 | 自动模式从立体影像开始，手动模式可直接使用已有密集点云 |
| `生成 正射影像` | DOM GeoTIFF/PNG | 默认全选项目影像，分辨率为 `0` 时自动沿用 DEM 网格 |

这三个入口的 UI 契约由 `test_gui_project_utils` 覆盖，避免后续改动把 DEM/DOM 错误耦合进三维重建流程。

### CLI 一键重建

`scripts/run_full_pipeline.py` 封装了与 GUI 等价的批处理入口，输入为影像和相机文件列表：

```bash
cmake --build build --target reconstruct_pipeline_cli -j$(nproc)
python scripts/run_full_pipeline.py path/to/input.lis \
  --build-dir build \
  --output-dir build/测试用临时文件/full_pipeline \
  --device cuda \
  --quality 3 \
  --dem-resolution 0
```

默认流程为 `SfM -> MVS 密集点云 -> 网格模型 -> DEM/DOM`。如只验证三维模型可加 `--skip-terrain`；如只验证地形产品前的点云/模型阶段可结合 `--skip-model` 或输出报告检查。

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
│   ├── camera/                # 相机模型 (Pinhole, Brown 畸变)
│   ├── feature_extractors/    # 8 种提取器, IExtractor 接口 + 工厂
│   │   ├── superpoint/        # SuperPoint (256d, TorchScript)
│   │   ├── disk/              # DISK (128d, TorchScript)
│   │   ├── aliked/            # ALIKED (128d, TorchScript)
│   │   └── tradition/         # SIFT / SURF / ORB / AKAZE (OpenCV)
│   ├── feature_match/         # 7 种匹配器, IMatcher 接口 + 工厂
│   │   ├── superglue/         # SuperGlue (GNN, 256d)
│   │   ├── lightglue/         # LightGlue (GNN)
│   │   ├── loftr/             # LoFTR (端到端密集)
│   │   └── tradition/         # BF / FLANN (OpenCV)
│   ├── sfm/                   # 增量式 SfM + 光束法平差 (Ceres)
│   ├── mvs/                   # PatchMatch 深度图 + 融合
│   ├── dense_match/           # MGM/SGM 密集立体匹配 (自研 CUDA)
│   ├── mesh/                  # Poisson 表面重建 + 纹理映射
│   ├── terrain/               # DEM + DOM 正射影像
│   ├── overlap/               # 影像重叠度分析
│   ├── intersection/          # 前方交汇精度检验
│   └── pipeline/              # SfM 服务层
├── gui/                       # Qt6 图形界面
│   ├── dialogs/               # 参数配置对话框
│   ├── widgets/               # 3D 画布 + 影像查看器
│   └── project/               # 项目管理 (.plascan 归档)
├── cli/                       # 命令行工具
└── common/                    # 通用工具 (日志, 数学, 结果包装)
3rdparty/
├── plapoint/   (submodule)    # 自研 GPU 点云库
└── plamatrix/  (submodule)    # 自研矩阵运算后端
```

## CLI 工具

### 特征提取 (`feature_extract_cli`)

```bash
# 8 种算法: superpoint, disk, aliked, sift, surf, orb, akaze, dedode
feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda
feature_extract_cli -a disk       -m disk.pt -i img.tif -o out.dsk --cuda
feature_extract_cli -a sift       -i img.tif -o out.sift -n 4096

# 批量处理目录
feature_extract_cli -a superpoint -m sp.pt -i ./images/ -o ./features/ --cuda
```

### 特征匹配 (`feature_match_cli`)

```bash
# 自动检测文件后缀
feature_match_cli --sp1 a.sp  --sp2 b.sp  -o out.match --cuda    # .sp → SuperGlue
feature_match_cli --sp1 a.dsk --sp2 b.dsk -o out.match --cuda    # .dsk → NN

# 端到端 (无需预先提取特征)
feature_match_cli -a loftr -L a.tif -R b.tif -o out.match --cuda
```

### 密集重建流水线

```bash
feature_extract_cli -a superpoint -m sp.pt -i A.tif -o A.sp --cuda
feature_extract_cli -a superpoint -m sp.pt -i B.tif -o B.sp --cuda
feature_match_cli   -a superglue -m sg.pt --sp1 A.sp --sp2 B.sp -o AB.match --cuda
rectify_cli         -L A.tif -R B.tif --camL A.txt --camR B.txt -o rect
dense_match_cli     -L rect_L.tif -R rect_R.tif -o disp.tif --cuda --algorithm mgm
triangulate_cli     -d disp.tif --rect rect.xml --camL A.txt --camR B.txt -o cloud.ply
```

## 模型文件

从 [Releases](https://github.com/guderianXu/plascan/releases) 下载预训练模型，放置到 `resources/models/`。

或通过导出脚本生成：

```bash
python scripts/export_superpoint.py                 # SuperPoint
python scripts/export_disk_aliked.py                # DISK + ALIKED
python scripts/export_models.py --loftr --roma      # LoFTR + RoMa
```

## 平台支持

| 功能 | Linux (NVIDIA) | macOS (Apple Silicon) |
|------|:---:|:---:|
| CUDA 加速 | ✅ | ❌ (MPS via PyTorch) |
| dense_match MGM/SGM | CUDA + CPU | CPU only |
| SuperPoint/DISK/ALIKED | CUDA + CPU | CPU |
| SIFT/ORB/AKAZE | CPU | CPU |
| 全部 CLI 工具 | ✅ | ✅ |
| Qt6 GUI | ✅ | ✅ |
| Docker 构建 | ✅ | — |

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
cmake --build build --target test_gui_project_utils test_mesh_reconstructor test_terrain_dem_dom plascan_gui -j$(nproc)
QT_QPA_PLATFORM=offscreen ./build/tests/test_gui_project_utils
./build/tests/test_mesh_reconstructor
./build/src/core/terrain/test_terrain_dem_dom
```

这些测试分别覆盖：

- GUI 一键入口边界：三维重建只负责点云/模型，DEM 和 DOM 使用独立按钮。
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
