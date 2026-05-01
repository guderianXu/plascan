# <img src="" width="28"> PlaScan

**行星表面摄影测量处理系统** — 从多视角影像生成高精度三维模型。

## 什么是 PlaScan

PlaScan 是一套完整的摄影测量流水线，用于从卫星或无人机影像重建行星表面三维模型。覆盖从影像输入到最终数字高程模型（DEM）的全流程——特征提取、特征匹配、运动恢复结构（SfM）、光束法平差、密集点云生成、网格重建和纹理映射。

```
  影像输入          稀疏重建               密集重建             最终输出
  ──────────────────────────────────────────────────────────────────────
  ┌──┐  ┌──┐    特征提取 → 匹配     极线校正 → 密集匹配     .ply  点云
  │  │  │  │    SuperPoint  SuperGlue  MGM/SGM   pointcloud  .obj  网格
  │  │  │  │    DISK        LightGlue  (CUDA)                .tif  DEM
  │  │  │  │    SIFT        LoFTR                             .png  正射影像
  └──┘  └──┘    ALIKED      BF/FLANN
  影像1  影像2   ORB/AKAZE
                     ↓
              SfM + 光束法平差
              (相机位姿 + 稀疏点云)
```

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

## 模块架构

```
src/
├── core/
│   ├── camera/                # 相机模型 (Pinhole, Brown 畸变)
│   ├── feature_extractors/    # 7 种提取器, IExtractor 接口 + 工厂
│   │   ├── superpoint/        # SuperPoint (256d, TorchScript)
│   │   ├── disk/              # DISK (128d, kornia)
│   │   ├── aliked/            # ALIKED (128d, lightglue)
│   │   └── tradition/         # SIFT / ORB / AKAZE (OpenCV)
│   ├── feature_match/         # 5 种匹配器, IMatcher 接口 + 工厂
│   │   ├── superglue/         # SuperGlue (GNN, 256d)
│   │   ├── lightglue/         # LightGlue (GNN, 256d/128d)
│   │   ├── loftr/             # LoFTR (端到端密集, LibTorch)
│   │   └── tradition/         # BF / FLANN (OpenCV)
│   ├── sfm/                   # 增量式 SfM + 光束法平差 (Ceres)
│   ├── mvs/                   # PatchMatch 深度图 + 融合
│   ├── dense_match/           # MGM/SGM 密集立体匹配 (自研 CUDA)
│   ├── pointcloud/            # 点云数据结构 + I/O (PLY/XYZ/LAS)
│   ├── mesh/                  # Poisson 表面重建 + 纹理映射
│   ├── terrain/               # DEM 生成 + DOM 正射影像
│   └── pipeline/              # SfM 服务层 (GUI 可调用)
├── gui/                       # Qt6 图形界面 (20+ 对话框)
│   ├── dialogs/               # 参数配置对话框
│   ├── widgets/               # 3D 画布 + 影像查看器
│   └── project/               # 项目管理 (.plascan 归档)
└── cli/                       # 命令行工具 (5 CLI, CLI11)
    ├── feature_extract_cli    # 统一特征提取
    ├── feature_match_cli      # 统一特征匹配
    ├── dense_match_cli        # 密集匹配
    ├── rectify_cli            # 极线校正
    └── triangulate_cli        # 视差三角化 → 点云
```

## 快速开始

### 从源码构建

```bash
git clone https://github.com/guderianXu/plascan.git
cd plascan

# Linux (CUDA)
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc)

# macOS (Apple Silicon M-series)
brew install cmake qt@6 opencv libtorch gdal libtiff libzip openmp
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
cmake --build . -j$(sysctl -n hw.logicalcpu)
```

### 运行测试

```bash
cd build && ctest --output-on-failure
# 密集匹配模块: 22 tests
```

### 模型文件

从 [Releases](https://github.com/guderianXu/plascan/releases) 下载预训练模型，放置到 `resources/models/`：

```
resources/models/
├── superpoint_v6_cuda.pt        # SuperPoint 检测器 (GPU)
├── superglue_outdoor_cuda.pt    # SuperGlue 匹配器 (室外)
├── lightglue_matcher_cuda.pt    # LightGlue 匹配器
├── superpoint_extractor_cuda.pt # SuperPoint 提取器
└── ...                           # 其他模型
```

## CLI 工具

### 特征提取 (`feature_extract_cli`)

```bash
# 自动选择算法 + 后缀, 无需手动指定
feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda
feature_extract_cli -a sift       -i img.tif -o out.sift -n 4096
feature_extract_cli -a disk       -m disk.pt -i img.tif -o out.dsk --cuda
feature_extract_cli -a aliked     -m aliked.pt -i img.tif -o out.alk
feature_extract_cli -a orb        -i img.tif -o out.orb -n 5000

# 批量处理目录
feature_extract_cli -a superpoint -m sp.pt -i ./images/ -o ./features/ --cuda
```

### 特征匹配 (`feature_match_cli`)

```bash
# 自动检测文件后缀, 无需 -a
feature_match_cli --sp1 a.sp  --sp2 b.sp  -o out.match --cuda   # .sp → SuperGlue
feature_match_cli --sp1 a.dsk --sp2 b.dsk -o out.match --cuda   # .dsk → NN

# 端到端 (无需特征提取)
feature_match_cli -a loftr -L a.tif -R b.tif -o out.match --cuda

# 手动指定
feature_match_cli -a superglue -m sg.pt --sp1 a.sp --sp2 b.sp -o out.match --cuda
feature_match_cli -a bf        --sp1 a.sp --sp2 b.sp -o out.match
```

### 完整密集重建流水线

```bash
# 稀疏: 特征提取 + 匹配
feature_extract_cli -a superpoint -m sp.pt -i A.tif -o A.sp --cuda
feature_extract_cli -a superpoint -m sp.pt -i B.tif -o B.sp --cuda
feature_match_cli   -a superglue  -m sg.pt --sp1 A.sp --sp2 B.sp -o AB.match --cuda

# 密集: 极线校正 → 立体匹配 → 点云
rectify_cli       -L A.tif -R B.tif --camL A.txt --camR B.txt -o rect
dense_match_cli   -L rect_L.tif -R rect_R.tif -o disp.tif --cuda --algorithm mgm
triangulate_cli   -d disp.tif --rect rect.xml --camL A.txt --camR B.txt -o cloud.ply
```

## 平台支持

| 功能 | Linux (NVIDIA) | macOS (Apple Silicon) |
|------|:---:|:---:|
| CUDA 加速 | ✅ | ❌ (MPS via PyTorch) |
| dense_match MGM/SGM | CUDA + CPU | CPU only |
| SuperPoint/DISK/ALIKED | CUDA + CPU | MPS + CPU |
| SIFT/ORB/AKAZE | CPU | CPU |
| 全部 CLI 工具 | ✅ | ✅ |
| Qt6 GUI | ✅ | ✅ |
| 构建指南 | — | `docs/BUILD_MACOS.md` |

## 开发

### 工作流

```bash
git checkout -b feat/<name>    # 从 main 创建
# ... 开发 (TDD: 红 → 绿 → 重构) ...
git checkout main && git merge feat/<name> --no-ff
git push origin main
```

### 代码规范

- 单文件 ≤ 400 行 | 嵌套 ≤ 4 层 | Allman 花括号
- 详细见 `CLAUDE.md`

## 文档

| 文档 | 内容 |
|------|------|
| `docs/PROJECT_ARCHITECTURE.md` | 完整目录树、模块职责、数据流、技术债务 |
| `docs/BUILD_MACOS.md` | macOS Apple Silicon 构建指南 |
| `docs/superpowers/specs/` | 功能规格说明 |
| `src/core/dense_match/README.md` | MGM/SGM 密集匹配算法文档 |
| `src/core/feature_extractors/README.md` | 7 种提取器对比、API、基准 |

## 依赖

| 库 | 版本 | 用途 |
|----|------|------|
| Qt6 | ≥6.5 | GUI |
| OpenCV | ≥4.8 | 图像处理、传统算法 |
| LibTorch | 2.5-2.11 | 深度学习推理 |
| GDAL | ≥3.5 | 地理空间数据 |
| CLI11 | 2.4 | CLI 参数解析 (header-only) |
| Google Test | ≥1.12 | 单元测试 |

## 许可

MIT License
