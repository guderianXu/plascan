# <img src="" width="28"> PlaScan

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

## 快速开始 (Docker, 推荐)

```bash
git clone https://github.com/guderianXu/plascan.git
cd plascan

# 1. 构建 Docker 镜像 (首次约 10 分钟, 含 CUDA 12.4 + LibTorch)
sudo docker build -t plascan-build -f docker/Dockerfile.ubuntu2404 .

# 2. 进入容器交互开发
./docker/shell.sh

# 3. 在容器内编译并测试
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

**一键构建+测试**：`./docker/build.sh`

**打包 .deb**：`./docker/package.sh`

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
│   ├── pointcloud/            # 点云数据结构 + I/O
│   ├── mesh/                  # Poisson 表面重建 + 纹理映射
│   ├── terrain/               # DEM + DOM 正射影像
│   └── pipeline/              # SfM 服务层
├── gui/                       # Qt6 图形界面
│   ├── dialogs/               # 参数配置对话框
│   ├── widgets/               # 3D 画布 + 影像查看器
│   └── project/               # 项目管理 (.plascan 归档)
└── cli/                       # 命令行工具 (6 CLI, CLI11)
    ├── feature_extract_cli    # 统一特征提取
    ├── feature_match_cli      # 统一特征匹配
    ├── dense_match_cli        # 密集匹配
    ├── rectify_cli            # 极线校正
    └── triangulate_cli        # 视差三角化 → 点云
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
# 自动检测文件后缀, 无需手动指定匹配器
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

或通过导出脚本生成（需要 conda env + lightglue/kornia）：

```bash
python scripts/export_superpoint.py                      # SuperPoint
python scripts/export_disk_aliked.py                     # DISK + ALIKED
python scripts/export_models.py --loftr --roma           # LoFTR + RoMa
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

### 工作流

```bash
git checkout -b feat/<name>    # 从 main 创建
# ... TDD: 红 → 绿 → 重构 ...
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
| `CONTEXT.md` | 当前环境、构建状态、系统依赖 |
| `docs/BUILD_MACOS.md` | macOS Apple Silicon 构建指南 |
| `docs/superpowers/specs/` | 功能规格说明 |

## 许可

MIT License
