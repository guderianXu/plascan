# PlaScan

行星表面摄影测量处理系统 — 从影像到三维模型的完整流水线。

## 核心流程

```
影像 → 特征提取 → 特征匹配 → SfM(BA) → 密集匹配 → 点云/网格/纹理
```

## 算法矩阵

### 特征提取

| 算法 | 描述子 | 设备 | 速度 | 卫星适用 |
|------|:---:|------|------|:---:|
| **SuperPoint** | 256d | GPU/CPU | ★★★ | ✅ |
| **DISK** | 128d | GPU/CPU | ★★★★ | ✅ |
| SIFT | 128d | CPU | ★★ | ✅ |
| ORB | 32d | CPU | ★★★★★ | △ |
| AKAZE | 61d | CPU | ★★ | △ |
| ALIKED | 128d | GPU/CPU | ★★★ | △ |
| DeDoDe | 256d | GPU/CPU | ★ | ❌ (DINOv2 域不匹配) |

### 特征匹配

| 算法 | 类型 | 设备 | 匹配质量 |
|------|------|------|:---:|
| **LoFTR** | 端到端密集 | GPU | ★★★★★ |
| **SuperGlue** | 稀疏(GNN) | GPU | ★★★★ |
| **LightGlue** | 稀疏(GNN) | GPU | ★★★★ |
| DISK+NN | 稀疏(NN) | GPU/CPU | ★★★ |
| BF(L2) | 稀疏(NN) | CPU | ★★ |
| FLANN | 稀疏(KNN) | CPU | ★★ |

### 实测对比 (4608×3456 卫星影像, RTX 4060)

| 提取+匹配 | 匹配点 | 耗时 |
|-----------|--------|------|
| LoFTR outdoor | 10,902 | 0.8s |
| DISK+NN | 1,308 | 0.4s |
| SP+SuperGlue | 508 | 2.8s |
| SP+LightGlue | 463 | 2.7s |
| SP+BF(L2) | 267 | 2.2s |
| SIFT+BF(L2) | 230 | 2.7s |
| ORB+BF(Hamming) | 135 | 0.1s |

## 快速开始

### 构建

```bash
# Linux + CUDA
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)

# macOS Apple Silicon
brew install cmake qt@6 opencv libtorch gdal libtiff libzip openmp
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH="$(brew --prefix qt@6)"
cmake --build . -j$(sysctl -n hw.logicalcpu)
```

### CLI 一键匹配

```bash
# 最优方案: LoFTR 端到端 (11000+ 匹配点)
feature_match_cli -a loftr -L img1.tif -R img2.tif -o out.match --cuda

# Lightning fast: DISK+NN (1300+ 匹配点)
feature_match_cli -a disk  -L img1.tif -R img2.tif -o out.match --cuda

# 精典方案: SuperPoint+SuperGlue (500+ 高质量匹配)
feature_extract_cli -a superpoint -m sp.pt -i img.tif -o out.sp --cuda
feature_match_cli   -a superglue  -m sg.pt --sp1 a.sp --sp2 b.sp -o out.match --cuda
```

### 完整密集重建

```bash
rectify_cli    -L a.tif -R b.tif --camL a.txt --camR b.txt -o rect
dense_match_cli -L rect_L.tif -R rect_R.tif -o disp.tif --cuda
triangulate_cli -d disp.tif --rect rect.xml --camL a.txt --camR b.txt -o cloud.ply
```

## 模块架构

```
src/
├── core/          # 核心算法
│   ├── camera/        # 相机模型 (Pinhole + 畸变)
│   ├── feature_extractors/  # 7 种提取器 (SP/SIFT/ORB/AKAZE/DISK/ALIKED/DeDoDe)
│   ├── feature_match/      # 6 种匹配器 (SG/LG/LoFTR/BF/FLANN/DISK)
│   ├── sfm/           # SfM + 光束法平差
│   ├── mvs/           # MVS 密集点云
│   ├── dense_match/   # CUDA MGM/SGM 密集匹配
│   ├── mesh/          # Poisson 网格重建
│   └── terrain/       # DEM/DOM 地形产品
├── gui/           # Qt6 GUI
└── cli/           # 命令行工具 (5 CLI)
```

## 文档

- `docs/PROJECT_ARCHITECTURE.md` — 完整项目结构
- `docs/BUILD_MACOS.md` — macOS 构建指南
- `src/core/dense_match/README.md` — 密集匹配模块
- `src/core/feature_extractors/README.md` — 特征提取模块

## 依赖

Qt6, OpenCV 4.x, LibTorch 2.x, GDAL, libtiff, libzip, OpenMP  
模型文件放置于 `resources/models/`
