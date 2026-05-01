# PlaScan

行星表面摄影测量处理系统。

## 功能

- **稀疏重建**: SuperPoint/SIFT/ORB/DISK/ALIKED 特征提取 + SuperGlue/LightGlue/LoFTR 匹配 + 增量 SfM + 光束法平差
- **密集重建**: 极线校正 + MGM/SGM/BM 密集匹配 (CUDA) + 视差三角化 → 密集点云
- **模型生成**: Poisson 网格重建 + 纹理映射 + 模型导出
- **地形产品**: DEM 生成 + DOM 正射影像

## 构建

```bash
# Linux + CUDA
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(nproc)

# macOS Apple Silicon
brew install cmake qt@6 opencv libtorch gdal libtiff libzip openmp
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.logicalcpu)
```

详见 `docs/BUILD_MACOS.md`

## CLI 工具

```bash
# 特征提取
feature_extract_cli -a superpoint -m model.pt -i img.tif -o out.sp --cuda
feature_extract_cli -a sift  -i img.tif -o out.sift -n 4096

# 特征匹配
feature_match_cli -a superglue -m sg.pt --sp1 a.sp --sp2 b.sp -o out.match --cuda
feature_match_cli -a loftr -L a.tif -R b.tif -o out.match --cuda

# 密集匹配
dense_match_cli -L a.tif -R b.tif -o disp.tif --cuda

# 极线校正 → 密集匹配 → 点云
rectify_cli -L a.tif -R b.tif --camL a.txt --camR b.txt -o rect
dense_match_cli -L rect_L.tif -R rect_R.tif -o disp.tif --cuda
triangulate_cli -d disp.tif --rect rect.xml --camL a.txt --camR b.txt -o cloud.ply
```

## 项目结构

详见 `docs/PROJECT_ARCHITECTURE.md`

## 许可

MIT
