# PlaScan 会话上下文 — 2026-05-01

## 环境
- Python: conda env `plascan`, PyTorch 2.5.1+cu124, CUDA 13.1 toolkit
- GPU: RTX 4060 8GB
- Build: `cd build && cmake .. -DBUILD_TESTS=ON [paths] && make -j22`
- 注意: conda GCC 注入 `-D_GLIBCXX_USE_CXX11_ABI=0`, 需要 `scripts/fix_cmake_abi.sh` 清除
- libmvec 修复: `sudo ln -sf ~/anaconda3/envs/plascan/x86_64-conda-linux-gnu/sysroot/lib64/libmvec-2.28.so /lib64/libmvec.so.1`
- Caffe2 cmake 需要 patch: Config FATAL_ERROR→WARNING, Targets 删除 torch::cudart

## 当前构建状态
✅ dense_match_cli, test_dense_match_unit (22/22)
✅ superpoint.so, feature_extractors_disk.a, feature_extractors_aliked.a
✅ test_disk_unit, test_aliked_unit
❌ feature_extract_cli (createExtractor 命名空间——.h 在 global, .cpp 在 xjw::feature_extractors)
❌ feature_match_cli (同上, createMatcher)
⚠️ pointcloud CUDA CCCL 冲突 (targets/cccl vs nvcc 13.1)

## 架构
```
feature_extractors/          feature_match/
├── IExtractor.h             ├── IMatcher.h
├── ExtractorFactory.h/cpp   ├── MatcherFactory.h/cpp
├── FeatureOutput.h          ├── superglue/
├── FeatureFileIO.h/cpp      ├── lightglue/
├── FeatureData.h/cpp        ├── loftr/  (Python子进程)
├── superpoint/              └── tradition/
├── disk/  (DiskExtractor)
├── aliked/ (AlikedExtractor)
├── tradition/ (SIFT/SURF/ORB/AKAZE)
├── loftr/  (README + run_loftr.py)
└── dedode/ (README + run_dedode.py)
```

## 提取器 (8种)
| CLI参数 | 算法 | 描述子 | 设备 |
|---------|------|:---:|------|
| superpoint | SuperPoint | 256d | GPU/CPU |
| disk | DISK | 128d | GPU/CPU |
| aliked | ALIKED | 128d | GPU/CPU |
| sift | SIFT | 128d | CPU |
| surf | SURF | 64/128d | CPU |
| orb | ORB | 32d | CPU |
| akaze | AKAZE | 61d | CPU |
| dedode | DeDoDe | 256d | Python子进程 |

## 匹配器 (7种)
superglue(C++) | lightglue(C++) | loftr(Python) | disk(Python) | aliked(Python) | bf(C++) | flann(C++)

## 待修复
1. createExtractor 命名空间不一致 (.h global, .cpp xjw::feature_extractors)
2. CLI 编译后链接失败
3. pointcloud CUDA CCCL 冲突
