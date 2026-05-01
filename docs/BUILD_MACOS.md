# PlaScan on macOS (Apple Silicon M-series)

## 前置依赖

```bash
# Homebrew (必须)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 系统依赖
brew install cmake qt@6 opencv libtorch gdal libtiff libzip openmp

# PyTorch (MPS 加速)
pip3 install torch torchvision
```

## 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(brew --prefix qt@6);$(brew --prefix opencv);$(brew --prefix libtorch)" \
         -DBUILD_TESTS=ON
cmake --build . -j$(sysctl -n hw.logicalcpu)
```

## 平台差异

| 功能 | Linux (NVIDIA) | macOS (Apple Silicon) |
|------|:---:|:---:|
| CUDA 加速 | ✅ | ❌ (MPS 替代) |
| dense_match | CUDA+CPU | CPU only |
| MVS PatchMatch | CUDA+CPU | CPU only |
| SuperPoint/DISK | CUDA+CPU | MPS+CPU |
| CLI 工具 | 全部 | 全部 (CPU fallback) |
| GUI | ✅ | ✅ (Native ARM64) |

## 已知限制

- **CUDA .cu 文件**: macOS 无 CUDA 编译器, cmake 自动跳过
- **密集匹配**: CPU fallback 路径已实现, 但速度较慢
- **LoFTR/DISK/ALIKED**: Python 脚本在 macOS 上通过 MPS 加速可用
- **OpenMP**: 需要 `brew install libomp`
