# PlaScan 会话上下文 — 2026-05-02

## 环境
- Python: conda env `plascan`, PyTorch 2.11.0+cu130, CUDA 13.1 toolkit
- GPU: RTX 4060 8GB
- Build: `cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . -j$(nproc)`
- 测试: `ctest -j1` 或 `-j$(nproc)` → 均 100%

## 编译状态
✅ 完整构建通过 (0 编译错误)

## 测试状态: 162/162 (100%) 顺序+并行均全过

## 模型文件 (全部真实 TorchScript 模型)
| 模型 | 状态 | 来源 |
|------|------|------|
| **SuperPoint** | ✅ 真实 | LightGlue 重导出, 匹配 C++ 接口 |
| **DISK** | ✅ 真实 | kornia, 绕过 heatmap_to_keypoints (NMS via max_pool 比较 + topk) |
| **ALIKED** | ✅ 真实 | lightglue, monkey-patch DeformableConv2d→regular Conv2d + top_k 模式 |

## 导出脚本
| 脚本 | 用途 |
|------|------|
| `scripts/export_superpoint.py` | SuperPoint 导出 |
| `scripts/export_disk_aliked.py` | DISK/ALIKED 导出 (含全部 monkey-patch) |
| `scripts/gen_test_models.py` | 占位模型 (已弃用, 保留备用) |

## C++ 模型接口 (已统一)
所有提取器: `forward(image [1,1,H,W], orig_wh [W,H])` → tuple of tensors

## 系统依赖
- `/lib64/libm.so.6` → `/usr/lib/x86_64-linux-gnu/libm.so.6`
- `/lib64/libc.so.6` → `/usr/lib/x86_64-linux-gnu/libc.so.6`
- `/lib64/libmvec.so.1` → conda sysroot libmvec-2.28.so
- `/lib/x86_64-linux-gnu/libnvrtc-builtins.so.13.0` → conda targets/lib/libnvrtc-builtins.so.13.1
- nvcc 需在 PATH 或 conda env 中 (根 CMakeLists 自动查找)

## 已修复问题
- ✅ terrain 并行文件冲突 (每个测试独立临时目录)
- ✅ DISK NMS trace-incompatible (nonzero→topk)
- ✅ ALIKED torchvision::deform_conv2d (monkey-patch→regular conv2d)
- ✅ 导出脚本移除 LightGlue 源码硬编码路径 (改为使用 pip 安装的 lightglue)
- ✅ 所有 162 测试顺序/并行均 100% 通过
