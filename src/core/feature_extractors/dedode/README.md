# DeDoDe — Detect, Don't Describe / Describe, Don't Detect

联合检测与描述深度学习特征提取器。使用 DINOv2-Large 骨干网络。

## 技术特点

- **描述子**: 256d float32, L2 归一化
- **骨干**: DINOv2 ViT-L/14
- **检测+描述联合**: 同时输出关键点坐标和密集描述子

## 实现方式

DeDoDe 使用 **kornia + PyTorch** (Python)，通过 C++ CLI 子进程调用。

- 脚本: `scripts/run_dedode.py`
- CLI: `feature_match_cli -a dedode -L a.tif -R b.tif -o out.match --cuda`

## 为何不是纯 C++

DeDoDe 的 DINOv2 encoder 内部使用了 `torch.nn.functional.interpolate` 配合动态 scale_factors，torch.jit.trace 无法正确处理。torch.jit.script 因动态模块列表长度而失败。

## 适用场景

DINOv2 在自然图像上预训练。卫星遥感影像（灰度、俯视视角）域不匹配，匹配点较少（~34）。适用于自然场景影像（无人机、地面拍摄）。

## 相关文件

- C++ wrapper: `src/core/feature_extractors/dedode/` (规划中)
- Python 脚本: `scripts/run_dedode.py`
- CLI 入口: `feature_extract_cli -a dedode`, `feature_match_cli -a dedode`
