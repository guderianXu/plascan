# LoFTR — 端到端密集特征匹配

LoFTR (Local Feature TRansformer) 是端到端的密集匹配器，直接从影像对生成匹配点，无需预先提取关键点。

## 技术特点

- **端到端**: `forward(img0, img1)` → 匹配点坐标
- **密集匹配**: 使用 Transformer 跨图像注意力
- **无需提取器**: 跳过关键点检测+描述子计算+匹配三个阶段

## 实现方式

LoFTR 使用 **kornia + PyTorch** (Python)，通过 C++ CLI 子进程调用。

- 模型: `loftr_outdoor_cuda.pt` (室外), `loftr_indoor_cuda.pt` (室内)
- 脚本: `scripts/run_loftr.py`
- CLI: `feature_match_cli -a loftr -L a.tif -R b.tif -o out.match --cuda`

## 为何不是纯 C++

LoFTR 模型包含动态模块列表和条件分支，无法通过 `torch.jit.trace` 或 `torch.jit.script` 导出为 TorchScript。C++ CLI 通过 QProcess 调用 Python 脚本透明执行。

## 性能

- 4608×3456 卫星影像: **10,902 匹配点**, 0.8s GPU (RTX 4060)
- 是 SuperGlue 匹配点数的 **22 倍**
