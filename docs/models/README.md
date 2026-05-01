# 深度学习模型说明

## 已有模型

| 文件 | 算法 | 用途 |
|------|------|------|
| `superpoint_v6_cuda.pt` / `_cpu.pt` | SuperPoint | 特征提取 |
| `superglue_outdoor_cuda.pt` 等 | SuperGlue | 特征匹配 |
| `lightglue_matcher_cuda.pt` / `_cpu.pt` | LightGlue | 特征匹配 |
| `loftr_outdoor_cuda.pt` 等 | LoFTR | 无检测器匹配（直接输入图像） |
| `loftr_indoor_cuda.pt` 等 | LoFTR | 无检测器匹配（室内/近景） |

## DISK / ALIKED（特征提取）

DISK 和 ALIKED 输出动态数量的关键点，无法导出为 TorchScript。使用 Python 脚本提取特征，输出与 SuperPoint 相同格式的 `.sp` 文件：

```bash
conda activate plascan

# DISK
python scripts/extract_features.py --algo disk \
    --images /path/to/images/*.jpg \
    --output /path/to/sp_output \
    --max-keypoints 2048

# ALIKED
python scripts/extract_features.py --algo aliked \
    --images /path/to/images/*.jpg \
    --output /path/to/sp_output \
    --max-keypoints 2048
```

输出的 `.sp` 文件可直接用于 LightGlue 匹配（在 FeatureMatchingDialog 中选择 LightGlue）。

## RoMa（无检测器匹配）

RoMa 同样无法导出为 TorchScript。使用 Python 脚本直接匹配：

```bash
conda activate plascan

python scripts/match_roma.py \
    --scene outdoor \
    --pairs img001__img002 img001__img003 \
    --image-dir /path/to/images \
    --output /path/to/match_output \
    --threshold 0.05 \
    --max-keypoints 10000
```

## LoFTR（无检测器匹配）

LoFTR 已导出为 TorchScript，可直接在 GUI 中使用（FeatureMatchingDialog → 选择 LoFTR）。

模型文件：
- `loftr_outdoor_cuda.pt` / `loftr_outdoor_cpu.pt` — 室外/航空影像
- `loftr_indoor_cuda.pt` / `loftr_indoor_cpu.pt` — 室内/近景

## 重新导出模型

如需重新导出 LoFTR：

```bash
conda activate plascan
python scripts/export_models.py --loftr
```

依赖：`pip install kornia`（plascan 环境已包含）
