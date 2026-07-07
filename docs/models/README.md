# 深度学习模型说明

## 已有模型

| 文件 | 算法 | 用途 |
|------|------|------|
| `superpoint_v6_cuda.pt` / `_cpu.pt` | SuperPoint | 特征提取 |
| `superglue_outdoor_cuda.pt` 等 | SuperGlue | 特征匹配 |
| `lightglue_matcher_cuda.pt` / `_cpu.pt` | LightGlue | 特征匹配 |
| `loftr_outdoor_cuda.pt` 等 | LoFTR | 无检测器匹配（直接输入图像） |
| `loftr_indoor_cuda.pt` 等 | LoFTR | 无检测器匹配（室内/近景） |
| `sam21_hiera_*_encoder_*.pt` / `sam21_hiera_*_decoder_*.pt` | SAM2.1 | GUI 照片蒙版生成 |
| `U2Net_v1.onnx` | U2Net | GUI 照片 AI 蒙版生成 |

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

## SAM2.1（AI 蒙版）

GUI 的“工具 → 生成蒙版...”支持 `AI: SAM2.1 TorchScript`。该模式使用整张照片的 box prompt
作为默认提示，不依赖黑背景，因此适用于普通航拍、行星表面和近景建模照片。输出仍采用 PlaScan
蒙版约定：`0` 表示保留区域，`255` 表示被遮罩区域。

模型文件默认放在 `resources/models/`，也可以通过 `PLASCAN_MODEL_DIR` 指定。文件名固定为：

- `sam21_hiera_tiny_encoder_cpu.pt`
- `sam21_hiera_tiny_decoder_cpu.pt`
- `sam21_hiera_tiny_encoder_cuda.pt`
- `sam21_hiera_tiny_decoder_cuda.pt`

`small`、`base_plus`、`large` 变体把文件名中的 `tiny` 分别替换为 `small`、`base_plus`、`large`。
GUI 默认优先使用 CUDA，勾选“CUDA 不可用时回退 CPU”后会在 CUDA 不可用或 CUDA 模型缺失时使用 CPU 模型。

GUI 会在“生成蒙版”对话框中显示每个 SAM2.1 变体的安装状态。未安装或缺少 CPU/CUDA
TorchScript 文件时，可以点击“安装模型...”下载官方 checkpoint，并调用项目内
`scripts/export_sam21_torchscript.py` 导出 TorchScript。该流程假定软件分发时已经内置包含
`torch` 和 `sam2` 的 Python 运行时；程序不会在 GUI 中自动安装 Python 包。

也可以用命令行安装指定变体：

```bash
python scripts/install_sam21_model.py \
    --variant small \
    --devices auto \
    --model-dir resources/models
```

可用变体为 `tiny`、`small`、`base_plus`、`large`。GUI 和 C++ 运行入口会优先使用
`PLASCAN_PYTHON_EXECUTABLE`，其次使用 `PLASCAN_PYTHON`，再使用项目根目录 `.venv`
（Windows 为 `.venv/Scripts/python.exe`，Linux 为 `.venv/bin/python`），然后才回退到打包运行时或
PATH 中的 Python。直接运行脚本时仍使用启动该脚本的当前 Python。

导出依赖 Meta SAM2 Python 包和 PyTorch：

```bash
conda activate plascan
pip install git+https://github.com/facebookresearch/sam2.git

# 默认导出 tiny，CPU；如果当前环境有 CUDA，会同时导出 CUDA
python scripts/export_sam21_torchscript.py \
    --variant tiny \
    --checkpoint resources/models/sam2.1_hiera_tiny.pt \
    --output-dir resources/models

# 强制导出 CPU + CUDA
python scripts/export_sam21_torchscript.py \
    --variant tiny \
    --checkpoint resources/models/sam2.1_hiera_tiny.pt \
    --devices cpu,cuda \
    --output-dir resources/models
```

## U2Net ONNX（AI 蒙版）

GUI 的“工具 → 生成蒙版...”支持 `AI: U2Net ONNX`。该模式使用内置 ONNX 模型做前景/背景分离，
适合类似 Metashape Automatic AI 的快速自动蒙版。输出沿用 PlaScan 蒙版约定：
`0` 表示保留区域，`255` 表示被遮罩区域。

模型文件默认放在 `resources/models/`，也可以通过 `PLASCAN_MODEL_DIR` 指定。安装包应内置：

- `U2Net_v1.onnx`

GUI 会显示 U2Net 模型是否已安装。U2Net 推理当前使用 OpenCV DNN 后端：CPU 在标准 OpenCV DNN
构建中可用；CUDA 需要 OpenCV 同时启用 CUDA DNN 后端并能发现 CUDA 设备。GUI 默认请求 CUDA，并勾选
“CUDA 不可用时回退 CPU”；如果当前 OpenCV 不支持 DNN CUDA，程序会自动回退 CPU。

Windows CUDA 构建中启用 U2Net ONNX CUDA 推理：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -EnableOpenCvDnnCuda
pwsh scripts\build_win\build_windows_cuda.ps1 -EnableOpenCvDnnCuda
```

第一条会让 vcpkg 用 `opencv-dnn-cuda` feature 重建 OpenCV；第二条用于后续正常配置/编译。若只安装了标准
OpenCV，GUI 仍可使用 U2Net CPU 推理，CUDA 按钮会在运行时回退或报告 OpenCV DNN CUDA 后端不可用。

## 重新导出模型

如需重新导出 LoFTR：

```bash
conda activate plascan
python scripts/export_models.py --loftr
```

依赖：`pip install kornia`（plascan 环境已包含）
