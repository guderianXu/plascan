# 模型与推理资源

当前生产构建只保留三类模型资源：SIFT + LightGlue 匹配所需的 TensorRT engine、SAM2.1 蒙版模型和
U2Net ONNX 蒙版模型。SuperPoint、SuperGlue、DISK、ALIKED、LoFTR、RoMa 和 DeDoDe 已从核心匹配链、
构建脚本及发行资源中删除。

## SIFT + LightGlue TensorRT

SIFT 由 CUDA 实现提取，不需要权重文件。LightGlue 只使用 TensorRT，不包含 TorchScript matcher 或
CPU 回退。engine 与 TensorRT 版本、GPU 架构、精度和固定关键点容量绑定，应在目标机器上生成，不能把
开发机 engine 当作通用发行资源。

先复用仓库根目录 `.venv` 并安装导出依赖：

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
.\.venv\Scripts\python.exe -m pip install tensorrt-cu13 nvidia-modelopt onnx onnxscript
```

默认从 LightGlue 官方发布读取 SIFT 权重并生成固定容量 engine：

```powershell
.\.venv\Scripts\python.exe scripts\models\export_lightglue_tensorrt.py `
    --engine build\model_cache\lightglue_tensorrt\lightglue_sift_fp32.engine `
    --precision fp32 `
    --bucket-keypoints 1024
```

`--weights` 可指定本地 `.pth`。`--bucket-keypoints` 是单张影像进入 LightGlue 的固定输入容量；运行时
超过该容量会明确报错，不会静默截断。生产默认 FP32；切换 FP16 前必须在业务数据上回归匹配集合、置信度、
几何内点数和最终注册率。

CMake 配置需要 TensorRT SDK：

```powershell
cmake -S . -B build\windows-vcpkg-cuda-release `
    -DTensorRT_ROOT=C:\path\to\TensorRT `
    -DBUILD_TESTS=ON
cmake --build build\windows-vcpkg-cuda-release --target feature_match_cli --parallel 32
```

`feature_match_cli` 直接接收影像，不再接收 `.sift/.sp` 等中间特征文件：

```powershell
build\windows-vcpkg-cuda-release\bin\feature_match_cli.exe `
    --left A.tif --right B.tif `
    --output-dir E:\project\assets\image_matches `
    --model build\model_cache\lightglue_tensorrt\lightglue_sift_fp32.engine
```

输出为每幅影像一个 `.pimatch` 二进制分片。关键点按像对算法变体隔离，像对、算法版本、配置/模型指纹、
几何内点和残差都在分片中；不会生成独立特征文件、成对 `.match` 或 JSON sidecar。

运行时按以下顺序寻找 engine：

1. `MatchPhotosOptions::lightGlueTensorRtEnginePath`；
2. 环境变量 `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE`；
3. 标准模型/构建缓存目录中的 `lightglue_sift_fp32.engine` 或容量后缀变体。

找不到 engine、TensorRT/GPU 不兼容或显式选择 CPU 时会失败并给出路径/设备原因，不会切换到另一算法。
Windows 部署还需要与构建版本一致的 `nvinfer_10.dll`。

## SAM2.1 蒙版

GUI 的“工具 -> 生成蒙版”支持 SAM2.1 TorchScript。输出遵循 PlaScan 蒙版约定：`0` 表示保留区域，
`255` 表示排除区域。默认资源目录是 `resources/models/`，也可通过 `PLASCAN_MODEL_DIR` 指定。

tiny 变体文件名：

- `sam21_hiera_tiny_encoder_cpu.pt`
- `sam21_hiera_tiny_decoder_cpu.pt`
- `sam21_hiera_tiny_encoder_cuda.pt`
- `sam21_hiera_tiny_decoder_cuda.pt`

`small`、`base_plus`、`large` 将文件名中的 `tiny` 替换为对应变体。GUI 默认优先 CUDA；只有勾选回退
选项时，CUDA 不可用或模型缺失才使用 CPU 模型。

安装或重新导出：

```powershell
.\.venv\Scripts\python.exe scripts\models\install_sam21_model.py `
    --variant tiny --devices auto --model-dir resources\models

.\.venv\Scripts\python.exe scripts\models\export_sam21_torchscript.py `
    --variant tiny `
    --checkpoint resources\models\sam2.1_hiera_tiny.pt `
    --devices cpu,cuda `
    --output-dir resources\models
```

运行入口优先读取 `PLASCAN_PYTHON_EXECUTABLE`，其次读取 `PLASCAN_PYTHON`，然后使用仓库 `.venv`、
打包运行时或 PATH 中的 Python。

## U2Net ONNX 蒙版

`resources/models/U2Net_v1.onnx` 用于快速前景/背景分离。标准 OpenCV DNN 构建支持 CPU；CUDA 需要
OpenCV 启用 DNN CUDA 后端。未启用时，只有用户允许回退才切换到 CPU。

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -EnableOpenCvDnnCuda
pwsh scripts\build_win\build_windows_cuda.ps1 -EnableOpenCvDnnCuda
```

模型加载失败必须报告实际解析路径、请求设备和回退状态，不能静默忽略缺失资源。
