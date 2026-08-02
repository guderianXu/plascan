# 模型与推理资源

PlaScan 生产构建使用 TensorRT 匹配资源和 U2Net ONNX 蒙版模型，不链接 LibTorch。Python PyTorch 只用于
开发机上的模型导出，不进入 C++ 运行时或安装包。

## SIFT + LightGlue TensorRT

SIFT 由 CUDA 实现提取，不需要权重文件。LightGlue 只使用 TensorRT，不包含 TorchScript matcher 或
CPU 回退。engine 与 TensorRT 版本、GPU 架构、精度和固定关键点容量绑定，应在目标机器上生成。

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
.\.venv\Scripts\python.exe -m pip install tensorrt-cu13 nvidia-modelopt onnx onnxscript
.\.venv\Scripts\python.exe scripts\models\export_lightglue_tensorrt.py `
    --engine build\model_cache\lightglue_tensorrt\lightglue_sift_fp32.engine `
    --precision fp32 --bucket-keypoints 4096
```

运行时按以下顺序寻找 engine：

1. `MatchPhotosOptions::lightGlueTensorRtEnginePath`；
2. 环境变量 `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE`；
3. 标准模型目录或构建缓存中的兼容 engine。

## LoMa-R TensorRT

LoMa-R 使用 DaD 检测器、DeDoDe-G/DINOv2 描述子和旋转不变 LoMa-R 匹配器。生产资源是一个 JSON
manifest 和两个固定形状 engine：

- `feature_engine`：输入 `[1,3,H,W]` RGB float，输出 `[1,K,2]` 归一化关键点、`[1,K]`
  置信度和 `[1,K,256]` 描述子；
- `matcher_engine`：输入两组关键点、描述子和有效位，输出 `[1,K,K]` 匹配概率矩阵；
- manifest：记录算法/格式版本、输入尺寸、K、描述子维度、精度和两个 engine 的 SHA-256。

导出需要本地 LoMa-R 源码以及以下官方权重，不会自动下载或提交权重：

- `loma_R.pth`
- `dad.pth`
- `dedode_descriptor_G.pth`
- `dinov2_vitl14_pretrain.pth`

```powershell
.\.venv\Scripts\python.exe -m pip install "einops>=0.8.1" tensorrt-cu13 onnx onnxscript
.\.venv\Scripts\python.exe scripts\models\export_loma_r_tensorrt.py `
    --loma-repo E:\code\matching-experiments\loma-r `
    --weights-dir E:\models\loma-r `
    --output-dir build\model_cache\loma_r_tensorrt `
    --input-size 784 --keypoints 2048 --precision fp16
```

运行时按以下顺序寻找 manifest：

1. `MatchPhotosOptions::lomaRTensorRtPackagePath`；
2. 环境变量 `PLASCAN_LOMA_R_TENSORRT_PACKAGE`；
3. 标准模型目录中的 `loma_r_tensorrt.json`、`loma_r_fp16.json` 或 `loma_r_fp32.json`。

LoMa-R 来源为 `davnords/loma`。其主体代码采用 MIT 许可，匹配器继承 LightGlue 的 Apache-2.0 许可；
权重再分发遵循上游项目条款，PlaScan 仓库不包含这些权重。

## 匹配输出

`feature_match_cli` 和空中三角测量最终都写入逐影像 `.pimatch` 二进制分片。关键点按算法变体隔离，
像对、算法版本、配置/模型指纹、几何内点和残差均保存在分片中；不会生成独立特征文件、成对
`.match` 或 JSON sidecar。GUI 在“工作流程 -> 设置 -> 空中三角测量”中选择算法与对应资源。

找不到资源、TensorRT/GPU 不兼容或显式选择 CPU 时会明确失败，不会静默切换算法。Windows 部署需要
与构建版本一致的 `nvinfer_10.dll`。

## U2Net ONNX 蒙版

`resources/models/U2Net_v1.onnx` 用于快速前景/背景分离。标准 OpenCV DNN 构建支持 CPU；CUDA 需要
OpenCV 启用 DNN CUDA 后端。未启用时，只有用户允许回退才切换到 CPU。

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -EnableOpenCvDnnCuda
pwsh scripts\build_win\build_windows_cuda.ps1 -EnableOpenCvDnnCuda
```

模型加载失败必须报告实际解析路径、请求设备和回退状态，不能静默忽略缺失资源。
