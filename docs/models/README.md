# 模型与推理资源

PlaScan 生产构建使用 TensorRT 匹配资源和 U2Net ONNX 蒙版模型，不链接 LibTorch。Python PyTorch 只用于
开发机上的模型导出，不进入 C++ 运行时或安装包。

## 预构建模型下载

工作流程设置会检测当前匹配算法的 TensorRT 资源。资源缺失时可点击“下载模型”，程序从
[`models-v1.0.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.0.0) Release 下载
engine/manifest，逐文件验证长度与 SHA-256 后直接使用，不需要本机 Python 或模型转换环境。

- 源码树运行：写入 `resources/models/lightglue_tensorrt` 或
  `resources/models/loma_r_tensorrt`；
- 安装版运行：写入 `QStandardPaths::AppLocalDataLocation/models` 下的算法子目录，避免安装目录无写权限；
- 设置 `PLASCAN_MODEL_DIR`：优先写入该目录下的算法子目录，适合共享模型盘或自定义部署。

首批 engine 由 NVIDIA GeForce RTX 5080（SM 12.0）和 TensorRT 10.16.1.11 构建。TensorRT engine
与 GPU 架构、TensorRT/CUDA 运行时绑定；Release 下载解决的是模型分发问题，不保证跨架构反序列化。
其他环境仍需发布对应兼容包，运行时加载失败会明确报告，不会调用 Python 现场转换或静默降级。
Release 全部资产的离线校验值见 `docs/models/models-v1.0.0.sha256`。

## SIFT + LightGlue TensorRT

SIFT 由 CUDA 实现提取，不需要权重文件。LightGlue 只使用 TensorRT，不包含 TorchScript matcher 或
CPU 回退。engine 与 TensorRT 版本、GPU 架构、精度和固定关键点容量绑定；兼容机器可直接使用 Release
预构建资源，其他机器应在目标环境生成并选择对应 engine。

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

分别以 `--keypoints 1024`、`2048` 和 `3840` 导出后，三个 manifest 和对应 engine 可以共存于
同一目录。manifest 文件名为 `loma_r_k<K>_<precision>.json`。TensorRT 10.x 的 `ITopK`
最多支持 3840，脚本会在请求更大 K 时直接给出错误，避免耗时导出后才构建失败。

运行时按以下顺序寻找 manifest：

1. `MatchPhotosOptions::lomaRTensorRtPackagePath`；
2. 环境变量 `PLASCAN_LOMA_R_TENSORRT_PACKAGE`；
3. 标准模型目录中的全部 `loma_r*.json`；运行时按手动档位或 GPU 总显存选择最合适的 K。

自动档位为：显存小于 8 GiB 使用 K1024，8 至 12 GiB 使用 K2048，12 GiB 及以上使用
K3840。GUI 的“工作流程设置 -> 空中三角测量 -> LoMa-R 特征档位”可以手动覆盖；显式 manifest
路径优先级最高。旧版通用文件名仍可被扫描，便于已有本机模型平滑迁移。

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

在“生成蒙版”中选择“AI: U2Net ONNX”时，GUI 会自动检测模型。模型缺失时可点击“下载 U2Net
模型”，PlaScan 从 GitHub Release `models-v1.0.0` 异步下载，并在写入最终文件前验证固定大小和
SHA-256：

```text
文件：U2Net_v1.onnx
大小：175997641 bytes
SHA-256：8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491
```

下载目录按运行形态选择：

- 设置 `PLASCAN_MODEL_DIR` 时写入该目录；
- 从仓库构建目录运行时写入源码树 `resources/models/`；
- 安装包或便携版运行时写入 Qt 的用户应用数据目录下 `models/`，不会尝试写入 `Program Files`
  或其它只读安装目录。Windows 通常为 `%LOCALAPPDATA%/PlaScan/models`，Linux 使用对应的用户
  数据目录。

下载过程使用同目录 `.part` 临时文件，文件大小和 SHA-256 均通过后才替换最终模型。U-2-Net
上游项目采用 Apache-2.0 许可；发布或再分发模型时必须保留上游来源与许可说明：
<https://github.com/xuebinqin/U-2-Net>。

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps -EnableOpenCvDnnCuda
pwsh scripts\build_win\build_windows_cuda.ps1 -EnableOpenCvDnnCuda
```

模型加载失败必须报告实际解析路径、请求设备和回退状态，不能静默忽略缺失资源。
