# 模型与推理资源

PlaScan 生产构建使用 TensorRT 匹配资源，以及 BiRefNet Dynamic/U2Net ONNX 蒙版模型，不链接
LibTorch。Python PyTorch 只用于开发机上的模型导出，不进入 C++ 运行时或安装包。

## 预构建模型下载

工作流程设置和“生成蒙版”对话框会检测当前算法的模型资源。资源缺失时可点击“下载模型”；程序按
独立且不可变的模型 Release 逐文件验证长度与 SHA-256：

- [`models-v1.1.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.1.0)：U2Net、LightGlue
  和 LoMa-R 既有资产；
- [`models-v1.2.0`](https://github.com/guderianXu/plascan/releases/tag/models-v1.2.0)：仅包含
  BiRefNet Dynamic ONNX 和 provenance，不重复或替换 v1.1.0 资产。

最终用户不需要 Python/PyTorch；PlaScan 的 C++ TensorRT Builder 会在首次使用时生成当前机器专用 engine。

- 源码树运行：写入 `resources/models/birefnet_dynamic`、`resources/models/lightglue_tensorrt` 或
  `resources/models/loma_r_tensorrt`；U2Net 位于 `resources/models`；
- 安装版运行：写入 `QStandardPaths::AppLocalDataLocation/models` 下的算法子目录，避免安装目录无写权限；
- 设置 `PLASCAN_MODEL_DIR`：优先写入该目录下的算法子目录，适合共享模型盘或自定义部署；调用方必须
  保证该目录可写，写入失败时程序不会静默改用其它缓存目录。

engine 缓存键包含 ONNX SHA-256、TensorRT 完整版本、GPU Compute Capability、精度、工作区和
优化级别。缓存根与模型发现路径解耦：安装版使用 `QStandardPaths::AppLocalDataLocation/models`；
U2Net 写入 `u2net/engines/<fingerprint>`，BiRefNet 写入 `birefnet_dynamic/engines/<fingerprint>`，
LightGlue 和 LoMa-R 写入各自的
`engines/<fingerprint>` 子目录。随包 ONNX 即使位于 `Program Files` 或 `/opt/plascan`，也不会被写入。
更换显卡或 TensorRT 后会进入新缓存目录并重新构建，不会反序列化旧机器的 plan。
Windows 安装包必须携带 `nvinfer`、`nvonnxparser` 和对应架构的
`nvinfer_builder_resource_*.dll`；源码构建则必须把完整 TensorRT SDK 传给 `TensorRT_ROOT`。
Release 全部资产的离线校验值分别见 `models-v1.1.0.sha256` 和 `models-v1.2.0.sha256`。

## CPack 内置模型

`PLASCAN_BUNDLE_ONNX_MODELS` 默认开启，安装 U2Net 与 LightGlue；Windows/Linux CUDA 打包 preset
开启 `PLASCAN_BUNDLE_BIREFNET_DYNAMIC`，Windows CUDA 还开启 `PLASCAN_BUNDLE_LOMA_R_MODELS`。
模型均使用只读安装布局：

```text
resources/models/U2Net_v1.onnx
resources/models/birefnet_dynamic/BiRefNet_dynamic_1024.onnx
resources/models/birefnet_dynamic/BiRefNet_dynamic_1024.provenance.json
resources/models/lightglue_tensorrt/lightglue_sift_bucket4096.onnx
resources/models/loma_r_tensorrt/loma_r_features_k3840_fp16.onnx
resources/models/loma_r_tensorrt/loma_r_matcher_dynamic_fp16.onnx
resources/models/loma_r_tensorrt/loma_r_k{1024,2048,3840}_fp16.json
share/plascan/models/{Apache-2.0.txt,U2Net_NOTICE.md,LightGlue_NOTICE.md,LoMa-R_NOTICE.md,models-v1.1.0.sha256}
share/plascan/models/{BiRefNet-MIT.txt,BiRefNet_NOTICE.md,models-v1.2.0.sha256}
```

默认输入为源码树中的同名文件；模型不进入 Git 历史。发布机构建前从 `models-v1.1.0` 准备原有模型，
从 `models-v1.2.0` 准备 BiRefNet 两项资产。
也可以设置 `PLASCAN_U2NET_ONNX_PATH`、`PLASCAN_LIGHTGLUE_ONNX_PATH`、
`PLASCAN_LOMA_R_MODEL_DIR` 或 `PLASCAN_BIREFNET_DYNAMIC_MODEL_DIR` 使用外部缓存。普通配置和编译
不强制读取大模型；安装/CPack 阶段会执行
完整校验，任一文件缺失、长度不符或 SHA-256 不符都会失败。LoMa-R 使用显式五文件白名单，BiRefNet
使用 ONNX/provenance 两文件白名单；任何本机 `.engine` 都不会进入安装包。开发用无模型安装树可显式
关闭对应开关，但不能标记为开箱可用发行包。

干净 clone 可在仓库根目录使用 GitHub CLI 准备默认输入：

```powershell
New-Item -ItemType Directory -Force `
  resources\models\birefnet_dynamic,resources\models\lightglue_tensorrt,resources\models\loma_r_tensorrt | Out-Null
gh release download models-v1.1.0 -R guderianXu/plascan `
  -p U2Net_v1.onnx -D resources/models --clobber
gh release download models-v1.1.0 -R guderianXu/plascan `
  -p lightglue_sift_bucket4096.onnx -D resources/models/lightglue_tensorrt --clobber
gh release download models-v1.1.0 -R guderianXu/plascan `
  -p 'loma_r_*' -D resources/models/loma_r_tensorrt --clobber
gh release download models-v1.2.0 -R guderianXu/plascan `
  -p 'BiRefNet_dynamic_1024.*' -D resources/models/birefnet_dynamic --clobber
```

内置 U2Net 可由 OpenCV DNN CPU 加载，也可像 BiRefNet、LightGlue 与 LoMa-R 一样在 NVIDIA GPU 上使用
TensorRT。BiRefNet 仅支持 TensorRT，不提供 OpenCV CPU 回退。TensorRT 路径要求 CUDA、ONNX Parser
和目标 GPU 架构对应的 builder resource；Windows 发布包应捆绑
这些运行时，Linux 包则需捆绑或明确要求目标机安装兼容版本。安装包只分发便携 ONNX，绝不能包含开发机
生成的 `.engine`，也不再为了 U2Net 携带 cuDNN。

## SIFT + LightGlue TensorRT

SIFT 由 CUDA 实现提取，不需要权重文件。LightGlue Release 发布固定 K4096 的 FP32 ONNX，不包含
TorchScript matcher 或 CPU 回退。程序首次匹配时在后台构建本机 engine，后续直接复用环境指纹缓存。
导出器会为基础 FP32 ONNX 及 ModelOpt 转换后的 FP16 ONNX 分别写入
`<artifact>.provenance.json`。sidecar 记录 artifact SHA-256、上游权重 SHA-256、LightGlue revision、
模型配置、输入/profile、opset、精度和 Torch/ONNX/ModelOpt/TensorRT 工具版本。已有 ONNX 只有在
sidecar 契约与当前导出参数完全一致且 artifact 哈希匹配时才会复用；`--skip-export` 不满足契约时直接报错。

```powershell
python scripts\env\setup_python_runtime.py --device cuda --cuda-wheel cu130
.\.venv\Scripts\python.exe -m pip install tensorrt-cu13 nvidia-modelopt onnx onnxscript
.\.venv\Scripts\python.exe scripts\models\export_lightglue_tensorrt.py `
    --onnx build\model_cache\lightglue_tensorrt\lightglue_sift_bucket4096.onnx `
    --precision fp32 --bucket-keypoints 4096 --skip-build
```

运行时按以下顺序寻找 ONNX 或历史本机 engine：

1. `MatchPhotosOptions::lightGlueTensorRtEnginePath`；
2. 环境变量 `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE`；
3. 标准模型目录中的 ONNX。

旧 `.engine` 只在用户显式选择时兼容，不参与自动发现，避免升级后继续命中另一块 GPU 生成的 plan。

## LoMa-R TensorRT

LoMa-R 使用 DaD 检测器、DeDoDe-G/DINOv2 描述子和旋转不变 LoMa-R 匹配器。便携资源由共享的
K3840 特征 ONNX、动态 K 匹配 ONNX 和三个 K 桶 manifest 组成：

- `feature_onnx`：输入 `[1,3,H,W]` RGB float，输出 `[1,3840,2]` 归一化关键点、`[1,3840]`
  置信度和 `[1,3840,256]` 描述子；
- `matcher_onnx`：输入动态 K 的两组关键点、描述子和有效位，输出 `[1,K,K]` 匹配概率矩阵；
- manifest：记录特征容量、匹配 K、输入尺寸、描述子维度、精度和两个 ONNX 的 SHA-256。

feature 与 matcher ONNX 各自携带独立的 provenance sidecar。feature 只归属 DaD、DeDoDe-G 和
DINOv2 checkpoints，matcher 只归属 `loma_R.pth`；二者都记录 LoMa-R 源码 revision、artifact 哈希、
模型配置、输入/profile、opset、精度和工具版本。包 manifest 只从实际消费的两个 ONNX 及其已校验
sidecar 推导，不复制本次 CLI 参数或把 matcher-only 的新 checkpoint 误记到旧 feature 上。

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
    --input-size 784 --keypoints 3840 --precision fp16 --onnx-only
```

上述唯一 package composer 一次生成 `loma_r_features_k3840_fp16.onnx`、
`loma_r_matcher_dynamic_fp16.onnx`，以及 `loma_r_k1024_fp16.json`、
`loma_r_k2048_fp16.json`、`loma_r_k3840_fp16.json` 三个清单，不需要也不允许手工重命名或修改 JSON。
已有产物 sidecar 缺失、契约变化或 artifact 哈希不符时会自动重导出；`--matcher-only` 会先严格校验
既有 feature provenance，不兼容时明确拒绝。也可单独运行 `compose_loma_r_package.py` 对已导出的
两个 ONNX 重新组合清单，但该入口同样要求文件名和 sidecar 契约完全匹配。
TensorRT 10.x 的 `ITopK` 最多支持 3840，特征 engine 始终输出 3840 个候选；运行时依据清单截取
特征并将动态 matcher 分别构建为 K1024、K2048 或 K3840 的固定 profile。
`MatchPhotosRuntime` 在调用 TensorRT Builder 前会重新计算两个 ONNX 的 SHA-256，并与 schema 2
manifest 的 `feature_onnx_sha256`、`matcher_onnx_sha256` 比较；缺失、格式错误或内容不匹配均会停止构建
并报告具体 artifact。

运行时按以下顺序寻找 manifest：

1. `MatchPhotosOptions::lomaRTensorRtPackagePath`；
2. 环境变量 `PLASCAN_LOMA_R_TENSORRT_PACKAGE`；
3. 标准模型目录中的全部 LoMa-R package manifest（自动排除 `.provenance.json` sidecar）；运行时按手动
   档位或 GPU 总显存选择最合适的 K。

自动档位为：显存小于 8 GiB 使用 K1024，8 至 12 GiB 使用 K2048，12 GiB 及以上使用
K3840。GUI 的“工作流程设置 -> 空中三角测量 -> LoMa-R 特征档位”可以手动覆盖；显式 manifest
路径优先级最高。旧版通用文件名仍可被扫描，便于已有本机模型平滑迁移。

LoMa-R 来源为 `davnords/loma`。其主体代码采用 MIT 许可，匹配器继承 LightGlue 的 Apache-2.0 许可；
权重再分发遵循上游项目条款。PlaScan Git 仓库不提交这些大文件，发布机构建从独立 Model Release
下载并校验，归属说明见 [`LoMa-R_NOTICE.md`](LoMa-R_NOTICE.md)。

## 匹配输出

`feature_match_cli` 和空中三角测量最终都写入逐影像 `.pimatch` 二进制分片。关键点按算法变体隔离，
像对、算法版本、配置/模型指纹、几何内点和残差均保存在分片中；不会生成独立特征文件、成对
`.match` 或 JSON sidecar。GUI 在“工作流程 -> 设置 -> 空中三角测量”中选择算法与对应资源。

找不到资源、TensorRT/GPU 不兼容或显式选择 CPU 时会明确失败，不会静默切换算法。Windows 部署需要
与构建版本一致的 `nvinfer_10.dll`、`nvonnxparser_10.dll` 和 builder resource。

## BiRefNet Dynamic ONNX 蒙版

“生成蒙版”中的“AI: BiRefNet Dynamic（推荐）”使用 `ZhengPeng7/BiRefNet_dynamic` 的固定生产部署契约：

- 输入 `input_image`：`1×3×1024×1024` RGB float32，opset 17；
- 输出 `output_image`：`1×1×1024×1024` float32 前景 logits；
- 原图保持宽高比缩放并居中 letterbox，空白区域填 0；RGB 转为 `[0,1]` 后使用 ImageNet
  `mean=(0.485,0.456,0.406)`、`std=(0.229,0.224,0.225)` 归一化；
- C++ 后处理对 raw logits 逐像素执行 sigmoid，不做逐图 min/max 归一化；随后裁掉 letterbox padding、
  恢复原图尺寸并按前景阈值生成 PlaScan 排除蒙版。

该模型只支持 TensorRT GPU。运行时优先构建 FP16 engine，不支持 FP16 时尝试 FP32；TensorRT、CUDA 或
受支持 NVIDIA GPU 不可用时明确失败，不会回退 OpenCV CPU。安装包和最终用户运行时不需要 Python、
PyTorch、LibTorch 或 Hugging Face 依赖；它们只用于开发机导出和等价性验证。

独立 Release `models-v1.2.0` **只包含以下两个资产**；U2Net、LightGlue 和 LoMa-R 仍从
`models-v1.1.0` 获取：

| 资产 | 字节数 | SHA-256 |
|------|-------:|---------|
| `BiRefNet_dynamic_1024.onnx` | 972558911 | `3af7fe29f80be80e12595671293c877af6767cae71566a8765face68965f0742` |
| `BiRefNet_dynamic_1024.provenance.json` | 1688 | `9e100509b59aedfeabd0aabc7277009b0d620803b27f482abb2e28220de8d4ff` |

provenance 固定上游 revision `280306042f57b7a33854319da62fd86aaa89ec4c`、原始
`model.safetensors` 的大小/SHA-256、导出器来源、opset、I/O 契约、工具版本和等价性指标。发布模型可由
以下脚本重新导出；`--skip-checker` 和 `--skip-runtime-check` 不能用于 Release 资产：

本次 Release provenance 记录的实际工具链为 Python 3.11.9、PyTorch 2.5.1+cpu、torchvision
0.20.1+cpu、transformers 4.45.2、ONNX 1.17.0 和 ONNX Runtime 1.20.1；需要逐字节复现时应固定这些版本。

```powershell
python scripts\env\setup_python_runtime.py --device cpu
.\.venv\Scripts\python.exe -m pip install `
  "transformers==4.45.2" "onnx==1.17.0" "onnxruntime==1.20.1"
.\.venv\Scripts\python.exe scripts\models\export_birefnet_dynamic_onnx.py `
  --output resources\models\birefnet_dynamic\BiRefNet_dynamic_1024.onnx
```

本次已完成的导出验证：

- `onnx.checker.check_model` 通过，模型为单文件、无 external data 和自定义算子域；
- ONNX Runtime CPU 与 PyTorch raw logits 对比：最大绝对误差 `0.000234127`，平均绝对误差
  `0.0000155839`。

真实 TensorRT 干净环境部署验证已通过：RTX 4060 Laptop 8 GiB、TensorRT 10.15 使用 FP16，首次
engine 构建加推理耗时 `2631483 ms`（43 分 51 秒），生成的 engine 为 `540031644` bytes；第二个进程
复用同一 engine 并完成推理耗时 `33573 ms`。实际后端为 TensorRT，输出张量为 `output_image`；engine
位于隔离的用户临时缓存，模型目录和安装树均未产生 `.engine`。首次构建受 GPU、TensorRT 和磁盘性能
影响，在同级 8 GiB GPU 上应预留约 45 分钟。发布前使用以下命令复现门禁：

```powershell
pwsh scripts\build_win\build_windows_cuda.ps1 `
  -Target test_mask_generation -RunBiRefNetTensorRtDeploymentTest
```

该门禁从 `package-smoke` 读取已安装运行时和模型，测试程序及 GTest DLL 只放在包外临时夹具；在清除
外部 CUDA/TensorRT/vcpkg/Python 路径并隔离用户缓存后执行两次进程：第一次必须新建 engine，第二次
必须复用同一路径，同时确认 engine 只写用户缓存且安装树不变。
模型来源、固定权重和 MIT 许可说明见 [`BiRefNet_NOTICE.md`](BiRefNet_NOTICE.md)。

## U2Net ONNX 蒙版

`resources/models/U2Net_v1.onnx` 用于快速前景/背景分离。标准 Windows CUDA 构建使用 TensorRT
FP16/FP32 加速，OpenCV 永久保持 CPU-only，只作为无 TensorRT/GPU 环境的回退后端。自动模式优先
TensorRT；强制 TensorRT 时默认禁止静默回退，只有用户明确允许后才会切换到 OpenCV CPU。随附模型
使用固定 `1×3×320×320` 输入，GUI 会显示并锁定该真实生效尺寸。

在“生成蒙版”中选择“AI: U2Net ONNX”时，GUI 会自动检测模型。模型缺失时可点击“下载 U2Net
模型”，PlaScan 从 GitHub Release `models-v1.1.0` 异步下载，并在写入最终文件前验证固定大小和
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
pwsh scripts\build_win\build_windows_cuda.ps1 -InstallDeps
pwsh scripts\build_win\build_windows_cuda.ps1
```

发布前使用 `-Target test_mask_generation -RunU2NetTensorRtDeploymentTest` 验证部署目录。该测试检查
OpenCV vcpkg ABI 不含 `cuda`、`cudnn`、`dnn-cuda`，并确认部署目录具有 TensorRT runtime、ONNX parser、
plugin、builder resource 及 CUDA 运行库，且不含 cuDNN 或预生成 engine。随后清空子进程中的外部
CUDA/TensorRT/vcpkg PATH，在全新用户缓存中从 ONNX 首次构建 engine 并执行真实 TensorRT 推理；用例
被跳过也会视为部署失败。

模型加载失败必须报告实际解析路径、请求设备和回退状态，不能静默忽略缺失资源。
