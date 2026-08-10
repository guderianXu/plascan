# PlaScan Models v1.2.0

发布日期：2026-08-10

> 这是增量模型 Release，只包含 BiRefNet Dynamic 的两个资产。U2Net、LightGlue 和 LoMa-R 不在本
> Release 重复发布，继续由不可变的 `models-v1.1.0` 提供。

## 新增

- 发布 `BiRefNet_dynamic_1024.onnx`，作为“生成蒙版 → AI: BiRefNet Dynamic（推荐）”的便携模型。
- 发布配套 `BiRefNet_dynamic_1024.provenance.json`，固定上游模型、权重哈希、导出器、opset、I/O
  契约、工具版本和 PyTorch/ONNX Runtime 等价性结果。
- 部署契约为固定 `1×3×1024×1024` RGB float32 输入和 `1×1×1024×1024` raw foreground logits
  输出；运行时使用保持宽高比的 letterbox、ImageNet 归一化和显式 sigmoid。

资产白名单：

| 资产 | 字节数 | SHA-256 |
|------|-------:|---------|
| `BiRefNet_dynamic_1024.onnx` | 972558911 | `3af7fe29f80be80e12595671293c877af6767cae71566a8765face68965f0742` |
| `BiRefNet_dynamic_1024.provenance.json` | 1688 | `9e100509b59aedfeabd0aabc7277009b0d620803b27f482abb2e28220de8d4ff` |

可供 `sha256sum -c`/`Get-FileHash` 复核的离线清单见
[`models-v1.2.0.sha256`](../models/models-v1.2.0.sha256)。

## 优化

- 生产运行仅依赖 PlaScan C++、CUDA/TensorRT 和便携 ONNX，不依赖 Python、PyTorch、LibTorch 或
  Hugging Face 运行时。
- 不发布绑定开发机的 TensorRT engine。PlaScan 首次使用时根据 ONNX SHA-256、TensorRT 完整版本、
  GPU Compute Capability、精度和构建参数在用户缓存中创建 engine，后续进程复用兼容缓存。
- BiRefNet 使用独立 `models-v1.2.0`，不改变 `models-v1.1.0` 的 U2Net、LightGlue、LoMa-R 下载 URL
  和校验值。

## 修复

- 无；这是新增模型资产 Release，不替换既有模型字节。

## 验证

- `onnx.checker.check_model`：通过；固定 opset 17、float32 I/O、无 external data、无自定义算子域。
- ONNX Runtime CPU 与 PyTorch raw logits 等价性：最大绝对误差 `0.000234127`，平均绝对误差
  `0.0000155839`。
- 文件校验：ONNX 为 `972558911` bytes / SHA-256
  `3af7fe29f80be80e12595671293c877af6767cae71566a8765face68965f0742`；provenance 为 `1688` bytes /
  SHA-256 `9e100509b59aedfeabd0aabc7277009b0d620803b27f482abb2e28220de8d4ff`。
- RTX 4060 Laptop 8 GiB / TensorRT 10.15 / FP16 干净部署门禁：通过。首次 engine 构建加推理耗时
  `2631483 ms`（43 分 51 秒），engine 为 `540031644` bytes；第二个进程复用同一路径并完成推理耗时
  `33573 ms`。实际后端为 TensorRT，输出张量为 `output_image`；engine 位于隔离用户临时缓存，模型
  目录和安装树均无 `.engine`。验证命令为：

  ```powershell
  pwsh scripts\build_win\build_windows_cuda.ps1 `
    -Target test_mask_generation -RunBiRefNetTensorRtDeploymentTest
  ```

## 已知问题

- BiRefNet Dynamic 仅支持 TensorRT GPU，没有 OpenCV/PyTorch CPU 回退；无受支持 NVIDIA GPU 或
  TensorRT/CUDA 运行时的机器应继续使用 U2Net CPU。
- ONNX 约 927.50 MiB，生成的 FP16 engine 在本次环境为 540031644 bytes；首次下载和首次本机 engine
  构建需要较长时间及额外用户缓存空间。RTX 4060 Laptop 8 GiB 实测首次构建加推理约 43 分 51 秒，
  同级设备应预留约 45 分钟；不同 GPU、TensorRT 版本和磁盘性能会改变耗时及 engine 大小。
