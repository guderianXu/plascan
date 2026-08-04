# PlaScan Models v1.1.0

发布日期：2026-08-04

## 新增

- 发布 CUDA SIFT + LightGlue K4096 FP32 的便携 ONNX。
- 发布 LoMa-R 的共享 K3840 特征 ONNX、动态 K 匹配 ONNX，以及 K1024、K2048、K3840 三个清单。
- 继续发布 `U2Net_v1.onnx`，保证所有模型下载入口使用同一个 Release 版本。

## 优化

- PlaScan 使用 C++ TensorRT ONNX Parser 在目标机器首次使用时构建 engine，最终用户不需要 Python。
- engine 缓存指纹包含 ONNX SHA-256、TensorRT 完整版本、GPU Compute Capability、精度和构建参数；
  更换显卡或 TensorRT 后会自动构建新的缓存，不会错误复用旧 plan。
- LoMa-R 三个 K 桶共享 1.23 GiB 特征 ONNX 和 43.4 MiB 动态 matcher，切换档位只需下载很小的清单。
- Windows 构建与安装会部署 `nvonnxparser`、TensorRT builder resource 和 plugin 运行库。

## 修复

- 删除 `models-v1.0.0` 中绑定 RTX 5080 / TensorRT 10.16.1.11 的预构建 engine，修复 RTX 4060、
  不同 TensorRT 补丁版本等目标机器反序列化失败的问题。
- 工作流程设置仅检查便携模型资源，不会在 GUI 主线程同步构建大型 LoMa-R engine。
- 模型下载器按尚未存在的共享文件计算磁盘需求，切换 LoMa-R K 桶不再重复预留完整模型空间。

## 验证

- `onnx.checker.check_model`：LightGlue、LoMa-R feature、LoMa-R dynamic matcher 全部通过。
- Windows/MSVC：`test_matchphotos_runtime` 12/12、`test_model_file_resolver` 7/7 通过，完整 GUI 成功链接。
- RTX 5080 / TensorRT 10.16.1.11：LightGlue ONNX 首次构建、缓存复用及推理通过。
- LoMa-R K1024 C++ 首次构建约 701 秒；Dino 双影像得到 666 个原始匹配、551 个几何内点。
  第二次缓存复用的完整流程为 13.55 秒，匹配和内点数量一致。
- LightGlue ONNX 缓存复用的 Dino 双影像流程得到 487 个原始匹配、460 个几何内点。
- 全部发布资产的 SHA-256 见 `docs/models/models-v1.1.0.sha256`。

## 已知问题

- 首次构建 LoMa-R feature engine 需要较长时间和额外磁盘空间；后续运行直接复用本机缓存。
- TensorRT 构建需要完整运行组件，包括 ONNX Parser 和 builder resource；只有 `nvinfer` 推理 DLL 不足以构建。
- 当前 WSL 环境若未安装 Qt6/TensorRT 开发包，无法执行 Linux/GCC TensorRT 构建验证。
