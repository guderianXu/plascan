# PlaScan Models v1.0.0

发布日期：2026-08-03

## 新增

- 发布 CUDA SIFT + TensorRT LightGlue K4096 FP32 engine、元数据和开发用 ONNX。
- 发布 LoMa-R K1024、K2048、K3840 FP16 的 feature engine、matcher engine 和 manifest。
- 发布 `U2Net_v1.onnx`，供“生成蒙版 -> AI: U2Net ONNX”直接下载。
- GUI 按当前算法和 LoMa-R 档位下载可直接运行的文件，不需要 Python 导出或 TensorRT 现场构建。

## 优化

- 模型二进制与 Git 历史分离，Release 资产由固定 URL、文件长度和 SHA-256 标识。
- PlaScan 下载器使用同目录 `.part` 临时文件，显示整体字节进度，并在替换最终文件前校验长度与
  SHA-256。
- LoMa-R 自动档只下载当前 GPU 显存分档需要的一套 engine，避免无条件下载全部 TensorRT 资源。

## 修复

- 运行时补充 `lightglue_tensorrt` 和 `loma_r_tensorrt` 子目录搜索，支持新的模型目录布局。

## 验证

- `Get-FileHash -Algorithm SHA256`：全部 13 个模型文件与客户端目录中的固定值一致；Release 另附
  `models-v1.0.0.sha256` 汇总清单。
- Windows/MSVC Release 构建通过：`plascan.exe`、`test_model_file_resolver`、
  `test_matchphotos_runtime`、`test_workflow_settings_dialog` 和 `test_gui_project_utils` 均成功链接。
- 模型目录/资产目录 7 项、匹配运行时 10 项、工作流程设置 5 项以及缺失 U2Net 模型 GUI 2 项测试通过。
- `U2NetMaskGeneratorIntegrationTest.OnnxModelRunsOnCpuWhenPresent` 通过，发布的 ONNX 可由 OpenCV DNN
  CPU 后端执行。
- GitHub Release API 返回的 13 个模型资产 SHA-256 digest 与本地汇总清单逐项一致。

## 已知问题

- TensorRT engine 的预构建环境为 NVIDIA GeForce RTX 5080（SM 12.0）和 TensorRT 10.16.1.11，
  不保证跨 GPU 架构或 TensorRT 版本兼容；不兼容设备需要单独发布对应模型包。
- LoMa-R K3840 下载约 803 MiB，U2Net 下载约 167.8 MiB，下载时间取决于用户网络。
- U2Net CPU 推理可由标准 OpenCV DNN 使用；CUDA 推理需要 OpenCV DNN CUDA 后端。
- 当前 WSL 环境未安装 Qt6/Qt6Network，因此本次未执行 Linux/GCC GUI 构建。
