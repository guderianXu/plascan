# LoMa-R ONNX 模型说明

PlaScan 安装包中的 LoMa-R 便携模型由 `models-v1.1.0` 发布，包含 DaD 检测、DeDoDe-G/DINOv2
描述与旋转不变 LoMa-R 匹配所需的两个共享 ONNX，以及 K1024、K2048、K3840 三个运行清单。
安装包不分发任何开发机生成的 TensorRT `.engine`；PlaScan 会在目标机器首次使用时生成并缓存
与本机 TensorRT 版本和 GPU Compute Capability 匹配的 engine。

- LoMa-R 项目：<https://github.com/davnords/loma>（项目代码采用 MIT License）。
- LoMa-R 匹配器派生自 LightGlue；Apache License 2.0 全文见
  [`Apache-2.0.txt`](Apache-2.0.txt)。
- ONNX 还包含上游 DaD、DeDoDe-G 与 DINOv2 权重转换后的参数；再分发时仍需遵守各上游项目和
  权重发布页面的适用条款。

发布资产的固定文件名、长度和 SHA-256 见 [`models-v1.1.0.sha256`](models-v1.1.0.sha256)。
PlaScan 项目自身采用根目录 `LICENSE` 中的 MIT License；该许可不会改变上游模型各自的许可条款。
