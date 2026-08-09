# LightGlue SIFT K4096 ONNX 模型说明

- 文件：`lightglue_sift_bucket4096.onnx`
- 大小：`51072656` 字节
- SHA-256：`773d3de316c37e8d408312d39139352b45e2a93ba055e59cfa2806c5d54ede69`
- 上游项目：<https://github.com/cvg/LightGlue>
- 上游版权：Copyright 2023 ETH Zurich
- 上游许可：Apache License 2.0；随包全文见 [`Apache-2.0.txt`](Apache-2.0.txt)，上游副本见
  <https://github.com/cvg/LightGlue/blob/main/LICENSE>

该文件由 PlaScan 将上游 PyTorch LightGlue 权重转换为固定 K4096、CUDA SIFT 描述子接口的便携 ONNX，
相对上游文件修改了模型格式和输入输出接口。目标机器首次使用时会通过 TensorRT ONNX Parser 生成与
本机 TensorRT 版本和 GPU Compute Capability 匹配的 engine；生成的 engine 不属于可跨机器分发的
模型资产。模型可从 PlaScan `models-v1.1.0` Release 获取，也可随启用
`PLASCAN_BUNDLE_ONNX_MODELS` 的 CPack 安装包一并分发。
