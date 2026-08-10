# BiRefNet Dynamic 模型归属说明

PlaScan 的 BiRefNet Dynamic 蒙版功能使用由以下官方模型权重转换得到的 ONNX：

- 模型：`ZhengPeng7/BiRefNet_dynamic`
- 来源：<https://huggingface.co/ZhengPeng7/BiRefNet_dynamic>
- 固定 revision：`280306042f57b7a33854319da62fd86aaa89ec4c`
- 上游权重：`model.safetensors`
- 上游权重大小：`444473596` bytes
- 上游权重 SHA-256：`e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d`
- 官方实现：<https://github.com/ZhengPeng7/BiRefNet>

上游权重使用任意形状与分辨率范围训练。PlaScan 首个生产部署契约将其导出为
`BiRefNet_dynamic_1024.onnx`：固定 `1x3x1024x1024` RGB float32 输入、opset 17，输入名为
`input_image`，输出名为 `output_image`，输出未经过 sigmoid 的前景 logits，形状为
`1x1x1024x1024`。运行时保持原图宽高比并使用 letterbox 预处理。

PlaScan 只分发可移植 ONNX。TensorRT engine 会在目标机器首次使用时根据本机 GPU 和 TensorRT
版本生成并写入用户缓存，不能作为跨机器模型资产分发。

BiRefNet 的代码和官方模型卡标注为 MIT License，版权归 ZhengPeng 所有。完整许可证见
[`BiRefNet-MIT.txt`](BiRefNet-MIT.txt)。本说明不改变 PlaScan 自身的 MIT 许可，也不表示上游作者为
PlaScan 提供担保或背书。

论文引用：

> Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe.
> Bilateral Reference for High-Resolution Dichotomous Image Segmentation. CAAI Artificial Intelligence
> Research, 2024.
