# U2Net v1 ONNX 模型说明

- 文件：`U2Net_v1.onnx`
- 大小：`175997641` 字节
- SHA-256：`8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491`
- ONNX 元数据：`producer_name=pytorch`、`producer_version=1.9`
- 上游项目：<https://github.com/xuebinqin/U-2-Net>
- 上游许可：Apache License 2.0；随包全文见 [`Apache-2.0.txt`](Apache-2.0.txt)，上游副本见
  <https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE>

该 ONNX 是对上游 U-2-Net PyTorch 模型的格式和计算图转换。模型作为 PlaScan 的运行资源从
`models-v1.1.0` Release 获取，不进入普通 Git 历史；启用
`PLASCAN_BUNDLE_ONNX_MODELS` 时也会随 CPack 安装包分发。PlaScan 项目自身仍采用根目录 `LICENSE`
中的 MIT License；上游模型许可不因 PlaScan 的项目许可而改变。
