# image_matching 统一影像匹配模块

`image_matching` 是 PlaScan 唯一的局部特征提取、学习型匹配、两视几何验证和匹配结果持久化模块。
当前生产算法固定为 **CUDA SIFT + TensorRT LightGlue**。`matchphototask` 负责编排任务，SfM、GUI 和
CLI 只消费本模块的稳定结果契约，不直接依赖 SIFT 描述子或 TensorRT 数据布局。

## 模块边界

- `ImageMatchingAlgorithm` 定义算法能力、版本、配置指纹、模型指纹、特征提取和像对匹配接口。
- `ImageMatchingRegistry` 是算法注册入口。增加新算法时注册新的实现，不修改 SfM、项目格式或查看器。
- `sift/` 负责 CUDA SIFT。提取结果只保存在一次 `MatchPhotosTask` 的有界内存缓存中。
- `lightglue/` 负责 TensorRT engine 加载、执行和输出后处理，不提供 TorchScript 或 CPU 隐式回退。
- `sift_lightglue/` 组合上述前端并提供当前唯一注册算法 `sift_lightglue`。
- `geometry/` 负责基础矩阵/单应模型验证和逐匹配像素残差。
- `ImageMatchFile` 是 `.pimatch` 格式的唯一序列化入口。
- `ImageMatchRepository` 负责对称写入、按缓存键查找和批量清理逐影像分片。

## 为什么不保存独立特征文件

描述子只服务于当次 LightGlue 推理，SfM 和质量分析实际需要的是像点、跨影像对应、置信度、几何内点状态和
残差。持久化描述子会把下游绑定到特征维度及模型版本，并产生重复 I/O。因此本模块采用以下生命周期：

1. CUDA SIFT 提取关键点和描述子到任务级内存缓存。
2. 所有引用该影像的候选像对完成后，描述子可以释放。
3. 最终分片仅保存至少参与一个匹配的关键点观测和邻接匹配记录。
4. 再次运行时只有算法 ID、算法版本、配置指纹和模型指纹全部命中的 `.pimatch` 像对可复用；
   缺失结果会重新提取所需影像。

## `.pimatch` v1 数据组织

文件名是“影像可读 stem + 规范绝对路径 SHA-256 前缀”，只标识 owner 影像，不编码 peer 或算法。
同名但不同目录的影像不会冲突。存储遵循“一幅影像一个分片”，每个参与匹配的影像分片包含：

- owner 影像身份、尺寸、文件大小和修改时间；
- 每个相邻影像的一个或多个算法变体；
- 每个变体的 `algorithmId + algorithmVersion + configFingerprint + modelFingerprint`；
- 每个变体独立的 owner 关键点观测表：`featureId/x/y/scale/orientation/response`；
- 原始匹配数、几何内点数、连接点数、几何模型及 3x3 模型矩阵；
- 每条对应的两端 featureId、peer 坐标、置信度、像素残差和状态位。

容器使用固定 magic `PLIMATCH`、小端序、显式格式版本、长度边界和 payload SHA-256。当前读取器只接受
`kImageMatchFormatVersion == 1`，不会把旧 `.match`、JSON sidecar 或未知版本静默转换为当前结果。
观测表按算法变体隔离，因此两个算法或两套提取配置即使都从 `featureId=0` 编号，也不会覆盖彼此坐标。

## 扩展新算法

新增算法必须：

1. 实现 `ImageMatchingAlgorithm` 并使用稳定、唯一的 `algorithmId` 和单调递增版本号。
2. 将所有会改变结果的参数写入配置指纹，将 engine/权重内容写入模型指纹。
3. 输出统一 `FeatureSet` 与 `PairMatchData`，并为每条匹配填入置信度和可计算的几何残差。
4. 在 `ImageMatchingRegistry` 注册；不得通过文件扩展名或项目元数据增加算法分支。
5. 增加注册、往返序列化、缓存失效、几何验证和损坏文件测试。

只要满足该契约，项目格式、SfM 输入、匹配查看器和空三工作流无需了解新算法的描述子类型。
