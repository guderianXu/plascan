# MatchPhotoTask

`matchphototask` 是类 Metashape “匹配照片”流程的编排层。
它拥有高层任务边界，但复用底层模块，不把其它模块搬进本目录。
它也是工作流中创建连接点的唯一所有者；`aerial_triangulation` 只能通过
`MatchPhotosTask` 请求特征、匹配和轨迹，不能直接实例化提取器或匹配器。

当前框架：

- `algorithm/` 负责类 Metashape 策略到注册算法计划的映射，当前支持 `sift_lightglue` 和 `loma_r`。
- `pair_selection/` 负责影像对类型、影像对选择策略和 `PairSelector`。
- `runtime/` 负责任务级 SIFT 内存缓存、ONNX 资源解析、本机 TensorRT engine 缓存、蒙版约束和 GUI 写回所需记录。
- `task/` 负责 `MatchPhotosTask`、选项、上下文和结果报告。
- `stages/` 负责流程阶段，按算法能力加载灰度或彩色输入并执行特征与两两匹配。
- `tie_points/` 负责最终多视图连接点 track 的构建、筛选和统计摘要。
- `tests/` 放置本模块自己的单元测试。

当前行为：

- `MatchPhotosAlgorithmSelector` 校验工作流选择的注册算法、设备要求与质量预设。
- SIFT 负责提供尺度和旋转鲁棒性，LightGlue 直接消费任务内存中的 SIFT 关键点与描述子。
- 连接点用途的 SIFT 检测阈值为 `0.0005`（快速模式为 `0.003`）；CUDA SIFT 使用等价的库内阈值映射。
  阈值写入前端签名，修改后旧缓存会自动失效。
- 特征阶段不再写入或读取独立特征文件。一次任务内每幅影像最多提取一次，描述子在最后一个相关像对完成后释放；
  缩放提取时会把关键点坐标还原到原始影像坐标。
- `maskApplyMode=keypoints` 时，特征阶段按项目蒙版过滤关键点和描述子，并强制重新提取特征；
  `maskApplyMode=tiepoints` 时，匹配阶段过滤任一端落入蒙版排除区的连接点。蒙版约定为 `0` 有效、非 `0` 排除。
- `maxKeypoints` 对应每张影像的关键点总量限制；调用方设置 `useExplicitKeypointLimit=true` 时，
  `0` 表示不限制。空中三角测量界面的“关键点限制”始终使用这一语义，不会因启用指导匹配而按像素数放大。
- `keypointLimitPerMegapixel` 是底层调用方可显式启用的独立限制。设置后，每张影像的关键点上限按
  `每百万像素关键点限制 * 影像百万像素数` 计算；它不由 `enableGuidedMatching` 自动开启。
- `enableGuidedMatching` 是显式用户开关，质量档位不能把未勾选状态自动改为开启；实际值写入
  `.pimatch` 配置指纹，后续 SfM 只消费同一结果变体。
- 设备为 `Auto` 时 SIFT 可优先选择 CUDA，但 LightGlue 固定使用 TensorRT/CUDA；选择 CPU、CUDA
  不可用或 engine 不兼容时会返回明确错误，不做 TorchScript/CPU 回退。
- LightGlue ONNX 可通过 `lightGlueTensorRtEnginePath` 显式传入，也可由
  `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE` 指定。运行时按 ONNX 内容、本机 TensorRT 完整版本和 GPU
  Compute Capability 构建环境指纹缓存；历史本机 engine 仅保留读取兼容。
- 匹配阶段加载本机缓存的 TensorRT engine。最终结果由 `ImageMatchRepository` 对称提交为“一幅影像一个
  `.pimatch` 分片”，同一文件保存所有相邻影像，并按算法版本与配置变体隔离本影像观测、模型指纹、
  置信度和残差，避免不同特征编号空间互相覆盖。
- 几何验证阶段使用统一 `MatchGeometryVerifier` 的 USAC/MAGSAC 基础矩阵内点。为抑制重复结构伪造的弱几何边，
  少于 64 个内点的像对还必须达到 70% 内点率；强支持像对仍按最小内点数通过。原始匹配数、内点数和内点率会记入像对设置。
- TensorRT 匹配并发按显存和 engine 固定关键点桶限制解析；检测到 OOM 时释放并发上下文并串行重试
  未完成像对。流程不会切换到另一匹配算法，也不会生成第二种缓存格式。
- 轨迹阶段通过 `TiePointTrackManager` 管理最终多视图连接点，并复用 `MultiViewTrackBuilder` 合并 track。
- 轨迹阶段成功后会写出 `assets/tie_points/latest_tie_points.json`，记录影像、track、观测点、
  参数摘要和统计信息，供项目下次打开或后续空三流程复用。
- `maxTiePointsPerImage` 对应连接点限制；
  `0` 表示关闭连接点数量稀疏。
- `excludeStationaryTiePoints` 会剔除在多张影像中像方坐标几乎固定的 track，用于过滤转台背景、
  传感器污点或镜头伪影类假连接点。
- `PairSelector` 合并来自手动输入、全量匹配、序列窗口、相机重叠、词汇召回和未来引导重匹配的候选影像对。
- 无相机通用预选由 `VocabularyOverlapRetriever` 调用 `OverlapPairGraphPlanner`，
  在词汇相似度候选上保留互选 TopK、补足单向 TopK、桥接连通分量，并固定补充序列窗口邻接候选，
  避免 BoW 图看似连通但实际匹配/SfM 有效图断裂。
- `MatchPhotosTask` 执行算法选择、影像对选择、特征提取、两两匹配、几何验证、轨迹构建和引导匹配报告；
  当前引导匹配 v1 通过关键点密度扩展提高初始匹配候选，尚未做姿态恢复后的二次补匹配。

`src/core/overlap` 保持为可复用的候选生成模块，由 `PairSelector` 调用；
它不会被改名，也不会被嵌入到本模块目录下。
