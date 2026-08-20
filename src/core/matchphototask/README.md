# MatchPhotoTask

`matchphototask` 是类 Metashape “匹配照片”流程的编排层。
它拥有高层任务边界，但复用底层模块，不把其它模块搬进本目录。
它也是工作流中创建连接点的唯一所有者；`aerial_triangulation` 只能通过
`MatchPhotosTask` 请求特征、匹配和轨迹，不能直接实例化提取器或匹配器。

当前框架：

- `algorithm/` 负责类 Metashape 策略到注册算法计划的映射，当前支持
  `auto_sift`、`sift_lightglue` 和 `loma_r`。
- `auto_sift` 是默认算法，自动设备顺序为 CUDA、Metal、OpenCL、OpenCV CPU。显式选择的设备
  不可用时明确失败；不保留 `cuda_sift` 算法别名。
- `pair_selection/` 负责影像对类型、影像对选择策略和 `PairSelector`。
- `runtime/` 负责任务级 SIFT 内存缓存、ONNX 资源解析、本机 TensorRT engine 缓存、蒙版约束和 GUI 写回所需记录。
- `task/` 负责 `MatchPhotosTask`、选项、上下文和结果报告。
- `stages/` 负责流程阶段，按算法能力加载灰度或彩色输入并执行特征与两两匹配。
- `tie_points/` 负责最终多视图连接点 track 的构建、筛选和统计摘要。
- `tests/` 放置本模块自己的单元测试。

当前行为：

- `MatchPhotosAlgorithmSelector` 校验工作流选择的注册算法、设备要求与质量预设。
- SIFT 负责提供尺度和旋转鲁棒性，LightGlue 直接消费任务内存中的 SIFT 关键点与描述子。
- CUDA SIFT 检测阈值为 `0.0005`（快速模式为 `0.003`）；OpenCV CPU 使用独立的 contrast threshold
  `0.02`（快速模式 `0.04`），避免把两个实现的不同量纲混为一个参数。
  阈值写入前端签名，修改后旧缓存会自动失效。
- Auto SIFT 对最短边小于 800 像素的影像使用 2 倍首层和分级降低阈值；对超过质量档最长边的大图先做
  全图粗尺度，再在原分辨率重叠瓦片补充细节。候选经 8×8 空间配额、近邻去重和 RootSIFT 归一化后进入匹配。
- 特征阶段不再写入或读取独立特征文件。一次任务内每幅影像最多提取一次，描述子在最后一个相关像对完成后释放；
  多尺度和瓦片提取都会把关键点坐标还原到原始影像坐标。
- `maskApplyMode=keypoints` 时，特征阶段按项目蒙版过滤关键点和描述子，并强制重新提取特征；
  `maskApplyMode=tiepoints` 时，匹配阶段过滤任一端落入蒙版排除区的连接点。蒙版约定为 `0` 有效、非 `0` 排除。
- `maxKeypoints` 对应每张影像的关键点总量限制；调用方设置 `useExplicitKeypointLimit=true` 时，
  `0` 表示不限制。空中三角测量界面的“关键点限制”始终使用这一语义，不会因启用指导匹配而按像素数放大。
- `keypointLimitPerMegapixel` 是底层调用方可显式启用的独立限制。设置后，每张影像的关键点上限按
  `每百万像素关键点限制 * 影像百万像素数` 计算；它不由 `enableGuidedMatching` 自动开启。
- `enableGuidedMatching` 是显式用户开关，质量档位不能把未勾选状态自动改为开启；实际值写入
  `.pimatch` 配置指纹，后续 SfM 只消费同一结果变体。
- 设备为 `Auto` 时 SIFT 按 CUDA、Metal、OpenCL、CPU 选择；LightGlue 固定使用 TensorRT/CUDA；选择 CPU、CUDA
  不可用或 engine 不兼容时会返回明确错误，不做 TorchScript/CPU 回退。
- LightGlue ONNX 可通过 `lightGlueTensorRtEnginePath` 显式传入，也可由
  `PLASCAN_LIGHTGLUE_TENSORRT_ENGINE` 指定。运行时按 ONNX 内容、本机 TensorRT 完整版本和 GPU
  Compute Capability 构建环境指纹缓存；历史本机 engine 仅保留读取兼容。
- 准备 TensorRT engine 时会显示 ONNX 校验、缓存检查、解析、构建、I/O 校验和写入阶段；缓存未命中会
  给出格式升级或环境字段变化等具体原因。TensorRT 内核搜索没有可信百分比，该阶段使用忙碌进度并持续
  显示真实已耗时，同时每 30 秒写入一次控制台日志。
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
- 通用预选与最终匹配使用不同预算：词汇分配默认从每张影像均匀抽取最多 4096 条描述子，
  最终匹配仍消费任务缓存中的完整特征。SIFT 128 维和 LoMa-R 256 维描述子只在各自任务内训练词汇，
  不跨算法复用词汇中心；LoMa-R 的最高 3840 关键点档默认不会被预选预算截断。
- `HierarchicalVocabularyTree` 使用真正的层次 K-means：每个内部节点独立聚类出子节点，描述子从根节点
  逐层选择最近分支直到叶词。默认分支因子 10、深度 3，每条描述子约比较 30 个节点中心；不再构建
  1000 个扁平中心，也不使用 FLANN KD-tree 冒充词汇树。训练节点和影像批次之间均可响应取消。
- `MatchPhotosTask` 执行算法选择、影像对选择、特征提取、两两匹配、几何验证、引导重匹配和轨迹构建。
  引导阶段在初始基础矩阵的极线带内检查描述子 top-k 候选，执行双向一致性和宽松 ratio 门控，新增对应后重新
  运行 USAC，再把结果交给多视轨迹构建。

`src/core/overlap` 保持为可复用的候选生成模块，由 `PairSelector` 调用；
它不会被改名，也不会被嵌入到本模块目录下。
