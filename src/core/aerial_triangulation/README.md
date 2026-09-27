# core/aerial_triangulation 模块

本模块实现与 Metashape “对齐照片”职责对应的空中三角测量工作流。它包含标准数值入口 `aerial_triangulation_engine` 和工程工作流
`aerial_triangulation` CMake target，GUI 空三入口、独立 CLI 和三维重建 CLI 都只能通过
`AerialTriangulationWorkflow` 进入，不直接调用连接点阶段或 SfM 内部组件。

## 目录与职责

- `engine/`：标准路径、相机、观测、取消和进度契约；生产针孔/RPC runner 都调用同步数值入口。
  `TiePointGraphReader` 使用 nlohmann::json 读取 v1/v2/v3 连接点，工程路径 token 通过回调适配。
- `model/`：GUI、CLI 工程工作流共用的 Qt 输入、解析后配置和结果 DTO。
- `workflow/AerialTriangulationWorkflow.*`：唯一公开入口；解析用户参数，决定复用、补齐或重建连接点，
  然后调用正式重建管线。
- `workflow/AerialTriangulationPipeline.*`：只消费已经持久化的连接点图，执行 SfM、无先验焦距搜索和结果发布。
- `preparation/TiePointPreparation.*`：对 `MatchPhotosTask` 的薄封装。特征提取、影像对生成、匹配、几何验证、
  guided image matching 和多视轨迹整理仍由 `src/core/matchphototask` 独占。
- `preparation/MatchResultCatalog.*`：读取逐影像 `.pimatch` 分片，在文件内部编目像对及算法变体；
  不扫描 sidecar，也不从文件名推断影像对。
- `preparation/ReconstructionPrerequisiteReport.*`：生成特征、匹配和图连通性的结构化前置检查结果。
- `reconstruction/SfmAttemptRunner.*`：解析项目/外部 PlaCamera 相机和标记点先验，配置 BA，调用持有原生实例的针孔航三入口。
- `reconstruction/CameraIntrinsicPriorSanitizer.*`：无外部相机文件且重置对齐时，直接在 PlaCamera
  面阵实例上修正明显偏离主焦距群的旧 SfM 内参，保留实例身份、位姿与畸变。
- `reconstruction/MarkerPriorLoader.*`：从项目标记点 sidecar 装载控制点、检查点、比例尺和像点投影先验。
- `reconstruction/SfmPairPlanner.h`：保留工程路径和 Qt 字符串适配的候选对规划。
- `reconstruction/SfmMatchDiagnostics.h`：使用 C++ 标准容器和字符串的纯匹配图诊断、引导匹配候选规划。
- `search/AdaptiveFocalSearch.*`、`SfmSearchPolicy.*`：无完整相机先验时的焦距候选排序和资源预算；候选列表使用 `std::vector`，GUI/CLI 只在边界转换。
- `reporting/`：写出稀疏点云、质量元数据和工作流报告；原生 PlaCamera 相机实例随结果返回，
  由 GUI/CLI 在正式写出成功后按 canonical ImageId 原子提交。
  `SparsePlyWriter` 使用标准流编码 little-endian PLY，文件提交由 `common/file` 独占。

## 数据流

1. `resolveConfig()` 把用户质量、关键点/连接点限制、蒙版、预选、设备和相机设置分别解析为
   `MatchPhotosOptions` 与 `PreparedAerialTriangulationInput`。
2. 缺少 `assets/tie_points/latest_tie_points.json` 时自动调用创建连接点。“重置当前对齐”只让 SfM
   忽略旧相机外方位并重新解算；默认勾选“重用现有匹配”，直接复用兼容的特征、匹配与连接点缓存。
   只有取消“重用现有匹配”时，才删除旧 `.pimatch` 分片和连接点文件并重新提取、匹配与整理轨迹。
   PlaMatch-HCT 的完整特征以 `.pihctcache` 原子持久化；参数与影像身份一致的后续运行可在解码影像前直接命中，
   旧项目只有 `.pifeature` 时会在首次运行补建一次完整缓存。
3. `AerialTriangulationPipeline` 只读取本次解析出的连接点路径，不扫描另一套特征或匹配缓存。
   同一次焦距搜索只解析一次连接点 JSON，所有候选共享只读图，避免重复内存和文件 I/O。
4. `SfmAttemptRunner` 装载连接点、多视观测和标记点先验，调用 `IncrementalSfm` 与统一 BA 后端。
5. `AerialTriangulationResultWriter` 把全部可用算法点发布为 `sfm_sparse.ply`，将用于交互显示的弱点/空间离群
   清理结果另存为 `sfm_sparse_display.ply`；`sfm_sparse_points.json` 使用严格的
   `plascan.sfm_sparse_points.v3` 格式保留全部算法点质量。顶层影像表为一次性的局部
   `camera_index` 与 canonical `image_id` 对照表，逐点观测必须是对象并同时携带这两个字段；路径只用于
   栅格定位，不能覆盖或推断影像身份。v2、固定数值行和旧对象兼容分支均不再读取。
   后续重算/MVS 使用主稀疏云，显示清理不删除或覆盖算法点。GUI 只在任务完成且未取消后写回项目。

## GPU 与 CPU 执行边界

- Auto SIFT 使用单个 CUDA、Metal 或 OpenCL GPU 提取通道；CPU 以有界队列提前完成下一张影像的读取、灰度化和缩放，
  使影像解码与 GPU 提取重叠。队列深度默认自动解析，取消后不会继续准备新影像。
- LightGlue TensorRT 按可用显存、固定 engine 桶容量和候选对数量自动解析并发数。每个 worker 独占
  `TensorRtLightGlueMatcher` 执行上下文，当前最多 4 路；检测到 CUDA OOM 时释放并发 worker，
  对未完成影像对进行串行重试，不把整个 GUI 进程带崩。
- 基础矩阵几何验证直接消费任务内存中的像点；多个影像对使用最多 8 路 CPU 并发执行确定性
  USAC-MAGSAC。PlaMatch-HCT 完整描述子由连接点模块持久化并复用，空三模块仍只读取最终连接点图。
- PlaMatch-HCT 的 CPU HCT 索引按实际 CPU 匹配请求延迟构建；CUDA/OpenCL 匹配不会再为每幅影像额外构建
  不会使用的 CPU 索引。
- PnP、增量三角化和连接点轨迹图继续使用 CPU。当前典型 16 影像规模下，这些阶段的总耗时低于
  CUDA 启动和数据搬运成本；只有基准证明端到端收益达到 20% 后才允许增加默认 GPU 路径。
- 已有兼容缓存命中时直接复用。不会为了提高任务管理器中的 GPU 利用率而重新提取或重新匹配。

## 连接点参数语义

- `keypointLimit` 是每幅影像的关键点上限；`0` 表示不限制。
- 指导图像匹配只启用几何引导的补充匹配，不改变 `keypointLimit` 的每幅影像上限语义；
  每百万像素关键点限制是 `MatchPhotosOptions` 的独立参数，空三界面当前不混用这两个预算。
- `tiepointLimit` 是每幅影像参与正式多视轨迹的连接点上限，GUI、CLI 和核心默认统一为参考值 `4000`；
  筛选优先保留长轨迹和高置信度轨迹。连接点文件头会保存实际配额，当前设置与缓存配额不同时只复用兼容的
  特征/逐对匹配并重新执行多视轨迹选择，不会把旧配额点网静默交给 SfM。
  前端解析出的每图、每网格上限会原样传给 SfM 输入保护，不能再被 SfM 的通用默认值二次截断。
- 照片序列的 `highest` 预设使用前后各 16 帧候选窗口，以增加三视及以上共视轨迹；其它质量档仍使用较小窗口控制耗时。
- `maskApplyMode=keypoints` 在 SIFT 提取后立即过滤蒙版外关键点；`tiepoints` 在匹配后过滤任一端位于排除区的观测。
- 通用预选、参考预选和照片序列只改变 `MatchPhotosTask` 的候选对策略，不在空三模块内另建匹配链路。
- 当前 guided image matching 是连接点阶段的显式选项，不包含 SfM 恢复位姿后的第二轮私有重匹配。

## 相机、焦距与先验

`SfmAttemptRunner` 在进入针孔航三入口前会从工程 `camera_instances` 解码 canonical 相机集合。`PinholeInput`
直接持有 PlaCamera 面阵实例；旧数值状态只在尚未迁移的 `IncrementalSfm` 求解器接收相机时构造。只要所选影像均有
规范实例，就按其 `image_uuid`、实例 ID 和 world frame 自动传播 `SolverCameraBinding`；调用方显式传入的绑定必须与
canonical 集合一致。工程外部相机或集合不完整时不会从路径、文件名或序号生成身份，若使用外部姿态先验则直接失败并
要求调用方提供完整绑定。Pipeline 只有在 PlaCamera 面阵实例齐全时才依据工程来源标记判断可信内参或已知位姿；
旧影像 JSON 即使保留 `intrinsic_source=imported` 或针孔数值字段，也不能绕过实例校验、焦距搜索或位姿重建。

- 完整 `.tsai` 相机列表由 PlaCamera 解析并绑定 canonical 影像/实例身份，再走已知相机路径；规范工程
  PlaCamera 实例可作为初值。重置对齐时只复用内参而不复用旧外参；
  若无外部相机文件且至少 70% 相机形成稳定焦距群，会在 PlaCamera 实例上清洗偏离该群超过 2 倍的历史焦距，
  最后进入当前 SfM 求解器时才生成旧数值状态。
- 无可信内参时，每幅影像先按显式 canonical `cameraBindings` 构建 PlaCamera 初始面阵实例，再在求解器输入边界
  生成旧数值状态；缺少实例身份的输入会在读取连接点之前失败，不从路径或 SfM 序号合成身份。
- 参考预选和 guided matching 使用 `ReferenceCameraGeometryMap`、`ReferenceCameraPositionMap` 两个
  `ImageId` 键控集合。`images`/路径只负责输入顺序和定位；外部 `.tsai` 只有在与 canonical 面阵针孔
  PlaCamera 实例合并身份、影像尺寸和 world frame 后才可成为投影参考；该路径不再转回旧针孔数值状态。
  RPC 空三先通过统一的
  PlaCamera `CameraInstanceSet::requireCapabilities()` 校验 canonical RPC 实例能力，再解码专用 RPC 数值相机；RPC/推扫实例不会被伪造为针孔，
  位置-only 参考也不会被送入极线投影器。
- 无完整相机先验时，焦距尺度表示 `焦距像素 / 影像最长边像素`。Pipeline 始终评估
  `0.55、0.70、0.85、0.95、1.0、1.05、1.2、1.6、2.0、2.4、2.8、3.2、4.0、5.2、6.4、8.0、9.0、10.0`
  的广域候选，即使低焦距基准已注册全部影像，也不会跳过窄视场检查。粗搜索按本机逻辑线程数自动
  创建候选 worker，每个候选先占一个线程，余数再分配给候选内部 BA；显式线程请求也不会超过硬件
  线程数。GUI、CLI 和 Python 工作流都在任务启动时按当前机器解析自动预算，不复用其它电脑保存的
  固定线程数。默认焦距和其它候选进入同一并行队列，界面进度显示运行数、CPU 线程预算和整体完成比例。
- 粗搜索候选只同步记录警告和错误；逐相机 INFO 日志由胜出候选的正式重建输出，避免并发 worker
  在每一行日志后争抢全局文件锁并强制刷新磁盘。
- 候选先比较注册覆盖，再联合评价多视网络、交会角、观测网格覆盖、重投影误差和闭环序列连续性。
  序列项采用有界质量分数，不允许一个异常相邻基线比覆盖其它全部几何证据。
- “自适应相机模型拟合”只控制正式 BA 是否释放共享焦距；关闭它不会跳过无相机先验所必需的初始焦距搜索，
  最佳初始焦距会作为固定内参参与正式重建。
- 手工标记点投影与比例尺通过 `MarkerPriorLoader` 注入同一个 `IncrementalSfm`，不会形成旁路 BA。
  `MarkerPriorLoader` 只接受 canonical `image_uuid` 到当前 solver `ImageId` 的显式映射；路径快照
  不能回退绑定，缺少或未知身份会在进入 SfM 前失败。
- BA 后端使用 `Auto`：小型固定焦距问题保留 CPU/OpenMP，共享焦距使用 PlaMatrix 联合优化，大问题达到相机和观测阈值后
  可选择 PlaMatrix CUDA/OpenCL。
  每次局部/全局 BA 都记录实际后端、状态、问题规模、RMS、耗时和回退原因。
  最终诊断同时记录真正生效的内参 mask、标定组/自标定阶段数，以及相对稳定标定组参考的最终值和变化量。
- 普通空三不根据航摄形状自动添加相机共面或“穹顶修正”约束；可信焦距只作为共享标定先验，仍按参考阶段释放
  内参并使用标准 similarity gauge。底层 `cameraPlaneConstraint` 只保留给调用方明确请求的专业约束流程。

## 边界

- 本模块不直接包含任何特征提取器或匹配器实现。
- 本模块不依赖 GUI 工程、项目窗口管理器或界面控件。
- 项目路径与标记点 sidecar 由 `src/common/project` 提供；项目元数据写回由 GUI 负责。
- 独立 CLI 的 `--output-dir` 只控制稀疏云和报告产物，`--assets-dir` 只控制 `.pimatch` 与连接点缓存；
  指定外部缓存目录时不会把重建产物写回该缓存所属项目。
- MVS、深度图、网格、DEM 和 DOM 不属于本模块。

## 图像 I/O 与 Qt 边界

空三生产库不链接 QtGui。输入尺寸、批次 EXIF 焦距先验和质量报告的尺寸读取统一调用
`common/io/ImageIO.h` 的 `readImageSize()`：读取 GDAL 影像头，BMP 使用固定长度头部回退，
不解码整幅图像、不应用 EXIF 旋转。支持范围以已启用的 GDAL 栅格驱动及 BMP 为准；
失败返回无效 QSize，可选错误参数包含路径和原因。质量报告沿用关键点范围回退。

针孔与 RPC 点云颜色通过公共 ImageIO 读取 8 位 BGR 图像并转成 RGB，显式忽略 EXIF 旋转，
保持匹配像素坐标、qRound 取整、边界裁剪及首个有效观测采样规则。
针孔路径按影像分组读取，RPC 缓存成功及失败读取；不可读影像记录警告并保留默认中性灰几何。
16 位 TIFF 沿用公共 GDAL 的 1/256 转换为 8 位约定；颜色不参与几何解算。
QtCore 的路径、工程 DTO、JSON 和尺寸值类型仍保留；匹配图诊断与自适应焦距候选排序已经不再暴露 Qt 容器或 QString。

本次 RPC 回归同时修复 imageCoordinate() 对 QMap::value() 临时关键点容器的悬空引用，
改为从 constFind() 获取原容器；实际 RPC 点云颜色按 sidecar 观测索引逐点校验。

## 标准库边界

连接点图采用 `std::filesystem::path`、`std::map` 和 `std::vector`，数值求解输入不包含项目 JSON。
工作流计时使用 `std::chrono::steady_clock`；针孔/RPC 的 PLY 和 sidecar 原子发布共用
`plascan_common_file`，连接点 JSON 使用 header-only 的 nlohmann 库。输出 schema 与算法参数保持兼容。

目前工程门面、相机/标记数据适配、候选规划和质量报告仍使用 Qt Core，包括 sidecar 的 QJson 构建。
数值入口自身无 Qt 类型，但 camera/common IO 和 SfM/common log 仍有传递链接依赖，不能视为完全 Qt-free target。
