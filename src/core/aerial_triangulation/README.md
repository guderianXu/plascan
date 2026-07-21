# core/aerial_triangulation 模块

本模块实现与 Metashape “对齐照片”职责对应的空中三角测量工作流。它是独立的
`aerial_triangulation` CMake target，GUI、独立 CLI 和三维重建 CLI 都只能通过
`AerialTriangulationWorkflow` 进入，不直接调用连接点阶段或 SfM 内部组件。

## 目录与职责

- `model/`：GUI、CLI 与核心共享的输入、解析后配置和结果 DTO。
- `workflow/AerialTriangulationWorkflow.*`：唯一公开入口；解析用户参数，决定复用、补齐或重建连接点，
  然后调用正式重建管线。
- `workflow/AerialTriangulationPipeline.*`：只消费已经持久化的连接点图，执行 SfM、无先验焦距搜索和结果发布。
- `preparation/TiePointPreparation.*`：对 `MatchPhotosTask` 的薄封装。特征提取、影像对生成、匹配、几何验证、
  guided image matching 和多视轨迹整理仍由 `src/core/matchphototask` 独占。
- `preparation/MatchResultCatalog.*`：编目 `.match` 及 sidecar，选择与当前影像和前端签名兼容的结果。
- `preparation/ReconstructionPrerequisiteReport.*`：生成特征、匹配和图连通性的结构化前置检查结果。
- `reconstruction/SfmAttemptRunner.*`：把连接点 JSON 转换为 `IncrementalSfm` 输入，配置 BA 并执行一次 SfM 尝试。
- `reconstruction/MarkerPriorLoader.*`：从项目标记点 sidecar 装载控制点、检查点、比例尺和像点投影先验。
- `reconstruction/SfmPairPlanner.h`、`SfmMatchDiagnostics.h`：候选对规划和匹配图诊断类型。
- `search/AdaptiveFocalSearch.*`、`SfmSearchPolicy.*`：无完整相机先验时的焦距候选排序和资源预算。
- `reporting/`：写出稀疏点云、相机更新、质量元数据和工作流报告。

## 数据流

1. `resolveConfig()` 把用户质量、关键点/连接点限制、蒙版、预选、设备和相机设置分别解析为
   `MatchPhotosOptions` 与 `PreparedAerialTriangulationInput`。
2. 缺少 `assets/tie_points/latest_tie_points.json` 时自动调用创建连接点；勾选“重置当前对齐”时，
   先删除旧匹配负缓存、匹配文件和连接点文件，再重新提取与匹配。
3. `AerialTriangulationPipeline` 只读取本次解析出的连接点路径，不扫描另一套特征或匹配缓存。
4. `SfmAttemptRunner` 装载连接点、多视观测和标记点先验，调用 `IncrementalSfm` 与统一 BA 后端。
5. `AerialTriangulationResultWriter` 发布稀疏点云、待写回相机、质量指标和诊断。GUI 在任务完成且未取消后，
   才把这些 DTO 写回项目元数据。

## 连接点参数语义

- `keypointLimit` 是每幅影像的关键点上限；`0` 表示不限制。
- 指导图像匹配只启用几何引导的补充匹配，不改变 `keypointLimit` 的每幅影像上限语义；
  每百万像素关键点限制是 `MatchPhotosOptions` 的独立参数，空三界面当前不混用这两个预算。
- `tiepointLimit` 是每幅影像参与正式多视轨迹的连接点上限；筛选优先保留长轨迹和高置信度轨迹。
- `maskApplyMode=keypoints` 会禁用旧特征复用并重新提取；`tiepoints` 在匹配后过滤任一端位于排除区的观测。
- 通用预选、参考预选和照片序列只改变 `MatchPhotosTask` 的候选对策略，不在空三模块内另建匹配链路。
- 当前 guided image matching 是连接点阶段的显式选项，不包含 SfM 恢复位姿后的第二轮私有重匹配。

## 相机、焦距与先验

- 完整 `.tsai` 相机列表走已知相机路径；项目元数据相机可作为初值，重置对齐时不会复用。
- 无完整相机先验时，焦距尺度表示 `焦距像素 / 影像最长边像素`。Pipeline 始终评估
  `0.55、0.70、0.85、1.2、1.6、2.0、2.4、2.8、3.2、4.0、5.2、6.4` 的广域候选，即使低焦距基准已注册全部影像，
  也不会跳过窄视场检查。候选按注册覆盖、多视网络强度、点数和重投影质量排序，最佳非默认尺度会以正式配置重放。
- “自适应相机模型拟合”只控制正式 BA 是否释放共享焦距；关闭它不会跳过无相机先验所必需的初始焦距搜索，
  最佳初始焦距会作为固定内参参与正式重建。
- 手工标记点投影与比例尺通过 `MarkerPriorLoader` 注入同一个 `IncrementalSfm`，不会形成旁路 BA。
- BA 后端使用 `Auto`：小问题保留 CPU/OpenMP，大问题达到相机和观测阈值后可选择 Ceres CUDA 或 native CUDA，
  并保留质量门控和回退原因。

## 边界

- 本模块不直接包含任何特征提取器或匹配器实现。
- 本模块不依赖 GUI 工程、项目窗口管理器或界面控件。
- 项目路径与标记点 sidecar 由 `src/common/project` 提供；项目元数据写回由 GUI 负责。
- 独立 CLI 的 `--output-dir` 只控制稀疏云和报告产物，`--assets-dir` 只控制特征、匹配与连接点缓存；
  指定外部缓存目录时不会把重建产物写回该缓存所属项目。
- MVS、深度图、网格、DEM 和 DOM 不属于本模块。
