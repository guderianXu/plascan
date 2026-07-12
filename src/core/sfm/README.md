# core/sfm 模块结构

当前 `sfm` 目录按职责逐步拆分为以下子模块：

- `common/`
  - `SfmTypes.h`
- `graph/`
  - `CorrespondenceGraph.*`
  - `ObservationNetworkBuilder.*`
- `reconstruction/`
  - `SfmReconstruction.*`
- `pipeline/`
  - `IncrementalSfm.*`
- `triangulation/`
  - `Triangulator.*`
  - `InitialSparsePointCloudTriangulator.*`
- `pose/`
  - `PnpSolver.*`
- `filtering/`
  - `SfmPointCloudFilter.*`

当前已经完成从根目录兼容头到真实模块头的切换。

`sfm` 子模块内部和外部调用方现在统一直接引用新路径，例如：

- `common/SfmTypes.h`
- `graph/CorrespondenceGraph.h`
- `reconstruction/SfmReconstruction.h`
- `pose/PnpSolver.h`
- `triangulation/Triangulator.h`
- `triangulation/InitialSparsePointCloudTriangulator.h`
- `filtering/SfmPointCloudFilter.h`

根目录兼容头已清退，后续新增代码应直接使用这些模块化路径。

## 无相机粗筛与正式精化

无相机文件且没有用户内参时，空三服务使用两级搜索：

- 粗筛阶段并发评估六个焦距尺度，每个 worker 持有独立 `IncrementalSfm`，最多使用六个初始像对。
- `SfmExecutionProfile::CoarseEvaluation` 将 BA 外层迭代限制为 5、全局精化限制为 1 轮，并把局部 BA 间隔放宽到 6 张影像。
- 候选排序优先比较注册影像数，其次比较有限且更低的重投影 RMS，最后比较三维点数。
- 正式阶段只重放最佳焦距，但不锁死粗筛初始像对；Guided matching 改变匹配图后必须重新自动选种子。
- 粗筛只读已经准备好的特征和匹配缓存，不写稀疏点云、项目记录或匹配质量报告。
- 初始对 E/F/H 估计和增量 PnP 使用由影像 ID 派生的稳定 RANSAC 种子；并行粗筛不会再改变正式 SfM 的随机状态。
- 照片序列插值只用于生成 PnP 初值。没有通过真实 3D-2D PnP 与序列几何门控的影像保持未注册，不能作为正式相机写回项目。

小型 BA 保持使用 Legacy CPU/OpenMP。只有相机数和观测数达到现有 Auto 门槛时才选择 Ceres CUDA 或 native CUDA；
正式 BA 日志会记录相机、轨迹、观测、线程、实际后端和选择原因。

## 已知相机与项目元数据

- GUI 项目元数据中的相机参数通常来自 EXIF/GPS 或前置估计，只作为增量 SfM 的相机初值和内参输入，后续 PnP/BA 允许调整位姿。
- 只有调用方显式提供完整 `.tsai` 相机文件列表时，才进入固定已知外参的直接三角化路径。
- 固定已知外参路径如果输入存在多视 track 但输出退化为全两视稀疏点云，应视为失败，不能发布为正式空三结果。

## 匹配配对规划与诊断

空三由 `AerialTriangulationWorkflow` 调用时，连接点阶段和 SfM 阶段必须共享显式的
`assetsDir`、`featureDir`、`matchDir`。`AerialTriangulationService` 只有在旧调用方未传目录时
才回退到 `.plascan` 项目旁的 `assets/ip` 和 `assets/matches`。特征 sidecar 中记录的路径必须与
SfM 实际加载的特征文件一致，否则索引不能进入观测网络。

匹配缓存中 `feature_format_version >= 2`、两侧索引数组为空且 `num_matches == 0` 的 sidecar
表示该影像对已确认无匹配。它会作为负缓存计入已处理配对，不进入待生成队列；缺少明确
`num_matches` 的旧空 sidecar 仍按无效缓存拒绝。

`src/core/aerial_triangulation/SfmPairPlanner.h` 负责大规模项目的 SfM 匹配候选规划。大项目默认不做无约束 N^2 全匹配，而是按以下来源生成候选并合并去重：

- `known_camera_overlap`：已知相机足迹重叠候选，优先级最高。
- `known_camera_spatial_neighbors`：已知相机中心邻域候选，用于跨航带/空间近邻补充。
- `sequence_window`：文件序列窗口候选，用于航线内连续影像。
- `manual_restricted`：调用方显式传入的配对列表。

每个候选 pair 会记录 `sourceTypes`、`priorityScore`、序列距离、相机中心距离和各来源得分。`AerialTriangulationService` 按规划后的 `allowedPairKeys` 顺序检查和补生成匹配，因此高优先级 pair 会优先进入缓存检查、自动补匹配和后续诊断。

SfM 匹配阶段会在项目 `assets/reports/` 下输出：

- `matching_quality_report.json`：候选图、实际匹配图、来源统计、候选样本和 BA 前诊断摘要。
- `matching_quality_report.csv`：完整 pair 明细，包括状态、匹配数、几何内点数、来源、优先级、失败原因。

GUI 的工作流报告会读取 `sfm_diagnostics.pair_plan`，展示规划候选数和来源分布，方便判断当前数据是足迹重叠、空间邻域还是顺序窗口在主导匹配。
