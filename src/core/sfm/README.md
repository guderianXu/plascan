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

## 已知相机与项目元数据

- GUI 项目元数据中的相机参数通常来自 EXIF/GPS 或前置估计，只作为增量 SfM 的相机初值和内参输入，后续 PnP/BA 允许调整位姿。
- 只有调用方显式提供完整 `.tsai` 相机文件列表时，才进入固定已知外参的直接三角化路径。
- 固定已知外参路径如果输入存在多视 track 但输出退化为全两视稀疏点云，应视为失败，不能发布为正式空三结果。

## 匹配配对规划与诊断

`src/core/pipeline/SfmPairPlanner.h` 负责大规模项目的 SfM 匹配候选规划。大项目默认不做无约束 N^2 全匹配，而是按以下来源生成候选并合并去重：

- `known_camera_overlap`：已知相机足迹重叠候选，优先级最高。
- `known_camera_spatial_neighbors`：已知相机中心邻域候选，用于跨航带/空间近邻补充。
- `sequence_window`：文件序列窗口候选，用于航线内连续影像。
- `manual_restricted`：调用方显式传入的配对列表。

每个候选 pair 会记录 `sourceTypes`、`priorityScore`、序列距离、相机中心距离和各来源得分。`SFMService` 按规划后的 `allowedPairKeys` 顺序检查和补生成匹配，因此高优先级 pair 会优先进入缓存检查、自动补匹配和后续诊断。

SfM 匹配阶段会在项目 `assets/reports/` 下输出：

- `matching_quality_report.json`：候选图、实际匹配图、来源统计、候选样本和 BA 前诊断摘要。
- `matching_quality_report.csv`：完整 pair 明细，包括状态、匹配数、几何内点数、来源、优先级、失败原因。

GUI 的工作流报告会读取 `sfm_diagnostics.pair_plan`，展示规划候选数和来源分布，方便判断当前数据是足迹重叠、空间邻域还是顺序窗口在主导匹配。
