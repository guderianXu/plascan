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
