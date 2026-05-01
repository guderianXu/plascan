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
