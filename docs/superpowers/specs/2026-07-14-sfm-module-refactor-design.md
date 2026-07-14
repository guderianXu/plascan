# SfM 模块收敛重构设计

## 1. 背景

PlaScan 的 `src/core/sfm` 已经具备完整的增量式 SfM、三角化、BA 编排、稀疏点云过滤和项目适配能力，
但长期迭代后出现了以下结构问题：

- `IncrementalSfm.cpp` 和 `BaInputBuilder.cpp` 承担过多职责；
- 存在两套稀疏点云过滤入口；
- 存在未使用类型、旧名称别名和兼容入口；
- 重投影、深度约定和三角化质量判断分散在多个实现中；
- `ObservationNetworkBuilder` 重复实现了 PlaPoint 已有的二维 KDTree；
- 纯算法代码与 Qt JSON、项目文件和结果序列化混在同一个构建目标中。

本次重构不改变 SfM 算法行为和项目格式，目标是消除冗余、明确依赖方向，并将算法类彻底移出 Qt 依赖。

## 2. 目标

1. 删除没有生产用途的类型和文件。
2. 删除旧类名、旧头文件、转发头和 `using` 兼容别名，并一次性迁移仓库内调用者。
3. 由 PlaPoint 统一承担统计离群、半径密度、体素和 KDTree 等通用点云能力。
4. SfM 只保留重投影误差、轨迹长度、三角交会角、深度和多视一致性等摄影测量规则。
5. 抽取统一的投影、OpenCV 相机转换和三角化质量组件。
6. 将 `BaInputBuilder` 和 `IncrementalSfm` 拆成职责单一、可独立测试的组件。
7. 将纯算法、后处理和项目适配拆成单向依赖的构建目标。
8. 保持现有 GUI、CLI、项目元数据和输出结果行为一致。

## 3. 非目标

- 不更换增量式 SfM 算法路线；
- 不修改连接点提取和匹配算法；
- 不改变 BA 后端选择、阈值或迭代次数；
- 不修改项目 JSON、PLY、质量报告等外部格式；
- 不在本次重构中引入新的第三方依赖；
- 不保留旧 API 兼容层。

## 4. 数值兼容要求

在相同输入、配置和运行环境下：

- 注册影像数必须一致；
- 三维点集合、轨迹归属和观测关系必须一致；
- 仅因无序容器或遍历调整造成的输出顺序变化可以接受；
- BA 输入、阈值、后端选择和迭代配置必须一致；
- 重投影误差只允许浮点舍入级差异；
- 两视预览云的候选点和过滤结果必须一致；
- GUI/CLI 设置键、项目结果记录和输出文件格式必须保持一致；
- 稳定 RANSAC seed 和随机调用顺序不得改变。

## 5. PlaPoint 与 SfM 的职责边界

PlaPoint 已提供：

- `statisticalOutlierRemoval`；
- `radiusOutlierRemoval`；
- `voxelDownsample`；
- CPU、GPU 和 Auto 处理设备选择；
- `SpatialKdTree<Dimension, Scalar>`；
- 通用 KDTree、法向估计和均匀降采样。

因此通用空间算法必须调用 PlaPoint。SfM 只负责：

- 重投影误差过滤；
- 轨迹长度过滤；
- 三角交会角过滤；
- 正深度检查；
- 多视观测和轨迹一致性；
- 将 PlaPoint 返回的保留/删除索引映射回摄影测量属性。

PlaPoint 当前没有通用并查集，因此 SfM 可以保留一个小型内部 `DisjointSet`，供轨迹构建和 MST 共用。

## 6. 最终目录结构

```text
sfm/
├── common/
│   ├── SfmTypes.h
│   └── DisjointSet.h
├── geometry/
│   ├── ProjectionGeometry.h/cpp
│   ├── TriangulationQuality.h/cpp
│   └── OpenCvCameraAdapter.h/cpp
├── graph/
│   ├── CorrespondenceGraph.h/cpp
│   └── ObservationNetworkBuilder.h/cpp
├── tracks/
│   ├── MultiViewTrackBuilder.h/cpp
│   └── CorrespondenceTrackThinner.h/cpp
├── pose/
│   └── PnpSolver.h/cpp
├── triangulation/
│   ├── Triangulator.h/cpp
│   └── InitialSparsePointFilter.h/cpp
├── reconstruction/
│   └── SfmReconstruction.h/cpp
├── pipeline/
│   ├── IncrementalSfm.h/cpp
│   ├── InitialPairInitializer.h/cpp
│   ├── ImageRegistrationEngine.h/cpp
│   ├── KnownPoseReconstructor.h/cpp
│   └── SfmBundleAdjustCoordinator.h/cpp
├── filtering/
│   ├── SparsePointCloudProcessor.h/cpp
│   └── SparsePointCloudWorkspace.h/cpp
├── quality/
│   ├── SfmQualityMetrics.h/cpp
│   └── SfmError.h
├── test/
│   ├── CMakeLists.txt
│   ├── test_correspondence_graph.cpp
│   ├── test_multiview_track_builder.cpp
│   ├── test_projection_geometry.cpp
│   ├── test_triangulation.cpp
│   ├── test_pnp_solver.cpp
│   ├── test_incremental_sfm.cpp
│   └── test_sparse_point_cloud_processor.cpp
└── project/
    ├── BaInputBuilder.h/cpp
    ├── ProjectMatchInputReader.h/cpp
    ├── BaTrackBuilder.h/cpp
    ├── SurveyControlBaAdapter.h/cpp
    ├── MarkerBaAdapter.h/cpp
    ├── SfmQualityJsonSerializer.h/cpp
    └── TriangulationService.h/cpp
```

文件按真实职责拆分，不做仅为了降低行数的机械拆分。

所有 SfM 专属单元测试和模块集成测试统一放在 `src/core/sfm/test`，由该目录的
`CMakeLists.txt` 注册。仓库根目录 `tests/` 只保留跨模块测试、GUI/CLI 测试和全仓源码契约测试。

## 7. 算法层禁止依赖 Qt

`sfm_core` 和 `sfm_postprocess` 中禁止出现：

- Qt 头文件；
- `QString`、`QJsonObject`、`QJsonArray`、`QMap`、`QHash`；
- `QFile`、`QDir`、`QObject` 和信号槽；
- 面向 UI 的本地化错误文案。

算法接口使用：

- `std::string`；
- `std::vector`、`std::array`；
- `std::map`、`std::unordered_map`；
- `std::filesystem::path`；
- 普通函数回调。

Qt 类型只能存在于 `sfm_project` 或 GUI/CLI 适配层，并在调用算法前转换为标准 C++ 类型。

## 8. 清退项

直接删除并迁移所有仓库内调用者：

- `common/SparsePointCloud.h`；
- `common/PhotogrammetryPointAttributes.h`；
- `filtering/SfmPointCloudFilter.h/.cpp`；
- `InitialSparsePointCloudTriangulator` 旧名称和别名；
- `SparseCloudLocalOptimOptions`；
- `SparseCloudLocalOptimResult`；
- `SparsePointCloudProcessor::localOptim()`。

初始预览点过滤统一命名为 `InitialSparsePointFilter`，空间清理统一使用
`SparsePointCloudSpatialCleanupOptions`、`SparsePointCloudSpatialCleanupResult` 和 `spatialCleanup()`。

## 9. 公共摄影测量几何

### 9.1 ProjectionGeometry

负责：

- 普通投影和 signed projection 回退；
- 单观测重投影误差；
- 正深度检查；
- 投影结果有限性检查。

### 9.2 TriangulationQuality

负责：

- 多相机最大和最小三角交会角；
- RMS 重投影误差；
- 观测数、深度、角度和重投影联合门控；
- 返回结构化拒绝原因。

### 9.3 OpenCvCameraAdapter

负责：

- `Camera` 到 OpenCV `K`、`R`、`t` 和畸变系数转换；
- `depthAxisFlipped`、U/V 轴方向处理；
- OpenCV 位姿结果转回 `Camera`。

迁移必须保持当前计算顺序、符号约定和阈值，避免改变浮点结果。

## 10. BaInputBuilder 拆分

```text
项目 JSON
   ├── ProjectMatchInputReader ── 相机、特征和匹配
   ├── SurveyControlBaAdapter ─── GCP、检查点和比例尺
   └── MarkerBaAdapter ────────── marker 观测
                    │
                    ▼
              BaTrackBuilder
                    │
                    ▼
             BaInputBuildResult
```

`BaInputBuilder` 最终只负责输入校验和组件编排。项目 JSON 解析只能存在于 `sfm_project`。
摄影测量几何由 `sfm_core/geometry` 提供，不得在适配器中重复实现。

## 11. IncrementalSfm 拆分

外部使用方式保持不变：

```cpp
IncrementalSfm sfm(options);
sfm.addImage(...);
sfm.addMatches(...);
const IncrementalSfmResult result = sfm.run(callback);
```

内部组件为：

- `InitialPairInitializer`：候选初始像对、E/F/H 估计、相对位姿恢复和模型评分；
- `ImageRegistrationEngine`：下一影像选择、PnP、序列位姿初值和失败重试；
- `KnownPoseReconstructor`：完整已知相机位姿重建；
- `SfmBundleAdjustCoordinator`：局部/全局 BA、控制网约束、BA 后过滤和重三角化。

组件共享同一个 `SfmReconstruction` 和 `CorrespondenceGraph`，不得复制完整重建状态。
`IncrementalSfm` 只负责阶段顺序、取消、进度和最终结果组装。

## 12. 错误处理

算法层使用结构化错误码：

```cpp
enum class SfmErrorCode
{
    None,
    InvalidInput,
    InsufficientMatches,
    InitialPairFailed,
    ImageRegistrationFailed,
    BundleAdjustmentFailed,
    Cancelled
};
```

算法结果可以带 `std::string diagnostic`，但不得生成中文 UI 文案。Qt 项目适配层或 GUI 根据错误码生成用户文案。
文件读取、项目 JSON、PLY 输出等失败只由 `sfm_project` 处理。

## 13. 构建目标

最终构建目标：

```text
sfm_core
  纯 SfM 算法、几何、轨迹和重建状态

sfm_postprocess
  稀疏点云过滤和纯 C++ 质量指标，依赖 PlaPoint 与 sfm_core

sfm_project
  项目 JSON、BA 输入和预览云输出，依赖 Qt、sfm_core、sfm_postprocess

sfm
  INTERFACE 聚合目标
```

依赖方向只能是：

```text
sfm_core <- sfm_postprocess <- sfm_project
```

`sfm` 聚合目标只提供构建迁移便利，不提供旧 C++ API、转发头或类型别名。

## 14. 实施顺序

1. 添加结构契约和数值基准测试；
2. 删除死代码、旧名称和兼容别名；
3. 合并过滤入口并统一 PlaPoint 通用算法；
4. 用 PlaPoint `SpatialKdTree<2, double>` 替换自建二维 KDTree；
5. 抽取公共摄影测量几何；
6. 拆分 `BaInputBuilder`；
7. 拆分 `IncrementalSfm`；
8. 拆分质量指标与 Qt JSON 序列化；
9. 把 SfM 专属测试迁移到 `src/core/sfm/test`，并删除根 `tests/` 中对应源文件和注册项；
10. 最后拆分 CMake target 和迁移调用者；
11. 更新架构文档并执行全量验证。

每个阶段必须保持可构建、可测试。不得把代码拆分和数值行为调整混在同一阶段。

## 15. 测试与验收

### 15.1 测试类型

- 死代码和旧 API 不存在的源码契约测试；
- 算法目录无 Qt include、Qt 类型和 Qt target 依赖的契约测试；
- PlaPoint 统计过滤、半径过滤和二维 KDTree 集成测试；
- 投影、深度符号和三角化质量单元测试；
- 初始像对、影像注册、已知位姿和 BA 编排单元测试；
- 拆分前后的固定数据数值基准测试；
- GUI/CLI 项目 JSON 和输出格式回归测试；
- 全量构建与 `ctest --output-on-failure`。

SfM 专属测试的判定标准是测试主体只依赖 `src/core/sfm`、Camera、Intersection、BundleAdjust、
ControlPoints 或 PlaPoint 等算法依赖。涉及 GUI、项目管理器、CLI 参数和跨模块架构约束的测试继续保留在根 `tests/`。

### 15.2 验收条件

- 所有旧文件、旧类型、别名和兼容入口均已删除；
- 算法目标不依赖 Qt；
- 通用空间算法不再在 SfM 中重复实现；
- `IncrementalSfm` 和 `BaInputBuilder` 只保留编排职责；
- 现有项目、GUI、CLI 和测试调用者均迁移到新接口；
- SfM 专属测试全部位于 `src/core/sfm/test`，根 `tests/` 不再重复注册这些测试；
- 数值兼容测试满足第 4 节要求；
- 全量构建成功，全部启用测试通过。
