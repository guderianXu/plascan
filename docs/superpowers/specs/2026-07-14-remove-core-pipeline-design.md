# 移除 core/pipeline 设计说明

## 目标

删除 `src/core/pipeline` 这一名不副实的历史目录，同时保留仍在使用的 LightGlue 显存预算能力，
并将完整 DEM 流程从旧 `FeatureMatchRunner` 迁移到当前统一的 `MatchPhotosTask`。

## 已确认方案

采用“能力归位并迁移调用方”的方案，不保留兼容头、类型别名或转发层：

1. 将 `LightGlueFeatureBudget.h` 移至 `src/core/feature_match/lightglue/`。
2. 将其命名空间从 `xjw::pipeline` 改为 `xjw::feature_match`，直接更新所有生产代码和测试调用点。
3. 完整 DEM 流程直接调用 `MatchPhotosTask`，固定走项目主线 `SIFT + LightGlue`。
4. 删除 `FeatureMatchRunner.h/.cpp`，并删除全部 `src/core/pipeline` include 路径和源码注册。
5. 同步更新测试以及 `docs/PROJECT_ARCHITECTURE.md`。

## 架构边界

### LightGlue 显存预算

显存预算、关键点空间采样、OOM 重试预算和索引恢复属于 LightGlue 匹配实现的共享策略，
因此归入 `feature_match/lightglue`。调用方包括：

- `matchphototask/stages/MatchingStage.cpp`
- `aerial_triangulation/AerialTriangulationService.cpp`
- 对应单元测试

该头文件继续保持无状态、header-only，不引入 GUI 或项目管理依赖。

### 完整 DEM 匹配流程

`ProjectTerrainProductsManager` 不再分别编排 `FeatureExtractionRunner`、`FeaturePairPlanner` 和
`FeatureMatchRunner`。它构造 `MatchPhotosOptions` 与 `MatchPhotosContext`，由 `MatchPhotosTask`
依次完成特征提取、影像对选择、LightGlue 匹配、几何验证和轨迹构建。

上下文使用当前项目的：

- 项目路径与 assets 工作目录
- `ipfind` 和 `ipmatch` 输出目录
- 全部输入影像
- 可用的参考相机
- 影像蒙版
- 取消标志与进度回调

任务成功后，将返回的特征记录和匹配记录批量写回 `ProjectManager`；写回前检查项目仍然打开且
未发生切换。任务失败或取消时发出 `demPipelineFinished(false, error)`，不继续启动 MVS/DEM。

## 行为变化

- 完整 DEM 的连接点主线统一为 `SIFT + LightGlue`。
- 删除 `CreateDemDialog` 中历史遗留的 `disk_lightglue` matcher 设置，以及 DEM context 中不再生效的
  `featureAlgorithm`、`matchAlgorithm` 字段。
- 普通“匹配照片”、空中三角测量和 CLI 的现有 `MatchPhotosTask` 行为不变。
- 不保留旧 `FeatureMatchRunner` API 的兼容入口；仓库内调用方全部同步修改。

## 错误处理

- 少于两张影像时继续由 DEM 入口拒绝执行。
- `MatchPhotosTask` 的模型缺失、CUDA/CPU 运行环境、特征读写、匹配和几何验证错误原样汇总到
  DEM 失败消息。
- 后台任务完成后若项目已经关闭或切换，放弃项目元数据写回，避免污染其他项目。
- 只有连接点任务成功并完成结果写回调度后，才进入既有稀疏云检查、MVS 和 DEM 信号链。

## 测试策略

先修改测试形成预期失败，再修改生产代码：

1. `test_lightglue_feature_budget` 改为从新目录包含头文件并使用 `xjw::feature_match`，验证迁移前编译失败。
2. 源码契约测试要求 DEM 流程包含 `MatchPhotosTask`、不再包含 `FeatureMatchRunner`，并要求 GUI/CMake
   不再注册 `src/core/pipeline`。
3. 更新旧 `FeatureMatchRunner.cpp` 源码契约，使其改为验证 `MatchingStage` 或新的 DEM 调用边界。
4. 构建并运行预算单元测试、matchphototask 测试、GUI 项目工具测试和源码契约测试。
5. 完成后重新配置并编译项目；若环境允许，再运行相关 `ctest` 过滤集。

## 完成条件

- `src/core/pipeline` 目录不存在。
- `rg "FeatureMatchRunner|src/core/pipeline" src tests docs` 不再发现有效代码或文档引用；
  `LightGlueFeatureBudget` 的调用点不再使用 `xjw::pipeline`。其他目录中仍属于
  `MatchResultCatalog` 的 `xjw::pipeline` 命名空间不在本次范围内。
- DEM 完整流程只通过 `MatchPhotosTask` 生成连接点。
- 所有相关构建目标和测试通过，或明确记录与本次改动无关的环境/历史失败。
