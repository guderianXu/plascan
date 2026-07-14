# Remove Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 `src/core/pipeline`，把 LightGlue 预算策略归入 `feature_match/lightglue`，并让完整 DEM 流程统一调用 `MatchPhotosTask`。

**Architecture:** LightGlue 预算保持 header-only，但进入 `xjw::feature_match` 命名空间并由真实匹配模块公开。GUI 的 DEM 管理器只负责构造 `MatchPhotosContext`、运行任务、把结果安全写回项目，以及在成功后衔接既有 MVS/DEM 流程。

**Tech Stack:** C++17、Qt6、LibTorch、OpenCV、CMake、GoogleTest。

## Global Constraints

- 不增加兼容头、类型别名或转发层。
- 完整 DEM 的连接点算法固定为 `SIFT + LightGlue`。
- 不回滚工作区内已有用户改动，只修改本任务相关位置。
- 未经用户明确要求不创建 Git commit。
- C++ 保持 4 空格、Allman 花括号和 `_lowerCamelCase` 私有成员命名。

---

### Task 1: 将 LightGlue 显存预算归入 feature_match

**Files:**
- Move: `src/core/pipeline/LightGlueFeatureBudget.h` → `src/core/feature_match/lightglue/LightGlueFeatureBudget.h`
- Modify: `src/core/matchphototask/stages/MatchingStage.cpp`
- Modify: `src/core/aerial_triangulation/AerialTriangulationService.cpp`
- Modify: `src/core/matchphototask/CMakeLists.txt`
- Modify: `tests/test_lightglue_feature_budget.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces: `xjw::feature_match::LightGlueGpuMemoryInfo`、`BudgetedFeatureData`、`resolveLightGlueKeypointBudget(...)`、`resolveLightGlueMatchThreshold(...)`、`lightGlueRetryKeypointBudgets(...)`、`budgetFeatureDataForLightGlue(...)`、`remapLightGlueMatchResultToOriginal(...)`。
- Consumes: 既有 `FeatureData` 与 `MatchResult`，函数签名和算法行为不变，仅路径和命名空间改变。

- [ ] **Step 1: 先把预算测试切到目标接口**

```cpp
#include "lightglue/LightGlueFeatureBudget.h"

EXPECT_EQ(xjw::feature_match::resolveLightGlueKeypointBudget(
              QStringLiteral("sift"), QStringLiteral("lightglue"), true, 0),
          2048);
```

将该测试文件中所有 `xjw::pipeline` 替换为 `xjw::feature_match`，并从测试 target 删除
`src/core/pipeline` include 目录。

- [ ] **Step 2: 构建测试并确认 RED**

Run: `cmake --build build --target test_lightglue_feature_budget -j 4`

Expected: 编译失败，明确报告找不到 `lightglue/LightGlueFeatureBudget.h` 或目标命名空间尚不存在。

- [ ] **Step 3: 移动头文件并更新生产调用方**

```cpp
namespace xjw::feature_match
{
// 原有预算类型和函数保持不变。
} // namespace xjw::feature_match
```

`MatchingStage.cpp` 和 `AerialTriangulationService.cpp` 使用
`#include "lightglue/LightGlueFeatureBudget.h"`，并把预算 API 调用改为 `xjw::feature_match::*`。
从 `matchphototask/CMakeLists.txt` 删除 `../pipeline` include 目录。

- [ ] **Step 4: 构建并运行 GREEN**

Run: `cmake --build build --target test_lightglue_feature_budget -j 4`

Run: `ctest --test-dir build --output-on-failure -R '^test_lightglue_feature_budget$'`

Expected: target 构建成功，预算测试全部通过。

- [ ] **Step 5: 检查本任务差异**

Run: `git diff -- src/core/feature_match/lightglue/LightGlueFeatureBudget.h src/core/matchphototask/stages/MatchingStage.cpp src/core/aerial_triangulation/AerialTriangulationService.cpp src/core/matchphototask/CMakeLists.txt tests/test_lightglue_feature_budget.cpp tests/CMakeLists.txt`

Expected: 只有文件归位、命名空间和 include 配置变化，没有算法行为变化。

---

### Task 2: 让完整 DEM 流程调用 MatchPhotosTask

**Files:**
- Modify: `tests/test_source_contracts.cpp`
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/project/manager/ProjectTerrainProductsManager.h`
- Modify: `src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `src/gui/dialogs/CreateDemDialog.cpp`

**Interfaces:**
- Consumes: `xjw::matchphotos::MatchPhotosTask::run(const MatchPhotosContext&)`。
- Produces: DEM 后台流程仍通过 `demPipelineProgressChanged` 与 `demPipelineFinished` 报告状态；成功时通过 `ProjectManager::appendIpfindResults(...)` 与 `appendIpmatchResults(...)` 写回结果。

- [ ] **Step 1: 添加 DEM 迁移源码契约测试**

```cpp
TEST(DemPipelineContractTest, UsesMatchPhotosTaskAndWritesResultsBeforeDenseStage)
{
    const QString source = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectTerrainProductsManager.cpp"));
    expectContainsAll(source, {
        "MatchPhotosTask",
        "MatchPhotosContext",
        "appendIpfindResults",
        "appendIpmatchResults",
        "result.success",
    });
    expectNotContainsAll(source, {
        "FeatureMatchRunner",
        "FeatureExtractionRunner",
        "FeaturePairPlanner",
        "canonicalFeatureAlgorithmFromMatcher",
        "canonicalMatchAlgorithmFromMatcher",
    });
}
```

同时更新旧 runner 专用测试：保留仍由 `MatchingStage` 或 `AerialTriangulationService` 提供的 sidecar、
V2 索引、取消和 Torch 警告契约；删除只针对旧 Python runner、多特征 suffix 与半转重试的契约。

- [ ] **Step 2: 运行源码契约测试并确认 RED**

Run: `cmake --build build --target test_source_contracts test_gui_project_utils -j 4`

Run: `ctest --test-dir build --output-on-failure -R 'test_source_contracts|test_gui_project_utils'`

Expected: 新 DEM 契约因源码仍包含 `FeatureMatchRunner`/`FeatureExtractionRunner` 而失败。

- [ ] **Step 3: 精简 DEM context**

```cpp
struct DemPipelineContext
{
    QString projectPath;
    QStringList images;
    QString outputDir;
    QMap<QString, xjw::Camera> referenceCameras;
    QMap<QString, QString> maskPaths;
    double demResolution = 0.0;
    QString demType;
};
```

删除不再生效的 `cameraPaths`、`featureAlgorithm`、`matchAlgorithm` 和 `knownCameraCenters`。
在 GUI 主线程构造 context 时读取项目路径、参考相机和蒙版路径。

- [ ] **Step 4: 用 MatchPhotosTask 替换旧两个 runner**

```cpp
xjw::matchphotos::MatchPhotosOptions options;
options.planOnly = false;
options.featureAlgorithm = QStringLiteral("sift");
options.matcherAlgorithm = QStringLiteral("lightglue");
options.device = xjw::matchphotos::ComputeDevice::Auto;
options.pairPolicy.exhaustiveMaxImages = 80;
options.pairPolicy.sequenceWindow = 4;
options.useReferencePreselection = !ctx.referenceCameras.isEmpty();

xjw::matchphotos::MatchPhotosContext context;
context.projectPath = ctx.projectPath;
context.workingDirectory = ProjectIO::projectAssetsDir(ctx.projectPath);
context.featureDirectory = ProjectIO::ipfindOutputDir(ctx.projectPath);
context.matchDirectory = ProjectIO::ipmatchOutputDir(ctx.projectPath);
context.pairInput.images = ctx.images;
context.referenceCameras = ctx.referenceCameras;
context.maskPaths = ctx.maskPaths;
context.cancelFlag = &cancelFlag;
context.progressCount = &progressCount;

const xjw::matchphotos::MatchPhotosResult result =
    xjw::matchphotos::MatchPhotosTask(options).run(context);
if (!result.success)
{
    emit demPipelineFinished(false, result.errorMessage);
    return;
}
```

把 result 中的 feature/match 记录转换成项目记录。使用 `QMetaObject::invokeMethod` 在
`ProjectManager` 所在线程批量写回，并在写回前检查 `currentProjectPath()` 仍等于 `ctx.projectPath`。

- [ ] **Step 5: 删除虚假的 DEM matcher 设置**

从 `CreateDemDialog.cpp` 删除：

```cpp
settings[QStringLiteral("matcher")] = QStringLiteral("disk_lightglue");
```

DEM manager 不再读取该设置，日志明确报告 `SIFT + LightGlue`。

- [ ] **Step 6: 运行 GREEN**

Run: `cmake --build build --target test_source_contracts test_gui_project_utils -j 4`

Run: `ctest --test-dir build --output-on-failure -R 'test_source_contracts|test_gui_project_utils'`

Expected: 新 DEM 契约通过，保留的 sidecar、取消和项目写回契约通过。

---

### Task 3: 删除目录并清理构建、测试和文档

**Files:**
- Delete: `src/core/pipeline/FeatureMatchRunner.h`
- Delete: `src/core/pipeline/FeatureMatchRunner.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `src/gui/CMakeLists.txt`
- Modify: `src/cli/CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`
- Modify: `src/gui/widgets/DualImageViewer.cpp`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `docs/design/enhanced_dem_workflow.md`

**Interfaces:**
- Produces: 仓库不再注册或引用 `src/core/pipeline`；`MatchResultCatalog` 的既有 `xjw::pipeline` 命名空间保持不变。

- [ ] **Step 1: 添加目录清理契约**

在源码契约测试中读取 GUI、CLI、matchphototask 和 tests CMake 文本，断言均不包含：

```cpp
EXPECT_FALSE(cmakeText.contains(QStringLiteral("src/core/pipeline")));
EXPECT_FALSE(guiSources.contains(QStringLiteral("FeatureMatchRunner.cpp")));
```

- [ ] **Step 2: 运行测试并确认 RED**

Run: `ctest --test-dir build --output-on-failure -R 'test_source_contracts|test_gui_project_utils'`

Expected: CMake 和 GUI source list 仍含旧目录，清理契约失败。

- [ ] **Step 3: 删除 runner 并清理所有路径**

从 `GuiSources.cmake` 删除 runner 源文件；从 GUI、CLI 和 tests CMake 删除所有
`src/core/pipeline` include 目录。删除已经迁出的旧目录文件，确保空目录自然消失。

- [ ] **Step 4: 更新文档和注释**

`PROJECT_ARCHITECTURE.md` 删除 core/pipeline 树节点，把预算工具列入 feature_match/lightglue，
并把 GUI task 说明改为 `MatchPhotosTask`。`enhanced_dem_workflow.md` 的旧伪代码改为当前任务调用；
`DualImageViewer.cpp` 注释改为“匹配流程写入的 sidecar”。

- [ ] **Step 5: 运行引用清理检查**

Run: `rg -n "FeatureMatchRunner|src/core/pipeline|../pipeline" src tests docs`

Expected: 无有效引用。设计/计划文档中的历史迁移说明允许保留。

Run: `rg -n "LightGlueFeatureBudget|xjw::pipeline::(resolveLightGlue|LightGlueGpuMemoryInfo|BudgetedFeatureData|budgetFeatureDataForLightGlue|remapLightGlue)" src tests`

Expected: 预算头仅位于 `feature_match/lightglue`，且没有预算 API 的旧命名空间调用。

- [ ] **Step 6: 重新配置和构建相关目标**

Run: `cmake -S . -B build -DBUILD_TESTS=ON`

Run: `cmake --build build -j 4`

Expected: CMake 配置和完整编译成功；若现有工作区其他未完成改动导致失败，记录首个与本任务无关的错误并继续运行能构建的相关目标。

- [ ] **Step 7: 运行相关测试**

Run: `ctest --test-dir build --output-on-failure -R 'LightGlueFeatureBudget|MatchPhotos|SourceContract|GuiProjectUtils|AerialTriangulation'`

Expected: 本次迁移相关测试通过；任何失败都记录测试名和错误摘要。

- [ ] **Step 8: 最终状态审计**

Run: `git status --short`

Run: `git diff --check`

Expected: 没有空白错误；只汇报本任务修改，不把工作区其他用户改动归入本次成果。
