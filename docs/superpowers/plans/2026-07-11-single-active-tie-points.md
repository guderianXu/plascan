# Single Active Tie Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将连接点改为单一当前结果，新结果安全覆盖旧结果，并在工作区按 Metashape 形式显示一个不可展开的“连接点（N个点）”资源节点。

**Architecture:** 新增 `ProjectTiePointResultService` 统一选择、替换、删除和文件保护逻辑；项目格式继续使用数组以兼容旧项目，但新写入只保留一条记录。`ProjectManager` 对外提供规范化的当前结果视图，`DataTreeWidget` 只负责把当前记录渲染成可交互的顶层叶节点。

**Tech Stack:** C++17、Qt6 Core/Widgets、QJson、QFile/QDir、CMake、GTest、CTest

---

本仓库只在用户明确要求时提交，因此本计划不包含自动 `git commit` 步骤。工作区已有大量未提交改动，实施时只能补丁式修改下列文件，不得回滚其他改动。

## 文件结构

- Create: `src/gui/project/services/ProjectTiePointResultService.h` — 当前结果选择、替换和删除接口及结果结构。
- Create: `src/gui/project/services/ProjectTiePointResultService.cpp` — 路径规范化、引用保护和安全清理实现。
- Modify: `src/gui/project/support/ProjectMetadataOperations.h/.cpp` — 将 `appendAtResult` 改为明确的单结果替换入口。
- Modify: `src/gui/project/manager/ProjectManager.h/.cpp` — 暴露 `replaceTiePointResult`，转发规范化元数据快照。
- Modify: `src/gui/project/support/ProjectSfmWorkflow.cpp` — 直接项目写回改用替换入口。
- Modify: `src/gui/project/manager/ProjectSparseReconstructionManager.cpp` — 三角化、稀释、清理和精修统一替换当前结果。
- Modify: `src/gui/main_window/MenuWorkflowController.cpp` — 两个 SfM 写回入口统一替换当前结果。
- Modify: `src/gui/widgets/DataTreeWidget.h/.cpp` — 创建单个顶层连接点资源节点并支持顶层资源交互。
- Modify: `src/gui/cmake/GuiSources.cmake` — 加入新服务源文件。
- Modify: `tests/CMakeLists.txt` — 将新服务加入 `test_gui_project_utils`。
- Modify: `tests/test_gui_project_utils.cpp` — 服务、集成和工作区回归测试。
- Modify: `docs/PROJECT_ARCHITECTURE.md` — 记录单一连接点结果服务边界。

### Task 1: 当前连接点选择与旧项目规范化视图

**Files:**
- Create: `src/gui/project/services/ProjectTiePointResultService.h`
- Create: `src/gui/project/services/ProjectTiePointResultService.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写当前结果选择失败测试**

在 `tests/test_gui_project_utils.cpp` 引入服务头文件，并增加以下测试结构。测试必须在临时项目目录中真实创建 PLY 文件，不能只判断非空路径：

```cpp
TEST(TiePointResultServiceTest, SelectsLatestExistingSparseCloudFromLegacyHistory)
{
    QTemporaryDir tempDir;
    ASSERT_TRUE(tempDir.isValid());
    const QString projectPath = QDir(tempDir.path()).filePath(QStringLiteral("legacy.plascan"));
    const QString oldPath = QDir(tempDir.path()).filePath(QStringLiteral("old/sparse.ply"));
    const QString latestMissing = QDir(tempDir.path()).filePath(QStringLiteral("missing/sparse.ply"));
    ASSERT_TRUE(QDir().mkpath(QFileInfo(oldPath).absolutePath()));
    QFile oldFile(oldPath);
    ASSERT_TRUE(oldFile.open(QIODevice::WriteOnly));
    oldFile.write("ply");
    oldFile.close();

    const QJsonObject oldRecord = makeSparseRecord(oldPath, 2314);
    const QJsonObject missingRecord = makeSparseRecord(latestMissing, 9999);
    const QJsonObject meta{
        {QStringLiteral("aerial_triangulation_results"), QJsonArray{oldRecord, missingRecord}}
    };

    const auto selected = ProjectTiePointResultService::selectCurrent(meta, projectPath);
    ASSERT_TRUE(selected.isValid());
    EXPECT_EQ(selected.sourceIndex, 0);
    EXPECT_EQ(selected.sparseCloudPath, QDir::cleanPath(oldPath));
    EXPECT_EQ(selected.pointCount, 2314);
}
```

同时增加 `ReturnsInvalidSelectionWhenNoSparseCloudExists`，验证全部路径缺失时 `isValid()` 为 `false`。

- [ ] **Step 2: 配置测试目标并确认 RED**

将新服务 `.cpp` 加入 `test_gui_project_utils`，将 `.h/.cpp` 加入 `GuiSources.cmake`。运行：

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe `
  --gtest_filter=TiePointResultServiceTest.SelectsLatestExistingSparseCloudFromLegacyHistory:TiePointResultServiceTest.ReturnsInvalidSelectionWhenNoSparseCloudExists
```

Expected: 编译或链接失败，因为 `ProjectTiePointResultService` 尚未实现。

- [ ] **Step 3: 定义服务接口**

`ProjectTiePointResultService.h` 使用完整、稳定的接口：

```cpp
#pragma once

#include <QJsonObject>
#include <QString>
#include <QStringList>

class ProjectData;

namespace xjw::gui::project
{

struct TiePointResultSelection
{
    int sourceIndex = -1;
    int pointCount = -1;
    QJsonObject record;
    QString sparseCloudPath;

    bool isValid() const { return sourceIndex >= 0 && !record.isEmpty() && !sparseCloudPath.isEmpty(); }
};

struct TiePointMutationResult
{
    bool success = false;
    int removedRecordCount = 0;
    QString errorMessage;
    QStringList cleanupWarnings;
};

class ProjectTiePointResultService
{
public:
    static TiePointResultSelection selectCurrent(const QJsonObject &metadata,
                                                 const QString &projectPath);
    static QJsonObject metadataWithCurrentOnly(const QJsonObject &metadata,
                                               const QString &projectPath);
    static TiePointMutationResult replaceCurrent(ProjectData *projectData,
                                                 const QJsonObject &newRecord);
    static TiePointMutationResult deleteAll(ProjectData *projectData);
};

} // namespace xjw::gui::project
```

- [ ] **Step 4: 实现只读选择和规范化视图**

在 `.cpp` 中从数组末尾向前检查 `files.sparse_cloud_xyz`，使用项目路径父目录解析相对路径，并要求 `QFileInfo(path).isFile()`。点数读取顺序固定为：

```cpp
int sparsePointCount(const QJsonObject &record)
{
    int count = record.value(QStringLiteral("sparse_point_count")).toInt(-1);
    if (count < 0) count = record.value(QStringLiteral("point_count")).toInt(-1);
    if (count < 0) count = record.value(QStringLiteral("quality")).toObject()
        .value(QStringLiteral("point_count")).toInt(-1);
    return count;
}
```

`metadataWithCurrentOnly()` 只修改返回副本：有效时数组为 `QJsonArray{selection.record}`，无有效结果时为空数组。不得写回 `ProjectData` 或删除文件。

- [ ] **Step 5: 运行选择测试确认 GREEN**

运行 Step 2 的相同命令。Expected: 两项测试通过。

### Task 2: 安全替换和真实删除

**Files:**
- Modify: `src/gui/project/services/ProjectTiePointResultService.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写覆盖、共享路径和删除失败测试**

新增以下 GTest：

```cpp
TEST(TiePointResultServiceTest, ReplaceKeepsOnlyNewRecordAndRemovesUnreferencedArtifacts)
TEST(TiePointResultServiceTest, ReplaceProtectsNewFilesInsideSharedOutputDirectory)
TEST(TiePointResultServiceTest, DeleteAllClearsMetadataAfterFilesAreRemoved)
TEST(TiePointResultServiceTest, DeleteFailureKeepsMetadataForRetry)
```

测试要求：

- 使用 `ProjectData::createProject()` 创建真实项目。
- 旧记录至少包含 `sparse_cloud_xyz`、`sparse_cloud_points_json` 和 `output_dir`。
- 新记录文件必须真实存在。
- 共享目录测试让旧记录和新记录使用同一目录，断言新 PLY/JSON 均存在且目录未被递归删除。
- 删除失败测试使用项目目录外路径，断言服务拒绝删除、`success == false`，原数组保持不变。

- [ ] **Step 2: 运行测试确认 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=TiePointResultServiceTest.Replace*:TiePointResultServiceTest.Delete*
```

Expected: `replaceCurrent()` / `deleteAll()` 返回未实现或断言失败。

- [ ] **Step 3: 实现路径收集与保护集合**

服务内部增加仅在 `.cpp` 可见的辅助函数：

```cpp
QString normalizedProjectPath(const QString &projectRoot, const QString &path);
void collectTiePointArtifacts(const QJsonObject &record,
                              const QString &projectRoot,
                              QSet<QString> *files,
                              QSet<QString> *directories);
void collectReferencedPaths(const QJsonObject &metadata,
                            const QString &projectRoot,
                            QSet<QString> *protectedPaths);
bool isWithinProjectRoot(const QString &path, const QString &projectRoot);
bool directoryContainsProtectedPath(const QString &directory, const QSet<QString> &protectedPaths);
```

Windows 路径比较使用 `QDir::cleanPath(...).toCaseFolded()` 作为键。目录删除必须同时满足：位于项目根目录内、不是项目根目录本身、不包含任何受保护路径。

- [ ] **Step 4: 实现替换事务顺序**

`replaceCurrent()` 必须：

1. 验证 `newRecord.files.sparse_cloud_xyz` 指向存在的普通文件。
2. 快照旧数组并生成旧文件集合。
3. 将新记录作为唯一数组写入 `ProjectData::updateMetadata(meta, true)`。
4. 从写回后的完整元数据生成保护集合。
5. 删除不受保护的旧文件；只递归删除通过项目根目录和保护集合检查的专属目录。
6. 清理失败仅写入 `cleanupWarnings`，新记录保持当前状态。

新记录验证失败或元数据对象不可用时返回 `success == false`，不得修改旧数组。

- [ ] **Step 5: 实现删除的失败保留语义**

`deleteAll()` 先收集并删除全部可安全处理的连接点文件。遇到项目外目录、文件删除失败或目录安全校验失败时立即返回失败，不清空元数据。全部处理成功后才写入空数组并标记项目已修改。

- [ ] **Step 6: 运行替换和删除测试确认 GREEN**

运行 Step 2 命令。Expected: 四项测试全部通过。

### Task 3: 所有连接点写回统一走替换入口

**Files:**
- Modify: `src/gui/project/support/ProjectMetadataOperations.h`
- Modify: `src/gui/project/support/ProjectMetadataOperations.cpp`
- Modify: `src/gui/project/manager/ProjectManager.h`
- Modify: `src/gui/project/manager/ProjectManager.cpp`
- Modify: `src/gui/project/support/ProjectSfmWorkflow.cpp`
- Modify: `src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Modify: `src/gui/main_window/MenuWorkflowController.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 增加源码契约测试**

新增 `TiePointResultIntegrationTest.AllSparseWritersUseReplacementEntryPoint`，读取上述源文件并断言：

```cpp
EXPECT_FALSE(allSources.contains(QStringLiteral("appendAtResult(")));
EXPECT_TRUE(projectManagerHeader.contains(QStringLiteral("replaceTiePointResult(")));
EXPECT_TRUE(metadataOperationsHeader.contains(QStringLiteral("replaceTiePointResult(")));
EXPECT_TRUE(allSources.count(QStringLiteral("replaceTiePointResult(")) >= 7);
```

另增运行时测试，验证 `ProjectManager::currentMeta()` 对旧多记录项目只返回一个最新有效记录，但 `ProjectData::metadata()` 在只读打开后仍保留原数组。

- [ ] **Step 2: 运行集成测试确认 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=TiePointResultIntegrationTest.*
```

Expected: 仍检测到 `appendAtResult()`，且 `currentMeta()` 返回多条记录。

- [ ] **Step 3: 替换元数据入口**

将 `ProjectMetadataOperations::appendAtResult` 改为：

```cpp
TiePointMutationResult replaceTiePointResult(ProjectData *projectData,
                                             const QString &sparseCloudPath,
                                             int sparsePointCount,
                                             const QStringList &selectedImages,
                                             const QString &outputDir,
                                             const QJsonObject &extraRecord = {});
```

函数继续用 `makeAtResultRecord()` 和 `sparseOperationDisplayName()` 构建记录，然后调用 `ProjectTiePointResultService::replaceCurrent()`。删除 `replaceIndex` 参数，因为单结果语义下没有历史索引替换。

- [ ] **Step 4: 更新 ProjectManager 和元数据转发**

`ProjectManager::replaceTiePointResult()` 返回 `bool`，失败时通过输出参数传递错误，清理警告写日志并刷新质量报告。`currentMeta()` 返回：

```cpp
return _projectData
    ? ProjectTiePointResultService::metadataWithCurrentOnly(_projectData->metadata(), currentProjectPath())
    : QJsonObject();
```

将 `ProjectData::metadataChanged` 改为 lambda 转发相同的规范化副本，确保 Dashboard、DataTree 和后续 UI 消费者看到同一个当前结果视图。不得将规范化副本写回旧项目。

- [ ] **Step 5: 更新全部调用点**

更新两个 `MenuWorkflowController` SfM 写回、`ProjectSparseReconstructionManager` 三角化与后处理、`ProjectManager` 两个 BA 写回，以及 `ProjectSfmWorkflow`。每个调用点在替换失败时停止报告成功，并显示或返回服务错误；任务失败和取消路径不得调用替换入口。

- [ ] **Step 6: 运行集成测试确认 GREEN**

运行 Step 2 命令。Expected: 所有源码契约和运行时测试通过。

### Task 4: Metashape 单叶节点和顶层资源交互

**Files:**
- Modify: `src/gui/widgets/DataTreeWidget.h`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: 编写工作区 RED 测试**

新增 `DataTreeWidgetTest.ShowsSingleMetashapeTiePointLeafForLegacyHistory`：

```cpp
QStandardItem *tiePoints = findTopLevelItem(model, QStringLiteral("连接点（2,314个点）"));
ASSERT_NE(tiePoints, nullptr);
EXPECT_EQ(tiePoints->rowCount(), 0);
EXPECT_FALSE(view->isExpandable(model->indexFromItem(tiePoints)));
EXPECT_EQ(model->item(tiePoints->row(), 1)->text(), latestPath);
EXPECT_EQ(model->item(tiePoints->row(), 2)->text(), QStringLiteral("generated"));
```

元数据放入两条记录，断言树中没有 `#0`、`#1`、`[当前]` 或操作名称子项。再用 `QSignalSpy` 双击/激活该顶层节点，验证 `resourceActivated("连接点", latestPath)`。

- [ ] **Step 2: 运行工作区测试确认 RED**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=DataTreeWidgetTest.ShowsSingleMetashapeTiePointLeafForLegacyHistory
```

Expected: 当前树仍显示可展开分组和历史子项，测试失败。

- [ ] **Step 3: 增加顶层资源角色和辅助函数**

在 `DataTreeWidget.cpp` 定义专用角色：

```cpp
constexpr int SectionRole = Qt::UserRole + 1;
constexpr int ResourcePathRole = Qt::UserRole + 2;
```

新增私有方法：

```cpp
QStandardItem *appendTopLevelResource(const QString &label,
                                      WorkspaceSection section,
                                      const QString &sectionName,
                                      const QString &path,
                                      const QString &storage);
```

该方法创建名称、路径、存储三列，将分区名和路径写入名称项角色，并设置现有 `workspaceSectionIcon(TiePoints)`。

- [ ] **Step 4: 重构资源索引和右键菜单识别**

`resourceFromIndex()` 优先读取名称项的 `SectionRole/ResourcePathRole`，支持顶层资源；无角色时继续使用父分组和隐藏路径列处理现有子项。

`onContextMenuRequested()` 不再无条件忽略顶层行，而是对每个选中行调用 `resourceFromIndex()`。普通分组根仍因无资源角色而返回 `false`，连接点叶节点可正常提供打开、定位、属性和删除数据操作。

- [ ] **Step 5: 用唯一连接点叶节点替代历史循环**

删除 `sparseResultCount` 分组创建和 912–944 行历史子项循环。反向选择最后一个带非空 `sparse_cloud_xyz` 的显示记录，格式化标签：

```cpp
QString label = QStringLiteral("连接点");
if (pointCount >= 0)
{
    label = QStringLiteral("连接点（%1个点）").arg(QLocale().toString(pointCount));
}
appendTopLevelResource(label,
                       WorkspaceSection::TiePoints,
                       QStringLiteral("连接点"),
                       sparsePath,
                       QStringLiteral("generated"));
```

- [ ] **Step 6: 更新既有断言并确认 GREEN**

将旧断言 `连接点 (1873)` 更新为 `连接点（1,873个点）`。运行：

```powershell
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=DataTreeWidgetTest.*TiePoint*:DataTreeWidgetTest.ShowsAlignedPhotoRatioAndHidesEmptySections
```

Expected: 相关测试全部通过。

### Task 5: 删除入口、文档和最终验证

**Files:**
- Modify: `src/gui/project/manager/ProjectManager.cpp`
- Modify: `src/gui/project/services/ProjectResourceCleanupService.cpp` only if shared helpers are extracted
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Verify: all files above

- [ ] **Step 1: 将连接点删除路由到专用服务**

在 `ProjectManager::deleteGeneratedData()` 中对 `section == "连接点"` 使用 `ProjectTiePointResultService::deleteAll()`，不再调用会先清元数据再删文件的通用清理路径。失败时显示具体错误和失败路径；成功后刷新元数据视图、质量报告和重建动作状态。

- [ ] **Step 2: 增加删除入口运行时测试**

使用临时项目连接 `DataTreeWidget::deleteDataRequested` 或直接调用项目管理入口，验证删除成功后 `aerial_triangulation_results` 为空；不安全路径失败时数组仍为原记录。

- [ ] **Step 3: 更新架构文档**

在 `docs/PROJECT_ARCHITECTURE.md` 的 GUI project services 部分加入：

```text
ProjectTiePointResultService：维护单一当前连接点结果，负责旧项目当前结果选择、覆盖清理、共享路径保护和真实删除；DataTreeWidget 仅渲染规范化元数据。
```

- [ ] **Step 4: 运行专项测试**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target test_gui_project_utils --parallel 8
build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=TiePointResultServiceTest.*:TiePointResultIntegrationTest.*:DataTreeWidgetTest.*TiePoint*:DataTreeWidgetTest.ShowsAlignedPhotoRatioAndHidesEmptySections
ctest --test-dir build/windows-vcpkg-cuda-release -C Release --output-on-failure -R "GuiProjectUtils|TiePoint"
```

Expected: 专项测试全部通过，0 failures。

- [ ] **Step 5: 构建 Release GUI 并检查补丁**

```powershell
cmake --build build/windows-vcpkg-cuda-release --config Release --target plascan.exe --parallel 8
git diff --check -- src/gui/project/services/ProjectTiePointResultService.h src/gui/project/services/ProjectTiePointResultService.cpp src/gui/project/support/ProjectMetadataOperations.h src/gui/project/support/ProjectMetadataOperations.cpp src/gui/project/manager/ProjectManager.h src/gui/project/manager/ProjectManager.cpp src/gui/project/support/ProjectSfmWorkflow.cpp src/gui/project/manager/ProjectSparseReconstructionManager.cpp src/gui/main_window/MenuWorkflowController.cpp src/gui/widgets/DataTreeWidget.h src/gui/widgets/DataTreeWidget.cpp src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_gui_project_utils.cpp docs/PROJECT_ARCHITECTURE.md
```

Expected: Release GUI 链接成功；`git diff --check` 仅允许既有 Windows 行尾转换警告，不允许空白错误。

- [ ] **Step 6: 人工检查**

打开包含多条旧连接点记录的项目，确认只显示一个不可展开的 `连接点（N个点）` 节点。执行一次连接点精修后确认节点数量不增加、路径和点数更新、旧专属文件被清理；删除节点后确认节点消失且没有历史结果回退。
