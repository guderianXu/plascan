# Depth Overlay, Workspace, and Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the partially implemented photo depth overlay, remove standalone depth resources from the workspace, and make MVS coverage reporting accurate and timely.

**Architecture:** Keep `depth_map_results` and raw artifacts as the authoritative project state. `CanvasWidget` enters a temporary depth-inspection state that suppresses feature diagnostics without changing user preferences, while `DataTreeWidget` stops materializing depth nodes. Reconstruction quality reads a backward-compatible canonical coverage value and refreshes immediately after depth completion.

**Tech Stack:** C++17, Qt 6, OpenCV, QJson, GTest, CMake/CTest, native Windows PowerShell.

**Execution constraints:** Use only `E:\code\plascan\build\windows-vcpkg-cuda-release`. Do not reset, checkout, clean, delete user/build data, commit, or push. Preserve all unrelated dirty-worktree changes.

---

## File map

- Modify `src/gui/widgets/CanvasWidget.h/.cpp`: own temporary depth-inspection suppression and restore feature diagnostics.
- Modify `src/gui/widgets/DataTreeWidget.cpp`: stop creating depth-map workspace sections while retaining metadata normalization.
- Modify `src/core/mvs/DepthMapGenerator.cpp`: publish canonical per-frame `valid_coverage`.
- Modify `src/core/mvs/MvsWorkspaceManifest.h/.cpp`: persist canonical coverage for resumed projects.
- Modify `src/core/qc/ReconstructionQualityReport.cpp`: resolve new, nested, legacy, and computed coverage; represent unavailable distinctly.
- Modify `src/gui/project/support/ProjectWorkflowReports.cpp`: omit unavailable coverage from registered report records.
- Modify `src/gui/widgets/ProjectDashboardWidget.cpp`: show unavailable coverage as `—` and suppress false alerts.
- Modify `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`: refresh quality after depth completion and streamed fusion completion.
- Modify `tests/test_gui_project_utils.cpp`: GUI, tree, dashboard, and source-contract regression tests.
- Modify `src/core/mvs/tests/test_mvs_workspace_manifest.cpp`: manifest coverage round-trip test.
- Modify `src/core/qc/tests/test_reconstruction_quality_report.cpp`: schema compatibility and unavailable-value tests.
- Modify `docs/PROJECT_ARCHITECTURE.md` and `src/core/mvs/README.md`: document final behavior and metadata.

### Task 1: Establish the focused baseline

- [ ] **Step 1: Record only task-related dirty files**

Run:

```powershell
git status --short -- `
  src/gui/widgets/CanvasWidget.h `
  src/gui/widgets/CanvasWidget.cpp `
  src/gui/widgets/DataTreeWidget.cpp `
  src/core/mvs/DepthMapGenerator.cpp `
  src/core/mvs/MvsWorkspaceManifest.h `
  src/core/mvs/MvsWorkspaceManifest.cpp `
  src/core/qc/ReconstructionQualityReport.cpp `
  src/gui/project/support/ProjectWorkflowReports.cpp `
  src/gui/widgets/ProjectDashboardWidget.cpp `
  src/gui/project/manager/ProjectDenseReconstructionManager.cpp `
  tests/test_gui_project_utils.cpp `
  src/core/mvs/tests/test_mvs_workspace_manifest.cpp `
  src/core/qc/tests/test_reconstruction_quality_report.cpp
```

Expected: existing user changes may appear; no file is reverted or staged.

- [ ] **Step 2: Build existing focused targets before edits**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_gui_project_utils test_mvs_workspace_manifest test_reconstruction_quality_report
```

Expected: either all three targets build, or the exact pre-existing compiler error is recorded before task edits.

### Task 2: Suppress feature diagnostics during depth inspection

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/widgets/CanvasWidget.h`
- Modify: `src/gui/widgets/CanvasWidget.cpp`

- [ ] **Step 1: Write the failing state-preservation test**

Add next to the existing `CanvasDepthOverlayTest` cases. This uses the real image loader, artifact resolver, depth loader, and overlay controller; it adds no test-only production method:

```cpp
TEST(CanvasDepthOverlayTest, SuppressesDiagnosticsWithoutChangingUserPreferences)
{
    QTemporaryDir directory;
    ASSERT_TRUE(directory.isValid());
    const QString image_path = directory.filePath(QStringLiteral("image.png"));
    const QString depth_path = directory.filePath(QStringLiteral("depth_0.bin"));
    const QString mask_path = directory.filePath(QStringLiteral("depth_0_mask.png"));
    ASSERT_TRUE(QImage(16, 12, QImage::Format_RGB32).save(image_path));
    const cv::Mat depth(12, 16, CV_32FC1, cv::Scalar(2.0f));
    ASSERT_TRUE(xjw::core::project::writeDepthMatStorage(depth_path, depth).ok);
    ASSERT_TRUE(cv::imwrite(mask_path.toStdString(),
                           cv::Mat(12, 16, CV_8UC1, cv::Scalar(255))));

    const QJsonObject record{
        {QStringLiteral("ref_image"), image_path},
        {QStringLiteral("raw_depth_path"), depth_path},
        {QStringLiteral("valid_mask_path"), mask_path},
        {QStringLiteral("grid_width"), 16},
        {QStringLiteral("grid_height"), 12}
    };
    CanvasWidget canvas;
    canvas.setProperty("currentProjectPath", directory.filePath(QStringLiteral("test.plascan")));
    canvas.setProjectMetadata(QJsonObject{
        {QStringLiteral("depth_map_results"), QJsonArray{record}}
    });
    LayerRenderer::FeatureDisplayOptions options;
    options.showPoints = true;
    options.showResiduals = true;
    canvas.applyFeatureDisplayOptions(options);
    canvas.showImage(image_path);
    QTRY_VERIFY(canvas.hasDisplayImage());

    canvas.setDepthOverlayEnabled(true);
    QTRY_VERIFY(canvas.depthOverlayVisible());
    EXPECT_TRUE(canvas.showsInterestPoints());
    EXPECT_TRUE(canvas.showsFeatureResiduals());
    EXPECT_TRUE(canvas.featureDiagnosticsSuppressed());

    canvas.setDepthOverlayEnabled(false);
    EXPECT_FALSE(canvas.featureDiagnosticsSuppressed());
    EXPECT_TRUE(canvas.showsInterestPoints());
    EXPECT_TRUE(canvas.showsFeatureResiduals());
}
```

- [ ] **Step 2: Run the new test and verify failure**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=CanvasDepthOverlayTest.SuppressesDiagnosticsWithoutChangingUserPreferences
```

Expected: compile fails because the production visibility/suppression accessors do not exist.

- [ ] **Step 3: Add an explicit transient inspection state**

Add to `CanvasWidget.h`:

```cpp
public:
    bool featureDiagnosticsSuppressed() const { return _depthInspectionActive; }
    bool depthOverlayVisible() const { return _depthOverlayVisible; }

private:
    void setDepthInspectionActive(bool active);
    bool shouldRenderFeatureDiagnostics() const { return !_depthInspectionActive; }

    bool _depthInspectionActive{false};
```

Implement the state transition in `CanvasWidget.cpp`:

```cpp
void CanvasWidget::setDepthInspectionActive(bool active)
{
    if (_depthInspectionActive == active || !_layerRenderer)
    {
        return;
    }

    _depthInspectionActive = active;
    ++_featureLoadGeneration;
    ++_residualLoadGeneration;
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearFeatureResidualLayers();

    if (active || _currentImagePath.trimmed().isEmpty())
    {
        return;
    }
    if (_showInterestPoints && _currentFeatureOpts.showPoints)
    {
        startSpLoadForImage(_currentImagePath);
    }
    if (_currentFeatureOpts.showResiduals)
    {
        startResidualLoadForImage(_currentImagePath);
    }
}
```

In `refreshDepthOverlay()`, call `setDepthInspectionActive(false)` on disabled/unavailable paths and `setDepthInspectionActive(true)` immediately before requesting the overlay. In the `overlayFailed` connection, call `setDepthInspectionActive(false)` after clearing the depth pixmap.

- [ ] **Step 4: Guard every feature/residual load and callback**

Use the same predicate at request and completion points:

```cpp
if (!shouldRenderFeatureDiagnostics())
{
    _layerRenderer->clearFeatureLayers();
    _layerRenderer->clearFeatureResidualLayers();
    return;
}
```

And inside asynchronous completion lambdas:

```cpp
if (!self || !self->shouldRenderFeatureDiagnostics())
{
    return;
}
```

Apply this to `applyFeatureDisplayOptions()`, automatic loads after `showImage()`, `setShowInterestPoints()`, `setShowFeatureResiduals()`, `startSpLoadForImage()`, and `startResidualLoadForImage()`.

- [ ] **Step 5: Run the overlay tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_gui_project_utils
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=CanvasDepthOverlayTest.*:DepthOverlayControllerTest.*:LayerRendererDepthOverlayTest.*
```

Expected: all selected tests pass; feature preferences remain true while diagnostics are transiently suppressed.

### Task 3: Remove standalone depth resources from the workspace

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/widgets/DataTreeWidget.cpp`

- [ ] **Step 1: Convert the result-only depth-section test into a non-materialization test**

Rename `ResultOnlyMetadataUpdateRefreshesDepthMapSection` and replace its final assertions with:

```cpp
TEST(DataTreeWidgetTest, ResultOnlyMetadataDoesNotMaterializeDepthResources)
{
    DataTreeWidget tree;
    QJsonObject image{{QStringLiteral("path"), QStringLiteral("/tmp/ref_image_001.jpg")}};
    tree.loadFromJson(QJsonObject{{QStringLiteral("images"), QJsonArray{image}}});

    const QJsonObject depth_record{
        {QStringLiteral("result_type"), QStringLiteral("mvs_depth")},
        {QStringLiteral("depth_png"), QStringLiteral("/tmp/mvs_output/depth_0.png")},
        {QStringLiteral("raw_depth_path"), QStringLiteral("/tmp/mvs_output/depth_0.bin")},
        {QStringLiteral("ref_image"), QStringLiteral("/tmp/ref_image_001.jpg")},
        {QStringLiteral("grid_width"), 6000},
        {QStringLiteral("grid_height"), 4000}
    };
    tree.loadFromJson(QJsonObject{
        {QStringLiteral("depth_map_results"), QJsonArray{depth_record}}
    });

    auto *view = tree.findChild<QTreeView *>();
    ASSERT_NE(view, nullptr);
    auto *model = qobject_cast<QStandardItemModel *>(view->model());
    ASSERT_NE(model, nullptr);
    for (int row = 0; row < model->rowCount(); ++row)
    {
        ASSERT_FALSE(model->item(row, 0)->text().contains(QStringLiteral("深度图")));
    }
}
```

- [ ] **Step 2: Run and verify the old depth section causes failure**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=DataTreeWidgetTest.ResultOnlyMetadataDoesNotMaterializeDepthResources
```

Expected: FAIL because `DataTreeWidget` still creates the depth section.

- [ ] **Step 3: Stop materializing depth nodes without changing metadata recognition**

In `populateFromMeta()`:

```cpp
const QJsonArray depthResults = normalized.value(QStringLiteral("depth_map_results")).toArray();
Q_UNUSED(depthResults);
```

Remove the `createSection(QStringLiteral("深度图"), depthMapCount, WorkspaceSection::DepthMaps)` call and the loop that appends final and pyramid preview nodes. Keep `depth_map_results` in `isResultKey()` and metadata normalization so result-only updates, cleanup, overlay lookup, and project archiving continue to work.

- [ ] **Step 4: Run all DataTree tests**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=DataTreeWidgetTest.*
```

Expected: all DataTree tests pass and none expects a depth workspace section.

### Task 4: Canonicalize and persist depth coverage

**Files:**
- Modify: `src/core/mvs/tests/test_mvs_workspace_manifest.cpp`
- Modify: `src/core/mvs/MvsWorkspaceManifest.h`
- Modify: `src/core/mvs/MvsWorkspaceManifest.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`

- [ ] **Step 1: Write a failing manifest round-trip test**

Add to the existing frame serialization test:

```cpp
record.validCoverage = 0.625;
const QJsonObject json = xjw::mvs::mvsWorkspaceFrameToJson(record);
EXPECT_DOUBLE_EQ(json.value(QStringLiteral("valid_coverage")).toDouble(), 0.625);

const auto loaded = xjw::mvs::mvsWorkspaceFrameFromJson(json);
EXPECT_DOUBLE_EQ(loaded.validCoverage, 0.625);
```

- [ ] **Step 2: Run the manifest test and verify failure**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_mvs_workspace_manifest
```

Expected: compile fails because `validCoverage` does not exist.

- [ ] **Step 3: Add the canonical manifest field**

Add to `MvsWorkspaceFrameRecord`:

```cpp
double validCoverage = -1.0;
```

Serialize only valid values and read missing legacy values as `-1.0`:

```cpp
if (record.validCoverage >= 0.0 && std::isfinite(record.validCoverage))
{
    object.insert(QStringLiteral("valid_coverage"), record.validCoverage);
}

record.validCoverage = object.value(QStringLiteral("valid_coverage")).toDouble(-1.0);
```

- [ ] **Step 4: Publish coverage in generated artifacts and manifest records**

In `DepthMapGenerator::saveDepthFrameArtifacts()` add:

```cpp
artifact[QStringLiteral("valid_coverage")] =
    static_cast<double>(result.qualityMetrics.validCoverage);
record.validCoverage = static_cast<double>(result.qualityMetrics.validCoverage);
```

`makeProjectDepthRecordFromArtifact()` already copies every artifact field, so no second conversion is added.

- [ ] **Step 5: Run MVS manifest and pipeline tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_mvs_workspace_manifest test_mvs_pipeline
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R "MvsWorkspace|MvsPipeline"
```

Expected: round-trip and existing MVS tests pass.

### Task 5: Make reconstruction coverage backward-compatible and nullable

**Files:**
- Modify: `src/core/qc/tests/test_reconstruction_quality_report.cpp`
- Modify: `src/core/qc/ReconstructionQualityReport.cpp`
- Modify: `src/gui/project/support/ProjectWorkflowReports.cpp`

- [ ] **Step 1: Add schema compatibility and unavailable tests**

Add:

```cpp
TEST(ReconstructionQualityReport, ReadsCanonicalNestedLegacyAndComputedDepthCoverage)
{
    QJsonObject canonical{{QStringLiteral("status"), QStringLiteral("completed")},
                          {QStringLiteral("valid_coverage"), 0.8}};
    QJsonObject nested{{QStringLiteral("status"), QStringLiteral("completed")},
                       {QStringLiteral("depth_quality"),
                        QJsonObject{{QStringLiteral("valid_coverage"), 0.6}}}};
    QJsonObject legacy{{QStringLiteral("status"), QStringLiteral("completed")},
                       {QStringLiteral("valid_ratio"), 0.4}};
    QJsonObject computed{{QStringLiteral("status"), QStringLiteral("completed")},
                         {QStringLiteral("valid_pixel_count"), 25},
                         {QStringLiteral("grid_width"), 10},
                         {QStringLiteral("grid_height"), 10}};
    const QJsonObject meta{{QStringLiteral("depth_map_results"),
                            QJsonArray{canonical, nested, legacy, computed}}};

    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    EXPECT_NEAR(report.value(QStringLiteral("mvs_valid_coverage")).toDouble(), 0.5125, 1e-9);
}

TEST(ReconstructionQualityReport, LeavesCoverageUnavailableWhenNoFrameHasAMeasurement)
{
    const QJsonObject meta{{QStringLiteral("depth_map_results"),
                            QJsonArray{QJsonObject{{QStringLiteral("status"),
                                                   QStringLiteral("completed")}}}}};
    const QJsonObject report = ReconstructionQualityReport::buildFromProjectMeta(meta);
    EXPECT_FALSE(report.contains(QStringLiteral("mvs_valid_coverage")));
}
```

- [ ] **Step 2: Run and verify the tests fail**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_reconstruction_quality_report
E:\code\plascan\build\windows-vcpkg-cuda-release\src\core\qc\Release\test_reconstruction_quality_report.exe
```

Expected: nested/canonical records are ignored and missing coverage is reported as zero.

- [ ] **Step 3: Implement one coverage resolver**

Replace the `double` average helper with:

```cpp
std::optional<double> depthCoverage(const QJsonObject &record)
{
    const auto finiteRatio = [](const QJsonValue &value) -> std::optional<double>
    {
        if (!value.isDouble())
        {
            return std::nullopt;
        }
        const double ratio = value.toDouble();
        return std::isfinite(ratio) && ratio >= 0.0 && ratio <= 1.0
            ? std::optional<double>(ratio)
            : std::nullopt;
    };

    if (const auto value = finiteRatio(record.value(QStringLiteral("valid_coverage"))))
    {
        return value;
    }
    if (const auto value = finiteRatio(
            record.value(QStringLiteral("depth_quality")).toObject().value(
                QStringLiteral("valid_coverage"))))
    {
        return value;
    }
    if (const auto value = finiteRatio(record.value(QStringLiteral("valid_ratio"))))
    {
        return value;
    }
    const int width = record.value(QStringLiteral("grid_width")).toInt(0);
    const int height = record.value(QStringLiteral("grid_height")).toInt(0);
    const int valid = record.value(QStringLiteral("valid_pixel_count")).toInt(-1);
    if (width > 0 && height > 0 && valid >= 0)
    {
        return std::clamp(static_cast<double>(valid) /
                              static_cast<double>(width) / static_cast<double>(height),
                          0.0,
                          1.0);
    }
    return std::nullopt;
}

std::optional<double> averageCompletedDepthCoverage(const QJsonObject &projectMeta)
{
    double sum = 0.0;
    int count = 0;
    for (const QJsonValue &value : projectMeta.value(
             QStringLiteral("depth_map_results")).toArray())
    {
        const QJsonObject record = value.toObject();
        const QString status = record.value(QStringLiteral("status")).toString();
        if (!status.isEmpty() && status != QStringLiteral("completed"))
        {
            continue;
        }
        if (const auto coverage = depthCoverage(record))
        {
            sum += *coverage;
            ++count;
        }
    }
    return count > 0 ? std::optional<double>(sum / count) : std::nullopt;
}
```

Insert `mvs_valid_coverage` into the report only when the optional contains a value. Add `<optional>` and `<algorithm>` includes if absent.

- [ ] **Step 4: Do not register an unavailable numeric value**

In `writeReconstructionQualityProjectReport()`:

```cpp
const QJsonValue mvs_coverage = writeResult.report.value(
    QStringLiteral("mvs_valid_coverage"));
if (mvs_coverage.isDouble())
{
    record[QStringLiteral("mvs_valid_coverage")] = mvs_coverage;
}
```

- [ ] **Step 5: Run all reconstruction-quality tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_reconstruction_quality_report
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R ReconstructionQualityReport
```

Expected: canonical, nested, legacy, computed, and unavailable cases pass.

### Task 6: Prevent false dashboard alerts

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/widgets/ProjectDashboardWidget.cpp`

- [ ] **Step 1: Add an unavailable-coverage widget test**

```cpp
TEST(ProjectDashboardWidgetTest, DoesNotWarnWhenMvsCoverageIsUnavailable)
{
    ProjectDashboardWidget widget;
    widget.loadFromJson(QJsonObject{
        {QStringLiteral("report_results"),
         QJsonArray{QJsonObject{
             {QStringLiteral("type"), QStringLiteral("reconstruction_quality")},
             {QStringLiteral("path"), QStringLiteral("quality.json")},
             {QStringLiteral("total_image_count"), 16},
             {QStringLiteral("registered_image_count"), 16}}}}});

    auto *alerts = widget.findChild<QTableWidget *>(
        QStringLiteral("dashboardQualityAlertTable"));
    ASSERT_NE(alerts, nullptr);
    for (int row = 0; row < alerts->rowCount(); ++row)
    {
        EXPECT_FALSE(alerts->item(row, 2)->text().contains(QStringLiteral("MVS覆盖")));
    }
}
```

- [ ] **Step 2: Run and verify current behavior**

Run the single test. Expected: the new test documents unavailable semantics; if it already passes, retain it as regression coverage.

- [ ] **Step 3: Render unavailable values explicitly**

Use:

```cpp
const QJsonValue coverage = report.value(QStringLiteral("mvs_valid_coverage"));
appendMetricRow(_qualityTable,
                &row,
                tr("MVS覆盖"),
                coverage.isDouble() ? metricValueText(coverage, true) : tr("—"));
```

Keep the alert predicate gated by `coverage.isDouble()`:

```cpp
if (coverage.isDouble() && coverage.toDouble() < 0.6)
{
    appendAlertRow(_qualityAlertTable,
                   &row,
                   tr("注意"),
                   tr("重建质量"),
                   tr("MVS覆盖 %1").arg(metricValueText(coverage, true)));
}
```

- [ ] **Step 4: Run dashboard tests**

Run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=ProjectDashboardWidgetTest.*
```

Expected: genuine low coverage still warns; unavailable coverage does not.

### Task 7: Refresh quality at the correct workflow boundaries

**Files:**
- Modify: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`

- [ ] **Step 1: Add a source-contract regression test**

```cpp
TEST(ProjectDenseReconstructionManagerContractTest, RefreshesQualityBeforeDepthToFusionTransition)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    const int finished_connection = source.indexOf(
        QStringLiteral("&DepthMapGenerator::finished"));
    const int refresh = source.indexOf(
        QStringLiteral("refreshReconstructionQualityReport()"), finished_connection);
    const int transition = source.indexOf(
        QStringLiteral("startFuseDepthMapsAsync(settings)"), finished_connection);
    ASSERT_GE(finished_connection, 0);
    ASSERT_GE(refresh, 0);
    ASSERT_GE(transition, 0);
    EXPECT_LT(refresh, transition);
}
```

- [ ] **Step 2: Run and verify failure**

Expected: FAIL because the pipeline `finished` connection transitions to fusion without refreshing.

- [ ] **Step 3: Refresh after depth records and before transition**

In the `DepthMapGenerator::finished` lambda used by `startGenerateDenseCloudAsync()`:

```cpp
if (success)
{
    self->_owner->refreshReconstructionQualityReport();
}
const bool shouldStartFusion = success && (continueMissingMode || pipelineMode);
```

Place this after the active-project check and before calculating/dispatching the fusion transition.

- [ ] **Step 4: Refresh after streamed fusion registration**

In the queued completion lambda of `startFuseDepthMapsAsync()`, after the optional dense record upsert and before completion signals:

```cpp
if (self->_owner)
{
    self->_owner->refreshReconstructionQualityReport();
}
emit self->denseCloudResultReady(outputPly, pointCount);
emit self->mvsProgressFinished(true);
```

- [ ] **Step 5: Run focused GUI/project tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_gui_project_utils
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=ProjectDenseReconstructionManagerContractTest.*:ProjectDashboardWidgetTest.*
```

Expected: report refresh is ordered before fusion and dashboard tests pass.

### Task 8: Documentation and phase verification

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/mvs/README.md`

- [ ] **Step 1: Document final GUI and metadata semantics**

Add these concrete statements:

```markdown
- 深度结果不作为独立工作区资源显示；照片工具栏的“显示深度图”按 `ref_image` 精确匹配并叠加显示。
- 深度检查期间特征点和匹配残差仅临时隐藏，关闭深度检查后恢复用户原有显示偏好。
- `valid_coverage` 是深度覆盖率规范字段；读取器兼容 `depth_quality.valid_coverage` 和旧 `valid_ratio`。
- 未统计覆盖率显示为“—”，不得解释为 0%。
```

- [ ] **Step 2: Build all affected targets**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  plascan_gui test_gui_project_utils test_mvs_workspace_manifest `
  test_reconstruction_quality_report test_mvs_pipeline
```

Expected: all listed targets build.

- [ ] **Step 3: Run focused tests**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R "DepthOverlay|DataTreeWidget|ProjectDashboard|ReconstructionQualityReport|MvsWorkspace|MvsPipeline"
```

Expected: all selected tests pass.

- [ ] **Step 4: Verify the Temple archive evidence after opening or regenerating depth**

Use the repository environment only:

```powershell
.\.venv\Scripts\python.exe -c "import json,zipfile,pathlib; p=pathlib.Path(r'E:\code\test\temple\temple.plascan'); z=zipfile.ZipFile(p); d=json.loads(z.read('project_results.json')); r=[x for x in d.get('report_results',[]) if x.get('type')=='reconstruction_quality']; print('depth_records=',len(d.get('depth_map_results',[])),'mvs_coverage=',r[-1].get('mvs_valid_coverage') if r else None)"
```

Expected after a successful refresh: `depth_records=16` and `mvs_coverage` is present and nonzero; before regeneration, legacy artifacts may resolve through nested or computed coverage.

- [ ] **Step 5: Inspect scoped changes without staging**

Run:

```powershell
git diff --check -- `
  src/gui/widgets/CanvasWidget.h src/gui/widgets/CanvasWidget.cpp `
  src/gui/widgets/DataTreeWidget.cpp src/core/mvs/DepthMapGenerator.cpp `
  src/core/mvs/MvsWorkspaceManifest.h src/core/mvs/MvsWorkspaceManifest.cpp `
  src/core/qc/ReconstructionQualityReport.cpp `
  src/gui/project/support/ProjectWorkflowReports.cpp `
  src/gui/widgets/ProjectDashboardWidget.cpp `
  src/gui/project/manager/ProjectDenseReconstructionManager.cpp `
  tests/test_gui_project_utils.cpp src/core/mvs/tests/test_mvs_workspace_manifest.cpp `
  src/core/qc/tests/test_reconstruction_quality_report.cpp
```

Expected: no whitespace errors. Do not stage, commit, or push.
