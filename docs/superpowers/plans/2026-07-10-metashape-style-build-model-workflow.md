# Metashape Style Build Model Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PlaScan `工作流程 -> 生成模型` behave like Metashape Build Model: when source data is `深度图`, reusable depth maps are used if valid, missing depth maps are generated automatically, depth maps are fused into a dense cloud when needed, and the model is generated without requiring the user to manually run a separate depth-map step.

**Architecture:** Add a small workflow policy layer that decides whether model generation can run directly, needs depth-map fusion, or needs depth-map estimation plus fusion. Add a GUI workflow orchestrator owned by `ProjectReconstructionManager` to chain existing `ProjectDenseReconstructionManager` and `ProjectModelManager` without blocking the UI or duplicating MVS algorithms. Keep heavy MVS and mesh algorithms in existing core managers.

**Tech Stack:** C++17, Qt6 signals/slots, QJsonObject project metadata, existing `DepthMapGenerator`, `DepthFrameUtils`, `ProjectDenseReconstructionManager`, `ProjectModelManager`, GTest contract tests.

---

## Metashape Mode Definition

PlaScan should treat depth maps as an internal reusable intermediate, matching the Metashape mental model:

1. User opens `生成模型`.
2. User selects `源数据 = 深度图`.
3. If compatible depth maps already exist and `重用深度图` is checked, PlaScan reuses them.
4. If a compatible dense cloud already exists for those depth maps, PlaScan uses it to build the mesh.
5. If depth maps exist but no dense cloud exists, PlaScan automatically runs depth-map fusion first.
6. If depth maps are missing or incomplete, PlaScan automatically runs depth-map estimation, then fusion, then model generation.
7. The user sees one workflow progress stream and can cancel through the existing MVS cancel path.

This plan does not replace the existing standalone `深度图估计` and `深度图融合` tools. Those remain available as advanced/manual workflow entries.

## File Structure

- Create `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.h`
  - Pure decision types for model generation.
  - No QObject dependency.
  - Determines direct mesh vs fuse existing depth maps vs estimate missing depth maps.
- Create `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.cpp`
  - Implements metadata/settings inspection.
  - Uses existing `collectLatestStoredDepthFrames()` rules.
- Create `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.h`
  - QObject orchestrator that chains dense manager and model manager.
- Create `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.cpp`
  - Implements Metashape-style source workflow.
  - Owns transient state for one active model-generation run.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.h`
  - Owns `ProjectModelGenerationWorkflow`.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.cpp`
  - Route `Task::MeshReconstruction` through workflow instead of directly to model manager.
  - Forward workflow progress signals.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.h`
  - Add a helper for pipeline-mode dense cloud generation status if needed.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
  - Ensure `denseCloudResultReady(path, pointCount)` is emitted for the pipeline path used by model generation.
  - Ensure pipeline mode does not show intermediate success dialogs.
- Modify `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
  - Align depth frame discovery with current `.bin` artifact names.
- Modify `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
  - Use real `depth_*.bin`, `depth_*_conf.bin`, `depth_*.png`, and `depth_*_mask.png` naming.
- Modify `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`
  - Add optional reusable dense cloud path to `DepthMapMeshBuildRequest`.
- Modify `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`
  - Accept the reusable dense path from the GUI workflow.
- Modify `E:/code/plascan/src/gui/dialogs/GenerateModelDialog.cpp`
  - Clarify UI status text for `深度图`: missing depth maps will be generated automatically.
  - Keep `重用深度图` checked by default.
- Modify `E:/code/plascan/tests/test_gui_project_utils.cpp`
  - Add policy and workflow contract tests.
- Modify `E:/code/plascan/tests/test_source_contracts.cpp`
  - Add source-level contract tests so future changes cannot bypass the workflow.
- Modify `E:/code/plascan/tests/test_mesh_reconstructor.cpp`
  - Update depth frame discovery tests from `.raw` to the current `.bin` artifact format.

---

### Task 1: Add Workflow Policy Tests

**Files:**
- Modify: `E:/code/plascan/tests/test_gui_project_utils.cpp`
- Create later in Task 2: `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.h`

- [ ] **Step 1: Add failing tests for source decision behavior**

Append these tests near existing MVS/depth-frame metadata tests in `E:/code/plascan/tests/test_gui_project_utils.cpp`:

```cpp
#include "ProjectModelWorkflowPolicy.h"

TEST(ModelWorkflowPolicyTest, PointCloudSourceRunsMeshDirectly)
{
    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("point_cloud");
    settings[QStringLiteral("source_path")] = QStringLiteral("E:/tmp/dense_cloud.ply");

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_path")).toString(),
              QStringLiteral("E:/tmp/dense_cloud.ply"));
}

TEST(ModelWorkflowPolicyTest, DepthMapsWithReusableDenseCloudRunMeshDirectly)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());
    const QString denseCloud = QDir(dir.path()).filePath(QStringLiteral("dense_cloud.ply"));
    QFile file(denseCloud);
    ASSERT_TRUE(file.open(QIODevice::WriteOnly | QIODevice::Text));
    file.write("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n");
    file.close();

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
    EXPECT_EQ(decision.reusableDenseCloudPath, denseCloud);
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("depth_maps"));
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_point_cloud_path")).toString(),
              denseCloud);
}

TEST(ModelWorkflowPolicyTest, DepthMapsWithStoredFramesFuseBeforeMesh)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    const QString depth0 = QDir(dir.path()).filePath(QStringLiteral("depth_0.png"));
    const QString raw0 = QDir(dir.path()).filePath(QStringLiteral("depth_0.bin"));
    const QString conf0 = QDir(dir.path()).filePath(QStringLiteral("depth_0_conf.bin"));
    const QString depth1 = QDir(dir.path()).filePath(QStringLiteral("depth_1.png"));
    const QString raw1 = QDir(dir.path()).filePath(QStringLiteral("depth_1.bin"));
    const QString conf1 = QDir(dir.path()).filePath(QStringLiteral("depth_1_conf.bin"));
    for (const QString &path : {depth0, raw0, conf0, depth1, raw1, conf1})
    {
        QFile file(path);
        ASSERT_TRUE(file.open(QIODevice::WriteOnly));
        file.write("x");
    }

    QJsonObject frame0;
    frame0[QStringLiteral("status")] = QStringLiteral("completed");
    frame0[QStringLiteral("ref_image")] = QStringLiteral("image_000.jpg");
    frame0[QStringLiteral("depth_png")] = depth0;
    frame0[QStringLiteral("raw_depth_path")] = raw0;
    frame0[QStringLiteral("raw_confidence_path")] = conf0;
    frame0[QStringLiteral("grid_width")] = 1;
    frame0[QStringLiteral("grid_height")] = 1;

    QJsonObject frame1 = frame0;
    frame1[QStringLiteral("ref_image")] = QStringLiteral("image_001.jpg");
    frame1[QStringLiteral("depth_png")] = depth1;
    frame1[QStringLiteral("raw_depth_path")] = raw1;
    frame1[QStringLiteral("raw_confidence_path")] = conf1;

    QJsonObject meta;
    meta[QStringLiteral("depth_map_results")] = QJsonArray{frame0, frame1};

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, meta);

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::FuseDepthMapsThenMesh);
    EXPECT_EQ(decision.denseSettings.value(QStringLiteral("pipeline_mode")).toBool(), true);
    EXPECT_EQ(decision.denseSettings.value(QStringLiteral("output_dir")).toString(), dir.path());
}

TEST(ModelWorkflowPolicyTest, DepthMapsMissingFramesEstimateThenFuseThenMesh)
{
    QTemporaryDir dir;
    ASSERT_TRUE(dir.isValid());

    QJsonObject settings;
    settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    settings[QStringLiteral("source_path")] = dir.path();
    settings[QStringLiteral("depthMapSourcePath")] = dir.path();
    settings[QStringLiteral("reuseDepthMaps")] = true;
    settings[QStringLiteral("quality")] = QStringLiteral("high");

    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, QJsonObject());

    EXPECT_EQ(decision.action, xjw::gui::project::ModelWorkflowAction::GenerateDenseCloudThenMesh);
    EXPECT_EQ(decision.denseSettings.value(QStringLiteral("pipeline_mode")).toBool(), true);
    EXPECT_EQ(decision.denseSettings.value(QStringLiteral("output_dir")).toString(), dir.path());
    EXPECT_EQ(decision.denseSettings.value(QStringLiteral("qualityProfile")).toString(),
              QStringLiteral("high_quality"));
    EXPECT_EQ(decision.modelSettings.value(QStringLiteral("source_data")).toString(),
              QStringLiteral("depth_maps"));
}
```

- [ ] **Step 2: Run the policy tests and verify they fail**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_gui_project_utils --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ModelWorkflowPolicyTest.*
```

Expected: compile failure because `ProjectModelWorkflowPolicy.h` and `decideModelGenerationWorkflow()` do not exist.

- [ ] **Step 3: Commit the failing tests only if working in a feature branch**

Run only when the branch policy allows intermediate commits:

```powershell
git add E:/code/plascan/tests/test_gui_project_utils.cpp
git commit -m "test: define Metashape-style model workflow policy"
```

---

### Task 2: Implement Model Workflow Policy

**Files:**
- Create: `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.h`
- Create: `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/CMakeLists.txt` if this support directory has its own target list; otherwise modify the GUI target list that owns support sources.

- [ ] **Step 1: Add the policy header**

Create `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.h`:

```cpp
#pragma once

#include <QJsonObject>
#include <QString>

namespace xjw::gui::project
{

enum class ModelWorkflowAction
{
    RunMeshDirectly,
    FuseDepthMapsThenMesh,
    GenerateDenseCloudThenMesh
};

struct ModelWorkflowDecision
{
    ModelWorkflowAction action = ModelWorkflowAction::RunMeshDirectly;
    QJsonObject modelSettings;
    QJsonObject denseSettings;
    QString depthMapSourcePath;
    QString reusableDenseCloudPath;
    QString reason;
};

ModelWorkflowDecision decideModelGenerationWorkflow(const QJsonObject &settings,
                                                    const QJsonObject &projectMeta);

QJsonObject denseSettingsFromModelSettings(const QJsonObject &settings,
                                           const QString &depthMapSourcePath);

} // namespace xjw::gui::project
```

- [ ] **Step 2: Add the policy implementation**

Create `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.cpp`:

```cpp
#include "ProjectModelWorkflowPolicy.h"

#include "DepthFrameUtils.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::gui::project
{

namespace
{

QString modelQualityToDenseProfile(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra") || normalized == QStringLiteral("high"))
    {
        return QStringLiteral("high_quality");
    }
    if (normalized == QStringLiteral("low"))
    {
        return QStringLiteral("fast_preview");
    }
    return QStringLiteral("standard");
}

QString modelQualityToDenseResolutionScale(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra"))
    {
        return QStringLiteral("1.0");
    }
    if (normalized == QStringLiteral("low"))
    {
        return QStringLiteral("0.25");
    }
    return QStringLiteral("0.5");
}

int modelQualityToDenseIterations(const QString &quality)
{
    const QString normalized = quality.trimmed().toLower();
    if (normalized == QStringLiteral("ultra"))
    {
        return 10;
    }
    if (normalized == QStringLiteral("high"))
    {
        return 8;
    }
    if (normalized == QStringLiteral("low"))
    {
        return 4;
    }
    return 6;
}

QString depthSourcePathFromSettings(const QJsonObject &settings)
{
    const QString explicitPath = settings.value(QStringLiteral("depthMapSourcePath")).toString().trimmed();
    if (!explicitPath.isEmpty())
    {
        return explicitPath;
    }
    return settings.value(QStringLiteral("source_path")).toString().trimmed();
}

QString reusableDenseCloudForDepthSource(const QString &depthMapSourcePath)
{
    if (depthMapSourcePath.trimmed().isEmpty())
    {
        return QString();
    }

    const QFileInfo info(depthMapSourcePath);
    const QString dirPath = info.isDir() ? info.absoluteFilePath() : info.absolutePath();
    const QString denseCloudPath = QDir(dirPath).filePath(QStringLiteral("dense_cloud.ply"));
    return QFileInfo::exists(denseCloudPath) ? denseCloudPath : QString();
}

} // namespace

QJsonObject denseSettingsFromModelSettings(const QJsonObject &settings,
                                           const QString &depthMapSourcePath)
{
    const QString quality = settings.value(QStringLiteral("quality")).toString(QStringLiteral("high"));
    const int threads = qMax(1, settings.value(QStringLiteral("threads")).toInt(8));
    const bool cuda = settings.value(QStringLiteral("cuda")).toBool(true);

    QJsonObject denseSettings;
    denseSettings[QStringLiteral("pipeline_mode")] = true;
    denseSettings[QStringLiteral("output_dir")] = depthMapSourcePath;
    denseSettings[QStringLiteral("qualityProfile")] = modelQualityToDenseProfile(quality);
    denseSettings[QStringLiteral("resScale")] = modelQualityToDenseResolutionScale(quality).toDouble();
    denseSettings[QStringLiteral("iterations")] = modelQualityToDenseIterations(quality);
    denseSettings[QStringLiteral("threads")] = threads;
    denseSettings[QStringLiteral("cuda")] = cuda;
    denseSettings[QStringLiteral("keepColor")] = settings.value(QStringLiteral("calculateVertexColors")).toBool(true);
    denseSettings[QStringLiteral("keepNormals")] = true;
    denseSettings[QStringLiteral("minViews")] = settings.value(QStringLiteral("minViews")).toInt(6);
    denseSettings[QStringLiteral("minConsistentViews")] =
        settings.value(QStringLiteral("minConsistentViews")).toInt(3);
    denseSettings[QStringLiteral("confidence")] =
        settings.value(QStringLiteral("depthPatchConfidence")).toDouble(0.60);
    denseSettings[QStringLiteral("minConfidence")] =
        settings.value(QStringLiteral("depthFusionConfidence")).toDouble(0.65);
    denseSettings[QStringLiteral("fusionMaxImageDim")] =
        settings.value(QStringLiteral("fusionMaxImageDim")).toInt(2048);
    denseSettings[QStringLiteral("geomConsistency")] =
        settings.value(QStringLiteral("geomConsistency")).toBool(true);
    denseSettings[QStringLiteral("maxReprojError")] =
        settings.value(QStringLiteral("maxReprojError")).toDouble(1.5);
    denseSettings[QStringLiteral("depthConsistency")] =
        settings.value(QStringLiteral("depthConsistency")).toDouble(1.5);
    denseSettings[QStringLiteral("speckleMinArea")] =
        settings.value(QStringLiteral("speckleMinArea")).toInt(16);
    return denseSettings;
}

ModelWorkflowDecision decideModelGenerationWorkflow(const QJsonObject &settings,
                                                    const QJsonObject &projectMeta)
{
    ModelWorkflowDecision decision;
    decision.modelSettings = settings;

    const QString sourceData =
        settings.value(QStringLiteral("source_data")).toString(QStringLiteral("point_cloud"));
    if (sourceData != QStringLiteral("depth_maps"))
    {
        decision.action = ModelWorkflowAction::RunMeshDirectly;
        decision.reason = QStringLiteral("源数据不是深度图，直接执行模型生成。");
        return decision;
    }

    decision.depthMapSourcePath = depthSourcePathFromSettings(settings);
    decision.modelSettings[QStringLiteral("depthMapSourcePath")] = decision.depthMapSourcePath;

    const QString denseCloudPath = reusableDenseCloudForDepthSource(decision.depthMapSourcePath);
    if (!denseCloudPath.isEmpty())
    {
        decision.action = ModelWorkflowAction::RunMeshDirectly;
        decision.reusableDenseCloudPath = denseCloudPath;
        decision.modelSettings[QStringLiteral("source_point_cloud_path")] = denseCloudPath;
        decision.reason = QStringLiteral("深度图目录已有可复用 dense_cloud.ply。");
        return decision;
    }

    const bool reuseDepthMaps = settings.value(QStringLiteral("reuseDepthMaps")).toBool(true);
    const auto storedFramesResult = xjw::core::project::collectLatestStoredDepthFrames(projectMeta);
    if (reuseDepthMaps && storedFramesResult.status.ok && storedFramesResult.frames.size() >= 2)
    {
        decision.action = ModelWorkflowAction::FuseDepthMapsThenMesh;
        decision.denseSettings = denseSettingsFromModelSettings(settings, decision.depthMapSourcePath);
        decision.reason = QStringLiteral("已有可复用深度图，先自动融合为密集点云。");
        return decision;
    }

    decision.action = ModelWorkflowAction::GenerateDenseCloudThenMesh;
    decision.denseSettings = denseSettingsFromModelSettings(settings, decision.depthMapSourcePath);
    decision.reason = QStringLiteral("缺少可复用深度图，先自动估计深度图并融合。");
    return decision;
}

} // namespace xjw::gui::project
```

- [ ] **Step 3: Add the policy source to the build**

Add `ProjectModelWorkflowPolicy.cpp` and `ProjectModelWorkflowPolicy.h` to the same GUI/support source list that already compiles `ProjectDenseWorkflowConfig.cpp`.

If the relevant list is in `E:/code/plascan/src/gui/CMakeLists.txt`, add:

```cmake
    project/support/ProjectModelWorkflowPolicy.cpp
    project/support/ProjectModelWorkflowPolicy.h
```

- [ ] **Step 4: Run policy tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_gui_project_utils --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ModelWorkflowPolicyTest.*
```

Expected: all `ModelWorkflowPolicyTest.*` tests pass.

- [ ] **Step 5: Commit**

```powershell
git add E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.h `
        E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.cpp `
        E:/code/plascan/src/gui/CMakeLists.txt `
        E:/code/plascan/tests/test_gui_project_utils.cpp
git commit -m "feat: add model generation workflow policy"
```

---

### Task 3: Align Depth Map Mesh Builder With Real Artifacts

**Files:**
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`
- Modify: `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`
- Modify: `E:/code/plascan/tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Update tests from `.raw` to current `.bin` artifact names**

In `E:/code/plascan/tests/test_mesh_reconstructor.cpp`, update `DepthMapMeshBuilderTest.DiscoversDepthFramesFromOutputDirectory` so it creates these files:

```cpp
const auto depth0 = root / "depth_0.bin";
const auto conf0 = root / "depth_0_conf.bin";
const auto preview0 = root / "depth_0.png";
const auto mask0 = root / "depth_0_mask.png";
const auto depth1 = root / "depth_1.bin";
const auto conf1 = root / "depth_1_conf.bin";
const auto preview1 = root / "depth_1.png";
const auto mask1 = root / "depth_1_mask.png";
for (const auto &path : {depth0, conf0, preview0, mask0, depth1, conf1, preview1, mask1})
{
    std::ofstream(path.string()) << "x";
}
```

Assert:

```cpp
ASSERT_EQ(frames.size(), 2);
EXPECT_TRUE(frames[0].depthPath.endsWith(QStringLiteral("depth_0.bin")));
EXPECT_TRUE(frames[0].confidencePath.endsWith(QStringLiteral("depth_0_conf.bin")));
EXPECT_TRUE(frames[0].previewPath.endsWith(QStringLiteral("depth_0.png")));
EXPECT_TRUE(frames[0].validMaskPath.endsWith(QStringLiteral("depth_0_mask.png")));
```

- [ ] **Step 2: Run the updated mesh tests and verify failure**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.*
```

Expected: failure until `DepthMapMeshBuilder` discovers `.bin` artifacts.

- [ ] **Step 3: Extend `DepthFrameArtifact`**

Update `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`:

```cpp
struct DepthFrameArtifact
{
    QString depthPath;
    QString confidencePath;
    QString previewPath;
    QString validMaskPath;
};
```

- [ ] **Step 4: Discover current depth artifacts**

Update `DepthMapMeshBuilder::discoverDepthFrames()` in `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`:

```cpp
QVector<DepthFrameArtifact> DepthMapMeshBuilder::discoverDepthFrames(const QString &sourcePath)
{
    const QFileInfo sourceInfo(sourcePath);
    const QDir dir(sourceInfo.isDir() ? sourceInfo.absoluteFilePath() : sourceInfo.absolutePath());
    QVector<DepthFrameArtifact> frames;

    const QStringList depthFiles =
        dir.entryList(QStringList{QStringLiteral("depth_*.bin")}, QDir::Files, QDir::Name);
    for (const QString &depthFile : depthFiles)
    {
        if (depthFile.endsWith(QStringLiteral("_conf.bin")))
        {
            continue;
        }

        const QString stem = QFileInfo(depthFile).completeBaseName();
        DepthFrameArtifact frame;
        frame.depthPath = dir.filePath(depthFile);
        frame.confidencePath = dir.filePath(stem + QStringLiteral("_conf.bin"));
        frame.previewPath = dir.filePath(stem + QStringLiteral(".png"));
        frame.validMaskPath = dir.filePath(stem + QStringLiteral("_mask.png"));

        if (QFileInfo::exists(frame.depthPath)
            && QFileInfo::exists(frame.confidencePath)
            && QFileInfo::exists(frame.previewPath))
        {
            frames.push_back(frame);
        }
    }

    return frames;
}
```

- [ ] **Step 5: Allow an explicit reusable dense cloud path**

In `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`, add:

```cpp
QString reusableDenseCloudPath;
```

to `DepthMapMeshBuildRequest`.

In `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`, update `buildMeshFromDepthMaps()` dense cloud resolution:

```cpp
QString densePath = request.reusableDenseCloudPath.trimmed();
if (densePath.isEmpty())
{
    QString resolveError;
    densePath = xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(
        request.depthMapSourcePath, &resolveError);
}
```

- [ ] **Step 6: Run mesh tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.*:MeshWorkflowSettingsTest.*
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```powershell
git add E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h `
        E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp `
        E:/code/plascan/src/core/mesh/ModelWorkflowService.h `
        E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp `
        E:/code/plascan/tests/test_mesh_reconstructor.cpp
git commit -m "fix: align depth-map model source with saved artifacts"
```

---

### Task 4: Add Model Generation Workflow Orchestrator

**Files:**
- Create: `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.h`
- Create: `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.cpp`
- Modify: `E:/code/plascan/src/gui/CMakeLists.txt`
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Add source contract test for the orchestrator**

Add this test to `E:/code/plascan/tests/test_source_contracts.cpp`:

```cpp
TEST(GuiAlgorithmAlignmentContractTest, GenerateModelUsesMetashapeStyleWorkflowOrchestrator)
{
    const QString header = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.h"));
    const QString source = readSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectReconstructionManager.cpp"));
    ASSERT_FALSE(header.isEmpty());
    ASSERT_FALSE(source.isEmpty());

    expectContainsAll(header, {
        "class ProjectModelGenerationWorkflow;",
        "ProjectModelGenerationWorkflow *_modelWorkflow = nullptr;"
    });
    expectContainsAll(source, {
        "#include \"ProjectModelGenerationWorkflow.h\"",
        "_modelWorkflow(new ProjectModelGenerationWorkflow(",
        "case Task::MeshReconstruction:",
        "_modelWorkflow->start(settings);"
    });

    const int taskStart = source.indexOf(QStringLiteral("void ProjectReconstructionManager::startTask"));
    ASSERT_GE(taskStart, 0);
    const QString taskBlock = source.mid(taskStart);
    EXPECT_FALSE(taskBlock.contains(QStringLiteral("_modelManager->startMeshReconstructionAsync(settings);")))
        << "Generate Model must use the workflow orchestrator so depth maps can be generated automatically.";
}
```

- [ ] **Step 2: Run the contract test and verify failure**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelUsesMetashapeStyleWorkflowOrchestrator
```

Expected: failure until the new orchestrator is wired.

- [ ] **Step 3: Create orchestrator header**

Create `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.h`:

```cpp
#pragma once

#include <QObject>
#include <QJsonObject>
#include <QPointer>
#include <QString>

class ProjectData;
class ProjectDenseReconstructionManager;
class ProjectManager;
class ProjectModelManager;
class QWidget;

class ProjectModelGenerationWorkflow : public QObject
{
    Q_OBJECT

public:
    explicit ProjectModelGenerationWorkflow(ProjectManager *owner,
                                            ProjectData *projectData,
                                            QWidget *parentWidget,
                                            ProjectDenseReconstructionManager *denseManager,
                                            ProjectModelManager *modelManager,
                                            QObject *parent = nullptr);

    void start(const QJsonObject &settings);
    bool isRunning() const;

signals:
    void meshProgressChanged(const QString &stage, int percent);
    void meshProgressFinished(bool success);

private:
    void startModelStage(const QJsonObject &settings);
    void startDenseStage(const QJsonObject &denseSettings,
                         const QJsonObject &modelSettings);
    void clearActiveRun();

    ProjectManager *_owner = nullptr;
    ProjectData *_projectData = nullptr;
    QWidget *_parentWidget = nullptr;
    ProjectDenseReconstructionManager *_denseManager = nullptr;
    ProjectModelManager *_modelManager = nullptr;

    bool _running = false;
    QString _projectPath;
    QJsonObject _pendingModelSettings;
};
```

- [ ] **Step 4: Create orchestrator implementation**

Create `E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.cpp`:

```cpp
#include "ProjectModelGenerationWorkflow.h"

#include "ProjectDenseReconstructionManager.h"
#include "ProjectModelManager.h"
#include "ProjectModelWorkflowPolicy.h"
#include "ProjectData.h"
#include "ProjectManager.h"

#include <QMessageBox>

ProjectModelGenerationWorkflow::ProjectModelGenerationWorkflow(
    ProjectManager *owner,
    ProjectData *projectData,
    QWidget *parentWidget,
    ProjectDenseReconstructionManager *denseManager,
    ProjectModelManager *modelManager,
    QObject *parent)
    : QObject(parent)
    , _owner(owner)
    , _projectData(projectData)
    , _parentWidget(parentWidget)
    , _denseManager(denseManager)
    , _modelManager(modelManager)
{
    connect(_modelManager, &ProjectModelManager::meshProgressChanged,
            this, &ProjectModelGenerationWorkflow::meshProgressChanged);
    connect(_modelManager, &ProjectModelManager::meshProgressFinished,
            this, [this](bool success)
    {
        if (_running)
        {
            clearActiveRun();
        }
        emit meshProgressFinished(success);
    });

    connect(_denseManager, &ProjectDenseReconstructionManager::denseCloudResultReady,
            this, [this](const QString &denseCloudPath, int)
    {
        if (!_running)
        {
            return;
        }

        QJsonObject modelSettings = _pendingModelSettings;
        modelSettings[QStringLiteral("source_point_cloud_path")] = denseCloudPath;
        modelSettings[QStringLiteral("source_path")] =
            modelSettings.value(QStringLiteral("depthMapSourcePath")).toString();
        modelSettings[QStringLiteral("pipeline_mode")] = true;
        startModelStage(modelSettings);
    });

    connect(_denseManager, &ProjectDenseReconstructionManager::mvsProgressFinished,
            this, [this](bool success)
    {
        if (!_running)
        {
            return;
        }
        if (!success)
        {
            clearActiveRun();
            emit meshProgressFinished(false);
        }
    });
}

bool ProjectModelGenerationWorkflow::isRunning() const
{
    return _running;
}

void ProjectModelGenerationWorkflow::start(const QJsonObject &settings)
{
    if (!_owner || !_projectData || !_denseManager || !_modelManager)
    {
        QMessageBox::warning(_parentWidget,
                             QStringLiteral("生成模型"),
                             QStringLiteral("模型生成工作流未正确初始化。"));
        return;
    }

    const QJsonObject meta = _projectData->metadata();
    const auto decision = xjw::gui::project::decideModelGenerationWorkflow(settings, meta);

    if (decision.action == xjw::gui::project::ModelWorkflowAction::RunMeshDirectly)
    {
        startModelStage(decision.modelSettings);
        return;
    }

    startDenseStage(decision.denseSettings, decision.modelSettings);
}

void ProjectModelGenerationWorkflow::startModelStage(const QJsonObject &settings)
{
    _modelManager->startMeshReconstructionAsync(settings);
}

void ProjectModelGenerationWorkflow::startDenseStage(const QJsonObject &denseSettings,
                                                     const QJsonObject &modelSettings)
{
    _running = true;
    _pendingModelSettings = modelSettings;
    _projectPath = _owner ? _owner->currentProjectPath() : QString();
    emit meshProgressChanged(QStringLiteral("正在准备深度图..."), 0);

    const QString action = denseSettings.value(QStringLiteral("workflow_action")).toString();
    if (action == QStringLiteral("fuse_existing_depth_maps"))
    {
        _denseManager->startFuseDepthMapsAsync(denseSettings);
        return;
    }

    _denseManager->startGenerateDenseCloudAsync(denseSettings);
}

void ProjectModelGenerationWorkflow::clearActiveRun()
{
    _running = false;
    _projectPath.clear();
    _pendingModelSettings = QJsonObject();
}
```

- [ ] **Step 5: Add action marker in policy dense settings**

In `E:/code/plascan/src/gui/project/support/ProjectModelWorkflowPolicy.cpp`, set:

```cpp
decision.denseSettings[QStringLiteral("workflow_action")] =
    QStringLiteral("fuse_existing_depth_maps");
```

for `FuseDepthMapsThenMesh`, and:

```cpp
decision.denseSettings[QStringLiteral("workflow_action")] =
    QStringLiteral("generate_dense_cloud");
```

for `GenerateDenseCloudThenMesh`.

- [ ] **Step 6: Add orchestrator sources to CMake**

Add to the GUI source list:

```cmake
    project/manager/ProjectModelGenerationWorkflow.cpp
    project/manager/ProjectModelGenerationWorkflow.h
```

- [ ] **Step 7: Run contract test**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelUsesMetashapeStyleWorkflowOrchestrator
```

Expected: failure until Task 5 wires `ProjectReconstructionManager`.

---

### Task 5: Wire ProjectReconstructionManager Through the Workflow

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.h`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.cpp`

- [ ] **Step 1: Add orchestrator member**

Modify `ProjectReconstructionManager.h`:

```cpp
class ProjectModelGenerationWorkflow;
```

and add:

```cpp
ProjectModelGenerationWorkflow *_modelWorkflow = nullptr;
```

- [ ] **Step 2: Construct and connect orchestrator**

Modify `ProjectReconstructionManager.cpp`:

```cpp
#include "ProjectModelGenerationWorkflow.h"
```

In the constructor initializer list:

```cpp
, _modelWorkflow(new ProjectModelGenerationWorkflow(owner,
                                                    projectData,
                                                    parentWidget,
                                                    _denseManager,
                                                    _modelManager,
                                                    this))
```

After existing model manager connections, add:

```cpp
connect(_modelWorkflow, &ProjectModelGenerationWorkflow::meshProgressChanged,
        this, &ProjectReconstructionManager::meshProgressChanged);
connect(_modelWorkflow, &ProjectModelGenerationWorkflow::meshProgressFinished,
        this, &ProjectReconstructionManager::meshProgressFinished);
```

- [ ] **Step 3: Route model generation through workflow**

Modify `ProjectReconstructionManager::startTask()`:

```cpp
case Task::MeshReconstruction:
    _modelWorkflow->start(settings);
    break;
```

Keep `Task::GenerateModel` as:

```cpp
case Task::GenerateModel:
    _modelManager->startGenerateModelAsync();
    break;
```

because that legacy toolbar action currently supplies no Metashape dialog settings.

- [ ] **Step 4: Run contract tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelUsesMetashapeStyleWorkflowOrchestrator
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.h `
        E:/code/plascan/src/gui/project/manager/ProjectModelGenerationWorkflow.cpp `
        E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.h `
        E:/code/plascan/src/gui/project/manager/ProjectReconstructionManager.cpp `
        E:/code/plascan/src/gui/CMakeLists.txt `
        E:/code/plascan/tests/test_source_contracts.cpp
git commit -m "feat: orchestrate model generation from depth maps"
```

---

### Task 6: Ensure Dense Pipeline Emits the Right Continuation Signal

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add regression test for dense cloud signal in pipeline path**

Add this source-level regression test to `E:/code/plascan/tests/test_gui_project_utils.cpp`:

```cpp
TEST(ModelWorkflowContractTest, GenerateDenseCloudPipelineEmitsDenseCloudReadyForModelWorkflow)
{
    const QString source = readProjectSourceFile(
        QStringLiteral("src/gui/project/manager/ProjectDenseReconstructionManager.cpp"));
    ASSERT_FALSE(source.isEmpty());

    const int start = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startGenerateDenseCloudAsync"));
    ASSERT_GE(start, 0);
    const int end = source.indexOf(
        QStringLiteral("void ProjectDenseReconstructionManager::startDenseCloudRefineAsync"), start);
    ASSERT_GT(end, start);
    const QString block = source.mid(start, end - start);

    EXPECT_TRUE(block.contains(QStringLiteral("emit self->denseCloudResultReady(")))
        << "Metashape-style Generate Model must be able to continue when dense cloud generation finishes.";
    EXPECT_TRUE(block.contains(QStringLiteral("pipelineMode")))
        << "Pipeline mode must suppress intermediate modal success dialogs.";
}
```

- [ ] **Step 2: Run the regression test**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_gui_project_utils --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ModelWorkflowContractTest.GenerateDenseCloudPipelineEmitsDenseCloudReadyForModelWorkflow
```

Expected: pass if existing dense generation already emits `denseCloudResultReady`; otherwise fail.

- [ ] **Step 3: Patch missing signal emission only if the test fails**

In the dense cloud generation completion path that writes `dense_cloud.ply`, ensure:

```cpp
emit self->denseCloudResultReady(outputPly, pointCount);
```

is emitted after the dense cloud record is persisted.

Do not emit this signal for failed, cancelled, or empty-cloud results.

- [ ] **Step 4: Run the regression test again**

Run:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ModelWorkflowContractTest.GenerateDenseCloudPipelineEmitsDenseCloudReadyForModelWorkflow
```

Expected: pass.

- [ ] **Step 5: Commit**

```powershell
git add E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp `
        E:/code/plascan/tests/test_gui_project_utils.cpp
git commit -m "fix: continue model workflow after dense cloud generation"
```

---

### Task 7: Improve Generate Model Dialog UX for Automatic Depth Maps

**Files:**
- Modify: `E:/code/plascan/src/gui/dialogs/GenerateModelDialog.cpp`
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Add source contract test for UI text**

Add to `E:/code/plascan/tests/test_source_contracts.cpp`:

```cpp
TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDialogExplainsAutomaticDepthMapGeneration)
{
    const QString source = readSourceFile(QStringLiteral("src/gui/dialogs/GenerateModelDialog.cpp"));
    ASSERT_FALSE(source.isEmpty());

    expectContainsAll(source, {
        "缺少深度图时将自动估计深度图",
        "重用深度图",
        "settings[QStringLiteral(\"reuseDepthMaps\")]",
        "settings[QStringLiteral(\"depthMapSourcePath\")] = sourcePath"
    });
}
```

- [ ] **Step 2: Run the contract test and verify failure**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelDialogExplainsAutomaticDepthMapGeneration
```

Expected: failure until the dialog contains the explanatory text.

- [ ] **Step 3: Update dialog status text**

In `GenerateModelDialog::updateStatus()` or the equivalent source-change update method, add this depth-map branch:

```cpp
if (sourceData == QStringLiteral("depth_maps"))
{
    _statusLabel->setText(tr("源数据为深度图：缺少深度图时将自动估计深度图；"
                             "已有兼容深度图时将按“重用深度图”设置复用，并自动融合后生成模型。"));
}
```

Keep existing source support checks and OK button behavior.

- [ ] **Step 4: Keep reuse enabled by default**

Ensure constructor default remains:

```cpp
_reuseDepthMapsCheck->setChecked(true);
```

If `applySettings()` reads older settings without `reuseDepthMaps`, use true as the default:

```cpp
_reuseDepthMapsCheck->setChecked(settings.value(QStringLiteral("reuseDepthMaps")).toBool(true));
```

- [ ] **Step 5: Run dialog contract test**

Run:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelDialogExplainsAutomaticDepthMapGeneration
```

Expected: pass.

- [ ] **Step 6: Commit**

```powershell
git add E:/code/plascan/src/gui/dialogs/GenerateModelDialog.cpp `
        E:/code/plascan/tests/test_source_contracts.cpp
git commit -m "ux: explain automatic depth maps in model dialog"
```

---

### Task 8: End-to-End Verification on Small and Full Projects

**Files:**
- No code changes unless failures expose defects.

- [ ] **Step 1: Build the GUI and focused tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target plascan_gui test_gui_project_utils test_source_contracts test_mesh_reconstructor --config Release -j 8
```

Expected: build succeeds.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_gui_project_utils.exe --gtest_filter=ModelWorkflowPolicyTest.*:ModelWorkflowContractTest.*
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModel*
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.*:MeshWorkflowSettingsTest.*
```

Expected: all focused tests pass.

- [ ] **Step 3: Manual GUI test with existing depth maps**

Use project:

```text
E:/code/test/agisoft_aerial_gcps/agisoft_aerial_gcps.plascan
```

Manual checks:

1. Open `工作流程 -> 生成模型`.
2. Select `源数据 = 深度图`.
3. Keep `重用深度图` checked.
4. Click `OK`.
5. If `mvs_output/dense_cloud.ply` exists, expected result is direct model generation.
6. If only depth maps exist, expected result is automatic fusion, then model generation.

Expected UI behavior:

- No manual `深度图估计` click is required.
- Progress label mentions depth preparation/fusion/model generation.
- No intermediate success modal appears before final model generation in pipeline mode.
- `3D模型` tree item refreshes after final model generation.

- [ ] **Step 4: Manual GUI test with missing depth maps**

Use the 9-image test project:

```text
E:/code/test/agisoft_aerial_gcps_small_9_cli
```

Manual checks:

1. Delete or move only that project output directory's old `mvs_output/depth_*.png`, `depth_*.bin`, and `dense_cloud.ply`.
2. Open project in PlaScan.
3. Run `工作流程 -> 生成模型`.
4. Select `源数据 = 深度图`.
5. Keep `重用深度图` checked.
6. Click `OK`.

Expected workflow:

```text
生成模型
-> 自动深度图估计
-> 自动深度图融合
-> 自动网格生成
-> 3D模型 metadata/tree refresh
```

Expected artifacts:

```text
<project-output>/mvs_output/depth_0.png
<project-output>/mvs_output/depth_0.bin
<project-output>/mvs_output/depth_0_conf.bin
<project-output>/mvs_output/dense_cloud.ply
<project-output>/products/model_from_mesh.ply
```

- [ ] **Step 5: Run a broader regression subset**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "Mvs|DepthFrame|DenseCloud|Mesh|Gui|Source" --output-on-failure
```

Expected: no new failures. If the historical `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` failure appears during broader testing, document it separately and do not mark full test suite as clean.

- [ ] **Step 6: Record verification result**

Update the implementation notes or final response with:

```text
Verified:
- cmake --build ... --target plascan_gui test_gui_project_utils test_source_contracts test_mesh_reconstructor --config Release -j 8
- test_gui_project_utils.exe --gtest_filter=ModelWorkflowPolicyTest.*:ModelWorkflowContractTest.*
- test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModel*
- test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.*:MeshWorkflowSettingsTest.*

Manual GUI:
- Existing depth maps: reused/fused automatically
- Missing depth maps: generated/fused/meshed automatically
```

---

## Implementation Notes

- Do not move depth estimation into `ModelWorkflowService`; that core mesh service should not depend on GUI managers or project UI.
- Do not make `GenerateModelDialog` run algorithms. It only emits settings.
- Do not scan the output directory as the source of truth when project metadata has compatible `depth_map_results`; use metadata first.
- Avoid accepting old dense clouds accidentally. Continuation after automatic MVS should use `denseCloudResultReady(path, pointCount)` from the current run.
- Keep standalone `深度图估计` and `深度图融合` behavior unchanged.
- Keep cancellation using existing `ProjectDenseReconstructionManager::cancelMvs()`. The first implementation does not need a separate model-workflow cancel button.

## Self-Review

- **Spec coverage:** The plan covers automatic depth-map generation, depth-map reuse, fusion before mesh, dialog status, metadata decisions, and verification.
- **Placeholder scan:** No placeholder task is left; each implementation task includes concrete files, code shape, commands, and expected results.
- **Type consistency:** `ModelWorkflowAction`, `ModelWorkflowDecision`, `decideModelGenerationWorkflow()`, `ProjectModelGenerationWorkflow`, and `denseSettingsFromModelSettings()` are introduced before use.
- **Scope:** The plan intentionally does not redesign depth-map algorithms or mesh reconstruction algorithms; it only changes workflow orchestration to match Metashape behavior.
