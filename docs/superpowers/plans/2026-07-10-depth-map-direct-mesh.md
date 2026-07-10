# Depth Map Direct Mesh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the workflow `生成模型` support a Metashape-like path where `深度图` is a first-class model source and can generate a mesh through reusable depth maps, streaming fusion, block processing, and real parameter mapping.

**Architecture:** Keep the existing GUI entry and `ProjectModelManager` orchestration, but move depth-map-to-mesh production into core mesh/MVS services. The first production implementation should not rebuild depth maps; it should reuse existing depth map metadata, stream-fuse depth maps into point tiles or a dense cloud, then call the same mesh builder with surface-type-aware settings.

**Tech Stack:** C++17, Qt6, OpenCV, PlaPoint, existing `src/core/mvs`, existing `src/core/mesh`, GTest, Windows CUDA build directory `E:/code/plascan/build/windows-vcpkg-cuda-release`.

---

## File Structure

- Modify `E:/code/plascan/src/gui/dialogs/GenerateModelDialog.cpp`
  - UI behavior for depth map source, quality, interpolation, filtering, block controls, and emitted JSON.
- Modify `E:/code/plascan/src/gui/main_window/ReconstructionWorkflowController.cpp`
  - Source candidate ordering and metadata passed to the dialog.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectModelManager.cpp`
  - Resolve `depth_maps` source to the new direct depth-map mesh request instead of only looking for `dense_cloud.ply`.
- Modify `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`
- Modify `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`
  - Add a public workflow API for depth-map source meshing.
- Create `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
- Create `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
  - Own depth map artifact discovery, streaming fusion, block dispatch, and mesh request assembly.
- Modify `E:/code/plascan/src/core/mesh/CMakeLists.txt`
  - Add the new builder.
- Modify `E:/code/plascan/src/core/mesh/MeshTypes.h`
  - Add only stable config fields needed by depth-map meshing if current config is insufficient.
- Modify `E:/code/plascan/src/core/mvs/DepthFrameUtils.h`
- Modify `E:/code/plascan/src/core/mvs/DepthFrameUtils.cpp`
  - Expose a small loader contract if the current helpers cannot enumerate depth frames by metadata/output directory.
- Modify `E:/code/plascan/tests/test_mesh_reconstructor.cpp`
  - Unit tests for mesh settings and builder behavior.
- Modify `E:/code/plascan/tests/test_source_contracts.cpp`
  - Contract tests for GUI/workflow source behavior.
- Modify `E:/code/plascan/tests/CMakeLists.txt`
  - Add any new focused test target if direct unit testing inside `test_mesh_reconstructor` becomes too large.

---

### Task 1: Lock the Direct Depth-Map Source Contract

**Files:**
- Modify: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Write the failing contract test**

Add a test under `GuiAlgorithmAlignmentContractTest`:

```cpp
TEST(GuiAlgorithmAlignmentContractTest, GenerateModelDepthMapsUseDirectMeshWorkflow)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/GenerateModelDialog.cpp"));
    const QString manager = readSourceFile(QStringLiteral("src/gui/project/manager/ProjectModelManager.cpp"));
    const QString workflow = readSourceFile(QStringLiteral("src/core/mesh/ModelWorkflowService.cpp"));

    expectContainsAll(dialog, {
        R"(settings[QStringLiteral("depthMapSourcePath")] = sourcePath)",
        R"(settings[QStringLiteral("reuseDepthMaps")] = _reuseDepthMapsCheck->isChecked())",
    });

    expectContainsAll(manager, {
        "buildMeshFromDepthMaps",
        "DepthMapMeshBuildRequest",
        "settings.value(QStringLiteral(\"source_data\")).toString(QStringLiteral(\"point_cloud\"))",
    });

    expectContainsAll(workflow, {
        "WorkflowResult buildMeshFromDepthMaps",
        "DepthMapMeshBuilder",
        "request.depthMapSourcePath",
    });

    expectNotContainsAll(manager, {
        "深度图源需要先融合为密集点云，但未找到可复用的 dense_cloud.ply",
    });
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelDepthMapsUseDirectMeshWorkflow
```

Expected: FAIL because `buildMeshFromDepthMaps`, `DepthMapMeshBuildRequest`, and `DepthMapMeshBuilder` do not exist yet.

- [ ] **Step 3: Commit after the test is red**

```powershell
git add E:/code/plascan/tests/test_source_contracts.cpp
git commit -m "test: require direct depth map mesh workflow"
```

---

### Task 2: Add Depth Map Mesh Request Types

**Files:**
- Modify: `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`
- Modify: `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`

- [ ] **Step 1: Write the failing unit test**

In `E:/code/plascan/tests/test_mesh_reconstructor.cpp`, add:

```cpp
TEST(MeshWorkflowSettingsTest, DepthMapMeshRequestPreservesSourceAndSettings)
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QStringLiteral("E:/tmp/mvs_output");
    request.outputRoot = QStringLiteral("E:/tmp/model");
    request.settings[QStringLiteral("surface_type")] = QStringLiteral("height_field");
    request.settings[QStringLiteral("quality")] = QStringLiteral("high");
    request.settings[QStringLiteral("reuseDepthMaps")] = true;

    EXPECT_EQ(request.depthMapSourcePath, QStringLiteral("E:/tmp/mvs_output"));
    EXPECT_EQ(request.outputRoot, QStringLiteral("E:/tmp/model"));
    EXPECT_TRUE(request.settings.value(QStringLiteral("reuseDepthMaps")).toBool());
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
```

Expected: compile FAIL because `DepthMapMeshBuildRequest` does not exist.

- [ ] **Step 3: Add the request type**

Add to `E:/code/plascan/src/core/mesh/ModelWorkflowService.h`:

```cpp
struct DepthMapMeshBuildRequest
{
    QString depthMapSourcePath;
    QString outputRoot;
    QJsonObject settings;
    xjw::mesh::ReconstructionConfig reconstruction;
    bool exportObj = false;
    xjw::mesh::TextureMappingConfig texture;
    std::function<void(const QString &, int)> progress;
};

WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request);
```

- [ ] **Step 4: Add a stub implementation**

Add to `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`:

```cpp
WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request)
{
    WorkflowResult result;
    if (request.depthMapSourcePath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("深度图源路径为空");
        return result;
    }
    result.errorMessage = QStringLiteral("深度图直接生成模型尚未实现");
    return result;
}
```

- [ ] **Step 5: Run the test and verify it passes**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=MeshWorkflowSettingsTest.DepthMapMeshRequestPreservesSourceAndSettings
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add E:/code/plascan/src/core/mesh/ModelWorkflowService.h E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp E:/code/plascan/tests/test_mesh_reconstructor.cpp
git commit -m "feat: add depth map mesh workflow request"
```

---

### Task 3: Implement Depth Map Artifact Discovery

**Files:**
- Create: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
- Create: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `E:/code/plascan/src/core/mesh/CMakeLists.txt`
- Test: `E:/code/plascan/tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write the failing discovery test**

Add to `test_mesh_reconstructor.cpp`:

```cpp
TEST(DepthMapMeshBuilderTest, DiscoversDepthFramesFromOutputDirectory)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_discovery_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.raw").put('\0');
    std::ofstream(root / "confidence_001.raw").put('\0');
    std::ofstream(root / "depth_002.raw").put('\0');

    const auto frames = xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(
        QString::fromStdString(root.string()));

    ASSERT_EQ(frames.size(), 2);
    EXPECT_TRUE(frames.at(0).depthPath.endsWith(QStringLiteral("depth_001.raw")));
    EXPECT_TRUE(frames.at(0).confidencePath.endsWith(QStringLiteral("confidence_001.raw")));
    EXPECT_TRUE(frames.at(1).depthPath.endsWith(QStringLiteral("depth_002.raw")));
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
```

Expected: compile FAIL because `DepthMapMeshBuilder` does not exist.

- [ ] **Step 3: Create `DepthMapMeshBuilder.h`**

```cpp
#pragma once

#include <QString>
#include <QVector>

namespace xjw::mesh
{

struct DepthFrameArtifact
{
    QString depthPath;
    QString confidencePath;
    QString previewPath;
};

class DepthMapMeshBuilder
{
public:
    static QVector<DepthFrameArtifact> discoverDepthFrames(const QString &sourcePath);
};

} // namespace xjw::mesh
```

- [ ] **Step 4: Create `DepthMapMeshBuilder.cpp`**

```cpp
#include "DepthMapMeshBuilder.h"

#include <QDir>
#include <QFileInfo>

namespace xjw::mesh
{

QVector<DepthFrameArtifact> DepthMapMeshBuilder::discoverDepthFrames(const QString &sourcePath)
{
    const QFileInfo info(sourcePath);
    const QDir dir(info.isDir() ? info.absoluteFilePath() : info.absolutePath());
    QVector<DepthFrameArtifact> frames;

    const QStringList depthFiles = dir.entryList(QStringList() << QStringLiteral("depth_*.raw"),
                                                 QDir::Files,
                                                 QDir::Name);
    frames.reserve(depthFiles.size());
    for (const QString &fileName : depthFiles)
    {
        const QString suffix = fileName.mid(QStringLiteral("depth_").size());
        DepthFrameArtifact frame;
        frame.depthPath = dir.filePath(fileName);
        const QString confidenceName = QStringLiteral("confidence_%1").arg(suffix);
        if (QFileInfo::exists(dir.filePath(confidenceName)))
        {
            frame.confidencePath = dir.filePath(confidenceName);
        }
        const QString previewName = QStringLiteral("depth_%1.png").arg(QFileInfo(suffix).completeBaseName());
        if (QFileInfo::exists(dir.filePath(previewName)))
        {
            frame.previewPath = dir.filePath(previewName);
        }
        frames.push_back(frame);
    }
    return frames;
}

} // namespace xjw::mesh
```

- [ ] **Step 5: Add to CMake**

Modify `E:/code/plascan/src/core/mesh/CMakeLists.txt` and add:

```cmake
DepthMapMeshBuilder.cpp
```

to the `meshing` source list.

- [ ] **Step 6: Run the test and verify it passes**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.DiscoversDepthFramesFromOutputDirectory
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp E:/code/plascan/src/core/mesh/CMakeLists.txt E:/code/plascan/tests/test_mesh_reconstructor.cpp
git commit -m "feat: discover depth map artifacts for meshing"
```

---

### Task 4: Add Streaming Depth Fusion Adapter

**Files:**
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp`
- Test: `E:/code/plascan/tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write the failing fallback test**

Add:

```cpp
TEST(DepthMapMeshBuilderTest, UsesExistingDenseCloudWhenPresent)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_existing_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path densePath = writeDenseGridPointCloud(root);
    fs::rename(densePath, root / "dense_cloud.ply");

    QString error;
    const QString resolved = xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(
        QString::fromStdString(root.string()),
        &error);

    EXPECT_TRUE(error.isEmpty());
    EXPECT_TRUE(resolved.endsWith(QStringLiteral("dense_cloud.ply")));
}
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
```

Expected: compile FAIL because `resolveReusableDenseCloud` does not exist.

- [ ] **Step 3: Add the reusable dense cloud resolver**

Add to `DepthMapMeshBuilder.h`:

```cpp
static QString resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage = nullptr);
```

Add to `DepthMapMeshBuilder.cpp`:

```cpp
QString DepthMapMeshBuilder::resolveReusableDenseCloud(const QString &sourcePath, QString *errorMessage)
{
    const QFileInfo info(sourcePath);
    const QDir dir(info.isDir() ? info.absoluteFilePath() : info.absolutePath());
    const QString densePath = dir.filePath(QStringLiteral("dense_cloud.ply"));
    if (QFileInfo::exists(densePath))
    {
        return QDir::cleanPath(densePath);
    }
    if (errorMessage)
    {
        *errorMessage = QStringLiteral("未找到可复用的深度图融合点云: %1").arg(densePath);
    }
    return QString();
}
```

- [ ] **Step 4: Wire `buildMeshFromDepthMaps` to reusable dense cloud**

In `ModelWorkflowService.cpp`, replace the stub body with:

```cpp
WorkflowResult buildMeshFromDepthMaps(const DepthMapMeshBuildRequest &request)
{
    WorkflowResult result;
    if (request.depthMapSourcePath.trimmed().isEmpty())
    {
        result.errorMessage = QStringLiteral("深度图源路径为空");
        return result;
    }

    QString resolveError;
    const QString densePath =
        xjw::mesh::DepthMapMeshBuilder::resolveReusableDenseCloud(request.depthMapSourcePath, &resolveError);
    if (densePath.isEmpty())
    {
        result.errorMessage = resolveError;
        return result;
    }

    MeshBuildRequest meshRequest;
    meshRequest.pointCloudPath = densePath;
    meshRequest.outputRoot = request.outputRoot.isEmpty()
        ? QFileInfo(densePath).absolutePath()
        : request.outputRoot;
    meshRequest.reconstruction = request.reconstruction;
    meshRequest.exportObj = request.exportObj;
    meshRequest.texture = request.texture;
    meshRequest.progress = request.progress;
    result = buildMeshAndOptionalTexture(meshRequest);
    if (result.ok)
    {
        result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
        result.payload[QStringLiteral("source_point_cloud_path")] = densePath;
        result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    }
    return result;
}
```

- [ ] **Step 5: Run tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.*:MeshWorkflowSettingsTest.*
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.* E:/code/plascan/src/core/mesh/ModelWorkflowService.* E:/code/plascan/tests/test_mesh_reconstructor.cpp
git commit -m "feat: reuse fused depth cloud for depth map mesh source"
```

---

### Task 5: Route GUI `depth_maps` Through the New Workflow

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectModelManager.cpp`
- Test: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Write the failing source contract**

Extend `GenerateModelDepthMapsUseDirectMeshWorkflow` to require:

```cpp
expectContainsAll(manager, {
    "xjw::mesh::workflow::DepthMapMeshBuildRequest depthRequest",
    "depthRequest.depthMapSourcePath",
    "buildMeshFromDepthMaps(depthRequest)",
});
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelDepthMapsUseDirectMeshWorkflow
```

Expected: FAIL until manager uses `DepthMapMeshBuildRequest`.

- [ ] **Step 3: Modify the model manager worker branch**

In `ProjectModelManager::startMeshReconstructionAsync`, inside the worker lambda, before creating `MeshBuildRequest`, add:

```cpp
const QString sourceData =
    effectiveSettings.value(QStringLiteral("source_data")).toString(QStringLiteral("point_cloud"));
if (sourceData == QStringLiteral("depth_maps"))
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest depthRequest;
    depthRequest.depthMapSourcePath =
        effectiveSettings.value(QStringLiteral("depthMapSourcePath"))
            .toString(effectiveSettings.value(QStringLiteral("source_path")).toString());
    depthRequest.outputRoot = resolvedSource.outputRoot;
    depthRequest.settings = effectiveSettings;
    depthRequest.reconstruction = cfg;
    depthRequest.exportObj = xjw::mesh::workflow::exportObjRequested(effectiveSettings);
    depthRequest.texture = xjw::mesh::workflow::defaultTextureConfig();
    depthRequest.progress = makeProgressReporter(self, ownerGuard, projectPath);

    const xjw::mesh::workflow::WorkflowResult workflowResult =
        xjw::mesh::workflow::buildMeshFromDepthMaps(depthRequest);
    applyWorkflowResult(&task, workflowResult);
    task.result[QStringLiteral("source_path")] = depthRequest.depthMapSourcePath;
    task.result[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
    return task;
}
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts plascan_gui --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelDepthMapsUseDirectMeshWorkflow
```

Expected: PASS and `plascan_gui` links.

- [ ] **Step 5: Commit**

```powershell
git add E:/code/plascan/src/gui/project/manager/ProjectModelManager.cpp E:/code/plascan/tests/test_source_contracts.cpp
git commit -m "feat: route depth map model source through mesh workflow"
```

---

### Task 6: Implement True Streaming Fusion When `dense_cloud.ply` Is Missing

**Files:**
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.h`
- Modify: `E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthFrameUtils.h`
- Modify: `E:/code/plascan/src/core/mvs/DepthFrameUtils.cpp`
- Test: `E:/code/plascan/tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write the failing no-dense-cloud test**

Add:

```cpp
TEST(DepthMapMeshBuilderTest, ReportsActionableErrorWhenDepthFramesCannotBeFused)
{
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "plascan_depth_mesh_no_dense_test";
    fs::remove_all(root);
    fs::create_directories(root);
    std::ofstream(root / "depth_001.raw").put('\0');

    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QString::fromStdString(root.string());
    request.outputRoot = QString::fromStdString(root.string());
    request.reconstruction = fallbackMeshConfig();

    const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);

    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("缺少深度图 metadata")));
}
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.ReportsActionableErrorWhenDepthFramesCannotBeFused
```

Expected: FAIL until `buildMeshFromDepthMaps` returns the actionable error.

- [ ] **Step 3: Add the explicit error path**

In `buildMeshFromDepthMaps`, after reusable dense cloud resolution fails, add:

```cpp
const auto frames = xjw::mesh::DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
if (frames.isEmpty())
{
    result.errorMessage = QStringLiteral("未找到可用于生成模型的深度图文件");
    return result;
}

result.errorMessage = QStringLiteral(
    "缺少深度图 metadata，无法从 raw depth 直接恢复相机、尺寸和尺度；"
    "请重新运行深度图估计或先执行深度图融合。");
return result;
```

- [ ] **Step 4: Run and verify pass**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_mesh_reconstructor --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe --gtest_filter=DepthMapMeshBuilderTest.ReportsActionableErrorWhenDepthFramesCannotBeFused
```

Expected: PASS.

- [ ] **Step 5: Add real fusion as a separate follow-up**

Do not implement ad-hoc raw-depth parsing in this task. Open a follow-up plan item to reuse existing `DepthMapFusion`/`DenseCloudBuilder` metadata types, because correct fusion requires camera pose, intrinsics, image dimensions, depth scale, valid mask, confidence, and source view selection.

- [ ] **Step 6: Commit**

```powershell
git add E:/code/plascan/src/core/mesh/DepthMapMeshBuilder.* E:/code/plascan/src/core/mesh/ModelWorkflowService.cpp E:/code/plascan/tests/test_mesh_reconstructor.cpp
git commit -m "fix: report missing metadata for direct depth map meshing"
```

---

### Task 7: Make Block Controls Honest

**Files:**
- Modify: `E:/code/plascan/src/gui/dialogs/GenerateModelDialog.h`
- Modify: `E:/code/plascan/src/gui/dialogs/GenerateModelDialog.cpp`
- Test: `E:/code/plascan/tests/test_source_contracts.cpp`

- [ ] **Step 1: Write the failing source contract**

Add:

```cpp
TEST(GuiAlgorithmAlignmentContractTest, GenerateModelBlockControlsAreBoundToSettings)
{
    const QString dialog = readSourceFile(QStringLiteral("src/gui/dialogs/GenerateModelDialog.cpp"));

    expectContainsAll(dialog, {
        "_splitRegionCheck",
        "_blockSizeSpin",
        R"(settings[QStringLiteral("splitIntoBlocks")] = _splitRegionCheck->isChecked())",
        R"(settings[QStringLiteral("blockSizeMeters")] = _blockSizeSpin->value())",
        "updateBlockControlsAvailability",
    });
}
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelBlockControlsAreBoundToSettings
```

Expected: FAIL because current block controls are local disabled placeholders.

- [ ] **Step 3: Promote block widgets to members**

Add to `GenerateModelDialog.h`:

```cpp
class QDoubleSpinBox;

QCheckBox *_splitRegionCheck = nullptr;
QDoubleSpinBox *_blockSizeSpin = nullptr;
QCheckBox *_skipBoundaryBlocksCheck = nullptr;
void updateBlockControlsAvailability();
```

- [ ] **Step 4: Bind settings in `GenerateModelDialog.cpp`**

Replace local placeholder widgets with member widgets and write settings:

```cpp
_splitRegionCheck = new QCheckBox(tr("分割成区块"), regionGroup);
_blockSizeSpin = new QDoubleSpinBox(regionGroup);
_blockSizeSpin->setRange(1.0, 100000.0);
_blockSizeSpin->setValue(250.0);
_blockSizeSpin->setSuffix(tr(" m"));
_skipBoundaryBlocksCheck = new QCheckBox(tr("跳过边界外的块"), regionGroup);
```

In `collectSettings()`:

```cpp
settings[QStringLiteral("splitIntoBlocks")] = _splitRegionCheck->isChecked();
settings[QStringLiteral("blockSizeMeters")] = _blockSizeSpin->value();
settings[QStringLiteral("skipBoundaryBlocks")] = _skipBoundaryBlocksCheck->isChecked();
```

Add:

```cpp
void GenerateModelDialog::updateBlockControlsAvailability()
{
    const QString sourceData = _sourceCombo->currentData().toString();
    const bool blockCapable =
        sourceData == QStringLiteral("depth_maps") || sourceData == QStringLiteral("point_cloud");
    _splitRegionCheck->setEnabled(blockCapable);
    _blockSizeSpin->setEnabled(blockCapable && _splitRegionCheck->isChecked());
    _skipBoundaryBlocksCheck->setEnabled(blockCapable && _splitRegionCheck->isChecked());
}
```

- [ ] **Step 5: Run and verify pass**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target test_source_contracts plascan_gui --config Release -j 8
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.GenerateModelBlockControlsAreBoundToSettings
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add E:/code/plascan/src/gui/dialogs/GenerateModelDialog.* E:/code/plascan/tests/test_source_contracts.cpp
git commit -m "feat: bind generate model block controls"
```

---

### Task 8: Verification and Regression Run

**Files:**
- No source changes unless a test fails.

- [ ] **Step 1: Build GUI and mesh tests**

Run:

```powershell
cmake --build E:/code/plascan/build/windows-vcpkg-cuda-release --target plascan_gui test_mesh_reconstructor test_source_contracts --config Release -j 8
```

Expected: exit code 0.

- [ ] **Step 2: Run targeted tests**

Run:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mesh_reconstructor.exe
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_source_contracts.exe --gtest_filter=GuiAlgorithmAlignmentContractTest.*:MeshWorkflowSettingsTest.*
```

Expected:
- `test_mesh_reconstructor.exe`: all tests pass.
- Targeted source contracts pass.

- [ ] **Step 3: Smoke test GUI manually**

Run:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/bin/plascan.exe
```

Manual checks:
- Open a project with existing `depth_map_results`.
- Click `工作流程 -> 生成模型`.
- `源数据` defaults to `深度图` when depth maps exist.
- `重用深度图` is checked.
- `表面类型 / 质量 / 面数 / 插值 / 深度过滤` can be changed.
- If no `dense_cloud.ply` or valid depth metadata exists, the error message says which prerequisite is missing.

- [ ] **Step 4: Record known unrelated failures**

If `test_source_contracts.exe` full run still fails at:

```text
SfmSourceContractTest.UsesAlgorithmAwareFeatureAndMatchPipeline
```

record it as unrelated to this plan unless the current task touched SfM feature extraction code.

- [ ] **Step 5: Commit verification docs only if needed**

If a release note is requested, update:

```text
E:/code/plascan/CHANGELOG.md
E:/code/plascan/docs/releases/vX.Y.Z.md
```

Commit:

```powershell
git add E:/code/plascan/CHANGELOG.md E:/code/plascan/docs/releases
git commit -m "docs: document depth map model generation workflow"
```

---

## Follow-Up Phase: True Direct Raw Depth Fusion

The plan above intentionally makes `深度图` a first-class source without guessing raw depth format. A full Metashape-like implementation needs a second plan after confirming current MVS metadata fields:

- Create a metadata loader for each depth frame: reference image, camera, intrinsics, image size, depth scale, confidence path, mask path.
- Stream depth pixels into 3D blocks using camera projection.
- Apply min confidence, min consistent views, reprojection consistency, and speckle filtering.
- Emit temporary block PLYs with color/confidence/source count.
- Mesh each block independently.
- Merge or register block mesh products into project metadata.

This should not be mixed into the first implementation because wrong depth scaling or camera loading will produce bad geometry while appearing to “work”.

---

## Self-Review

- **Spec coverage:** The plan covers Metashape-style `深度图` model source, dialog behavior, real parameter mapping, block controls, and backend routing. It explicitly separates true raw-depth fusion as a follow-up because it needs depth metadata and camera contracts.
- **Placeholder scan:** No `TBD`, `TODO`, or unspecified “write tests” steps remain. Each task has concrete files, snippets, commands, and expected outcomes.
- **Type consistency:** `DepthMapMeshBuildRequest`, `DepthMapMeshBuilder`, `DepthFrameArtifact`, and `buildMeshFromDepthMaps` are introduced before later tasks use them.
