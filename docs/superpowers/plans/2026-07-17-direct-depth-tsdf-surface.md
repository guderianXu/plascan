# Direct Depth TSDF Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make confidence-weighted TSDF fusion of persisted depth frames the default arbitrary-3D model path, with no dense-point-cloud intermediate and no silent reconstruction fallback.

**Architecture:** A new `DepthTsdfSurfaceBuilder` loads accepted depth/confidence/mask/camera frames, estimates robust world bounds, validates a resolution-320 memory layout, integrates a projective TSDF on CPU/OpenMP, and extracts a cleaned Marching Cubes mesh. The shared model workflow routes `depth_maps` directly to this builder; GUI model generation waits for depth artifacts rather than `denseCloudResultReady`, while visual hull and Poisson remain explicit legacy modes.

**Execution record (2026-07-17):** Implemented the direct-depth route, GUI/CLI contracts, verified-first
source diagnostics, 2% largest-component-relative cleanup, and fine-level coverage fallback. Focused mesh,
MVS, CLI, and GUI tests pass. The final Temple run produced a one-component TSDF mesh with largest face
ratio `1.0`, coverage `0.7218`, IoU `0.7176`, edge P90 `81.00 px`, and SSIM `0.5457`; reconstruction and
meshing passed, while the strict image-quality gate remains open. UAV9 completed 9/9 depth frames and all
nine SIFT extractions were capped at 40000 keypoints.

**Tech Stack:** C++17, Qt 6 Core/Gui, OpenCV, OpenMP, plapoint Marching Cubes and mesh filters, CMake/GTest, native Windows PowerShell, repository `.venv` for regression scripts.

**Execution constraints:** Use only `E:\code\plascan\build\windows-vcpkg-cuda-release`. Put MVS tests only in `src/core/mvs/tests`. Do not reset, checkout, clean, delete user/build data, commit, or push. Preserve unrelated dirty changes and existing submodule state.

---

## File map

- Create `src/core/mesh/DepthTsdfSurfaceBuilder.h`: TSDF frame, options, layout, statistics, result, and builder API.
- Create `src/core/mesh/DepthTsdfSurfaceBuilder.cpp`: artifact loading, robust bounds, memory validation, projective integration, Marching Cubes, colors, and cleanup.
- Modify `src/core/mesh/CMakeLists.txt`: compile the new builder into `meshing`.
- Modify `src/core/mesh/DepthMapMeshBuilder.h/.cpp`: expose complete frame artifacts and reuse manifest discovery; keep visual hull explicit.
- Modify `src/core/mesh/ModelWorkflowService.h/.cpp`: route `depth_maps` to `depth_tsdf` by default and record actual mode/statistics.
- Modify `tests/test_mesh_reconstructor.cpp`: synthetic TSDF, memory, mask, routing, and no-fallback tests.
- Modify `src/gui/project/support/ProjectModelWorkflowPolicy.h/.cpp`: replace dense-cloud preparation with depth-only preparation.
- Modify `src/gui/project/manager/ProjectDenseReconstructionManager.h/.cpp`: return start status and emit completed depth-batch identity.
- Modify `src/gui/project/manager/ProjectModelGenerationWorkflow.h/.cpp`: transition from depth completion directly to model generation.
- Modify `src/gui/dialogs/GenerateModelDialog.cpp`: expose truthful depth-direct behavior and settings.
- Modify `src/gui/project/manager/ProjectModelManager.cpp`: persist `depth_tsdf` source/mode without a dense-cloud path.
- Modify `tests/test_gui_project_utils.cpp`: model-policy, dialog, workflow, and manager contract tests.
- Modify `src/cli/cli_mesh_reconstruct.cpp`: make depth TSDF explicit in the shared CLI output/settings.
- Modify `tests/test_cli_contracts.cpp`: require direct depth mode and no dense input.
- Modify `scripts/validation/run_depth_overlay_regression.ps1`: run depth-only reconstruction, direct mesh CLI, and image/model quality.
- Modify `src/core/mvs/MvsSourcePlanner.h/.cpp`, `src/core/mvs/DepthMapGenerator.cpp`, and `src/core/mvs/tests/test_mvs_source_planner.cpp`: verified-first source backfill and shortfall diagnostics.
- Modify `docs/PROJECT_ARCHITECTURE.md`, `src/core/mvs/README.md`, and `README.md`: document default direct-depth modeling and validation.

### Task 1: Define and validate the TSDF volume contract

**Files:**
- Create: `src/core/mesh/DepthTsdfSurfaceBuilder.h`
- Create: `src/core/mesh/DepthTsdfSurfaceBuilder.cpp`
- Modify: `src/core/mesh/CMakeLists.txt`
- Modify: `tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write failing layout and memory tests**

Add:

```cpp
#include "DepthTsdfSurfaceBuilder.h"

TEST(DepthTsdfSurfaceBuilderTest, ResolutionAppliesToLongestPhysicalAxis)
{
    const std::array<float, 3> minimum{0.0f, 0.0f, 0.0f};
    const std::array<float, 3> maximum{2.0f, 1.0f, 0.5f};
    const auto layout = xjw::mesh::DepthTsdfSurfaceBuilder::makeLayout(
        minimum, maximum, 320, false);

    ASSERT_TRUE(layout.ok);
    EXPECT_EQ(layout.cells[0], 320);
    EXPECT_EQ(layout.cells[1], 160);
    EXPECT_EQ(layout.cells[2], 80);
    EXPECT_GT(layout.requiredBytes, 0u);
}

TEST(DepthTsdfSurfaceBuilderTest, MemoryFailureDoesNotLowerResolution)
{
    xjw::mesh::DepthTsdfOptions options;
    options.resolution = 320;
    options.availableMemoryBytes = 1024;

    const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::validateAllocation(
        std::array<float, 3>{0.0f, 0.0f, 0.0f},
        std::array<float, 3>{1.0f, 1.0f, 1.0f},
        options);

    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.layout.cells[0], 320);
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("320")));
    EXPECT_TRUE(result.errorMessage.contains(QStringLiteral("bytes")));
}
```

- [ ] **Step 2: Run and verify compile failure**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_mesh_reconstructor
```

Expected: compile fails because `DepthTsdfSurfaceBuilder` does not exist.

- [ ] **Step 3: Add the public data contract**

Create `DepthTsdfSurfaceBuilder.h` with these concrete types:

```cpp
#pragma once

#include "Camera.h"
#include "MeshTypes.h"

#include <QJsonObject>
#include <QString>
#include <QVector>
#include <opencv2/core/mat.hpp>

#include <array>
#include <cstdint>
#include <functional>

namespace xjw::mesh
{

struct DepthTsdfFrame
{
    int refIndex = -1;
    QString refImage;
    Camera camera;
    cv::Mat depth;
    cv::Mat confidence;
    cv::Mat depthValidMask;
    cv::Mat supportMask;
    cv::Mat colorBgr;
    float frameQualityWeight = 1.0f;
};

struct DepthTsdfOptions
{
    int resolution = 320;
    float truncationVoxels = 4.0f;
    float minimumConfidence = 0.25f;
    float minimumVoxelWeight = 1.0f;
    int minimumInputFrames = 3;
    int minimumDistinctCameraSupport = 2;
    int minimumComponentFaces = 64;
    bool calculateVertexColors = true;
    int workerCount = 0;
    std::uint64_t availableMemoryBytes = 0;
    std::function<bool()> isCancelled;
    std::function<void(const QString &, int)> progress;
};

struct DepthTsdfLayout
{
    bool ok = false;
    std::array<float, 3> boundsMin{};
    std::array<float, 3> boundsMax{};
    std::array<int, 3> cells{};
    std::array<float, 3> voxelSize{};
    std::uint64_t sampleCount = 0;
    std::uint64_t requiredBytes = 0;
};

struct DepthTsdfStatistics
{
    int inputFrameCount = 0;
    int acceptedFrameCount = 0;
    std::uint64_t integratedVoxelUpdates = 0;
    std::uint64_t rejectedProjectionCount = 0;
    std::uint64_t rejectedSupportMaskCount = 0;
    std::uint64_t rejectedDepthValidCount = 0;
    std::uint64_t rejectedDepthCount = 0;
    std::uint64_t rejectedConfidenceCount = 0;
    std::uint64_t supportedSampleCount = 0;
    int vertexCount = 0;
    int faceCount = 0;
    int componentCount = 0;
    double largestComponentFaceRatio = 0.0;
};

struct DepthTsdfResult
{
    bool ok = false;
    QString errorMessage;
    DepthTsdfLayout layout;
    DepthTsdfStatistics statistics;
    TriMesh mesh;
};

class DepthTsdfSurfaceBuilder
{
public:
    static DepthTsdfLayout makeLayout(const std::array<float, 3> &boundsMin,
                                      const std::array<float, 3> &boundsMax,
                                      int resolution,
                                      bool includeColor);
    static DepthTsdfResult validateAllocation(const std::array<float, 3> &boundsMin,
                                              const std::array<float, 3> &boundsMax,
                                              const DepthTsdfOptions &options);
    static DepthTsdfResult build(const QVector<DepthTsdfFrame> &frames,
                                 const DepthTsdfOptions &options);
    static QJsonObject statisticsToJson(const DepthTsdfResult &result);
};

} // namespace xjw::mesh
```

- [ ] **Step 4: Implement aspect-ratio layout and overflow-safe memory estimates**

Use checked multiplication for `(cells[0] + 1) * (cells[1] + 1) * (cells[2] + 1)`. Base bytes per sample are `float tsdf + float weight + uint16_t support`; color adds three float accumulators and one float color weight. Reject non-finite/degenerate bounds and resolution below 8.

The allocation guard is:

```cpp
const std::uint64_t available = options.availableMemoryBytes > 0
    ? options.availableMemoryBytes
    : availablePhysicalMemoryBytes();
const std::uint64_t budget = available > 0 ? available * 3 / 4 : 0;
if (budget > 0 && result.layout.requiredBytes > budget)
{
    result.errorMessage = QStringLiteral(
        "TSDF allocation rejected: resolution=%1 cells=%2x%3x%4 required=%5 bytes available=%6 bytes")
        .arg(options.resolution)
        .arg(result.layout.cells[0])
        .arg(result.layout.cells[1])
        .arg(result.layout.cells[2])
        .arg(result.layout.requiredBytes)
        .arg(available);
    return result;
}
```

On Windows, `availablePhysicalMemoryBytes()` uses `GlobalMemoryStatusEx`; a zero result means the OS query is unavailable, not permission to reduce resolution.

- [ ] **Step 5: Register the source and run tests**

Add `DepthTsdfSurfaceBuilder.cpp` to `src/core/mesh/CMakeLists.txt`, rebuild `test_mesh_reconstructor`, and run:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_mesh_reconstructor.exe `
  --gtest_filter=DepthTsdfSurfaceBuilderTest.ResolutionAppliesToLongestPhysicalAxis:DepthTsdfSurfaceBuilderTest.MemoryFailureDoesNotLowerResolution
```

Expected: both tests pass and resolution stays 320 on memory failure.

### Task 2: Load accepted depth frames and estimate robust bounds

**Files:**
- Modify: `src/core/mesh/DepthMapMeshBuilder.h`
- Modify: `src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `src/core/mesh/DepthTsdfSurfaceBuilder.h`
- Modify: `src/core/mesh/DepthTsdfSurfaceBuilder.cpp`
- Modify: `tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write failing artifact-load and depth-convention tests**

Create three 32x24 raw depth/confidence/depth-valid/support-mask artifacts in a temporary directory with cameras constructed through `setIntrinsics()` and `setPose()`. Assert:

```cpp
const auto loaded = xjw::mesh::DepthTsdfSurfaceBuilder::loadFrames(artifacts);
ASSERT_TRUE(loaded.ok) << loaded.errorMessage.toStdString();
ASSERT_EQ(loaded.frames.size(), 3);
EXPECT_EQ(loaded.frames.front().depth.type(), CV_32FC1);
EXPECT_EQ(loaded.frames.front().confidence.type(), CV_32FC1);
EXPECT_EQ(loaded.frames.front().depthValidMask.type(), CV_8UC1);
EXPECT_EQ(loaded.frames.front().supportMask.type(), CV_8UC1);
EXPECT_TRUE(loaded.frames.front().camera.isValid());

const auto bounds = xjw::mesh::DepthTsdfSurfaceBuilder::estimateBounds(loaded.frames);
ASSERT_TRUE(bounds.ok);
EXPECT_LT(bounds.minimum[2], 2.0f);
EXPECT_GT(bounds.maximum[2], 2.0f);
```

Use `xjw::core::project::writeDepthMatStorage()` for depth/confidence and `cv::imwrite()` for the mask so the test exercises production readers.

- [ ] **Step 2: Run and verify missing loader APIs**

Expected: compile fails on `loadFrames()` and `estimateBounds()`.

- [ ] **Step 3: Complete `DepthFrameArtifact` metadata**

Add:

```cpp
QString status;
QString acceptance;
QString supportMaskPath;
bool fusionEligible = true;
double validCoverage = -1.0;
double meanConfidence = -1.0;
int sourceViewCount = 0;
```

Populate these from `mvs_manifest.json`, falling back to nested `depth_quality` values. Continue to ignore frames whose status is neither empty nor `completed`.

Persist `support_mask_path` separately from `valid_mask_path` in `DepthMapGenerator` and `MvsWorkspaceManifest`. `valid_mask_path` continues to mean `depth > 0`; `support_mask_path` stores `DepthFrameResult::validMask`, whose zero pixels are the authoritative project/content exterior. If `DepthFrameResult::validMask` is empty, write an all-255 support mask with the depth dimensions.

- [ ] **Step 4: Add concrete loader result types and implementation**

Add to `DepthTsdfSurfaceBuilder.h`:

```cpp
struct DepthTsdfFrameLoadResult
{
    bool ok = false;
    QString errorMessage;
    QVector<DepthTsdfFrame> frames;
};

struct DepthTsdfBoundsResult
{
    bool ok = false;
    QString errorMessage;
    std::array<float, 3> minimum{};
    std::array<float, 3> maximum{};
    std::uint64_t sampleCount = 0;
};
```

Expose:

```cpp
static DepthTsdfFrameLoadResult loadFrames(const QVector<DepthFrameArtifact> &artifacts);
static DepthTsdfBoundsResult estimateBounds(const QVector<DepthTsdfFrame> &frames);
```

For each frame, require camera, `CV_32FC1` depth, matching confidence or an all-ones fallback, and matching `CV_8UC1` depth-valid mask. Load the separate `CV_8UC1` support mask when present. For a legacy frame without `support_mask_path`, use an all-255 support mask so missing depth is not interpreted as free space. Reject a corrupt referenced frame with all exact paths in the error; do not substitute a preview PNG.

Estimate bounds by sampling at most 6000 valid pixels per frame, calling `camera.unprojectPixel(pixel, depth, world)`, sorting each world axis, taking P1/P99, and padding every axis by 8% of its retained span. Require at least 500 finite samples and at least three accepted frames.

- [ ] **Step 5: Run artifact and bounds tests**

Run the new tests plus existing `DepthMapMeshBuilderTest.DiscoversDepthFramesFromOutputDirectory` and `LoadsDepthGridCameraFromWorkspaceManifest`.

Expected: raw storage, masks, camera-axis depth, and robust bounds pass without directory-preview fallback.

### Task 3: Integrate confidence-weighted projective TSDF

**Files:**
- Modify: `src/core/mesh/DepthTsdfSurfaceBuilder.cpp`
- Modify: `tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write a failing synthetic multi-camera plane test**

Create three cameras centered at `x = -0.1, 0.0, 0.1`, all viewing a fronto-parallel plane at physical positive depth 2.0. Fill confidence with 0.9 and mask with 255 internal-valid pixels. Configure resolution 48 and colors off for test speed:

```cpp
xjw::mesh::DepthTsdfOptions options;
options.resolution = 48;
options.calculateVertexColors = false;
options.availableMemoryBytes = 512ull * 1024ull * 1024ull;

const auto result = xjw::mesh::DepthTsdfSurfaceBuilder::build(frames, options);
ASSERT_TRUE(result.ok) << result.errorMessage.toStdString();
EXPECT_GT(result.statistics.integratedVoxelUpdates, 0u);
EXPECT_GT(result.statistics.supportedSampleCount, 0u);
EXPECT_FALSE(result.mesh.empty());
for (const auto &vertex : result.mesh.vertices)
{
    EXPECT_TRUE(std::isfinite(vertex.x));
    EXPECT_TRUE(std::isfinite(vertex.y));
    EXPECT_TRUE(std::isfinite(vertex.z));
}
```

- [ ] **Step 2: Write a failing confidence and mask rejection test**

Set the left half confidence to 0.1 and an intentional rectangular hole in the separate support mask to 0. Leave a different region at zero only in the depth-valid mask. Assert rejected-confidence, rejected-depth-valid, and rejected-support-mask counters are positive; the support-mask hole is carved, while the depth-valid-only hole is not treated as exterior free space.

- [ ] **Step 3: Run and verify empty/unimplemented fusion failure**

Expected: `build()` fails or returns no mesh because projective integration is not implemented.

- [ ] **Step 4: Implement projective voxel integration**

Allocate arrays initialized as:

```cpp
std::vector<float> tsdf(sample_count, 1.0f);
std::vector<float> weight(sample_count, 0.0f);
std::vector<std::uint16_t> support(sample_count, 0);
```

For each grid sample and frame, use only `Camera::projectWorldPointWithDepth()`:

```cpp
double pixel[2]{};
double voxel_depth = 0.0;
if (!frame.camera.projectWorldPointWithDepth(world, pixel, voxel_depth))
{
    ++local.rejectedProjectionCount;
    continue;
}
const int column = static_cast<int>(std::lround(pixel[0]));
const int row = static_cast<int>(std::lround(pixel[1]));
if (row < 0 || row >= frame.depth.rows || column < 0 || column >= frame.depth.cols)
{
    ++local.rejectedProjectionCount;
    continue;
}
if (frame.supportMask.at<std::uint8_t>(row, column) == 0)
{
    integrateWeighted(&tsdf[index], &weight[index], 1.0f, 0.25f);
    ++local.rejectedSupportMaskCount;
    continue;
}
if (frame.depthValidMask.at<std::uint8_t>(row, column) == 0)
{
    ++local.rejectedDepthValidCount;
    continue;
}
const float observed_depth = frame.depth.at<float>(row, column);
if (!std::isfinite(observed_depth) || observed_depth <= 0.0f)
{
    ++local.rejectedDepthCount;
    continue;
}
const float confidence = frame.confidence.empty()
    ? 1.0f
    : frame.confidence.at<float>(row, column);
if (!std::isfinite(confidence) || confidence < options.minimumConfidence)
{
    ++local.rejectedConfidenceCount;
    continue;
}
const float signed_distance = observed_depth - static_cast<float>(voxel_depth);
if (signed_distance < -truncation)
{
    continue;
}
const float normalized = std::clamp(signed_distance / truncation, -1.0f, 1.0f);
const float observation_weight = confidence * frame.frameQualityWeight;
integrateWeighted(&tsdf[index], &weight[index], normalized, observation_weight);
if (std::fabs(signed_distance) <= truncation)
{
    support[index] = static_cast<std::uint16_t>(
        std::min<int>(std::numeric_limits<std::uint16_t>::max(), support[index] + 1));
}
```

Parallelize the outer Z loop with OpenMP and per-thread counters, then reduce counters after the loop. Check cancellation between Z slabs. Do not update the same voxel from multiple threads; each parallel iteration owns its voxel indices.

- [ ] **Step 5: Gate the scalar field by accumulated evidence**

Before extraction, a sample is supported only when:

```cpp
supported[index] = weight[index] >= options.minimumVoxelWeight
    && support[index] >= options.minimumDistinctCameraSupport;
```

Unsupported samples return `+1.0f` to Marching Cubes. This prevents a one-camera sheet from entering the production surface.

- [ ] **Step 6: Run fusion tests**

Expected: synthetic plane produces finite geometry, weak confidence is counted, and the intentional mask hole remains open.

### Task 4: Extract and clean the zero surface

**Files:**
- Modify: `src/core/mesh/DepthTsdfSurfaceBuilder.cpp`
- Modify: `tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write failing extraction-statistics tests**

Assert the successful synthetic result has `vertexCount == mesh.vertexCount()`, `faceCount == mesh.faceCount()`, `componentCount >= 1`, `largestComponentFaceRatio >= 0.9`, and every normal has finite components and length within `1e-3` of one.

- [ ] **Step 2: Extract with the existing plapoint Marching Cubes**

Configure:

```cpp
plapoint::mesh::MarchingCubes<float> marching_cubes;
marching_cubes.setBounds(result.layout.boundsMin, result.layout.boundsMax);
marching_cubes.setResolution(result.layout.cells[0],
                             result.layout.cells[1],
                             result.layout.cells[2]);
marching_cubes.setIsoLevel(0.0f);
```

The scalar callback maps world coordinates to the nearest precomputed sample and returns `+1.0f` for unsupported samples. Convert returned matrices to `TriMesh` using the same checked index conversions as `VisualHullReconstructor.cpp`.

- [ ] **Step 3: Apply production cleanup without blanket hole filling**

Run, in order:

```cpp
detail::removeDegenerateFaces(&result.mesh);
detail::removeSmallConnectedComponents(&result.mesh,
                                       std::max(2, options.minimumComponentFaces));
detail::weldCoincidentVertices(&result.mesh, 1.0e-6f);
detail::recomputeNormals(&result.mesh);
```

Do not call `fillSmallBoundaryHoles()` or smoothing by default. Analyze connectivity with `VisualHullReconstructor::analyzeConnectivity()` and store counts in `DepthTsdfStatistics`.

- [ ] **Step 4: Sample optional vertex color independently**

For each vertex, project into eligible frames, reject invalid mask/depth, require the observed depth to agree within one truncation distance, and accumulate BGR using the same confidence weight. A vertex without a valid color keeps the neutral `200,200,200`; geometry acceptance never depends on image availability.

- [ ] **Step 5: Run all TSDF unit tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_mesh_reconstructor
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_mesh_reconstructor.exe `
  --gtest_filter=DepthTsdfSurfaceBuilderTest.*
```

Expected: layout, memory, loading, fusion, mask, extraction, normals, connectivity, and colors pass.

### Task 5: Route shared model generation directly to TSDF

**Files:**
- Modify: `src/core/mesh/ModelWorkflowService.h`
- Modify: `src/core/mesh/ModelWorkflowService.cpp`
- Modify: `tests/test_mesh_reconstructor.cpp`

- [ ] **Step 1: Write failing default-mode and no-fallback tests**

Add:

```cpp
TEST(MeshWorkflowSettingsTest, DepthMapsDefaultToDepthTsdf)
{
    QJsonObject settings{{QStringLiteral("source_data"), QStringLiteral("depth_maps")},
                         {QStringLiteral("surface_type"), QStringLiteral("arbitrary_3d")},
                         {QStringLiteral("meshResolution"), 320}};
    EXPECT_EQ(xjw::mesh::workflow::depthReconstructionModeFromSettings(settings),
              QStringLiteral("depth_tsdf"));
}

TEST(DepthMapMeshBuilderTest, TsdfFailureDoesNotFallBackToVisualHullOrDenseCloud)
{
    xjw::mesh::workflow::DepthMapMeshBuildRequest request;
    request.depthMapSourcePath = QStringLiteral("E:/missing/depth");
    request.settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
    request.reconstruction.resolution = 320;

    const auto result = xjw::mesh::workflow::buildMeshFromDepthMaps(request);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(result.payload.value(QStringLiteral("actual_mesh_algorithm")).toString(),
              QStringLiteral("depth_tsdf"));
    EXPECT_FALSE(result.payload.contains(QStringLiteral("source_point_cloud_path")));
    EXPECT_FALSE(result.payload.contains(QStringLiteral("visual_hull_fallback_reason")));
}
```

- [ ] **Step 2: Run and verify current visual-hull/dense routing fails the contract**

Expected: missing mode resolver and current payload/routing mismatch.

- [ ] **Step 3: Add an explicit mode resolver**

Declare and implement:

```cpp
QString depthReconstructionModeFromSettings(const QJsonObject &settings)
{
    const QString requested = settings.value(QStringLiteral("reconstruction_mode"))
                                  .toString()
                                  .trimmed()
                                  .toLower();
    if (!requested.isEmpty())
    {
        return requested;
    }
    return settings.value(QStringLiteral("surface_type")).toString()
               == QStringLiteral("height_field")
        ? QStringLiteral("poisson_legacy")
        : QStringLiteral("depth_tsdf");
}
```

- [ ] **Step 4: Make `buildMeshFromDepthMaps()` an explicit mode switch**

Use this structure:

```cpp
const QString mode = depthReconstructionModeFromSettings(request.settings);
result.payload[QStringLiteral("actual_mesh_algorithm")] = mode;
result.payload[QStringLiteral("reconstruction_mode")] = mode;
result.payload[QStringLiteral("depth_map_source_path")] = request.depthMapSourcePath;
result.payload[QStringLiteral("source_data")] = QStringLiteral("depth_maps");

if (mode == QStringLiteral("depth_tsdf"))
{
    const auto artifacts = DepthMapMeshBuilder::discoverDepthFrames(request.depthMapSourcePath);
    const auto loaded = DepthTsdfSurfaceBuilder::loadFrames(artifacts);
    if (!loaded.ok)
    {
        result.errorMessage = loaded.errorMessage;
        return result;
    }
    DepthTsdfOptions options = depthTsdfOptionsFromSettings(
        request.settings, request.reconstruction.resolution);
    const auto tsdf = DepthTsdfSurfaceBuilder::build(loaded.frames, options);
    const QJsonObject statistics = DepthTsdfSurfaceBuilder::statisticsToJson(tsdf);
    for (auto it = statistics.constBegin(); it != statistics.constEnd(); ++it)
    {
        result.payload[it.key()] = it.value();
    }
    if (!tsdf.ok)
    {
        result.errorMessage = tsdf.errorMessage;
        return result;
    }
    const QJsonObject diagnostic_payload = result.payload;
    result = saveMeshAndOptionalTexture(tsdf.mesh,
                                        "depth_tsdf",
                                        output_root,
                                        request.exportObj,
                                        request.texture,
                                        request.progress);
    for (auto it = diagnostic_payload.constBegin();
         it != diagnostic_payload.constEnd(); ++it)
    {
        result.payload[it.key()] = it.value();
    }
    return result;
}
if (mode != QStringLiteral("visual_hull") &&
    mode != QStringLiteral("poisson_legacy"))
{
    result.errorMessage = QStringLiteral("未知深度模型重建模式: %1").arg(mode);
    return result;
}
if (mode == QStringLiteral("visual_hull"))
{
    DepthMapVisualHullOptions options;
    options.strictVolumetricMasks = request.settings.value(
        QStringLiteral("strictVolumetricMasks")).toBool(false);
    const DepthMapVisualHullResult hull = DepthMapMeshBuilder::buildVisualHull(
        request.depthMapSourcePath,
        request.reconstruction.resolution,
        options,
        request.progress);
    if (!hull.applicable || !hull.ok)
    {
        result.errorMessage = hull.message.isEmpty()
            ? QStringLiteral("显式 Visual Hull 模式不可用于当前深度数据")
            : hull.message;
        return result;
    }
    result = saveMeshAndOptionalTexture(hull.mesh,
                                        "silhouette_visual_hull",
                                        output_root,
                                        request.exportObj,
                                        request.texture,
                                        request.progress);
    result.payload[QStringLiteral("reconstruction_mode")] =
        QStringLiteral("visual_hull");
    result.payload[QStringLiteral("visual_hull_views")] = hull.usableViewCount;
    result.payload[QStringLiteral("visual_hull_component_count")] =
        hull.connectivity.componentCount;
    result.payload[QStringLiteral("visual_hull_largest_component_ratio")] =
        hull.connectivity.largestComponentFaceRatio;
    return result;
}

QString resolve_error;
QString dense_path = request.reusableDenseCloudPath.trimmed();
if (dense_path.isEmpty())
{
    dense_path = DepthMapMeshBuilder::resolveReusableDenseCloud(
        request.depthMapSourcePath, &resolve_error);
}
if (dense_path.isEmpty())
{
    result.errorMessage = resolve_error.isEmpty()
        ? QStringLiteral("显式 poisson_legacy 模式缺少可复用密集点云")
        : resolve_error;
    return result;
}
MeshBuildRequest legacy_request;
legacy_request.pointCloudPath = dense_path;
legacy_request.outputRoot = output_root;
legacy_request.reconstruction = request.reconstruction;
legacy_request.exportObj = request.exportObj;
legacy_request.texture = request.texture;
legacy_request.progress = request.progress;
result = buildMeshAndOptionalTexture(legacy_request);
result.payload[QStringLiteral("reconstruction_mode")] =
    QStringLiteral("poisson_legacy");
result.payload[QStringLiteral("source_point_cloud_path")] = dense_path;
return result;
```

Replace the current implicit visual-hull-first and dense-cloud fallthrough with the three explicit branches shown. `depth_tsdf` returns before either legacy branch, and the diagnostic payload is merged back after mesh saving as shown.

- [ ] **Step 5: Map settings without silent downgrade**

`depthTsdfOptionsFromSettings()` sets resolution from `meshResolution`, colors from `calculateVertexColors`, filter-dependent confidence threshold, `minimumInputFrames=3`, and `minimumDistinctCameraSupport=2`. It never reduces resolution or changes the requested mode.

- [ ] **Step 6: Run all mesh workflow tests**

Expected: existing explicit Poisson/height-field tests still pass, depth maps default to `depth_tsdf`, and no-fallback tests pass.

### Task 6: Remove the GUI dense-cloud transition from model generation

**Files:**
- Modify: `src/gui/project/support/ProjectModelWorkflowPolicy.h`
- Modify: `src/gui/project/support/ProjectModelWorkflowPolicy.cpp`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.h`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `src/gui/project/manager/ProjectModelGenerationWorkflow.h`
- Modify: `src/gui/project/manager/ProjectModelGenerationWorkflow.cpp`
- Modify: `src/gui/dialogs/GenerateModelDialog.cpp`
- Modify: `src/gui/project/manager/ProjectModelManager.cpp`
- Modify: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write failing model-policy tests**

Add or update tests to assert:

```cpp
EXPECT_EQ(decision.action,
          xjw::gui::project::ModelWorkflowAction::RunMeshDirectly);
EXPECT_EQ(decision.modelSettings.value(QStringLiteral("reconstruction_mode")).toString(),
          QStringLiteral("depth_tsdf"));
EXPECT_FALSE(decision.modelSettings.contains(QStringLiteral("source_point_cloud_path")));
```

for a complete reusable depth batch, and:

```cpp
EXPECT_EQ(decision.action,
          xjw::gui::project::ModelWorkflowAction::GenerateDepthMapsThenMesh);
EXPECT_EQ(decision.depthSettings.value(QStringLiteral("workflow_action")).toString(),
          QStringLiteral("generate_depth_maps"));
```

for a missing/incomplete batch.

- [ ] **Step 2: Run and verify current dense actions fail**

Expected: policy returns `FuseDepthMapsThenMesh` or `GenerateDenseCloudThenMesh`.

- [ ] **Step 3: Replace dense preparation actions with depth-only preparation**

Use:

```cpp
enum class ModelWorkflowAction
{
    RunMeshDirectly,
    GenerateDepthMapsThenMesh
};

struct ModelWorkflowDecision
{
    ModelWorkflowAction action = ModelWorkflowAction::RunMeshDirectly;
    QJsonObject modelSettings;
    QJsonObject depthSettings;
    QString depthMapSourcePath;
    QString reason;
};
```

When `stored_batch_complete`, set `RunMeshDirectly`, `source_data=depth_maps`, `depthMapSourcePath`, and `reconstruction_mode=depth_tsdf`. Otherwise populate `depthSettings` from the existing dense settings mapper but force `workflow_action=generate_depth_maps`; do not request fusion.

- [ ] **Step 4: Emit a completed depth batch and return start status**

Change `ProjectDenseReconstructionManager::startEstimateDepthMapsAsync()` to return `bool`; update every early exit to `return false` and return true after `_activeMvsGenerator` is installed.

Add:

```cpp
void depthMapBatchReady(const QString &outputDirectory, int frameCount);
```

On successful `DepthMapGenerator::finished`, after report refresh:

```cpp
emit self->depthMapBatchReady(mvsOutDir, selectedImageCount);
emit self->mvsProgressFinished(true);
```

Keep existing callers valid by allowing them to ignore the returned bool.

- [ ] **Step 5: Make `ProjectModelGenerationWorkflow` wait for depth, not dense cloud**

Use stages `Idle`, `DepthMaps`, and `Model`; rename `_expectedDenseOutputDir` to `_expectedDepthOutputDir`. Connect `depthMapBatchReady`:

```cpp
if (_stage != Stage::DepthMaps || !belongsToActiveProject())
{
    return;
}
const QString result_directory = QDir::cleanPath(output_directory);
if (result_directory.compare(_expectedDepthOutputDir, Qt::CaseInsensitive) != 0)
{
    return;
}
QJsonObject model_settings = _pendingModelSettings;
model_settings[QStringLiteral("source_data")] = QStringLiteral("depth_maps");
model_settings[QStringLiteral("source_path")] = result_directory;
model_settings[QStringLiteral("depthMapSourcePath")] = result_directory;
model_settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
model_settings.remove(QStringLiteral("source_point_cloud_path"));
startModelStage(model_settings);
```

Remove the workflow dependency on `denseCloudResultReady`. Call `startEstimateDepthMapsAsync(decision.depthSettings)` for the depth preparation action.

- [ ] **Step 6: Make GUI text and settings truthful**

For arbitrary-3D depth sources, collect:

```cpp
settings[QStringLiteral("method")] = QStringLiteral("Depth TSDF");
settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
```

Change the automatic source note to:

```cpp
tr("当前没有可复用深度图；生成模型时将自动估计深度图并直接进行 TSDF 表面重建。")
```

In `buildMeshReconstructionRecord()`, keep `source_dense_cloud` empty for `source_data=depth_maps` and persist `reconstruction_mode` from the task result.

- [ ] **Step 7: Run GUI policy, dialog, and workflow tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target test_gui_project_utils
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_gui_project_utils.exe `
  --gtest_filter=*ModelWorkflow*:*GenerateModelDialog*:*ProjectModelGenerationWorkflow*
```

Expected: complete depth batches go directly to TSDF; incomplete batches estimate depth only and then go directly to TSDF.

### Task 7: Make CLI regression use the direct-depth path

**Files:**
- Modify: `src/cli/cli_mesh_reconstruct.cpp`
- Modify: `tests/test_cli_contracts.cpp`
- Modify: `scripts/validation/run_depth_overlay_regression.ps1`

- [ ] **Step 1: Add failing CLI contract assertions**

Require:

```cpp
expectContainsAll(meshSource, {
    "depth_maps",
    "reconstruction_mode",
    "depth_tsdf",
    "buildModel"
});
expectContainsAll(regressionScript, {
    "--mvs-depth-only",
    "mesh_reconstruct_cli.exe",
    "--source-data", "depth_maps",
    "--depth-map-dir",
    "reconstruction_mode",
    "depth_tsdf"
});
```

- [ ] **Step 2: Run and verify current one-process dense pipeline fails the contract**

Run `test_cli_contracts` filtered to mesh and depth-overlay regression cases. Expected: missing direct-depth tokens.

- [ ] **Step 3: Make `mesh_reconstruct_cli` default depth sources to TSDF**

After resolving `source_data_qt`:

```cpp
if (source_data_qt == QStringLiteral("depth_maps") &&
    !settings.contains(QStringLiteral("reconstruction_mode")))
{
    settings[QStringLiteral("reconstruction_mode")] = QStringLiteral("depth_tsdf");
}
```

Keep `--dense-cloud` only for explicit `poisson_legacy`; do not require or populate it for `depth_tsdf`.

- [ ] **Step 4: Split regression into depth-only and direct-mesh processes**

Resolve `mesh_reconstruct_cli.exe`. Add `--mvs-depth-only` to reconstruction arguments. Write a settings file inside the timestamped run directory:

```powershell
$modelSettingsPath = Join-Path $runDirectory "model-settings.json"
@{
    generate_model = @{
        source_data = "depth_maps"
        surface_type = "arbitrary_3d"
        reconstruction_mode = "depth_tsdf"
        meshResolution = 320
        calculateVertexColors = $true
    }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $modelSettingsPath -Encoding UTF8
```

Then run:

```powershell
$meshArguments = @(
    "--source-data", "depth_maps",
    "--depth-map-dir", (Join-Path $pipelineDirectory "mvs"),
    "--output-dir", (Join-Path $pipelineDirectory "model"),
    "--settings-json", $modelSettingsPath,
    "--settings-key", "generate_model"
)
```

Parse the mesh CLI JSON as the model report and record both process exit codes/peak memory. Assert `actual_mesh_algorithm == depth_tsdf` before quality evaluation.

- [ ] **Step 5: Run CLI contract tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  mesh_reconstruct_cli test_cli_contracts
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\Release\test_cli_contracts.exe `
  --gtest_filter=*MeshReconstruct*:*DepthOverlayRegression*
```

Expected: direct-depth CLI and script contracts pass.

### Task 8: Fill verified-source shortfalls safely and report them

**Files:**
- Modify: `src/core/mvs/MvsSourcePlanner.h`
- Modify: `src/core/mvs/MvsSourcePlanner.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/tests/test_mvs_source_planner.cpp`

- [ ] **Step 1: Write failing verified-first backfill tests**

Construct two verified candidates and four track/angle-qualified unverified candidates with `maxSources=6`. Assert selected order starts with both verified candidates, then qualified geometry backfills, no sequence fallback is used, and selected size is six. Add a second test where unverified candidates have fewer than 20 shared tracks or angles outside `[0.2, 35]`; assert they remain rejected and the shortfall is reported.

- [ ] **Step 2: Add source provenance to plan entries**

Add:

```cpp
enum class MvsSourceTier
{
    VerifiedPair,
    TrackGeometryBackfill,
    SequenceFallback
};

MvsSourceTier tier = MvsSourceTier::VerifiedPair;
```

Serialize `source_tier` as `verified_pair`, `track_geometry_backfill`, or `sequence_fallback` in `mvsSourcePlanEntryToJson()`.

- [ ] **Step 3: Run a strict first pass and bounded backfill pass**

In `DepthMapGenerator`, keep the current verified-pair plan as pass one. If selected size is below `numSourceViews`, run a second plan over unused candidates with:

```cpp
backfillOptions.requireVerifiedPairGeometry = false;
backfillOptions.allowSequenceFallback = false;
backfillOptions.allowWeakKnownOverlap = false;
backfillOptions.minSharedTracks = 20;
backfillOptions.minGeometricInliers = 0;
backfillOptions.minSourceQualityScore = 0.35f;
backfillOptions.maxSources = desiredSourceCount - static_cast<int>(sources.size());
```

Accept only candidates whose sampled angle is inside `[0.2, 35]`; mark them `TrackGeometryBackfill`. Never add zero-evidence sequence neighbors in this pass.

- [ ] **Step 4: Persist requested count and shortfall**

Add to each depth artifact:

```cpp
artifact[QStringLiteral("requested_source_view_count")] = desiredSourceCount;
artifact[QStringLiteral("source_view_shortfall")] =
    std::max(0, desiredSourceCount - sourceQualitySummary.sourceViewCount);
```

Include verified/backfill counts in `source_plan` and quality reports.

- [ ] **Step 5: Run source planner and MVS pipeline tests**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_mvs_source_planner test_mvs_pipeline test_mvs_types
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R "MvsSourcePlanner|MvsPipeline|MvsSourceViewSelection"
```

Expected: verified pairs remain first, safe backfill reaches the requested count when evidence exists, and weak candidates remain rejected.

### Task 9: Build and focused verification

- [ ] **Step 1: Reconfigure only the main build directory**

Run the repository's established Windows CUDA configure script with its current short-path vcpkg settings, targeting:

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release
```

Expected: configure completes without creating another build directory.

- [ ] **Step 2: Build required targets**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 3 --target `
  test_aerial_triangulation_workflow `
  test_mvs_depth_pyramid `
  test_mvs_types `
  test_mvs_pipeline `
  test_mvs_rectifier_unit `
  test_cli_contracts `
  test_mesh_reconstructor `
  test_gui_project_utils `
  mesh_reconstruct_cli `
  model_quality_cli `
  plascan_gui
```

Expected: all listed targets build.

- [ ] **Step 3: Run the six user-required binaries**

Run through CTest by exact executable/test regex or invoke binaries directly:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R "AerialTriangulation|MvsDepthPyramid|MvsTypes|MvsPipeline|MvsRectifier|CliContract"
```

Expected: required suites pass. Report any unrelated historical failure separately.

- [ ] **Step 4: Run TSDF, overlay, quality, and workflow tests**

Run:

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release `
  --output-on-failure -R "DepthTsdf|DepthOverlay|DataTreeWidget|ProjectDashboard|ReconstructionQuality|ModelWorkflow|MeshWorkflow"
```

Expected: all new behavior passes with no silent fallback.

### Task 10: Temple and UAV regression, documentation, and scoped handoff

**Files:**
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/mvs/README.md`
- Modify: `README.md`

- [ ] **Step 1: Run the Temple direct-depth regression**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  E:\code\plascan\scripts\validation\run_depth_overlay_regression.ps1 `
  -Project E:\code\test\temple\temple.plascan `
  -BuildDirectory E:\code\plascan\build\windows-vcpkg-cuda-release `
  -SceneProfile orbital_object `
  -MaximumFrames 16 `
  -MaximumDimension 1024
```

Expected report:

- `mesh.algorithm = depth_tsdf`;
- coverage at least `0.75`;
- IoU at least `0.70`;
- edge P90 below `65` pixels;
- SSIM at least `0.20`;
- largest component face ratio at least `0.90`;
- recognizable roof, columns, doorway, and base without broad layered ridges.

- [ ] **Step 2: Run the nine-image UAV regression**

Run the same validation script on the established nine-image UAV project with `-SceneProfile aerial_terrain -MaximumFrames 9`. Use the reconstruction report to assert every frame's SIFT keypoint count is at most 40000 and record depth source counts, shortfalls, coverage, TSDF connectivity, and image/model metrics.

Expected: no frame exceeds 40000 SIFT keypoints and terrain/model quality does not regress from the saved baseline.

- [ ] **Step 3: Inspect representative GUI behavior**

Open the built GUI, load Temple, enable **显示深度图**, switch final/L1/L2/L3 and intensity, and switch photos. Expected: no blue feature crosses during depth inspection, mask outline remains, original preferences return when disabled, and no standalone depth section appears in the workspace.

- [ ] **Step 4: Document final architecture**

Document:

```markdown
- 任意 3D 的深度源默认使用 `depth_tsdf`：raw depth + confidence + valid mask + camera → TSDF → Marching Cubes。
- 默认模型路径不生成或消费密集点云；密集点云是单独请求的产品。
- Visual Hull 与 Poisson 仅是显式 legacy/diagnostic 模式，TSDF 失败不得静默回退。
- `meshResolution=320` 表示最长物理轴 320 个体素，分配前执行内存检查且不静默降级。
```

Add the exact Temple/UAV validation commands and measured results after they run.

- [ ] **Step 5: Verify scoped diffs without staging**

Run:

```powershell
git diff --check -- `
  src/core/mesh src/gui/project/support/ProjectModelWorkflowPolicy.h `
  src/gui/project/support/ProjectModelWorkflowPolicy.cpp `
  src/gui/project/manager/ProjectDenseReconstructionManager.h `
  src/gui/project/manager/ProjectDenseReconstructionManager.cpp `
  src/gui/project/manager/ProjectModelGenerationWorkflow.h `
  src/gui/project/manager/ProjectModelGenerationWorkflow.cpp `
  src/gui/dialogs/GenerateModelDialog.cpp `
  src/gui/project/manager/ProjectModelManager.cpp `
  src/core/mvs/MvsSourcePlanner.h src/core/mvs/MvsSourcePlanner.cpp `
  src/core/mvs/DepthMapGenerator.cpp src/core/mvs/tests/test_mvs_source_planner.cpp `
  src/cli/cli_mesh_reconstruct.cpp tests/test_mesh_reconstructor.cpp `
  tests/test_gui_project_utils.cpp tests/test_cli_contracts.cpp `
  scripts/validation/run_depth_overlay_regression.ps1 `
  docs/PROJECT_ARCHITECTURE.md src/core/mvs/README.md README.md
```

Expected: no whitespace errors. Do not stage, commit, or push.
