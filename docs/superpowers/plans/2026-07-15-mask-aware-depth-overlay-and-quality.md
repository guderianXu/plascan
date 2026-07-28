# Mask-aware Depth Overlay and Reconstruction Quality Implementation Plan

> **Superseded for remaining work on 2026-07-17.** The approved implementation is split into
> `2026-07-17-depth-overlay-workspace-quality-fixes.md` followed by
> `2026-07-17-direct-depth-tsdf-surface.md`. The older fusion/Poisson and automatic commit steps
> below are retained only as historical context and must not be executed.

> **2026-07-26 后续模型质量工作：** 使用
> `2026-07-26-depth-surface-quality-optimization.md`。本文件 2026-07-26
> 章节仅保留已完成和已拒绝实验记录。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Metashape-style depth overlay to the active photo and make project masks, three-level depth estimation, fusion, and mesh gating preserve object boundaries and openings.

**Architecture:** A small metadata resolver and renderer produce cached transparent overlays for `CanvasWidget`, while `MainMenu` exposes the depth shortcut and level menu. The MVS core receives an authoritative per-view valid mask, carries it through every pyramid level and persistence, and reports enough quality data for mesh generation to reject fragmented inputs explicitly.

**Tech Stack:** C++17, Qt6 Widgets/QtConcurrent, OpenCV, existing PlaScan depth binary storage, CUDA PatchMatch, GTest, CMake.

---

## File map

- Create `src/gui/views/DepthOverlayData.h/.cpp`: depth-record resolution, level selection, raw depth loading, robust colorization, masked intensity composition.
- Create `src/gui/widgets/DepthOverlayController.h/.cpp`: asynchronous request generation, stale-result rejection, bounded LRU cache.
- Modify `src/gui/views/LayerRenderer.h/.cpp`: one dedicated depth overlay pixmap and optional intensity base replacement.
- Modify `src/gui/widgets/CanvasWidget.h/.cpp`: overlay state, current metadata, controller wiring, image-change refresh.
- Modify `src/gui/menu/MainMenu.h/.cpp`: checkable shortcut and exclusive depth-level/intensity actions.
- Modify `src/gui/main_window/MainWindow.cpp`: menu-to-canvas connections and availability synchronization.
- Modify `src/gui/cmake/GuiSources.cmake` and `tests/CMakeLists.txt`: register the new sources in GUI and tests.
- Modify `src/core/mvs/MvsTypes.h`: add the optional authoritative valid-mask path to `CameraView`.
- Modify `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`: populate masks from `ProjectIO` for both depth-generation entry points.
- Modify `src/core/mvs/DepthPyramidEstimator.h/.cpp`: carry and resize the authoritative valid mask per level.
- Modify `src/core/mvs/DepthMapGenerator.h/.cpp`: preload project masks, clip every result, persist mask source and level quality.
- Modify `src/core/mvs/DepthMapFusion.cpp` and `src/core/mesh/DepthMapMeshBuilder.cpp`: reject masked, unsupported, and inconsistent geometry before meshing.
- Modify `src/core/mvs/MvsQualityReport.h/.cpp`: record mask source, coverage, selected level, and rejection counts.
- Test in `tests/test_gui_project_utils.cpp`, `tests/test_mvs_depth.cpp`, `tests/test_mvs_depth_pyramid.cpp`, `tests/test_mesh_reconstructor.cpp`, and `tests/test_source_contracts.cpp`.

### Task 1: Depth artifact resolver and level semantics

**Files:**
- Create: `src/gui/views/DepthOverlayData.h`
- Create: `src/gui/views/DepthOverlayData.cpp`
- Test: `tests/test_gui_project_utils.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing record-resolution tests**

Add tests that construct two `depth_map_results` records with different normalized path spelling and verify exact reference-image matching, final-level selection, Level 1/2/3 selection, and no cross-image fallback:

```cpp
TEST(DepthOverlayDataTest, ResolvesExactReferenceAndRequestedLevel)
{
    QJsonObject level_one{{QStringLiteral("level"), 1},
                          {QStringLiteral("raw_depth_path"), QStringLiteral("L1.bin")},
                          {QStringLiteral("valid_mask_path"), QStringLiteral("L1_mask.png")}};
    QJsonObject record{{QStringLiteral("ref_image"), QStringLiteral("E:/images/a.png")},
                       {QStringLiteral("raw_depth_path"), QStringLiteral("final.bin")},
                       {QStringLiteral("valid_mask_path"), QStringLiteral("final_mask.png")},
                       {QStringLiteral("pyramid_levels"), QJsonArray{level_one}}};
    QJsonObject metadata{{QStringLiteral("depth_map_results"), QJsonArray{record}}};

    const auto selected = xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("e:\\images\\a.png"), xjw::gui::views::DepthOverlayLevel::Level1);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(selected->rawDepthPath, QStringLiteral("L1.bin"));
    EXPECT_FALSE(xjw::gui::views::resolveDepthOverlayArtifact(
        metadata, QStringLiteral("E:/images/b.png"), xjw::gui::views::DepthOverlayLevel::Final));
}
```

- [ ] **Step 2: Build the test and verify it fails**

Run:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_gui_project_utils -j 8
```

Expected: compilation fails because `DepthOverlayData.h` and `resolveDepthOverlayArtifact` do not exist.

- [ ] **Step 3: Implement the resolver types and exact path matching**

Define:

```cpp
enum class DepthOverlayLevel { Final, Level1, Level2, Level3 };

struct DepthOverlayArtifact
{
    QString referenceImage;
    QString rawDepthPath;
    QString validMaskPath;
    QString previewPath;
    int level = 0;
};

std::optional<DepthOverlayArtifact> resolveDepthOverlayArtifact(
    const QJsonObject &project_metadata,
    const QString &image_path,
    DepthOverlayLevel level);
```

Normalize paths with `QFileInfo::absoluteFilePath`, `QDir::cleanPath`, and case-insensitive comparison on Windows. `Final` reads top-level fields; explicit levels search `pyramid_levels[].level`. Return `std::nullopt` for an unavailable requested level.

- [ ] **Step 4: Rebuild and run the focused tests**

```powershell
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=DepthOverlayDataTest.*
```

Expected: all `DepthOverlayDataTest` cases pass.

- [ ] **Step 5: Commit the resolver**

```powershell
git add src/gui/views/DepthOverlayData.* src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_gui_project_utils.cpp
git commit -m "feat: resolve depth overlay artifacts"
```

### Task 2: Robust depth colorization and mask semantics

**Files:**
- Modify: `src/gui/views/DepthOverlayData.h`
- Modify: `src/gui/views/DepthOverlayData.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write failing image-generation tests**

Cover finite-value filtering, P2/P98 normalization, transparency outside the valid mask, excluded project-mask semantics, and deterministic output size:

```cpp
TEST(DepthOverlayDataTest, ColorizationMakesInvalidPixelsTransparent)
{
    cv::Mat depth = (cv::Mat_<float>(2, 3) << 1.0f, 2.0f, 0.0f,
                                                3.0f, std::nanf(""), 1000.0f);
    cv::Mat valid = (cv::Mat_<uchar>(2, 3) << 255, 255, 255, 255, 255, 0);
    const QImage overlay = xjw::gui::views::colorizeDepthOverlay(depth, valid, 150);
    ASSERT_EQ(overlay.size(), QSize(3, 2));
    EXPECT_EQ(qAlpha(overlay.pixel(2, 0)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(1, 1)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(2, 1)), 0);
    EXPECT_EQ(qAlpha(overlay.pixel(0, 0)), 150);
}
```

- [ ] **Step 2: Run the focused test to verify failure**

Run the Task 1 build and test command. Expected: compile failure for `colorizeDepthOverlay`.

- [ ] **Step 3: Implement loading, normalization, and composition**

Reuse `xjw::core::project::loadDepthMatStorage`. Compute P2/P98 only from finite positive pixels where `valid_mask != 0`. Apply an OpenCV TURBO color map and write `QImage::Format_RGBA8888`, setting invalid alpha to zero. Add:

```cpp
struct DepthOverlayRenderOptions
{
    int opacity = 150;
    bool showIntensity = false;
};

struct DepthOverlayRenderResult
{
    QImage overlay;
    QImage intensityBase;
    QString errorMessage;
};
```

When intensity mode is enabled, resize the source image to the depth grid with `INTER_AREA`, convert to grayscale, and clear excluded pixels. Do not interpret confidence as intensity.

- [ ] **Step 4: Run tests and verify exact alpha behavior**

Expected: all depth overlay colorization cases pass, including an all-invalid matrix returning a descriptive error and a null image.

- [ ] **Step 5: Commit rendering utilities**

```powershell
git add src/gui/views/DepthOverlayData.* tests/test_gui_project_utils.cpp
git commit -m "feat: render masked depth overlays"
```

### Task 3: Asynchronous overlay controller with stale-result guard

**Files:**
- Create: `src/gui/widgets/DepthOverlayController.h`
- Create: `src/gui/widgets/DepthOverlayController.cpp`
- Modify: `src/gui/cmake/GuiSources.cmake`
- Test: `tests/test_gui_project_utils.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing controller-state tests**

Test request generations without sleeping: issue request A, issue request B, then deliver A and verify it is rejected. Also verify cache keys include raw-depth modification time, level, opacity, and intensity mode.

- [ ] **Step 2: Build and confirm the missing-controller failure**

Expected: compilation fails for `DepthOverlayController`.

- [ ] **Step 3: Implement the controller**

Expose:

```cpp
class DepthOverlayController : public QObject
{
    Q_OBJECT
public:
    void setProjectMetadata(const QJsonObject &metadata);
    void request(const QString &image_path,
                 xjw::gui::views::DepthOverlayLevel level,
                 const xjw::gui::views::DepthOverlayRenderOptions &options);
    void cancelPending();
signals:
    void availabilityChanged(bool available);
    void overlayReady(QString image_path, QImage overlay, QImage intensity_base);
    void overlayFailed(QString image_path, QString error_message);
};
```

Use `QtConcurrent::run`, `QFutureWatcher`, `QPointer`, and a monotonic request generation. Bound the cache by total `QImage::sizeInBytes()` to 256 MiB and evict least-recently-used entries.

- [ ] **Step 4: Run focused tests**

Expected: stale request, cache invalidation, and availability tests pass.

- [ ] **Step 5: Commit controller implementation**

```powershell
git add src/gui/widgets/DepthOverlayController.* src/gui/cmake/GuiSources.cmake tests/CMakeLists.txt tests/test_gui_project_utils.cpp
git commit -m "feat: load depth overlays asynchronously"
```

### Task 4: Canvas and renderer integration

**Files:**
- Modify: `src/gui/views/LayerRenderer.h`
- Modify: `src/gui/views/LayerRenderer.cpp`
- Modify: `src/gui/widgets/CanvasWidget.h`
- Modify: `src/gui/widgets/CanvasWidget.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write failing layer-order and persistence tests**

Verify a base image at Z=0, depth overlay at Z=10, feature overlay above Z=20, and that `showImage()` clears the old depth item while preserving the enabled mode for the next request.

- [ ] **Step 2: Run tests to observe missing overlay API**

Expected: compile failure for `setDepthOverlay` and `clearDepthOverlay`.

- [ ] **Step 3: Add a dedicated renderer item**

Add `QGraphicsPixmapItem *_depthOverlayItem` and methods:

```cpp
bool setDepthOverlay(const QImage &overlay, int z = 10);
void clearDepthOverlay();
bool hasDepthOverlay() const noexcept;
```

Replacing the overlay must reuse or safely delete only this item. `clear()` also clears it. Do not append it to the base `_layers` list.

- [ ] **Step 4: Wire CanvasWidget state and controller**

Add slots `setDepthOverlayEnabled(bool)`, `setDepthOverlayLevel(DepthOverlayLevel)`, `setDepthIntensityVisible(bool)`, and `setProjectMetadata(const QJsonObject &)`. On `activeImageChanged`, request the exact matching record if enabled. Disable feature loading only for standalone depth-preview PNGs, not for an original photo with an overlay.

- [ ] **Step 5: Build and run tests**

Expected: renderer Z-order, image-switch clearing, and enabled-state persistence pass.

- [ ] **Step 6: Commit canvas integration**

```powershell
git add src/gui/views/LayerRenderer.* src/gui/widgets/CanvasWidget.* tests/test_gui_project_utils.cpp
git commit -m "feat: overlay depth on active photos"
```

### Task 5: Toolbar shortcut and level menu

**Files:**
- Modify: `src/gui/menu/MainMenu.h`
- Modify: `src/gui/menu/MainMenu.cpp`
- Modify: `src/gui/main_window/MainWindow.cpp`
- Test: `tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write failing action-contract tests**

Verify action text, checkable state, default-off state, exclusive final/Level 1/2/3 group, and the independent `显示强度` toggle.

- [ ] **Step 2: Build and confirm missing action accessors**

Expected: compile failure for `showDepthOverlayAction()` and level accessors.

- [ ] **Step 3: Implement the split shortcut**

Create one toolbar `QToolButton` with `MenuButtonPopup`, a checkable default action named `显示深度图`, and menu actions `所有级别`, `级别 1`, `级别 2`, `级别 3`, separator, `显示强度`. Use the existing icon library or toolbar icon helper; set tooltips rather than adding explanatory visible text.

- [ ] **Step 4: Connect actions to the canvas**

In `MainWindow::setupMenuConnections`, connect the toggle and action group to `CanvasWidget`. Track overall
depth availability separately from final/Level 1/2/3 availability: a missing selected level disables only that
level and must never disable the toolbar button or prevent switching back. Set the action unchecked when no
project is open while retaining the chosen level.

The workspace tree exposes one non-activatable aggregate depth-map node instead of per-frame preview images.
Its context menu deletes the complete depth batch, including persisted pyramid artifacts, while preserving
source photos. GUI depth estimation saves Level 2/3 visualization rasters by default so the level actions are
usable after a new run.

- [ ] **Step 5: Build and run GUI tests**

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target plascan_gui test_gui_project_utils -j 8
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_gui_project_utils.exe --gtest_filter=*DepthOverlay*
```

Expected: build succeeds and all depth-overlay GUI tests pass.

- [ ] **Step 6: Commit the shortcut**

```powershell
git add src/gui/menu/MainMenu.* src/gui/main_window/MainWindow.cpp tests/test_gui_project_utils.cpp
git commit -m "feat: add depth overlay shortcut"
```

### Task 6: Authoritative photo masks as MVS inputs

**Files:**
- Modify: `src/core/mvs/MvsTypes.h`
- Modify: `src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.h`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Test: `tests/test_mvs_depth.cpp`

- [ ] **Step 1: Write failing mask-conversion tests**

Create a project-style mask containing a foreground hole and verify conversion produces `255=valid`, `0=excluded`, preserves the hole, and uses nearest-neighbor resizing.

```cpp
TEST(MvsDepthMaskTest, ConvertsProjectExclusionMaskWithoutClosingOpenings)
{
    cv::Mat project_mask(7, 7, CV_8U, cv::Scalar(255));
    project_mask(cv::Rect(1, 1, 5, 5)).setTo(0);
    project_mask(cv::Rect(3, 2, 1, 3)).setTo(255);
    const cv::Mat valid = xjw::mvs::DepthMapGenerator::projectMaskToValidMask(project_mask, {7, 7});
    EXPECT_EQ(valid.at<uchar>(1, 1), 255);
    EXPECT_EQ(valid.at<uchar>(3, 3), 0);
}
```

- [ ] **Step 2: Run the focused test and confirm failure**

Expected: compile failure for `projectMaskToValidMask`.

- [ ] **Step 3: Add per-view mask paths**

Extend `CameraView` with `std::string validRegionMaskPath`. In both loops that build `CameraView` in `ProjectDenseReconstructionManager.cpp`, call `ProjectIO::findMaskForImage(project_path, image_path)` and assign the path when present.

- [ ] **Step 4: Preload authoritative masks**

Add `_authoritativeValidMasks`. Decode project masks in the existing image preload worker, invert once, and keep their original dimensions. Use automatic content masks only when a view has no project mask. Log mask source and coverage per frame.

- [ ] **Step 5: Run MVS mask tests**

Expected: conversion, no-mask fallback, size mismatch, and doorway-hole preservation tests pass.

- [ ] **Step 6: Commit MVS mask input support**

```powershell
git add src/core/mvs/MvsTypes.h src/core/mvs/DepthMapGenerator.* src/gui/project/manager/ProjectDenseReconstructionManager.cpp tests/test_mvs_depth.cpp
git commit -m "feat: pass photo masks into mvs"
```

### Task 7: Carry masks through all pyramid levels

**Files:**
- Modify: `src/core/mvs/DepthPyramidEstimator.h`
- Modify: `src/core/mvs/DepthPyramidEstimator.cpp`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Test: `tests/test_mvs_depth_pyramid.cpp`

- [ ] **Step 1: Write failing per-level mask tests**

Use a fake backend that intentionally returns valid depth everywhere. Pass a valid mask with a doorway-shaped exclusion and verify Level 3, Level 2, Level 1, and final output remain zero in that region.

- [ ] **Step 2: Run test and verify the current leak**

Expected: test fails because `DepthPyramidRequest` has no valid mask and backend depth survives outside the intended region.

- [ ] **Step 3: Extend pyramid requests**

Add `cv::Mat referenceValidMask` to `DepthPyramidRequest` and `PatchMatchBackendRequest`. Resize it to each working size with `INTER_NEAREST`. Before propagation, zero prior validity outside the mask. After every backend call, zero depth, confidence, support count, and uncertainty outside it.

- [ ] **Step 4: Apply the mask before and after unrectification**

For rectified stereo, rectify the mask with nearest-neighbor sampling. After `unrectifyDepth`, re-apply the original full-resolution valid mask. Never use morphological closing on the authoritative mask.

- [ ] **Step 5: Run pyramid and depth tests**

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release --target test_mvs_depth_pyramid test_mvs_depth -j 8
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "MvsDepthPyramid|MvsDepthMask"
```

Expected: all level masks preserve the exclusion and final output uses the finest successful level.

- [ ] **Step 6: Commit pyramid masking**

```powershell
git add src/core/mvs/DepthPyramidEstimator.* src/core/mvs/DepthMapGenerator.cpp tests/test_mvs_depth_pyramid.cpp
git commit -m "fix: constrain depth pyramid to photo masks"
```

### Task 8: Persist level quality and fallback reasons

**Files:**
- Modify: `src/core/mvs/DepthPyramidEstimator.h`
- Modify: `src/core/mvs/DepthMapGenerator.cpp`
- Modify: `src/core/mvs/MvsQualityReport.h`
- Modify: `src/core/mvs/MvsQualityReport.cpp`
- Modify: `src/core/mvs/MvsWorkspaceManifest.h`
- Modify: `src/core/mvs/MvsWorkspaceManifest.cpp`
- Test: `src/core/mvs/tests/test_mvs_workspace_manifest.cpp`
- Test: `tests/test_mvs_depth_pyramid.cpp`

- [ ] **Step 1: Write failing metadata round-trip tests**

Verify `mask_source`, `mask_coverage`, `selected_level`, `fallback_reason`, valid coverage, support statistics, discontinuity ratio, and rejection counts survive JSON serialization.

- [ ] **Step 2: Run tests to confirm fields are absent**

Expected: assertions fail on missing JSON keys.

- [ ] **Step 3: Add explicit quality fields**

Extend `DepthLevelSummary` with `meanSupportViews` and `depthDiscontinuityRatio`. Extend the persisted frame record with mask source/coverage, selected level, and fallback reason. Populate values from the accepted final level and post-processing statistics.

- [ ] **Step 4: Run metadata and pyramid tests**

Expected: JSON round trips exactly and coarse fallback is always visible in metadata.

- [ ] **Step 5: Commit diagnostics**

```powershell
git add src/core/mvs/DepthPyramidEstimator.h src/core/mvs/DepthMapGenerator.cpp src/core/mvs/MvsQualityReport.* src/core/mvs/MvsWorkspaceManifest.* src/core/mvs/tests/test_mvs_workspace_manifest.cpp tests/test_mvs_depth_pyramid.cpp
git commit -m "feat: report depth pyramid quality"
```

### Task 9: Fusion and mesh quality gate

**Files:**
- Modify: `src/core/mvs/DepthMapFusion.cpp`
- Modify: `src/core/mesh/DepthMapMeshBuilder.cpp`
- Modify: `src/core/mesh/SurfaceReconstructor.cpp`
- Test: `tests/test_mesh_reconstructor.cpp`
- Test: `tests/test_mvs_depth.cpp`

- [ ] **Step 1: Write failing fragmented-sheet tests**

Construct two small depth frames with an intentional doorway, one conflicting rear sheet, and a disconnected island. Verify the conflict and island are rejected while the opening remains empty and generated normals are finite.

- [ ] **Step 2: Run tests and observe current acceptance**

Expected: conflicting geometry or the disconnected island survives into mesh input.

- [ ] **Step 3: Add production geometry gates**

Require authoritative valid mask, configured support count, bounded reprojection disagreement, and local finite depth gradient before producing a fused point. Track rejection reasons. In `DepthMapMeshBuilder`, remove isolated point components and invalid normals before calling Poisson.

- [ ] **Step 4: Improve the failure message**

If fewer than the production minimum of finite oriented points remain, return an error containing input frames, valid depth pixels, accepted fused points, finite normals, and rejection counts. Do not silently use sparse tie points or coarse levels for arbitrary-3D mode.

- [ ] **Step 5: Run mesh and MVS tests**

Expected: opening preservation and finite-normal tests pass; existing aerial height-field tests remain unchanged.

- [ ] **Step 6: Commit fusion gating**

```powershell
git add src/core/mvs/DepthMapFusion.cpp src/core/mesh/DepthMapMeshBuilder.cpp src/core/mesh/SurfaceReconstructor.cpp tests/test_mesh_reconstructor.cpp tests/test_mvs_depth.cpp
git commit -m "fix: gate inconsistent depth geometry"
```

### Task 10: Temple and aerial end-to-end regression

**Files:**
- Create: `scripts/validation/run_depth_overlay_regression.ps1`
- Modify: `docs/PROJECT_ARCHITECTURE.md`
- Modify: `src/core/mvs/README.md`
- Test: `tests/test_cli_contracts.cpp`

- [ ] **Step 1: Add a failing script-contract test**

Require the validation script to accept project path, output directory, scene profile, and optional maximum dimension, and to emit a machine-readable comparison report.

- [ ] **Step 2: Implement the validation script**

The script must run the shared CLI depth/model pipeline for:

```powershell
-Project E:\code\test\temple\temple.plascan -SceneProfile orbital_object
-Project E:\code\test\agisoft_aerial_gcps_small\agisoft_aerial_gcps_small.plascan -SceneProfile aerial_terrain
```

Collect per-level coverage, selected level, accepted mesh points, finite normals, vertices/faces, elapsed time, and peak memory into JSON. Preserve outputs in unique timestamped directories.

- [ ] **Step 3: Build all affected targets**

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release --config Release -j 8
```

Expected: all CLI, GUI, CUDA, and test targets compile successfully.

- [ ] **Step 4: Run focused and broad tests**

```powershell
ctest --test-dir E:\code\plascan\build\windows-vcpkg-cuda-release -C Release --output-on-failure -R "Mvs|Depth|Mesh|Gui|Cli"
```

Expected: all selected tests pass. If the historical `TerrainDemDomTest.TerrainPipelineGeneratesDemDomFromDirectory` failure appears during full `ctest`, record it separately rather than describing the suite as fully passing.

- [ ] **Step 5: Run both visual regressions**

Run the new script for temple and aerial data. Inspect representative source/depth overlays and rendered mesh screenshots. Temple success requires connected roof, columns, doorway, and base without broad layered ridges; aerial success requires continuous terrain without mask-edge clipping.

- [ ] **Step 6: Update documentation**

Document toolbar behavior, metadata keys, project-mask semantics, level responsibilities, CLI validation commands, and residual risks in `docs/PROJECT_ARCHITECTURE.md` and `src/core/mvs/README.md`.

- [ ] **Step 7: Commit validation and documentation**

```powershell
git add scripts/validation/run_depth_overlay_regression.ps1 docs/PROJECT_ARCHITECTURE.md src/core/mvs/README.md tests/test_cli_contracts.cpp
git commit -m "test: validate mask-aware depth reconstruction"
```

## Final verification checklist

- [ ] `git diff --check` reports no whitespace errors in touched files.
- [ ] `plascan_gui` launches with the Windows CUDA runtime and the depth shortcut is initially off.
- [ ] Switching photos with overlay enabled never shows another photo's depth.
- [ ] Final/Level 1/2/3 and intensity modes display correctly over the source photo.
- [ ] Project mask holes remain holes at every pyramid level and in the mesh.
- [ ] Temple and 9-image aerial reports contain selected level, mask source, rejection counts, finite normals, and mesh statistics.
- [ ] No unrelated dirty-worktree changes are staged or reverted.

---

## 2026-07-26 update: direct depth-map global surface experiment

### Task 32: Visibility-aware implicit-field regularization before Marching Cubes

The production path remains `depth maps -> TSDF/implicit field -> Marching Cubes`;
no dense point cloud is introduced. `DepthImplicitFieldRegularizer` is an
experimental, explicitly enabled stage that records its own recovery/update/time
statistics. It uses shared geometry-source evidence, locks original zero values
and signs, and applies robust axial second-order predictions only where at least
two axes have two-sided support. Axial support-gap recovery is independently
configurable and defaults off.

Focused build and tests:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release `
  --target test_mesh_reconstructor mesh_reconstruct_cli -j 3
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_mesh_reconstructor.exe `
  --gtest_filter="DepthImplicitFieldRegularizerTest.*:MeshWorkflowSettingsTest.GlobalImplicitRegularizationIsExplicitAndBounded"
```

Result: passed.

Dino fail-fast A/B, using the same 16 accepted depth frames and GUI-equivalent
60k-face settings:

| Variant | MC faces | Final faces | Open edges | Aspect > 10 | Aspect > 20 | Edge P90 | IoU | SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v71 baseline | 2,173,654 | 116,050 | 944 | 5.244% | 2.489% | 9.243 px | 0.9410 | 0.8600 |
| v73 axial bridge only | 2,539,697 | 90,250 | 2,518 | 7.827% | 3.855% | not run | not run | not run |
| v74 first-order smoothing only | 2,410,290 | 78,093 | 955 | 7.503% | 3.551% | 9.193 px | 0.9375 | 0.8289 |
| v75 robust axial curvature only | 2,534,650 | 82,597 | 775 | 7.247% | 3.137% | 9.500 px | 0.9369 | 0.8248 |

Conclusion: the axial bridge exposes conflicting zero-crossing sheets and is
rejected. Robust curvature regularization reduces open edges but loses image
agreement and worsens triangle aspect ratios, so global implicit regularization
remains disabled by default and Temple is not run for this rejected parameter
family. The next implementation must use the paper's actual visibility
histograms plus adaptive 2:1 octree and primal-dual TGV variables; further
Laplacian/curvature parameter tuning on the regular TSDF grid is not an accepted
substitute.

---

## 2026-07-26 update: visibility histogram, adaptive octree, and sparse TGV

### Task 33: Formal sparse implicit-field experiment

Implemented an explicitly enabled pre-Marching-Cubes stage without introducing
a dense point cloud:

- `DepthVisibilityHistogram` stores eight quantized signed-distance bins in
  exactly eight bytes per dense sample and exposes total weight, robust median,
  dominant-bin ratio, and conflicting-sign ratio.
- `AdaptiveTsdfOctree` merges complete same-sign, low-range 2x2x2 blocks away
  from the zero band, re-splits coarse leaves until all face neighbors satisfy
  the 2:1 rule, and records the sparse face-neighbor graph.
- `SparseTgvSolver` uses primal/dual scalar, vector, and symmetric-tensor
  variables for first- and second-order TGV on that graph. The data term clamps
  the precise fused TSDF within the robust median bin instead of replacing it
  with the bin center.
- Unsupported samples may only be recovered by the existing cell-level
  positive/negative-corner vote with shared geometry sources and a histogram
  conflict limit. Per-voxel neighbor growth was rejected because it created
  small closed bubbles.
- Full-ray back-space signs are available only through the separate
  `tsdfAdaptiveTgvUseGlobalVisibilityField` experiment and default off.

The initial octree face scan allocated an `unordered_set` for every queried
face. Replacing that four-entry set with a fixed array and reusing the last
balance pass for solver neighbors reduced the same Dino octree build from
190.4 s to 117-121 s.

Focused validation:

```powershell
cmake --build E:\code\plascan\build\windows-vcpkg-cuda-release `
  --target test_mesh_reconstructor mesh_reconstruct_cli --config Release -j 3
E:\code\plascan\build\windows-vcpkg-cuda-release\tests\test_mesh_reconstructor.exe `
  --gtest_brief=1
```

Result: 121/121 mesh tests passed. The full Release build also completed.

Dino used the same 16 accepted depth frames, 384 TSDF resolution, and
GUI-equivalent 60k target. `model_quality_cli` used all 16 views and the contact
sheet was inspected after every accepted run:

| Variant | Octree / TGV | Final faces | Open edges | Edge P90 | IoU | SSIM | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| v71 baseline | off | 116,050 | 944 | 9.243 px | 0.9410 | 0.8600 | current reference |
| v76 median-bin center | 190.4 / 21.3 s | 201,684 | 2,264 | 11.500 px | 0.9341 | 0.8255 | reject: quantized zero shift |
| v77 precise sub-bin TSDF | 120.9 / 20.4 s | 94,437 | 373 | 10.542 px | 0.9347 | 0.8237 | topology better, image gate worse |
| v78 per-sample recovery | 121.1 / 5.3 s | 73,978 | 218 | 10.610 px | 0.9350 | 0.8147 | reject: bubble-like support |
| v79 full-ray sign field | 136.2 / 5.6 s | about 497k | 3,446 | 83.670 px | 0.6321 | 0.5228 | reject: inflated visual hull |
| v80 cell-vote recovery | 117.2 / 5.3 s | 95,941 | 295 | 10.651 px | 0.9348 | 0.8223 | topology better, image gate worse |

v80 compressed 33,840,307 histogram samples to 7,525,669 leaves, remained
2:1 balanced, and recovered 2,342 additional cell-verified samples. It still
failed the strict image gate. Contact sheets retained missing ear-root,
underside, and side-surface regions. The adaptive TGV path therefore remains
disabled by default and Temple was intentionally not run for this rejected
parameter family. The remaining blocker is missing or rejected cross-view depth
support at those structures; mesh regularization cannot reconstruct evidence
that never reaches a coherent signed-distance neighborhood.

## 2026-07-26 update: true-pose narrow-band TGV and baseline-managed final denoising

The Dino comparison now uses the imported Middlebury camera poses without
bundle adjustment drift. Moderate depth consistency requires two source-view
confirmations at 1.5% relative depth error. Robust frame rejection retained 11
of 14 loaded frames for TSDF fusion.

`ProcessingBaselineManager` now owns the immutable input snapshot fingerprint,
reference mesh metrics, thresholds, persistence, and candidate comparison. It
rejects input drift, excess faces/open edges, inflated normalized area,
non-manifold or fragmented topology, poor triangle aspect ratios, and adjacent
face normal roughness.

The adaptive TGV graph is restricted to `|TSDF| <= 0.85`. On Dino this reduced
octree construction from about 114 s to 12-17 s while retaining a strong field
update (mean absolute curvature about 0.213 to 0.055). A second normal-aware
surface denoising pass now runs after hole filling and simplification, followed
by one constrained tangential relaxation pass. Boundary vertices and their
protection ring remain fixed. The candidate is accepted only when topology is
unchanged, triangle aspect ratios are safe, and measured normal roughness
improves.

The accepted same-input A/B result was:

| Metric | Narrow-band TGV | + final denoise |
|---|---:|---:|
| Faces | 195,566 | 195,566 |
| Boundary edges | 31,546 | 31,546 |
| Adjacent normal median | 35.44 deg | 33.49 deg |
| Adjacent normal over 30 deg | 52.86% | 51.84% |
| Aspect ratio over 10 | 11.18% | 11.10% |
| Chamfer-L1 to Metashape | 0.0008536 | 0.0008433 |

The strict gate still fails. Metashape has 62,922 faces and only 122 boundary
edges, while PlaScan still has 195,566 faces and 31,546 boundary edges. The
remaining face-count and visual roughness gap is therefore dominated by
unsupported/open surface topology, not by the absence of another unconstrained
mesh smoothing pass. The final denoising change is retained because it improves
roughness and geometric agreement without changing the silhouette or opening
additional holes. At this checkpoint adaptive TGV remained explicit pending
the later contour-band cross-scene validation.

## 2026-07-26 update: contour-band zero-crossing diagnosis

Contour-band recovery was evaluated on the same true-pose Dino input with the
11 robustly retained frames. Restoring all 14 loaded frames increased
conflicting surfaces, while lowering the geometry cell vote from two to one
increased fragments; both variants were rejected.

The conservative contour-band variant retained the two-vote rule and reduced
Dino faces from 195,566 to 175,061, boundary edges from 31,546 to 26,287, and
the adjacent-normal median from 33.49 to 29.51 degrees. On Temple it produced
one component, reached 60,336 faces, reduced boundary edges from 7,164 to
4,634, and reduced aspect-ratio-over-10 faces from 13.08% to 6.21%.

New rejection counters split contour candidates by observation weight,
cross-view source overlap, inverse-depth spread, free-space consistency,
neighborhood continuity, geometry support, and signed-neighbor pairing. On
Dino, all 19,962 free-space rejections were caused by the absolute TSDF limit,
not by the surface-observation ratio.

An explicit contour-only absolute TSDF limit was added for controlled
experiments. Raising it from 0.45 to 0.55 improved Dino component count from
six to three and Chamfer-L1 from 0.0008652 to 0.0008578, but slightly increased
boundary edges and high-aspect triangles. More importantly, the same value
regressed Temple to 95,936 faces, 16 components, and 10,906 boundary edges.
Therefore 0.55 is rejected as a default, the validated 0.45 behavior remains
unchanged, and future relaxation must be derived from cross-view evidence
rather than a dataset-independent constant.

A follow-up two-stage experiment first evaluated each scene at 0.45 and only
allowed a 0.55 second pass when the conservative contour candidate rejection
ratio was at most 52% and its recovery ratio was at least 7%. It correctly
kept Temple at 0.45, but the accepted Dino second pass regressed boundary edges
from 26,287 to 26,703, aspect-ratio-over-10 faces from 11.79% to 12.00%, and
the adjacent-normal median from 29.51 to 30.07 degrees. The adaptive branch was
therefore removed. Voxel-level acceptance statistics are insufficient for
predicting final mesh quality; any future relaxation needs a mesh-level
candidate/rollback gate or stronger upstream cross-view depth evidence.

The stored depth artifacts show that about 99% of pixels inside each Dino
foreground mask retain a valid depth, but only 47-75% of one-pixel contour
samples have four or more geometry confirmations. Lowering boundary recovery
from four confirmations to three was tested while retaining the two-source,
1% inverse-depth-spread, and two-cell-vote gates. It regressed Temple to
96,840 faces, 18 components, and 10,892 boundary edges, and regressed Dino to
194,238 faces, eight components, and 31,094 boundary edges. The variant and
its temporary configuration were removed. The remaining missing surface is
not caused by a simple valid-depth shortage; weak contour observations disagree
about the signed surface and must be grouped into a coherent sheet before they
can safely contribute.

Increasing the visibility-verified final-hole limit from 192 edges / 48 voxels
to 256 edges / 64 voxels preserved Temple's single component and reduced its
boundary edges slightly from 4,634 to 4,593. Dino was bit-for-bit unchanged at
175,061 faces and 26,287 boundary edges: its remaining openings are not closed
loops in this size interval. The temporary configuration was removed and the
default limit was not changed.

The final same-coordinate comparison against the current high-detail orbital
GUI policy established that the conservative 0.45 contour-band variant with
narrow-band TGV is a material default improvement even though the MetaShape
strict gate is not yet met:

| Dino metric | Current GUI policy | TGV + contour 0.45 |
|---|---:|---:|
| Faces | 375,735 | 175,061 |
| Boundary edges | 51,263 | 26,287 |
| Adjacent-normal median | 44.01 deg | 29.51 deg |
| Components | 13 | 6 |
| Chamfer-L1 to MetaShape | 0.0008407 | 0.0008652 |

The topology and roughness gains are large while the absolute geometry-distance
change is small. High-detail orbital defaults now enable adaptive narrow-band
TGV and conservative contour-band zero-crossing support when the user has not
provided an explicit override. Explicit `false` settings continue to disable
either stage. The rejected 0.55, three-confirmation, and enlarged-hole variants
remain excluded.

### Rejected sample-level surface-sheet clustering

A follow-up experiment grouped weak contour candidates as connected TSDF
samples and required each group to touch multiple already-supported anchors.
The implementation and focused unit tests were used only to test whether this
could make the rejected three-confirmation boundary evidence safe.

It did not. Thin physical surfaces commonly cross several zero-crossing cells
without producing a face-connected set of candidate corner samples. The
sample graph therefore split one surface into small groups and removed useful
support:

| Scene / variant | Faces | Boundary edges | Components | High-aspect faces | Normal median |
|---|---:|---:|---:|---:|---:|
| Temple, conservative reference | 60,336 | 4,634 | 1 | 6.21% | 16.34 deg |
| Temple, three confirmations + sheets | 99,600 | 17,524 | 18 | 18.40% | 24.47 deg |
| Dino, conservative reference | 175,061 | 26,287 | 6 | 11.79% | 29.51 deg |
| Dino, four confirmations + sheets | 195,419 | 31,337 | 7 | 11.10% | 33.34 deg |

The candidate-level feature, settings, diagnostics, tests, and temporary
configuration were removed after the regression. The next viable formulation
must use zero-crossing cells or extracted faces as graph nodes, preserve all
corners belonging to an accepted sheet, and pass both Dino and Temple
mesh-level rollback gates before it can change the default.

### Zero-crossing cell-sheet recovery

The replacement implementation uses zero-crossing cells as graph nodes. Cells
are connected only across a shared face, must request the same missing TSDF
sign, and must share at least one geometry source. A component is accepted
only when it contains at least three cells and at least two distinct cells
touch an already extractable zero-crossing cell. All eligible corners belonging
to an accepted component are then restored together.

Compared with the conservative TGV + contour 0.45 reference:

| Metric | Dino reference | Dino cell sheets | Temple reference | Temple cell sheets |
|---|---:|---:|---:|---:|
| Faces | 175,061 | 170,106 | 60,336 | 60,278 |
| Boundary edges | 26,287 | 24,880 | 4,634 | 4,654 |
| Components | 6 | 6 | 1 | 1 |
| High-aspect faces | 11.79% | 11.64% | 6.21% | 6.19% |
| Adjacent-normal median | 29.51 deg | 29.71 deg | 16.34 deg | 16.15 deg |

Dino Chamfer-L1 to MetaShape changed from 0.0008652 to 0.0008833. The
registered contact sheet showed no silhouette collapse; the distance increase
is the controlled cost of restoring more complete thin-surface topology.

Two stricter variants were rejected. Requiring four cells and three anchors
reduced the Dino boundary improvement to only 139 edges while Chamfer still
increased to 0.0008766. Restricting single-cell-vote endpoints to
`|TSDF| <= 0.25` improved Chamfer to 0.0008475 but split Dino into eight
components and increased boundary edges to 30,830. The accepted high-detail
orbital default therefore uses the complete three-cell/two-anchor sheet and
keeps an explicit `tsdfGeometryZeroCrossingCellSheets=false` rollback switch.
