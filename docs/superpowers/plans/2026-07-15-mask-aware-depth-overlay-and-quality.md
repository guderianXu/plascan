# Mask-aware Depth Overlay and Reconstruction Quality Implementation Plan

> **Superseded for remaining work on 2026-07-17.** The approved implementation is split into
> `2026-07-17-depth-overlay-workspace-quality-fixes.md` followed by
> `2026-07-17-direct-depth-tsdf-surface.md`. The older fusion/Poisson and automatic commit steps
> below are retained only as historical context and must not be executed.

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
