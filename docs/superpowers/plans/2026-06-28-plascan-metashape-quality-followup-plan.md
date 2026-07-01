# PlaScan Metashape-Comparable Quality Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PlaScan's aerial reconstruction results explainable and progressively closer to Metashape output by fixing match reuse, sparse quality, MVS depth quality, dense fusion, terrain mesh generation, and production-grade reporting.

**Architecture:** Treat every reconstruction stage as a producer of explicit metadata and quality reports. Downstream stages must consume those reports instead of guessing from file presence, and GUI actions must schedule background jobs without rerunning expensive upstream work unless a report says data is missing or stale.

**Tech Stack:** C++17, Qt6, OpenCV, CUDA/LibTorch, GDAL, GTest, Python comparison scripts, Windows CUDA build script `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`.

---

## File Structure

- `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.*`
  - Owns sparse workflow preflight: existing valid matches, known failed pairs, missing pairs, stale inputs, and recommended action.
- `E:/code/plascan/src/core/pipeline/SFMService.*`
  - Reuses completed matching results, runs missing-pair-only gap filling, executes guided rematching, and writes SfM quality metadata.
- `E:/code/plascan/src/core/pipeline/GuidedRematchService.*`
  - Generates append-only pose/epipolar guided matches and never replaces stable matches.
- `E:/code/plascan/src/core/sfm/quality/*`
  - Stores graph connectivity, track length, reprojection error, triangulation angle, BA, and registration health metrics.
- `E:/code/plascan/src/core/mvs/DepthMapGenerator.*`
  - Owns source-view selection, memory-adaptive scheduling, depth confidence, valid masks, cancellation, and depth post-processing.
- `E:/code/plascan/src/core/mvs/DepthMapFusion.*`
  - Owns confidence-aware dense fusion, color retention, normal estimation, and multi-view consistency filtering.
- `E:/code/plascan/src/core/mvs/MvsQualityReport.*`
  - Writes depth, fusion, dense-cloud, and mesh quality reports consumed by GUI and downstream terrain products.
- `E:/code/plascan/src/core/mesh/*`
  - Should hold terrain-mesh specific reconstruction, smoothing, spike filtering, and product identity logic.
- `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
  - Schedules MVS/fusion/mesh jobs asynchronously, with progress, cancellation, output registration, and explicit metadata refresh.
- `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
  - Schedules DEM/DOM jobs asynchronously and consumes exact dense/mesh output IDs instead of `last()` metadata guesses.
- `E:/code/plascan/src/gui/widgets/DataTreeWidget.cpp`
  - Shows products by product kind and natural filename order.
- `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
  - Compares PlaScan dense cloud/mesh products with Metashape or LiDAR references.
- `E:/code/plascan/docs/reports/`
  - Stores reproducible reports for the Agisoft aerial GCP comparison.

---

## Phase 1: Establish A Reliable Baseline

### Task 1: Compare PlaScan dense products against Metashape output

**Files:**
- Modify: `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
- Modify: `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py`
- Create: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_metashape_comparison.md`

- [ ] Add comparison modes for point-to-point, point-to-plane, and 2.5D grid height residuals.
- [ ] Record RMSE, median, NMAD, P84, P95, max, coverage, density, and vertical roughness.
- [ ] Compare these inputs:

```text
PlaScan project: E:/code/test/agisoft_aerial_gcps
Metashape sample: E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply
Metashape full: E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_with_normals.ply
```

- [ ] Run:

```powershell
python -m pytest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py -q
```

- [ ] Expected result: comparison tests pass and the report states whether PlaScan's main error is sparse pose, depth noise, fusion noise, meshing, or visualization.

---

## Phase 2: Stop Repeating Matching Work

### Task 2: Make aerial triangulation reuse completed matches

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.h`
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Test: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] Classify pair state as `existing_valid_match`, `settled_no_match`, `failed_geometry`, `missing_match`, or `stale_due_to_feature_change`.
- [ ] If matching has already completed, aerial triangulation must run SfM from existing valid matches and not trigger full feature matching again.
- [ ] If only some pairs are missing, run gap filling only for missing/stale pairs.
- [ ] Log the preflight recommendation JSON in GUI before starting SfM.
- [ ] Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|SfmSparseResultMetadata|GuiProject'
```

- [ ] Expected result: tests pass and the GUI no longer starts a full rematch after a finished matching stage.

---

## Phase 3: Improve Sparse Reconstruction Quality

### Task 3: Upgrade guided rematching from append-only support to pose-guided recovery

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/GuidedRematchService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Test: `E:/code/plascan/tests/test_guided_rematch_service.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] Keep the current append-only rule: guided matches may add high-confidence observations but must not replace stable existing matches.
- [ ] Prefer camera-pose-derived epipolar geometry when both cameras are registered; fall back to F estimated from existing matches only when pose is unavailable.
- [ ] Add diagnostics for attempted pairs, added matches, rejected matches, second-pass registered images, and second-pass reprojection error.
- [ ] Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematch|SfmGuided|SfmSparseResultMetadata'
```

- [ ] Expected result: guided rematch improves or preserves registered image count and never degrades the accepted initial reconstruction.

### Task 4: Add track thinning, splitting, and scale-aware BA

**Files:**
- Modify: `E:/code/plascan/src/core/sfm/quality/*`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Test: `E:/code/plascan/tests/test_sfm_quality_report.cpp`

- [ ] Score tracks by length, reprojection residual, triangulation angle, matcher source, and spatial coverage.
- [ ] Split conflicting tracks instead of dropping entire connected components.
- [ ] Thin tracks per image grid so BA is not dominated by one textured patch.
- [ ] Weight BA observations by keypoint scale and track confidence.
- [ ] Expected result: sparse cloud has fewer line-like artifacts, more even image coverage, and a lower P95 reprojection residual.

---

## Phase 4: Improve MVS Depth Quality

### Task 5: Make source-view selection Metashape-like

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_source_planner.cpp`

- [ ] Rank source views by shared tracks, geometric inliers, baseline angle, overlap score, texture score, and pose uncertainty.
- [ ] Avoid using views with too little overlap or too small/large baseline.
- [ ] Store selected sources per depth frame in metadata.
- [ ] Expected result: fewer isolated red depth spikes and fewer invalid black holes caused by bad source views.

### Task 6: Add confidence, valid-mask, and speckle cleanup gates

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsTypes.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.cpp`
- Test: `E:/code/plascan/tests/test_mvs_types.cpp`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_workspace_manifest.cpp`

- [ ] Output raw depth, confidence, valid mask, and preview for every depth frame.
- [ ] Remove small isolated speckles and local extreme-depth spikes before fusion.
- [ ] Keep metadata for device, elapsed time, source images, valid ratio, and confidence histogram.
- [ ] Expected result: depth maps still preserve true terrain changes but suppress isolated high/low noise points.

---

## Phase 5: Fix Dense Fusion And Mesh Generation

### Task 7: Make dense fusion confidence-aware and color-safe

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapFusion.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.cpp`
- Test: `E:/code/plascan/tests/test_mvs_pipeline.cpp`

- [ ] Fuse only depth samples supported by enough consistent source views.
- [ ] Keep RGB color, confidence, normal, and source count in fused points.
- [ ] Record density and coverage, not just total point count.
- [ ] Expected result: dense cloud is denser than sparse cloud but not filled with unsupported noisy points.

### Task 8: Separate dense cloud, terrain mesh, and 3D model product identity

**Files:**
- Modify: `E:/code/plascan/src/gui/project/support/ProjectResultRecords.cpp`
- Modify: `E:/code/plascan/src/gui/widgets/DataTreeWidget.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] A point-only PLY must appear under dense cloud, not under 3D model.
- [ ] A real mesh product must contain faces and be registered as `mesh`.
- [ ] Terrain-mesh output should use 2.5D terrain constraints for nadir aerial data before generic 3D meshing.
- [ ] Expected result: the GUI never labels dense cloud as a mesh, and terrain mesh roughness is reduced before texture/DEM/DOM stages.

---

## Phase 6: Make GUI Workflows Production-Safe

### Task 9: Move long terrain and mesh jobs into cancellable background tasks

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectDashboardSummary.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] Use guarded `QPointer` callbacks for background tasks.
- [ ] Pass exact output IDs and paths through task completion signals.
- [ ] Progress bars must advance by actual stage progress, not by fixed placeholders.
- [ ] Cancel must stop scheduling new frames/jobs and return GUI state to idle.
- [ ] Expected result: depth, dense cloud, mesh, DEM, and DOM generation can be cancelled without hanging the GUI.

---

## Phase 7: Production Mapping Features

### Task 10: Add survey-grade control and reporting milestones

**Files:**
- Modify or create under: `E:/code/plascan/src/core/control/`
- Modify or create under: `E:/code/plascan/src/gui/dialogs/`
- Update: `E:/code/plascan/docs/PROJECT_ARCHITECTURE.md`

- [ ] Add GCP/check point tables with residuals, sigma, enabled/disabled state, and control/check classification.
- [ ] Add scale-bar constraints for close-range projects.
- [ ] Add CRS/geoid validation for DEM/DOM export.
- [ ] Add DOM color/seam/ghosting diagnostics after geometry stabilizes.
- [ ] Expected result: PlaScan starts moving from "can reconstruct" toward "can deliver survey products".

---

## Phase 8: Verification And Release

### Task 11: Run focused and full validation before tagging

**Files:**
- Update: `E:/code/plascan/CHANGELOG.md`
- Update or create: `E:/code/plascan/docs/releases/v1.1.7.md`

- [ ] Run focused tests:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematch|ReconstructionPrerequisite|SfmQuality|MvsQuality|MvsDepth|FusionDepth|MvsWorkspace|ProjectResult|GuiProject|FeaturePair'
```

- [ ] Run Python comparison tests:

```powershell
python -m pytest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py -q
```

- [ ] Run full CTest if focused tests pass:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure
```

- [ ] Run one real Agisoft aerial GCP reconstruction pass with fixed parameters and save reports under `E:/code/plascan/docs/reports/`.
- [ ] Only after evidence is recorded, update changelog, release notes, tag, and GitHub Release body.

---

## Priority Order

1. Stop redundant matching in aerial triangulation.
2. Produce a PlaScan-vs-Metashape comparison report for the current project.
3. Improve guided rematching and track quality.
4. Improve MVS source selection and depth cleanup.
5. Make dense fusion confidence-aware.
6. Generate terrain mesh with a terrain-specific path instead of treating dense cloud as arbitrary 3D surface.
7. Fix GUI cancellation/progress/product identity.
8. Add survey-grade control, CRS, and DOM production features.

This order keeps the pipeline measurable: first stop wasting work, then identify where the quality gap comes from, then improve the stage that actually causes the gap.
