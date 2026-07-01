# PlaScan Production Quality Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring PlaScan's aerial photogrammetry workflow closer to a production-grade Metashape-like pipeline while keeping the current Windows CUDA build stable.

**Architecture:** The near-term path is metadata-driven and test-first: preserve upstream products instead of recomputing them, attach quality reports to every stage, then use those reports to guide SfM, MVS, mesh, DEM, and DOM. GUI managers should orchestrate background tasks only; core algorithms should expose deterministic services and JSON reports that the GUI can display.

**Tech Stack:** C++17, Qt6, CMake/Ninja, OpenCV, LibTorch/CUDA, GDAL, GTest, Python unittest, Windows CUDA build script `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`.

---

## File Structure

- Modify `E:/code/plascan/src/core/pipeline/SFMService.cpp` and `SFMService.h` for matching reuse, missing-pair classification, stage diagnostics, and opt-in guided rematching.
- Modify `E:/code/plascan/src/core/pipeline/FeatureMatchRunner.cpp` and `.h` for pair state input/output, retry policy, and skipped/failed reason recording.
- Modify `E:/code/plascan/src/core/pipeline/SfmPairPlanner.h` and related tests so overlap planning is the single source of truth for GUI and CLI.
- Modify `E:/code/plascan/src/core/sfm/MultiViewTrackBuilder.*`, `BundleAdjuster.*`, and reporting helpers for track scoring, thinning, and scale-aware BA weighting.
- Modify `E:/code/plascan/src/core/mvs/DepthMapGenerator.*`, `DepthMapFusion.*`, `MvsSourcePlanner.*`, `MvsWorkspaceManifest.*`, and `MvsQualityReport.*` for source view quality, confidence/valid-mask metadata, cancellation, memory guardrails, and fusion confidence.
- Modify `E:/code/plascan/src/core/mesh/*` or create focused mesh product helpers if current mesh logic is embedded in managers; do not keep adding large blocks to GUI managers.
- Modify `E:/code/plascan/src/core/terrain/*` for DEM/DOM production quality, seam/ghost diagnostics, CRS/geotransform validation, and quality artifacts.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`, `ProjectDenseReconstructionManager.cpp`, and `ProjectTerrainProductsManager.cpp` only as orchestration layers.
- Modify `E:/code/plascan/src/gui/dialogs/*` only to expose real parameters and reports; avoid algorithm logic in dialogs.
- Add or update tests under `E:/code/plascan/tests/`, `E:/code/plascan/src/core/mvs/tests/`, and Python unittest files in `E:/code/plascan/tests/*.py`.
- Update `E:/code/plascan/README.md`, `E:/code/plascan/CHANGELOG.md`, `E:/code/plascan/docs/PROJECT_ARCHITECTURE.md`, and `E:/code/plascan/docs/releases/vX.Y.Z.md` when a stage becomes user-visible.

---

## Phase 1: Stop Wasteful Reprocessing And Make State Reliable

### Task 1: Reuse completed matching results during aerial triangulation

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.h`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write the failing test**

Add a test that simulates the current bug: project metadata contains completed feature/match records, then aerial triangulation must not schedule a full matching pass.

```cpp
TEST(AerialTriangulationWorkflowTest, CompletedMatchesDisableFullAutoRematch)
{
    const QString manager_path =
        QStringLiteral("E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp");
    QFile file(manager_path);
    ASSERT_TRUE(file.open(QIODevice::ReadOnly | QIODevice::Text));
    const QString body = QString::fromUtf8(file.readAll());

    EXPECT_TRUE(body.contains("options.autoGenerateMissingMatches = false"));
    EXPECT_TRUE(body.contains("autoGenerateMissingMatches = true"));
    EXPECT_TRUE(body.contains("missing") || body.contains("缺失"));
}
```

- [ ] **Step 2: Run test to verify it fails on missing guard**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'AerialTriangulationWorkflowTest.CompletedMatchesDisableFullAutoRematch'
```

Expected: FAIL if the manager cannot prove it disables full auto-rematch when matching prerequisites exist.

- [ ] **Step 3: Implement missing-pair-only behavior**

In `ProjectSparseReconstructionManager.cpp`, set `options.autoGenerateMissingMatches = false` for normal aerial triangulation. Only set it to true inside the explicit “自动补齐缺失步骤” path, after logging how many missing pairs are being generated.

In `SFMService.cpp`, separate pair states:

```cpp
enum class PairInputState
{
    ExistingValidMatch,
    ExistingInvalidMatch,
    MissingFeature,
    MissingMatch,
    SkippedByPlanner,
    FailedGeometry
};
```

Generate new matches only for `MissingMatch` pairs when `autoGenerateMissingMatches` is true. Never regenerate `ExistingValidMatch` pairs unless the user explicitly chooses a “重新匹配/清空并重跑” command.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'AerialTriangulationWorkflowTest|FeatureMatch|SfmPairPlanner'
```

Expected: PASS. Logs should show existing matches reused and only missing pairs queued.

### Task 2: Add a single reconstruction prerequisite report

**Files:**
- Create: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.h`
- Create: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/CMakeLists.txt`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`

- [ ] **Step 1: Write failing tests for complete, partial, and stale matching states**

Test cases:

```cpp
TEST(ReconstructionPrerequisiteReportTest, CompleteMatchesAreReadyForTriangulation)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 984;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 0;
    report.failedGeometryPairCount = 592;

    EXPECT_TRUE(report.hasEnoughUpstreamData());
    EXPECT_FALSE(report.shouldRunFullRematch());
}

TEST(ReconstructionPrerequisiteReportTest, MissingMatchesRequireOnlyGapFill)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1568;
    report.validMatchPairCount = 900;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 84;
    report.failedGeometryPairCount = 584;

    EXPECT_TRUE(report.hasEnoughUpstreamData());
    EXPECT_TRUE(report.shouldOfferGapFill());
    EXPECT_FALSE(report.shouldRunFullRematch());
}
```

- [ ] **Step 2: Implement report JSON**

Fields must include:

```json
{
  "image_count": 444,
  "planned_pair_count": 1568,
  "valid_match_pair_count": 984,
  "missing_feature_pair_count": 0,
  "missing_match_pair_count": 84,
  "failed_geometry_pair_count": 584,
  "recommended_action": "run_sfm_with_existing_matches"
}
```

- [ ] **Step 3: Display report in GUI log before AT starts**

GUI log must say one of:

- `空三上游数据就绪：复用已有匹配 ...`
- `空三缺少部分匹配：只补齐缺失 pair ...`
- `空三缺少特征/匹配：需要先运行特征提取/匹配 ...`

- [ ] **Step 4: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|AerialTriangulationWorkflowTest'
```

Expected: PASS.

---

## Phase 2: Improve Sparse Reconstruction Quality

### Task 3: Unify pair planning for GUI and CLI

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SfmPairPlanner.h`
- Modify: `E:/code/plascan/src/core/pipeline/SfmPairPlanner.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_sfm_pair_planner.cpp`

- [ ] **Step 1: Add pair source tests**

Require `PairPlan` records to include:

```cpp
EXPECT_EQ(pair.source, PairPlanSource::SequenceWindow);
EXPECT_EQ(pair.status, PairPlanStatus::Pending);
EXPECT_FALSE(pair.reason.empty());
```

Cover sequence-window, camera-neighbor, footprint-overlap, manual-overlap, and BoW candidates.

- [ ] **Step 2: Remove duplicate GUI pair-planning logic**

GUI should call core planner and consume `PairPlan` records. No GUI-only planner should make different decisions for the same project.

- [ ] **Step 3: Run planner tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmPairPlanner|FeaturePairPlanner'
```

Expected: PASS.

### Task 4: Add guided rematching v1 after initial SfM

**Files:**
- Create: `E:/code/plascan/src/core/pipeline/GuidedRematchService.h`
- Create: `E:/code/plascan/src/core/pipeline/GuidedRematchService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/core/feature_match/MatchOutlierRejector.*`
- Test: `E:/code/plascan/tests/test_guided_rematch_service.cpp`

- [ ] **Step 1: Add tests for epipolar-band search eligibility**

Test must require:

- both cameras registered,
- pair has overlap score above threshold,
- existing inliers below target,
- pair is not marked permanently invalid.

- [ ] **Step 2: Implement read-only guided candidate generation**

For v1, generate candidate pairs and local search windows only. Do not replace stable existing matches. New matches must be marked `source = guided_rematch`.

- [ ] **Step 3: Add GUI option**

Expose a checkbox in the sparse reconstruction settings:

- label: `初始空三后补充引导匹配`
- default: enabled for aerial datasets with known camera priors,
- log each guided rematch batch.

- [ ] **Step 4: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematch|Sfm|FeatureMatch|MatchOutlier'
```

Expected: PASS.

### Task 5: Track scoring, thinning, and scale-aware BA

**Files:**
- Modify: `E:/code/plascan/src/core/sfm/MultiViewTrackBuilder.h`
- Modify: `E:/code/plascan/src/core/sfm/MultiViewTrackBuilder.cpp`
- Modify: `E:/code/plascan/src/core/sfm/BundleAdjuster.h`
- Modify: `E:/code/plascan/src/core/sfm/BundleAdjuster.cpp`
- Test: `E:/code/plascan/tests/test_multiview_track_builder.cpp`
- Test: `E:/code/plascan/tests/test_bundle_adjuster_quality_weights.cpp`

- [ ] **Step 1: Add track quality tests**

Each track gets:

- `track_length`
- `mean_reprojection_error_px`
- `triangulation_angle_deg`
- `spatial_grid_cell`
- `source_matcher`
- `confidence`

Test that longer, well-distributed, lower-error tracks survive thinning.

- [ ] **Step 2: Add BA observation sigma**

BA residual weight:

```cpp
weight = track_confidence / std::max(0.25, sigma_px * sigma_px);
```

LoFTR/RoMa-like dense match observations must be capped so they do not dominate sparse keypoints.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MultiViewTrack|BundleAdjust|Sfm'
```

Expected: PASS.

---

## Phase 3: Improve MVS, Dense Cloud, And Mesh

### Task 6: Make MVS quality metadata first-class

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapFusion.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.cpp`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_workspace_manifest.cpp`
- Test: `E:/code/plascan/tests/test_mvs_pipeline.cpp`

- [ ] **Step 1: Require depth, confidence, valid mask, and preview records**

Every completed frame record must contain:

```json
{
  "depth_png": "depth_222.png",
  "raw_depth_path": "depth_222.f32",
  "raw_confidence_path": "confidence_222.f32",
  "valid_mask_path": "mask_222.png",
  "ref_image": "2019_08_06_...",
  "source_images": ["..."],
  "status": "completed",
  "elapsed_ms": 13380,
  "device": "GPU",
  "depth_quality": {}
}
```

- [ ] **Step 2: Keep cancellation responsive**

Add cancellation checks before and after:

- source image loading,
- hint generation,
- patchmatch iterations,
- filtering,
- disk write.

Expected user-facing log: `密集重建取消完成，已保存 N/M 个有效深度图记录。`

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsWorkspaceManifest|MvsDepthPostprocess|MvsPipeline'
```

Expected: PASS.

### Task 7: Fix mesh-from-dense-cloud production path

**Files:**
- Modify or create focused helpers under `E:/code/plascan/src/core/mesh/`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_mesh_reconstruction_quality.cpp`

- [ ] **Step 1: Add tests for mesh input type**

The GUI tree must not label raw dense point cloud as a 3D mesh. A mesh result must have faces:

```cpp
EXPECT_GT(mesh_result.faceCount, 0);
EXPECT_EQ(mesh_result.kind, ProjectAssetKind::Mesh);
EXPECT_NE(mesh_result.sourcePointCloudPath, mesh_result.meshPath);
```

- [ ] **Step 2: Add terrain mesh mode**

For aerial nadir data, prefer a 2.5D terrain mesh mode:

- grid or TIN from dense cloud,
- robust height aggregation,
- median/bilateral smoothing,
- optional texture projection,
- no giant spiky Poisson surface by default.

- [ ] **Step 3: Add memory-aware streaming**

Use tile scanning and chunked vertex buffers. For a 64 GB host, default to high-quality streaming terrain mesh, not full in-memory global reconstruction.

- [ ] **Step 4: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Mesh|DenseCloud|Terrain'
```

Expected: PASS.

---

## Phase 4: Add Production Photogrammetry Controls

### Task 8: GCP, checkpoints, and scale bars

**Files:**
- Create: `E:/code/plascan/src/core/control/ControlPointSet.h`
- Create: `E:/code/plascan/src/core/control/ControlPointSet.cpp`
- Create: `E:/code/plascan/src/core/control/ScaleBarSet.h`
- Create: `E:/code/plascan/src/core/control/ScaleBarSet.cpp`
- Modify: `E:/code/plascan/src/core/sfm/BundleAdjuster.*`
- Modify: GUI reference tab files under `E:/code/plascan/src/gui/`
- Test: `E:/code/plascan/tests/test_control_points.cpp`

- [ ] **Step 1: Add import and residual tests**

Control points must support:

- enabled/disabled,
- control/check classification,
- XYZ sigma,
- image observation sigma,
- residual report.

- [ ] **Step 2: Add BA constraints**

Use robust loss and sigma-aware residuals. Checkpoints must report residuals but not constrain BA.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ControlPoint|ScaleBar|BundleAdjust'
```

Expected: PASS.

### Task 9: CRS, geoid, and product metadata

**Files:**
- Modify: `E:/code/plascan/src/core/terrain/*`
- Modify: `E:/code/plascan/src/core/project/*`
- Modify: GUI project metadata panels
- Test: `E:/code/plascan/tests/test_project_crs_metadata.cpp`

- [ ] **Step 1: Add CRS metadata tests**

Project metadata must preserve:

- EPSG or WKT,
- horizontal unit,
- vertical datum,
- geoid grid name if used,
- product transform.

- [ ] **Step 2: Validate DEM/DOM exports**

DEM/DOM export must fail loudly if georeferencing is missing or inconsistent, and the dialog must show the effective CRS.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Crs|Geo|Terrain|Dem|Dom'
```

Expected: PASS.

---

## Phase 5: GUI Responsiveness And Large Data Handling

### Task 10: Standardize GUI background task lifecycle

**Files:**
- Create: `E:/code/plascan/src/gui/tasks/GuiTaskRunner.h`
- Create: `E:/code/plascan/src/gui/tasks/GuiTaskRunner.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add static lifecycle tests**

Tests must reject new raw `QtConcurrent::run([this` captures in managers. Background callbacks must use `QPointer`.

- [ ] **Step 2: Implement runner**

Runner responsibilities:

- capture `QPointer<QObject>`,
- expose cancellation token,
- deliver progress safely to GUI thread,
- ignore callbacks after project/window destruction,
- log task completion/cancel/failure consistently.

- [ ] **Step 3: Run GUI tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Gui|Project|DenseCloudDialog|AerialTriangulationWorkflow'
```

Expected: PASS.

### Task 11: Large PLY loading and display

**Files:**
- Modify: `E:/code/plascan/src/gui/dialogs/CameraModel3DDialog.cpp`
- Create focused PLY loader helper if needed under `E:/code/plascan/src/gui/scene/`
- Test: `E:/code/plascan/tests/test_large_ply_loading.cpp`

- [ ] **Step 1: Add tests for huge PLY header and streaming count**

The loader must report progress and avoid `bad allocation` for huge files by using chunked reads and display decimation.

- [ ] **Step 2: Separate asset type display**

Dense cloud, sparse cloud, and mesh must be distinct in the tree and viewer:

- dense cloud: points only,
- mesh: vertices plus faces,
- DEM/DOM: raster products.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Ply|CameraModel3D|ProjectWorkflow'
```

Expected: PASS.

---

## Phase 6: Baseline Evaluation Against Metashape Data

### Task 12: Add repeatable A/B quality report for `agisoft_aerial_gcps`

**Files:**
- Create: `E:/code/plascan/scripts/quality/compare_dense_clouds.py`
- Create: `E:/code/plascan/scripts/quality/run_agisoft_aerial_gcps_baseline.ps1`
- Modify: `E:/code/plascan/docs/releases/vX.Y.Z.md` when publishing results
- Test: `E:/code/plascan/tests/test_quality_scripts.py`

- [ ] **Step 1: Add script tests**

Tests must verify:

- missing input path fails with clear error,
- PLY header is parsed,
- metrics JSON contains `rmse`, `p50`, `p84`, `p95`, `coverage_ratio`, and `sample_count`.

- [ ] **Step 2: Implement comparison script**

Default inputs:

```powershell
$PlaScanCloud = "E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply"
$MetashapeCloud = "E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_with_normals.ply"
```

The script must support a sampled mode for quick CI and a full mode for local Windows validation.

- [ ] **Step 3: Run tests and local baseline**

Run:

```powershell
python -m unittest E:/code/plascan/tests/test_quality_scripts.py
& E:/code/plascan/scripts/quality/run_agisoft_aerial_gcps_baseline.ps1 -SampleRatio 0.01
```

Expected: tests PASS and a JSON report written under `E:/code/test/agisoft_aerial_gcps/reports/`.

---

## Verification Gate Before Each Release

- [ ] Build selected targets:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -Target 'plascan_gui;reconstruct_pipeline_cli;test_gui_project_utils;test_mvs_pipeline' `
  -BuildOnly `
  -Jobs 8
```

- [ ] Run focused tests:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Feature|Match|Sfm|Mvs|DenseCloud|Mesh|Terrain|Gui'
```

- [ ] Run Python config/script tests:

```powershell
python -m unittest discover E:/code/plascan/tests -p "test_*.py"
```

- [ ] If updating a version tag, update all of:

```text
E:/code/plascan/CHANGELOG.md
E:/code/plascan/docs/releases/vX.Y.Z.md
GitHub Release body with 新增 / 优化 / 修复 / 验证 / 已知问题
```

---

## Self-Review

- Spec coverage: The plan covers repeated matching, pair planning, guided matching, sparse quality, MVS metadata/cancel/memory, dense cloud/mesh quality, GCP/CRS production controls, GUI lifecycle, large PLY display, and Metashape comparison.
- Placeholder scan: No task uses “TBD” or “implement later”; each task names files, tests, commands, and expected behavior.
- Type consistency: Pair states, quality reports, depth records, and GUI runner concepts are introduced before use and are scoped to concrete files.
