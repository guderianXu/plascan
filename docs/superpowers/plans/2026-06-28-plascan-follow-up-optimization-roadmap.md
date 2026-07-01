# PlaScan Follow-Up Optimization Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize PlaScan's current aerial photogrammetry workflow by reusing completed matching work, quantifying dense-cloud roughness against Metashape output, and making MVS/mesh/GUI behavior production-ready.

**Architecture:** Keep algorithm decisions in `src/core` and GUI managers as orchestration only. Every reconstruction stage must produce metadata or a quality report that the next stage consumes instead of guessing from files on disk.

**Tech Stack:** C++17, Qt6, OpenCV, LibTorch/CUDA, GDAL, GTest, Python unittest, Windows CUDA build script `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`.

---

## File Structure

- Modify `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.*` for "reuse existing matches vs fill missing pairs vs full matching" decisions.
- Modify `E:/code/plascan/src/core/pipeline/SFMService.*` for match-state reuse, missing-pair-only generation, sparse quality reports, and guided rematching entry points.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp` so the GUI does not trigger a full rematch after matching has already run.
- Modify `E:/code/plascan/testData/compare_point_cloud_to_lidar.py` and `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py` for PlaScan-vs-Metashape distance and vertical roughness gates.
- Modify `E:/code/plascan/src/core/mvs/MvsQualityReport.*`, `DepthMapGenerator.*`, `DepthMapFusion.*`, and `MvsWorkspaceManifest.*` for depth quality, cancellation, memory guardrails, confidence, and valid-mask metadata.
- Modify or create focused helpers under `E:/code/plascan/src/core/mesh/` for dense-cloud-to-terrain-mesh generation.
- Modify `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp` and `ProjectTerrainProductsManager.cpp` only for task scheduling, progress, cancellation, and metadata refresh.
- Update `E:/code/plascan/docs/reports/`, `E:/code/plascan/CHANGELOG.md`, and `E:/code/plascan/docs/releases/vX.Y.Z.md` only after validation results exist.

---

## Phase 1: Fix Aerial Triangulation Reprocessing

### Task 1: Treat completed matching as valid upstream data

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.h`
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectSparseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Write the failing prerequisite test**

```cpp
TEST(ReconstructionPrerequisiteReportTest, ReusesCompletedMatchesWithoutFullRematch)
{
    ReconstructionPrerequisiteReport report;
    report.imageCount = 444;
    report.plannedPairCount = 1576;
    report.validMatchPairCount = 984;
    report.failedGeometryPairCount = 592;
    report.missingFeaturePairCount = 0;
    report.missingMatchPairCount = 0;

    EXPECT_TRUE(report.hasEnoughUpstreamData());
    EXPECT_EQ(report.recommendedAction(), "run_sfm_with_existing_matches");
    EXPECT_FALSE(report.shouldRunFullRematch());
    EXPECT_FALSE(report.shouldOfferGapFill());
}
```

- [ ] **Step 2: Verify the test fails before implementation**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport'
```

Expected: FAIL if the report still treats failed/no-match pairs as missing upstream work.

- [ ] **Step 3: Implement state classification**

Use these states in the report and SFM service:

```cpp
enum class PairInputState
{
    ExistingValidMatch,
    ExistingFailedGeometry,
    ExistingNoMatch,
    MissingFeature,
    MissingMatch,
    SkippedByPlanner
};
```

Only `MissingFeature` and `MissingMatch` can trigger automatic upstream work. `ExistingFailedGeometry` and `ExistingNoMatch` are diagnostic outcomes, not evidence that the whole matching phase did not run.

- [ ] **Step 4: Add GUI regression guard**

Add or update a static test that requires normal aerial triangulation to disable full auto-rematch:

```cpp
EXPECT_TRUE(body.contains("autoGenerateMissingMatches = false"));
EXPECT_TRUE(body.contains("自动补齐缺失"));
```

- [ ] **Step 5: Verify**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|AerialTriangulationWorkflowTest|FeatureMatch'
```

Expected: PASS. GUI logs should say `空三上游数据就绪：复用已有匹配`.

---

## Phase 2: Quantify Why PlaScan Dense Cloud Is Rougher Than Metashape

### Task 2: Finish vertical error quality gates

**Files:**
- Modify: `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
- Modify: `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py`
- Output: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_dense_sample.json`

- [ ] **Step 1: Write the failing vertical gate test**

```python
def test_vertical_error_quality_gate_catches_rough_surface(self):
    source = [
        comparator.Point(0.0, 0.0, 3.0),
        comparator.Point(1.0, 0.0, 3.5),
    ]
    reference = [
        comparator.Point(0.0, 0.0, 0.0),
        comparator.Point(1.0, 0.0, 0.0),
    ]

    comparison = comparator.compare_point_clouds(
        source,
        reference,
        max_vertical_rmse_m=1.0,
        max_vertical_p95_m=1.5,
    )

    gate = comparison["quality_gate"]
    self.assertFalse(gate["passed"])
    self.assertIn("vertical_rmse_above_threshold", gate["failure_codes"])
    self.assertIn("vertical_p95_above_threshold", gate["failure_codes"])
```

- [ ] **Step 2: Verify the red test**

Run:

```powershell
python -m unittest tests.test_compare_point_cloud_to_lidar.ComparePointCloudToLidarTest.test_vertical_error_quality_gate_catches_rough_surface
```

Expected: FAIL until `compare_point_clouds()` accepts vertical thresholds.

- [ ] **Step 3: Implement thresholds**

Add CLI arguments:

```python
parser.add_argument("--max-vertical-rmse-m", type=float)
parser.add_argument("--max-vertical-p95-m", type=float)
```

Add quality gate failure codes:

```python
"vertical_rmse_above_threshold"
"vertical_p95_above_threshold"
```

- [ ] **Step 4: Run Python tests**

Run:

```powershell
python -m unittest tests.test_compare_point_cloud_to_lidar
```

Expected: PASS.

- [ ] **Step 5: Compare the current PlaScan dense cloud to Metashape**

Run:

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --output-json E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_dense_sample.json `
  --nearest-neighbor-method kd-tree `
  --max-source-points 50000 `
  --max-reference-points 50000 `
  --coverage-radius-m 0.5 `
  --min-reference-coverage-percent 50 `
  --max-vertical-rmse-m 0.20 `
  --max-vertical-p95-m 0.50
```

Expected: JSON report includes `distance_m`, `vertical_error_m`, `reference_coverage`, and `quality_gate`.

---

## Phase 3: Improve Sparse Reconstruction Quality Before MVS

### Task 3: Add sparse quality gates that block bad MVS input

**Files:**
- Modify: `E:/code/plascan/src/core/sfm/quality/SfmQualityReport.h`
- Modify: `E:/code/plascan/src/core/sfm/quality/SfmQualityReport.cpp`
- Modify: `E:/code/plascan/src/common/project/SparseResultQuality.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Test: `E:/code/plascan/tests/test_sfm_quality_report.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Require warnings for weak sparse geometry**

Test warnings:

```cpp
EXPECT_THAT(report.warningCodes, Contains("high_reprojection_error"));
EXPECT_THAT(report.warningCodes, Contains("weak_triangulation_angle"));
EXPECT_THAT(report.warningCodes, Contains("poor_observation_spatial_coverage"));
```

- [ ] **Step 2: Persist quality gate in project metadata**

Expected JSON shape:

```json
{
  "quality_gate": {
    "acceptable_for_mvs": false,
    "warnings": [
      "high_reprojection_error",
      "poor_observation_spatial_coverage"
    ]
  }
}
```

- [ ] **Step 3: Make MVS preflight reject bad sparse results**

`SparseResultQuality::isProductionSparseResult()` must return false when `acceptable_for_mvs=false`.

- [ ] **Step 4: Verify**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmQualityReport|SparseResultQualityTest|ReconstructionPrerequisiteReport'
```

Expected: PASS.

### Task 4: Add guided rematching as an opt-in second pass

**Files:**
- Create: `E:/code/plascan/src/core/pipeline/GuidedRematchService.h`
- Create: `E:/code/plascan/src/core/pipeline/GuidedRematchService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Test: `E:/code/plascan/tests/test_guided_rematch_service.cpp`

- [ ] **Step 1: Test eligibility**

Guided rematching is allowed only when:

```cpp
EXPECT_TRUE(pair.hasRegisteredCameraA);
EXPECT_TRUE(pair.hasRegisteredCameraB);
EXPECT_GT(pair.overlapScore, 0.2);
EXPECT_LT(pair.geometricInlierCount, targetInlierCount);
EXPECT_FALSE(pair.permanentlyRejected);
```

- [ ] **Step 2: Implement epipolar-band candidate generation**

For v1, generate new candidate matches in a local epipolar band and mark them:

```cpp
match.source = MatchSource::GuidedRematch;
match.replacesExistingMatch = false;
```

- [ ] **Step 3: Verify**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'GuidedRematch|Sfm|MatchOutlier'
```

Expected: PASS.

---

## Phase 4: Stabilize MVS Depth, Fusion, And Memory

### Task 5: Detect and suppress local depth spikes

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Test: `E:/code/plascan/tests/test_mvs_types.cpp`

- [ ] **Step 1: Test local spike detection**

Synthetic depth map: all values 10 m except two 40 m spikes. Expected:

```cpp
EXPECT_EQ(report.localDepthOutlierCount, 2);
EXPECT_TRUE(report.hasLocalDepthOutliers);
EXPECT_GT(report.localDepthOutlierRatio, 0.02f);
```

- [ ] **Step 2: Add post-filter**

Suppress a pixel when:

```text
confidence < min_confidence
or abs(depth - local_median) > max(absolute_jump_m, relative_jump_ratio * local_median)
or valid_neighbor_count < min_valid_neighbors
```

- [ ] **Step 3: Verify**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsQualityReportTest|MvsDepthPostprocess|MvsWorkspaceManifest'
```

Expected: PASS.

### Task 6: Make cancellation and memory backpressure reliable

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_mvs_pipeline.cpp`
- Test: `E:/code/plascan/tests/test_mvs_scheduler_config.py`

- [ ] **Step 1: Add cancellation checkpoints**

Check the cancellation token before and after:

```text
source image load
sparse hint projection
CUDA upload
PatchMatch iteration
depth filtering
raw depth/confidence write
manifest update
```

- [ ] **Step 2: Add memory mode selection**

For 64 GB host memory:

```text
High quality default, but keep source-image cache bounded.
When available RAM drops below safety threshold, reduce preload window and flush completed frame buffers.
```

- [ ] **Step 3: Verify**

Run:

```powershell
python -m unittest tests.test_mvs_scheduler_config
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsPipeline|MvsWorkspaceManifest'
```

Expected: PASS.

---

## Phase 5: Fix Dense Cloud And Mesh Product Identity

### Task 7: Do not display dense cloud as 3D mesh

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectSupportUtils.cpp`
- Modify: `E:/code/plascan/src/gui/dialogs/CameraModel3DDialog.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Test product type metadata**

Expected metadata:

```json
{
  "kind": "dense_cloud",
  "path": "dense_cloud.ply",
  "vertex_count": 1058511291,
  "face_count": 0
}
```

Mesh metadata must differ:

```json
{
  "kind": "mesh",
  "path": "model_from_mesh.ply",
  "vertex_count": 3286949,
  "face_count": 6502504
}
```

- [ ] **Step 2: Make viewer mode explicit**

Dense cloud uses point renderer. Mesh uses triangle renderer. If `face_count == 0`, do not put the product under `3D模型`.

- [ ] **Step 3: Verify**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Gui|ProjectWorkflow|CameraModel3D'
```

Expected: PASS.

### Task 8: Add aerial terrain mesh path

**Files:**
- Create: `E:/code/plascan/src/core/mesh/TerrainMeshBuilder.h`
- Create: `E:/code/plascan/src/core/mesh/TerrainMeshBuilder.cpp`
- Test: `E:/code/plascan/tests/test_terrain_mesh_builder.cpp`

- [ ] **Step 1: Test robust planar terrain behavior**

Input dense points with a few high/low spikes. Expected mesh heights follow median local surface, not the spikes:

```cpp
EXPECT_LT(std::abs(mesh.heightAt(0.5, 0.5) - 10.0), 0.25);
EXPECT_GT(mesh.faceCount(), 0);
```

- [ ] **Step 2: Implement tile/grid/TIN terrain mesh**

Default aerial mode:

```text
tile dense cloud
aggregate cell height by median or trimmed mean
remove isolated spikes
smooth DEM-like surface lightly
triangulate grid or local TIN
project texture or vertex color
```

- [ ] **Step 3: Verify against Metashape reference**

Run the point-cloud comparison before and after mesh generation. The mesh should reduce vertical roughness, not hide coordinate-frame errors.

---

## Phase 6: GUI, CI, Packaging, And Release Hygiene

### Task 9: Fix CI so GitHub tests are useful again

**Files:**
- Modify: `E:/code/plascan/.github/workflows/ci.yml`
- Modify: `E:/code/plascan/cmake/PlascanPackages.cmake`
- Test locally with configure/build commands used by CI.

- [ ] **Step 1: Reproduce CI configure locally or in GitHub log**

Known failure classes:

```text
libzip imported target zipcmp missing
cuda_runtime.h required by plamatrix on non-CUDA Linux runner
style/static tests too broad or flaky
```

- [ ] **Step 2: Split CI jobs**

Use separate jobs:

```text
linux-cpu-configure-build-tests
windows-cuda-build-packaging
style-static-tests
python-tests
```

- [ ] **Step 3: Verify GitHub**

Expected: latest GitHub Actions run shows green or fails only on a real test with actionable logs.

### Task 10: Keep Windows installer reproducible

**Files:**
- Modify: `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`
- Modify or create: `E:/code/plascan/scripts/build_win/package_windows_installer.ps1`
- Update: `E:/code/plascan/docs/releases/RELEASE_PROCESS.md`

- [ ] **Step 1: Check bundled DLLs**

Installer must include Qt plugins, CUDA runtime DLLs, cuDNN, cuPTI, LibTorch CUDA DLLs, OpenCV, GDAL, libzip, OpenMP, and app resources.

- [ ] **Step 2: Add a smoke test**

Run installed app in a clean shell:

```powershell
& "C:/Program Files/PlaScan/plascan_gui.bin.exe" --version
```

Expected: no missing DLL dialog and no Qt platform plugin failure.

- [ ] **Step 3: Document release**

When tagging, GitHub Release must include:

```text
新增
优化
修复
验证
已知问题
```

---

## Release Gate

Before tagging the next version:

- [ ] Run Python tests:

```powershell
python -m unittest tests.test_compare_point_cloud_to_lidar
python -m unittest tests.test_mvs_scheduler_config
```

- [ ] Run focused Windows CUDA tests:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|SfmQualityReport|SparseResultQualityTest|MvsQualityReportTest|MvsDepthPostprocess|MvsWorkspaceManifest|Gui|ProjectWorkflow'
```

- [ ] Run real `agisoft_aerial_gcps` comparison and save the JSON report under:

```text
E:/code/plascan/docs/reports/
```

- [ ] Update release docs:

```text
E:/code/plascan/CHANGELOG.md
E:/code/plascan/docs/releases/vX.Y.Z.md
GitHub Release body
```

## Self-Review

- Spec coverage: This roadmap covers repeated matching, PlaScan-vs-Metashape comparison, sparse gates, MVS quality, dense/mesh identity, terrain mesh roughness, CI, installer, and release hygiene.
- Placeholder scan: No task is empty; every phase has concrete files, tests, commands, and expected behavior.
- Type consistency: Report names, pair states, depth quality fields, and product metadata fields are introduced before use.
