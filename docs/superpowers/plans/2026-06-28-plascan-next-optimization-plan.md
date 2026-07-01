# PlaScan Next Optimization Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current PlaScan aerial workflow from “can run end-to-end” into a stable, inspectable, Metashape-comparable production workflow.

**Architecture:** Keep the current core/GUI split: core modules own algorithms, reports, metadata, and deterministic tests; GUI managers only schedule background work and display reports. Every optimization must leave a measurable artifact such as JSON/CSV diagnostics, point-cloud distance statistics, MVS quality metadata, or GUI state logs.

**Tech Stack:** C++17, Qt6, OpenCV, LibTorch/CUDA, GDAL, GTest, Python unittest, Windows CUDA build script `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`.

---

## File Structure

- `E:/code/plascan/src/core/pipeline/SfmPairPlanner.*`: single source of truth for candidate pair planning and pair state.
- `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.*`: sparse reconstruction preflight report and recommended action.
- `E:/code/plascan/src/core/pipeline/SFMService.*`: reuse existing matches, gap-fill only missing pairs, guided rematching, track quality reporting.
- `E:/code/plascan/src/core/sfm/*`: track scoring, track splitting/thinning, scale-aware BA, prior residual reports.
- `E:/code/plascan/src/core/mvs/DepthMapGenerator.*`: source image scheduling, memory/cancel guardrails, depth/confidence/valid-mask output.
- `E:/code/plascan/src/core/mvs/DepthMapFusion.*`: confidence-aware fusion, color retention, normal estimation, consistency filtering.
- `E:/code/plascan/src/core/mvs/MvsQualityReport.*`: depth-map quality, dense-cloud quality, and actionable recommendations.
- `E:/code/plascan/src/core/mesh/*`: mesh-from-dense-cloud path, smoothing controls, outlier protection, product metadata.
- `E:/code/plascan/src/core/terrain/*`: DEM/DOM generation, CRS checks, seam/ghost/color diagnostics.
- `E:/code/plascan/src/gui/project/manager/*`: background task orchestration, progress, cancellation, metadata refresh.
- `E:/code/plascan/tests/*` and `E:/code/plascan/src/core/*/tests/*`: regression tests for every step.
- `E:/code/plascan/docs/releases/` and `E:/code/plascan/CHANGELOG.md`: user-visible release notes after verified milestones.

---

## Phase 1: Stop Repeated Work And Make Sparse State Trustworthy

### Task 1: Reuse existing matching results during aerial triangulation

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.h`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectSupportUtils.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`
- Test: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`

- [ ] **Step 1: Write a failing test for “matched-but-not-rematched”**

Add a test where `no_match_pairs.json` and existing valid match files are present. The expected recommendation is `run_sfm_with_existing_matches`, not `run_full_rematch`.

- [ ] **Step 2: Run the failing test**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|AerialTriangulationWorkflowTest'
```

Expected: FAIL until the report treats valid matches and settled no-match pairs as processed upstream data.

- [ ] **Step 3: Implement missing-pair-only behavior**

Use the prerequisite report to choose one of three actions:

- `run_sfm_with_existing_matches`
- `fill_missing_pairs_then_run_sfm`
- `run_feature_extraction_and_matching`

Do not call full feature matching from normal aerial triangulation when existing match state is complete enough.

- [ ] **Step 4: Verify sparse preflight**

Run the same CTest regex. Expected: PASS and GUI log includes the recommendation JSON.

### Task 2: Unify GUI and CLI pair planning

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SfmPairPlanner.h`
- Modify: `E:/code/plascan/src/core/pipeline/SfmPairPlanner.cpp`
- Modify: `E:/code/plascan/src/gui/dialogs/FeaturePairPlanner.h`
- Modify: `E:/code/plascan/src/gui/dialogs/FeaturePairPlanner.cpp`
- Test: `E:/code/plascan/tests/test_sfm_pair_planner.cpp`
- Test: `E:/code/plascan/tests/test_feature_pair_planner.cpp`

- [ ] **Step 1: Write the failing equivalence test**

For the same image list and camera centers, GUI planner output must equal core planner output after converting to pair keys.

- [ ] **Step 2: Run planner tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmPairPlanner|FeaturePairPlanner'
```

Expected: FAIL until GUI delegates pair generation to `SfmPairPlanner`.

- [ ] **Step 3: Implement delegation**

Keep GUI output format, but make `FeaturePairPlanner` consume `PairPlan` records from core. Preserve pair source, status, and failure reason for later reports.

- [ ] **Step 4: Verify no N² fallback on large projects**

Add a 444-image planner test that asserts candidate pair count stays near sequence/spatial/overlap planning and does not become `444 * 443 / 2`.

---

## Phase 2: Fix The Sparse Cloud Quality Gap

### Task 3: Build a Metashape comparison baseline

**Files:**
- Create: `E:/code/plascan/tools/compare_dense_clouds.py`
- Create: `E:/code/plascan/tests/test_compare_dense_clouds.py`
- Create: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_baseline.md`

- [ ] **Step 1: Write a Python test with two tiny synthetic point clouds**

The test must verify nearest-neighbor distance metrics: `count`, `rmse`, `p50`, `p84`, `p95`, `max`, and `coverage`.

- [ ] **Step 2: Implement the comparison script**

Inputs:

```powershell
python E:/code/plascan/tools/compare_dense_clouds.py `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --candidate E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --output E:/code/test/agisoft_aerial_gcps/reports/metashape_cloud_compare.json
```

- [ ] **Step 3: Run the real comparison**

Expected output: JSON with distance statistics and an explicit note if coordinate frames or scale are inconsistent.

### Task 4: Add sparse quality gates before BA and MVS

**Files:**
- Modify: `E:/code/plascan/src/core/sfm/MultiViewTrackBuilder.cpp`
- Modify: `E:/code/plascan/src/core/sfm/BundleAdjuster.cpp`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Test: `E:/code/plascan/tests/test_sfm_match_diagnostics.cpp`

- [ ] **Step 1: Test track quality classification**

Tracks must be classified by length, triangulation angle, reprojection residual, and grid coverage.

- [ ] **Step 2: Implement track thinning and diagnostics**

Before BA, keep tie points spatially balanced per image grid. Do not let repeated grass/tree texture dominate one region.

- [ ] **Step 3: Verify output report**

`matching_quality_report.json` must include:

- valid pair count
- geometric inlier distribution
- track length histogram
- registered image count
- BA RMS/P50/P84/P95
- rejected-track reasons

---

## Phase 3: Stabilize MVS, Dense Cloud, And Mesh

### Task 5: Improve depth-map quality and reduce isolated spikes

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.cpp`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_quality_report.cpp`

- [ ] **Step 1: Write a speckle regression test**

Synthetic depth maps with isolated high-depth red spikes must produce a `speckle_warning=true` quality flag.

- [ ] **Step 2: Implement local consistency filtering**

Use confidence, valid-mask neighborhood, and relative depth jumps to suppress isolated depth spikes before fusion.

- [ ] **Step 3: Verify metadata**

Each depth frame record must include `depth_png`, `raw_depth_path`, `raw_confidence_path`, `valid_mask_path`, `source_images`, `quality`, `elapsed_ms`, and `device`.

### Task 6: Rebuild mesh from filtered dense cloud, not sparse fallback

**Files:**
- Modify: `E:/code/plascan/src/core/mesh/*`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Test: `E:/code/plascan/tests/test_mesh_reconstruction.cpp`

- [ ] **Step 1: Test product identity**

When “3D 模型” is generated from dense cloud, metadata must say `input_type=dense_cloud` and vertex count must not equal sparse point count.

- [ ] **Step 2: Implement mesh quality presets**

For 64 GB RAM, default to an out-of-core dense-cloud-to-mesh path with tile streaming and smoothing enabled. Avoid loading all 100M+ dense points into one mesh builder allocation.

- [ ] **Step 3: Verify GUI display**

The 3D model tree item must show mesh face count and should not be displayed as a point-only object when faces exist.

---

## Phase 4: Production Controls And Surveying Features

### Task 7: Add GCP/checkpoint/scale-bar product layer

**Files:**
- Create: `E:/code/plascan/src/core/control/ControlPointTable.h`
- Create: `E:/code/plascan/src/core/control/ControlPointTable.cpp`
- Create: `E:/code/plascan/src/core/control/ControlQualityReport.h`
- Create: `E:/code/plascan/src/core/control/ControlQualityReport.cpp`
- Test: `E:/code/plascan/tests/test_control_points.cpp`

- [ ] **Step 1: Test control/checkpoint separation**

Control points affect BA. Checkpoints are excluded from BA and only appear in accuracy reports.

- [ ] **Step 2: Implement CSV import and residual report**

CSV columns: `id,x,y,z,sigma_x,sigma_y,sigma_z,type`.

- [ ] **Step 3: Add BA integration**

Use robust loss and per-point sigma. Default imported external poses remain soft priors, not fixed poses.

### Task 8: Add CRS and vertical datum validation

**Files:**
- Create: `E:/code/plascan/src/core/georef/CoordinateReferenceReport.h`
- Create: `E:/code/plascan/src/core/georef/CoordinateReferenceReport.cpp`
- Test: `E:/code/plascan/tests/test_coordinate_reference_report.cpp`

- [ ] **Step 1: Test missing CRS warnings**

DEM/DOM export with missing CRS must produce a warning and write it to project metadata.

- [ ] **Step 2: Implement WKT/EPSG report**

Report horizontal CRS, vertical datum status, unit, geotransform, and whether GCP/pose/dense products share the same coordinate frame.

---

## Phase 5: GUI Responsiveness And Large Data Handling

### Task 9: Guard all background callbacks with object lifetime checks

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add static regression tests**

The test must require `QPointer` in async callback-heavy managers and reject raw `QMetaObject::invokeMethod(this, ...)` from background lambdas.

- [ ] **Step 2: Implement guarded task callbacks**

Capture `QPointer<ManagerType> self(this)` and return early if `self.isNull()`.

- [ ] **Step 3: Verify cancellation**

Depth estimation and dense fusion cancellation must leave the GUI status bar, progress bar, and task state consistent.

### Task 10: Make project tree products metadata-driven and naturally sorted

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectDepthFrameUtils.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Test natural sorting**

`depth_2.png` must sort before `depth_10.png`.

- [ ] **Step 2: Implement metadata-first refresh**

Do not scan output directories as the primary truth. Read project metadata records, then verify paths exist.

---

## Verification Matrix

- Sparse preflight:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|AerialTriangulationWorkflowTest|SfmPairPlanner|FeaturePairPlanner'
```

- MVS and mesh:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'Mvs|DenseCloud|Mesh|Terrain'
```

- Python utilities:

```powershell
python -m unittest tests.test_mvs_scheduler_config
python -m unittest tests.test_compare_dense_clouds
```

- Full Windows CUDA build:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -Jobs 8
```

---

## Execution Order

1. Finish Phase 1 first, because it prevents repeated matching and makes sparse state reliable.
2. Run the Metashape comparison baseline before touching more MVS or mesh code.
3. Fix sparse quality gates before MVS tuning; poor tie points poison dense reconstruction.
4. Tune MVS confidence and fusion after sparse reports are trustworthy.
5. Rebuild mesh from filtered dense cloud using streaming/out-of-core behavior.
6. Add GCP/CRS production features after the main reconstruction chain is stable.
7. Update `CHANGELOG.md`, `docs/releases/vX.Y.Z.md`, and GitHub Release notes only after targeted tests and at least one real `agisoft_aerial_gcps` run are recorded.

## Self-Review

- Spec coverage: The plan covers repeated matching, pair planning, sparse quality, Metashape comparison, MVS spikes, dense mesh identity, production controls, CRS, GUI async safety, and project-tree sorting.
- Placeholder scan: No task uses “TBD” or an empty validation step; each phase has files and concrete verification commands.
- Type consistency: New reports are consistently named `ReconstructionPrerequisiteReport`, `MvsQualityReport`, `ControlQualityReport`, and `CoordinateReferenceReport`.
