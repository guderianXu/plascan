# PlaScan Post-v1.1.6 Quality Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the current Windows CUDA PlaScan pipeline from "can run the full chain" to "can explain and improve quality against Metashape on the Agisoft aerial GCP dataset."

**Architecture:** Use a report-first workflow: every stage writes deterministic metadata and quality reports, then the GUI and downstream stages consume those reports instead of guessing from files. Keep algorithm code in `src/core`, keep GUI managers as cancellable schedulers, and verify every quality change with focused tests plus at least one reproducible Agisoft aerial GCP comparison.

**Tech Stack:** C++17, Qt6, CMake/Ninja, OpenCV, CUDA/LibTorch, GDAL, GTest, Python pytest/unittest, Windows CUDA build script `E:/code/plascan/scripts/build_win/build_windows_cuda.ps1`.

---

## Current Evidence

The latest 64-image Agisoft aerial GCP CLI run confirms that the immediate quality gap is not sparse registration:

- Input list: `E:/code/test/agisoft_aerial_gcps_cli_ab_64/image_camera_64_abs_fwd.lis`
- CLI output: `E:/code/test/agisoft_aerial_gcps_cli_ab_64/run_current_abs_fwd`
- Registered images: `64/64`
- Sparse points: `12,210`
- Filtered sparse points: `9,875`
- Mean reprojection error: `0.8973 px`
- Match graph: `341` planned pairs, `225` usable pairs, `2` connected components, no isolated images
- Depth maps: selected frames have millions of valid depth pixels before postprocessing
- Dense cloud output: only `784` vertices in `mvs/dense_cloud.ply`
- Metashape sample comparison: source coverage at `0.25 m` radius is only `0.56%`

This means the next optimization must treat dense fusion as the first blocking defect. Depth estimation is producing data, but the strict multi-view fusion path collapses the output to a sparse-like cloud. Fixing matcher presets, meshing, DEM, or DOM before this will hide the root problem instead of solving it.

## Immediate Priority Order

1. Fix dense fusion low-yield collapse and make it report why points are rejected.
2. Stop redundant matching during aerial triangulation when matching has already completed.
3. Add sparse/MVS quality gates so bad upstream data blocks with a reason instead of producing misleading products.
4. Fix product identity so dense point clouds, meshes, DEM, and DOM are registered and displayed as different product types.
5. Rebuild terrain mesh through a 2.5D aerial path instead of a generic rough surface path.
6. Add Metashape comparison reports as a standing regression target.

## File Structure

- `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.*`: sparse preflight state, existing match reuse, missing/stale pair classification, recommended action.
- `E:/code/plascan/src/core/pipeline/SFMService.*`: SfM orchestration, existing match reuse, gap-fill-only matching, guided rematch, sparse report emission.
- `E:/code/plascan/src/core/pipeline/GuidedRematchService.*`: append-only pose/epipolar guided rematching.
- `E:/code/plascan/src/core/sfm/quality/*`: match graph, tracks, BA residuals, registration health, sparse quality gates.
- `E:/code/plascan/src/core/mvs/MvsSourcePlanner.*`: source image ranking by shared tracks, baseline, overlap, texture, and pose uncertainty.
- `E:/code/plascan/src/core/mvs/DepthMapGenerator.*`: depth generation, confidence, valid mask, cancel checks, memory-adaptive scheduling.
- `E:/code/plascan/src/core/mvs/DepthMapFusion.*`: multi-view consistent fusion, confidence, color, normal, source count.
- `E:/code/plascan/src/core/mvs/MvsQualityReport.*`: depth/fusion/dense product diagnostics and user-facing recommendations.
- `E:/code/plascan/src/core/mesh/*`: terrain mesh and generic mesh paths, smoothing, spike filtering, metadata.
- `E:/code/plascan/src/core/control/*`: future GCP, checkpoint, scale-bar, and residual report logic.
- `E:/code/plascan/src/core/georef/*`: future CRS, EPSG/WKT, vertical datum, and product georeference validation.
- `E:/code/plascan/src/gui/project/manager/*`: async scheduling, progress, cancel, metadata refresh, no algorithm logic.
- `E:/code/plascan/src/gui/widgets/DataTreeWidget.cpp`: product tree grouping, natural sorting, product kind display.
- `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`: current point-cloud comparison utility to extend for Metashape baselines.
- `E:/code/plascan/docs/reports/`: reproducible quality reports for `agisoft_aerial_gcps`.
- `E:/code/plascan/CHANGELOG.md` and `E:/code/plascan/docs/releases/vX.Y.Z.md`: release evidence after verified milestones.

---

## Phase 1: Establish A Reliable Quality Baseline

### Task 1: Generate a reproducible PlaScan-vs-Metashape comparison

**Files:**
- Modify: `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
- Modify: `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py`
- Create: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape.md`

- [ ] **Step 1: Add tests for comparison metrics**

Add tests that validate `rmse`, `p50`, `p84`, `p95`, `max`, `coverage_ratio`, `vertical_rmse`, and `vertical_p95_abs` on two tiny synthetic PLY inputs.

- [ ] **Step 2: Run the failing tests**

Run:

```powershell
python -m pytest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py -q
```

Expected: FAIL if any new metric or missing-input diagnostic is absent.

- [ ] **Step 3: Implement missing metrics and report writing**

The script must compare:

```text
PlaScan dense cloud:
E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply

Metashape sample:
E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply

Metashape full:
E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_with_normals.ply
```

- [ ] **Step 4: Record the baseline**

Run:

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --output-json E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_sample.json `
  --max-source-points 50000 `
  --max-reference-points 50000 `
  --coverage-radius-m 0.25 `
  --nearest-neighbor-method kd-tree
```

Expected: JSON report plus Markdown summary saying whether the dominant gap is pose/sparse, depth noise, fusion, meshing, or visualization.

### Task 2: Add a small repeatable CLI A/B run

**Files:**
- Create: `E:/code/plascan/scripts/quality/run_agisoft_aerial_gcps_cli64.ps1`
- Create: `E:/code/plascan/tests/test_quality_cli_scripts.py`

- [ ] **Step 1: Test absolute `.lis` generation**

The script must generate `E:/code/test/agisoft_aerial_gcps_cli_ab_64/image_camera_64_abs.lis` with absolute image and camera paths, because the CLI resolves relative paths relative to the list file.

- [ ] **Step 2: Run the script test**

Run:

```powershell
python -m pytest E:/code/plascan/tests/test_quality_cli_scripts.py -q
```

Expected: FAIL until the script creates valid absolute paths and rejects missing source data clearly.

- [ ] **Step 3: Implement the CLI wrapper**

The wrapper should run:

```powershell
& E:/code/plascan/build/windows-vcpkg-cuda-release/bin/reconstruct_pipeline_cli.exe `
  E:/code/test/agisoft_aerial_gcps_cli_ab_64/image_camera_64_abs.lis `
  --output-dir E:/code/test/agisoft_aerial_gcps_cli_ab_64/run_current `
  --device cuda `
  --quality 1 `
  --threads 8 `
  --cuda-parallel-pairs 1 `
  --feature-max-image-dim 2048 `
  --mvs-res-scale 0.25 `
  --mvs-iterations 3 `
  --mvs-confidence 0.60 `
  --mvs-fusion-confidence 0.65 `
  --mvs-gpu-frame-workers 1 `
  --mvs-cpu-frame-workers 0 `
  --mvs-max-frames 24 `
  --mvs-fusion-max-image-dim 2048 `
  --skip-mesh `
  --skip-terrain `
  --force
```

- [ ] **Step 4: Compare the CLI output**

Compare the new dense cloud to Metashape sample and write:

```text
E:/code/plascan/docs/reports/agisoft_aerial_gcps_cli64_vs_metashape_sample.json
```

Expected: The report gives a smaller, faster signal before running the full 444-image project.

---

## Phase 2: Stop Redundant Matching And Stabilize Sparse Reconstruction

### Task 3: Treat finished matching as authoritative upstream data

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.h`
- Modify: `E:/code/plascan/src/core/pipeline/ReconstructionPrerequisiteReport.cpp`
- Modify: `E:/code/plascan/src/gui/main_window/MenuWorkflowController.cpp`
- Modify: `E:/code/plascan/tests/test_reconstruction_prerequisites.cpp`
- Modify: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add failing tests for completed-but-unusable matching**

Cases:

- completed matching with valid match pairs -> run SfM from existing matches
- completed matching with known failed/no-match pairs -> do not rerun full matching
- completed matching with no usable graph -> stop and report quality problem
- missing or stale pairs -> offer gap-fill only

- [ ] **Step 2: Run sparse preflight tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisiteReport|AerialTriangulationWorkflowTest'
```

Expected: FAIL until the recommendation distinguishes `run_sfm_existing`, `gap_fill_missing`, `block_on_match_quality`, and `run_full_matching`.

- [ ] **Step 3: Implement the recommendation gate**

Normal 空三 must not trigger full feature matching when matching has already completed. The GUI should log the preflight JSON before starting SfM.

- [ ] **Step 4: Verify no repeated work**

Expected: In the current `agisoft_aerial_gcps` project, clicking 空三 after matching should reuse existing matches or block with a quality reason, never silently rerun matching for all pairs.

### Task 4: Improve pair planning and guided rematching

**Files:**
- Modify: `E:/code/plascan/src/core/pipeline/SfmPairPlanner.*`
- Modify: `E:/code/plascan/src/gui/dialogs/FeaturePairPlanner.*`
- Modify: `E:/code/plascan/src/core/pipeline/GuidedRematchService.*`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.*`
- Modify: `E:/code/plascan/tests/test_feature_pair_planner.cpp`
- Modify: `E:/code/plascan/tests/test_guided_rematch_service.cpp`

- [ ] **Step 1: Add planner equivalence tests**

The same image list and camera centers must generate identical pair keys from GUI and core planner.

- [ ] **Step 2: Add guided rematch eligibility tests**

Eligible pairs require registered cameras, overlap, low current inliers, and no permanent invalid flag.

- [ ] **Step 3: Implement core planner delegation and append-only guided rematch**

GUI should consume `PairPlan` from core. Guided rematch may add high-confidence matches but must not replace stable existing matches.

- [ ] **Step 4: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmPairPlanner|FeaturePairPlanner|GuidedRematch|SfmSparseResultMetadata'
```

Expected: PASS and candidate pair count for 444 images does not fall back to N squared matching.

### Task 5: Add sparse quality gates before MVS

**Files:**
- Modify: `E:/code/plascan/src/core/sfm/quality/*`
- Modify: `E:/code/plascan/src/core/pipeline/SFMService.cpp`
- Modify: `E:/code/plascan/tests/test_sfm_quality_report.cpp`

- [ ] **Step 1: Add track and BA quality tests**

Quality report must include registered image count, connected components, track length histogram, grid coverage, triangulation angle, reprojection RMS/P50/P84/P95, and rejected reasons.

- [ ] **Step 2: Implement track thinning and report gates**

Before MVS, reject or warn on reconstructions with poor graph connectivity, poor image coverage, or very low multi-view track ratio.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'SfmQuality|SfmSparseResultMetadata|BundleAdjust'
```

Expected: PASS and sparse results contain actionable diagnostics before dense reconstruction starts.

---

## Phase 3: Improve MVS Depth And Dense Fusion

### Task 6: Rank MVS source views with sparse quality signals

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/tests/test_mvs_source_planner.cpp`

- [ ] **Step 1: Add source ranking tests**

Rank by shared tracks, geometric inlier count, baseline angle, overlap score, texture score, and pose uncertainty.

- [ ] **Step 2: Implement bad-source rejection**

Reject sources with tiny overlap, extreme baseline, or too few shared tracks.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsSourcePlanner|MvsPipeline'
```

Expected: PASS and each depth frame records selected source images and scores.

### Task 7: Reduce isolated depth spikes and unsupported dense points

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapFusion.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsQualityReport.cpp`
- Modify: `E:/code/plascan/tests/test_mvs_pipeline.cpp`
- Modify: `E:/code/plascan/src/core/mvs/tests/test_mvs_workspace_manifest.cpp`

- [ ] **Step 1: Add synthetic spike and speckle tests**

Depth maps with isolated high or low spikes must flag local outliers and suppress those samples before fusion.

- [ ] **Step 2: Implement confidence-aware filtering**

Fusion should keep RGB, confidence, normal, and source-count fields. Points should require enough consistent source views.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsDepthPostprocess|MvsQuality|MvsWorkspace|MvsPipeline'
```

Expected: PASS and depth/dense reports show valid ratio, confidence histogram, local outlier count, and source support.

---

## Phase 4: Fix Mesh, DEM, DOM, And Product Identity

### Task 8: Separate dense cloud, terrain mesh, generic mesh, DEM, and DOM

**Files:**
- Modify: `E:/code/plascan/src/gui/project/support/ProjectResultRecords.cpp`
- Modify: `E:/code/plascan/src/gui/widgets/DataTreeWidget.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Modify: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add product identity tests**

A point-only PLY must be registered under `稠密点云`. A mesh PLY must have `face_count > 0` and be registered under `3D模型`.

- [ ] **Step 2: Implement metadata-first tree refresh**

The tree consumes project result metadata first, validates paths second, and sorts files naturally by name.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ProjectResult|DataTree|GuiProject'
```

Expected: PASS and dense clouds are no longer shown as 3D models.

### Task 9: Build aerial terrain mesh through a 2.5D path

**Files:**
- Create or modify: `E:/code/plascan/src/core/mesh/*`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Create: `E:/code/plascan/tests/test_terrain_mesh_reconstruction.cpp`

- [ ] **Step 1: Add roughness regression tests**

Synthetic flat terrain with small noise should produce a smoothed mesh without spike amplification.

- [ ] **Step 2: Implement terrain mesh defaults**

For nadir aerial data, use a 2.5D terrain mesh path: tile dense cloud, aggregate robust height, smooth with edge-preserving filter, triangulate grid/TIN, then project texture.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'TerrainMesh|Mesh|DenseCloud'
```

Expected: PASS and mesh result is visibly less rough than the current spike-heavy generic surface.

### Task 10: Make DEM and DOM consume exact upstream product IDs

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Add stale-output tests**

If a project has an old dense cloud and a new dense job completes, DEM generation must use the new output ID/path, not `dense_cloud_results.last()` from unrelated metadata updates.

- [ ] **Step 2: Implement completion-path handoff**

Dense cloud completion should emit exact output path and record ID. DEM/DOM should consume that exact path.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'TerrainPipelineAsync|Dem|Dom|ProjectTerrain'
```

Expected: PASS and DEM/DOM no longer accidentally use stale dense products.

---

## Phase 5: Production Photogrammetry Features

### Task 11: Add GCP, checkpoint, and scale-bar workflow

**Files:**
- Create: `E:/code/plascan/src/core/control/ControlPointSet.h`
- Create: `E:/code/plascan/src/core/control/ControlPointSet.cpp`
- Create: `E:/code/plascan/src/core/control/ControlQualityReport.h`
- Create: `E:/code/plascan/src/core/control/ControlQualityReport.cpp`
- Modify: `E:/code/plascan/src/core/sfm/BundleAdjuster.*`
- Create: `E:/code/plascan/tests/test_control_points.cpp`

- [ ] **Step 1: Add control/checkpoint tests**

Control points constrain BA. Checkpoints are excluded from BA and report residuals only.

- [ ] **Step 2: Implement CSV import and residual report**

CSV columns:

```text
id,x,y,z,sigma_x,sigma_y,sigma_z,type
```

- [ ] **Step 3: Add robust BA constraints**

Use sigma-aware residuals and robust loss. External camera priors remain soft by default.

- [ ] **Step 4: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ControlPoint|BundleAdjust|ScaleBar'
```

Expected: PASS and quality reports separate control residuals from checkpoint residuals.

### Task 12: Add CRS, vertical datum, and export validation

**Files:**
- Create: `E:/code/plascan/src/core/georef/CoordinateReferenceReport.h`
- Create: `E:/code/plascan/src/core/georef/CoordinateReferenceReport.cpp`
- Modify: `E:/code/plascan/src/core/terrain/*`
- Create: `E:/code/plascan/tests/test_coordinate_reference_report.cpp`

- [ ] **Step 1: Add CRS validation tests**

DEM/DOM export with missing CRS must warn. Mixed coordinate frames must block export until the user confirms or fixes metadata.

- [ ] **Step 2: Implement CRS report**

Report EPSG/WKT, horizontal unit, vertical datum, geoid grid name, geotransform, and whether pose/GCP/dense/terrain products share the same frame.

- [ ] **Step 3: Run tests**

Run:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'CoordinateReference|Crs|Geo|Terrain'
```

Expected: PASS and DEM/DOM export dialogs show the effective coordinate reference.

---

## Release And Verification Gate

Before tagging the next version:

- [ ] Run focused sparse tests:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'ReconstructionPrerequisite|GuidedRematch|SfmQuality|FeaturePair|AerialTriangulationWorkflow'
```

- [ ] Run focused dense/terrain tests:

```powershell
& E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 `
  -SourceDir E:/code/plascan `
  -BuildDir E:/code/plascan/build/windows-vcpkg-cuda-release `
  -RunTests `
  -CTestRegex 'MvsSourcePlanner|MvsQuality|MvsDepth|MvsWorkspace|MvsPipeline|DenseCloud|Mesh|Terrain|Dem|Dom'
```

- [ ] Run Python quality tests:

```powershell
python -m pytest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py -q
python -m pytest E:/code/plascan/tests/test_quality_cli_scripts.py -q
```

- [ ] Run full CTest:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure
```

- [ ] Run at least one real Agisoft aerial GCP baseline and save JSON/Markdown under:

```text
E:/code/plascan/docs/reports/
```

- [ ] If tagging a release, update:

```text
E:/code/plascan/CHANGELOG.md
E:/code/plascan/docs/releases/vX.Y.Z.md
GitHub Release body with 新增 / 优化 / 修复 / 验证 / 已知问题
```

---

## Recommended Execution Order

1. Finish the PlaScan-vs-Metashape baseline first; otherwise every visual judgment is too subjective.
2. Stop redundant matching in 空三; this directly addresses the current GUI behavior.
3. Add sparse quality gates; bad sparse geometry should not silently feed MVS.
4. Improve MVS source selection and depth cleanup; this targets depth speckles and rough dense cloud.
5. Fix product identity and terrain mesh; dense cloud and mesh must be different products.
6. Make DEM/DOM consume exact upstream output IDs; this removes stale product bugs.
7. Add GCP/checkpoint/scale-bar and CRS/geoid validation after the main chain is stable.
8. Only tag after focused tests, full CTest, and at least one saved real-data report.

## Self-Review

- Spec coverage: The plan covers current user-visible failures: repeated matching during 空三, questionable dense cloud quality, Metashape comparison, rough mesh, product identity, MVS spike noise, stale DEM/DOM handoff, and production surveying gaps.
- Placeholder scan: No task contains TBD or empty validation; each task names files, tests, commands, and expected behavior.
- Type consistency: Report and service names stay aligned with existing project names: `ReconstructionPrerequisiteReport`, `GuidedRematchService`, `MvsQualityReport`, `ControlQualityReport`, and `CoordinateReferenceReport`.
