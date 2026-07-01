# Dense Cloud Metashape Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PlaScan's Agisoft aerial GCP dense cloud and mesh outputs measurably closer to the Metashape reference by reducing local terrain thickness, multi-view inconsistency, and raw-cloud meshing artifacts.

**Architecture:** Keep the existing SfM/MVS pipeline, but split products into raw, refined, and production stages. Add measurable quality gates around depth maps, fusion, dense-cloud refinement, and mesh generation so the GUI always consumes metadata-backed products instead of guessing from output directories.

**Tech Stack:** C++17, Qt6, CMake, plapoint, OpenCV, CUDA MVS kernels, Python quality scripts, GTest/CTest.

---

## Current Evidence

- PlaScan raw dense cloud: `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply`, size about 9.1 GB.
- Metashape reference cloud: `E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply`.
- Latest comparison report: `E:/code/test/agisoft_aerial_gcps/reports/metashape_cloud_compare_50k_with_roughness_gate.json`.
- The report fails the quality gate with `local_z_range_p95_above_threshold`.
- PlaScan local terrain thickness p95 is `1.5543 m`; Metashape sample p95 is `0.4698 m`; current gate is `0.8 m`.
- Distance RMSE against Metashape sample is `0.2955 m`; vertical absolute p95 is `0.5911 m`.

## Root Cause Hypotheses

1. **Depth maps contain isolated high/low speckles.** The screenshots show red isolated depth noise in visually flat regions, which means PatchMatch confidence and local speckle filtering are not strict enough.
2. **Fusion accepts weakly supported observations.** Dense points are likely being fused even when supported by too few source views or inconsistent depths, creating a thick vertical slab instead of a terrain surface.
3. **Postprocessing is not production-stage enough.** Existing filtering removes some obvious outliers, but does not yet enforce local terrain thickness and support confidence strongly enough for nadir UAV mapping.
4. **Mesh generation sometimes uses the wrong stage.** Raw dense cloud or a point-only PLY can be routed into the 3D model path, causing jagged terrain or "mesh shown like point cloud" behavior.
5. **GUI metadata and task lifecycle are too fragile.** Product selection still relies on latest metadata/output-path heuristics in places, so old/raw products can be consumed by DEM/mesh workflows.

## Acceptance Metrics

- `source_local_roughness.z_range_in_cell.p95 <= 0.8 m` for the 50k sampled comparison, with a stretch target of `<= 0.6 m`.
- `vertical_error_m.p95_abs <= 0.5 m` for the 50k sampled comparison, with a stretch target of `<= 0.35 m`.
- Dense fusion report records support-count histogram, accepted/rejected counts, and top rejection reasons.
- Mesh generation consumes `dense_cloud_refined.ply` or a terrain DEM-derived surface by default, not `dense_cloud.ply`.
- GUI directory tree shows raw/refined/model products separately and sorted naturally by filename.

### Task 1: Baseline And Reproducible Quality Report

**Files:**
- Modify: `E:/code/plascan/testData/compare_point_cloud_to_lidar.py`
- Modify: `E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py`
- Create: `E:/code/test/agisoft_aerial_gcps/reports/dense_quality_baseline_current.json`

- [ ] **Step 1: Record the current raw-vs-Metashape baseline**

Run:

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --max-source-points 50000 `
  --max-reference-points 50000 `
  --local-roughness-grid-cells 120 `
  --local-roughness-min-count 5 `
  --max-local-z-range-p95-m 0.8 `
  --output-json E:/code/test/agisoft_aerial_gcps/reports/dense_quality_baseline_current.json
```

Expected: nonzero exit until quality improves; report contains `quality_gate.failure_codes` with `local_z_range_p95_above_threshold`.

- [ ] **Step 2: Add an improvement report fixture**

Add a unit test that builds two tiny synthetic point clouds: a thick raw cloud and a thinner refined cloud. Assert that `improvement.local_z_range_p95_reduction_percent` is positive and that quality passes only for the refined cloud.

- [ ] **Step 3: Verify the script tests**

Run:

```powershell
python -m unittest E:/code/plascan/tests/test_compare_point_cloud_to_lidar.py
```

Expected: all tests pass.

### Task 2: Standalone Dense Cloud Refinement CLI

**Files:**
- Create: `E:/code/plascan/src/cli/cli_dense_cloud_refine.cpp`
- Modify: `E:/code/plascan/src/cli/CMakeLists.txt`
- Test: `E:/code/plascan/tests/test_dense_cloud_refine_cli.py`

- [ ] **Step 1: Keep the existing failing test**

Run:

```powershell
python -m unittest E:/code/plascan/tests/test_dense_cloud_refine_cli.py
```

Expected before implementation: fails because `dense_cloud_refine_cli` is not registered and `cli_dense_cloud_refine.cpp` does not exist.

- [ ] **Step 2: Implement the CLI surface**

The CLI must expose:

```text
--input
--output
--report-json
--terrain-grid-cells
--terrain-min-cell-points
--terrain-min-height-threshold
--terrain-mad-multiplier
```

It must read PLY through plapoint, call `xjw::mvs::filterTerrainHeightSpikes`, write a binary little-endian PLY, and write a JSON report with `terrain_spike_filter`.

- [ ] **Step 3: Register the CMake target**

Add `dense_cloud_refine_cli` to `E:/code/plascan/src/cli/CMakeLists.txt`, link it against `mvs`, `Qt6::Core`, and the existing common CLI dependencies.

- [ ] **Step 4: Verify the static CLI test**

Run:

```powershell
python -m unittest E:/code/plascan/tests/test_dense_cloud_refine_cli.py
```

Expected: pass.

- [ ] **Step 5: Build the CLI**

Run:

```powershell
E:/code/plascan/scripts/build_win/build_windows_cuda.ps1 -Target dense_cloud_refine_cli
```

Expected: `E:/code/plascan/build/windows-vcpkg-cuda-release/bin/dense_cloud_refine_cli.exe` exists.

### Task 3: Depth Map Quality Filtering

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapGenerator.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsTypes.h`
- Test: `E:/code/plascan/tests/test_mvs_types.cpp`
- Test: `E:/code/plascan/tests/test_mvs_pipeline.cpp`

- [ ] **Step 1: Add explicit quality fields**

Extend depth frame metadata with:

```cpp
int sourceViewCount = 0;
float validPixelRatio = 0.0F;
float medianConfidence = 0.0F;
float p95DepthJump = 0.0F;
int speckleRemovedPixels = 0;
int consistencyRejectedPixels = 0;
```

- [ ] **Step 2: Add speckle removal after PatchMatch**

Use connected-component filtering on valid mask islands. Reject islands smaller than the configured minimum unless confidence and multi-view support are high.

- [ ] **Step 3: Add local depth jump filtering**

For each valid pixel, compare depth to a small neighborhood median. Reject pixels whose normalized jump exceeds the configured threshold and whose confidence is below the high-confidence bypass threshold.

- [ ] **Step 4: Verify MVS metadata tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "MvsPipeline|MvsTypes"
```

Expected: tests pass and new metadata fields are populated in generated reports.

### Task 4: Source View Selection With Support-Aware Scoring

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsSourcePlanner.h`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_source_planner.cpp`

- [ ] **Step 1: Add score terms**

Score candidate source views using shared tracks, geometric inliers, baseline angle, footprint overlap, image sequence distance, and previous depth success.

- [ ] **Step 2: Reject weak sources earlier**

Reject source views with too little overlap, too small baseline, too large baseline, or too few shared tracks unless they are needed as fallback coverage.

- [ ] **Step 3: Log source selection decisions**

For each reference image, write top accepted and rejected sources with score terms to the MVS quality report.

- [ ] **Step 4: Verify source planner tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "MvsSourcePlanner"
```

Expected: source ordering is stable and poor geometry candidates are ranked below strong overlap candidates.

### Task 5: Support-Aware Fusion And Product Staging

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DepthMapFusion.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DepthMapFusion.h`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.cpp`
- Modify: `E:/code/plascan/src/core/mvs/MvsWorkspaceManifest.h`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectResultRecords.cpp`
- Test: `E:/code/plascan/src/core/mvs/tests/test_mvs_workspace_manifest.cpp`

- [ ] **Step 1: Extend fused point metadata**

Each fused point must track support count, depth variance, reprojection consistency, source image count, and confidence.

- [ ] **Step 2: Add fusion gates**

Reject points with support count below the configured threshold, excessive depth variance, or inconsistent normal/color evidence.

- [ ] **Step 3: Write separate product records**

Register:

```text
dense_cloud_raw.ply      stage=raw
dense_cloud_refined.ply  stage=refined
dense_cloud_report.json  stage=quality_report
```

- [ ] **Step 4: Verify manifest and GUI metadata**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "MvsWorkspaceManifest|GuiProjectUtils"
```

Expected: refined cloud is preferred for mesh/DEM workflows when present.

### Task 6: Production Dense Cloud Refinement

**Files:**
- Modify: `E:/code/plascan/src/core/mvs/DenseCloudQualityFilter.cpp`
- Modify: `E:/code/plascan/src/core/mvs/DenseCloudQualityFilter.h`
- Modify: `E:/code/plascan/src/core/mvs/tests/test_dense_cloud_quality_filter.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`

- [ ] **Step 1: Preserve attributes through all filters**

Verify that every filter preserves RGB, normals, scalars, and source statistics.

- [ ] **Step 2: Add local terrain thickness filter**

For nadir/terrain mode, divide XY into adaptive grid cells, estimate robust local height center by median/MAD, and remove vertical spikes that exceed the configured absolute and MAD thresholds.

- [ ] **Step 3: Add normal-aware smoothing**

Smooth terrain height only within local planar neighborhoods; do not blur roads, tree edges, or sharp terrain boundaries across large discontinuities.

- [ ] **Step 4: Verify dense filter tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "DenseCloudQualityFilter"
```

Expected: spike points are removed, colors/normals remain attached to surviving points, and local z range improves.

### Task 7: Mesh Route Correction

**Files:**
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectTerrainProductsManager.cpp`
- Modify: `E:/code/plascan/src/gui/project/support/ProjectMetadataOperations.cpp`
- Modify: `E:/code/plascan/src/gui/project/manager/ProjectDenseReconstructionManager.cpp`
- Test: `E:/code/plascan/tests/test_gui_project_utils.cpp`

- [ ] **Step 1: Always prefer refined dense cloud for model generation**

If `stage=refined` exists, model generation must consume it. Raw dense cloud is only allowed when the user explicitly chooses a raw/debug option.

- [ ] **Step 2: Use terrain-first meshing for nadir UAV data**

For terrain projects, generate a DSM/DEM-like surface first, then triangulate the grid. Use direct point-cloud surface meshing only for non-terrain/oblique projects.

- [ ] **Step 3: Keep point-cloud and mesh products distinct**

The GUI tree must not label a point-only PLY as a 3D mesh. PLY with faces is a mesh; PLY with only vertices is a dense cloud.

- [ ] **Step 4: Verify GUI routing tests**

Run:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "GuiProjectUtils"
```

Expected: mesh routes to refined cloud and point-only PLY appears under dense cloud, not 3D model.

### Task 8: Full A/B Validation Against Metashape

**Files:**
- Create: `E:/code/test/agisoft_aerial_gcps/reports/dense_quality_refined_50k.json`
- Create: `E:/code/test/agisoft_aerial_gcps/reports/dense_quality_raw_vs_refined.json`
- Modify: `E:/code/plascan/docs/reports/`

- [ ] **Step 1: Run refinement on existing raw cloud**

Run the standalone CLI first on a small sampled PLY, then on the full raw cloud only after the sample passes.

- [ ] **Step 2: Compare refined cloud to Metashape**

Run:

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --baseline-source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud_refined.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --max-source-points 50000 `
  --max-reference-points 50000 `
  --local-roughness-grid-cells 120 `
  --local-roughness-min-count 5 `
  --max-local-z-range-p95-m 0.8 `
  --min-local-z-range-p95-improvement-percent 25 `
  --output-json E:/code/test/agisoft_aerial_gcps/reports/dense_quality_raw_vs_refined.json
```

Expected: refined cloud passes the local thickness gate or reports a measurable reduction with remaining failure reasons.

Current measured result on 2026-06-29:

- Raw cloud: `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply`.
- First refined cloud: `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud_refined.ply`, produced by streaming refine with `grid=220`, `minCellPoints=32`, `minHeightThreshold=0.35`, `madMultiplier=4.0`.
- Second refined cloud: `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud_refined_v2.ply`, produced by streaming refine with `grid=260`, `minCellPoints=32`, `minHeightThreshold=0.25`, `madMultiplier=3.0`.
- Comparison report: `E:/code/test/agisoft_aerial_gcps/reports/dense_quality_raw_vs_refined_v2.json`.
- Quality gate: pass.
- Local terrain thickness p95: `1.5543 m -> 0.7319 m`, a `52.9%` reduction.
- Distance RMSE to Metashape sample: `0.2955 m -> 0.2113 m`.
- Vertical absolute p95: `0.5911 m -> 0.2722 m`.
- Metashape reference local terrain thickness p95 remains lower at `0.4698 m`, so source MVS/fusion quality still needs follow-up optimization.

- [ ] **Step 3: Build and test**

Run:

```powershell
E:/code/plascan/scripts/build_win/build_windows_cuda.ps1
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "DenseCloudQualityFilter|MvsSourcePlanner|MvsPipeline|MvsWorkspaceManifest|GuiProjectUtils"
```

Expected: targeted tests pass; any known unrelated failures are listed separately with exact test names.

## Execution Order

1. Finish Task 1 and Task 2 first so existing 9 GB raw cloud can be refined without rerunning MVS.
2. Implement Task 6 to improve the standalone refined product.
3. Implement Task 7 so GUI and mesh workflows consume the refined product.
4. Implement Task 3, Task 4, and Task 5 to prevent bad points from entering the dense cloud in the first place.
5. Run Task 8 after each major improvement and record metrics in `E:/code/plascan/docs/reports/`.

## Known Risks

- Full 9 GB PLY refinement may still require streaming/tiled processing. If the standalone CLI hits memory pressure, add tiled PLY reading/writing before running full data.
- Terrain-first meshing improves nadir UAV products, but must remain optional for oblique/complex 3D scenes.
- Aggressive filtering can reduce coverage. Every filter must report removed counts and coverage impact.
