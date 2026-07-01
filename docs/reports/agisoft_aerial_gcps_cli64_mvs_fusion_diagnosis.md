# Agisoft Aerial GCP CLI64 MVS Fusion Diagnosis

Date: 2026-06-28

## Inputs

- Image/camera list: `E:/code/test/agisoft_aerial_gcps_cli_ab_64/image_camera_64_abs_fwd.lis`
- PlaScan output: `E:/code/test/agisoft_aerial_gcps_cli_ab_64/run_current_abs_fwd`
- Metashape reference sample: `E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply`
- Comparison JSON: `E:/code/plascan/docs/reports/agisoft_aerial_gcps_cli64_vs_metashape_sample.json`

The original `.lis` generated with Windows backslashes failed because the CLI list parser treats backslashes as escapes.
The successful run used forward slashes in absolute Windows paths.

## CLI Command

```powershell
& E:/code/plascan/build/windows-vcpkg-cuda-release/bin/reconstruct_pipeline_cli.exe `
  E:/code/test/agisoft_aerial_gcps_cli_ab_64/image_camera_64_abs_fwd.lis `
  --output-dir E:/code/test/agisoft_aerial_gcps_cli_ab_64/run_current_abs_fwd `
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

## Observed Results

- SFM registered images: `64/64`
- SFM sparse points: `12,210`
- Filtered sparse points: `9,875`
- Mean reprojection error: `0.8973 px`
- MVS selected frames: `24`
- Example depth maps contain millions of valid pixels, such as `4,207,623`, `3,166,979`, and `4,747,933` valid pixels.
- `mvs/dense_cloud.ply`: `784` vertices
- `mvs/refined_dense_cloud.ply`: `736` vertices

The sparse stage is therefore not the first blocking issue for this run. The dominant failure is that dense fusion collapses millions of valid depth pixels into a sparse-like point cloud.

## Metashape Comparison

Comparison against the Metashape sample produced:

- PlaScan source points: `784`
- Reference sample points: `50,000`
- NN RMSE: `0.4987 m`
- NN median: `0.1774 m`
- NN P95: `1.2007 m`
- Vertical RMSE: `0.4470 m`
- Vertical P95 absolute error: `1.1060 m`
- Reference coverage within `0.25 m`: `0.56%`

The distance statistics are less informative than the coverage statistic because the PlaScan cloud contains too few points. Coverage is the main regression metric until fusion produces a production-scale dense cloud.

## Root Cause Hypothesis

The stream fusion path `DepthMapFusion::fuseFirstFrameObservationsFast()` requires strict multi-view agreement. With `minNumPixels=3`, any point that has only one consistent neighbor is rejected even when the reference depth and one source depth are mutually consistent.

This is too brittle for the current aerial MVS output. It is safer to keep strict fusion as the first pass, but if strict fusion yields an abnormally low point/valid-depth ratio, fall back to a two-view-consistent pass. The fallback must never become pure single-view back-projection.

## Implemented Guardrail

Added a TDD regression test:

```text
MvsPipelineTest.StreamingFirstFrameFusionFallsBackToTwoViewAgreementWhenStrictYieldCollapses
```

The test constructs three stream-fusion frames:

- reference frame depth is valid
- one neighbor agrees
- one neighbor disagrees
- strict `minNumPixels=3` should collapse
- fallback should preserve two-view-supported observations

The adjacent safety test still passes:

```text
MvsPipelineTest.StreamingFirstFrameFusionRejectsDepthsWithoutNeighborAgreement
```

This confirms the fallback does not emit unsupported single-view points when all neighbors disagree.

## Verification

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "StreamingFirstFrameFusion(FallsBackToTwoViewAgreementWhenStrictYieldCollapses|RejectsDepthsWithoutNeighborAgreement)"
```

Result:

```text
100% tests passed, 0 tests failed out of 2
```

The full MVS pipeline test group also passes:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "MvsPipelineTest"
```

Result:

```text
100% tests passed, 0 tests failed out of 24
```

MVS source, workspace, depth postprocess, and quality-report tests pass:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release --output-on-failure -R "Mvs(SourcePlanner|WorkspaceManifest|DepthPostprocess|QualityReport)"
```

Result:

```text
100% tests passed, 0 tests failed out of 21
```

## Small Real-Data Check After Fix

Ran a smaller 16-image / 8-frame CLI check:

```powershell
& E:/code/plascan/build/windows-vcpkg-cuda-release/bin/reconstruct_pipeline_cli.exe `
  E:/code/test/agisoft_aerial_gcps_cli_ab_16/image_camera_16_abs_fwd.lis `
  --output-dir E:/code/test/agisoft_aerial_gcps_cli_ab_16/run_fusion_fallback `
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
  --mvs-max-frames 8 `
  --mvs-fusion-max-image-dim 2048 `
  --skip-mesh `
  --skip-terrain `
  --force
```

Observed stream-fusion logs:

```text
[StereoFusion] 严格流式融合产出过低: points=69 valid=130106 ratio=0.0005, fallback minNumPixels=2
[StereoFusion] 已采用双视一致 fallback: points=2151
```

Output:

- `E:/code/test/agisoft_aerial_gcps_cli_ab_16/run_fusion_fallback/mvs/dense_cloud.ply`: `13,056` vertices
- `E:/code/test/agisoft_aerial_gcps_cli_ab_16/run_fusion_fallback/mvs/dense_cloud_refined.ply`: `12,529` vertices
- Postprocess valid depth pixels across 8 frames after confidence/local filtering: `684,359`

This is not yet a production-density result, but it proves the previous strict-only fusion collapse is real and the guarded two-view fallback improves the real CLI path.

## Next Checks

1. Re-run the 64-image CLI benchmark with the same settings and compare point count/coverage against the original `784`-point baseline.
2. Recompute `agisoft_aerial_gcps_cli64_vs_metashape_sample.json`.
3. Add fusion reject-reason counters so the GUI can explain whether points were rejected by depth, reprojection, normal, bbox, or missing source pixels.
4. Treat success as improved dense point count and improved Metashape sample coverage, not only passing unit tests.
