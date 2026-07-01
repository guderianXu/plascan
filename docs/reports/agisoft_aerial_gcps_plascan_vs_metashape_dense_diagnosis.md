# Agisoft Aerial GCPs Dense Cloud Diagnosis

Date: 2026-06-28

## Inputs

- PlaScan project: `E:/code/test/agisoft_aerial_gcps`
- PlaScan dense cloud: `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply`
- Metashape reference sample: `E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply`

## Commands

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --output-json E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_dense_quick.json `
  --max-source-points 100000 `
  --max-reference-points 100000 `
  --nearest-neighbor-method kd-tree `
  --coverage-radius-m 0.15
```

## Findings

- Nearest-neighbor distance, PlaScan to Metashape sample:
  - mean: `0.1915 m`
  - RMSE: `0.2847 m`
  - median: `0.1313 m`
  - P95: `0.6170 m`
  - max: `3.2223 m`
- Vertical error:
  - mean signed: `0.0417 m`
  - mean absolute: `0.1521 m`
  - RMSE: `0.2612 m`
  - P50 absolute: `0.0903 m`
  - P95 absolute: `0.5813 m`
  - max absolute: `3.1953 m`
- Reference coverage within `0.15 m`: `51.42%`.

The dense cloud is not only visually rough. It has measurable vertical noise and incomplete agreement with the Metashape reference.

## Roughness Check

A deterministic sample was binned into `0.35 m` XY cells. Cells with fewer than 5 sampled points were ignored.

| Metric | PlaScan | Metashape sample |
| --- | ---: | ---: |
| Used cells | 7545 | 11382 |
| Z range P50 | 0.7198 m | 0.0594 m |
| Z range P84 | 1.4187 m | 0.3018 m |
| Z range P95 | 1.8421 m | 0.4161 m |
| Z std P50 | 0.1720 m | 0.0202 m |
| Z std P84 | 0.3664 m | 0.1227 m |
| Z std P95 | 0.5193 m | 0.1603 m |

This confirms that PlaScan is preserving many high-frequency vertical outliers. The mesh spikes are therefore a downstream symptom of dense cloud quality, not just a rendering issue.

## Pipeline Clues

- `aerial_triangulation_sfm_report.json` reports:
  - `num_registered = 444`
  - `num_points_3d = 252344`
  - `mean_reproj_error_px = 0.8865`
  - `ba_rms_after = 0.6491`
- `matching_quality_report.json` reports:
  - candidate graph: 444 images, 1576 edges, 1 component
  - actual match graph: 444 images, 984 edges, 13 components
  - failed pairs: 592
- `mvs_manifest.json` reports:
  - 444 completed depth frames
  - source views per frame: min/p50/p95/max = 3/3/3/3
  - valid pixels per frame: min/p50/p95/max = 6249769/23860899/24000000/24000000
  - total valid pixels before fusion: 10414030433
- `reconstruction_quality_report.json` is stale or incomplete:
  - `registered_image_count = 0`
  - `sparse_point_count = 0`
  - `dense_point_count = 0`
  - `mvs_completed_depth_frame_count = 444`

## Diagnosis

The most likely failure path is:

1. The actual tie-point graph is weak and split into 13 connected components. The workflow can still carry all 444 cameras because external priors exist, but MVS source selection is not backed by a strong global tie network.
2. MVS uses only 3 source views per reference image. That is too little redundancy for this low-altitude vegetation/field scene.
3. Depth estimation/fusion is too permissive: nearly full-frame valid masks are being accepted, and fusion produces a 336M-point cloud versus the 79M-point Metashape reference. The extra density is mostly noisy vertical variation rather than useful detail.
4. Mesh reconstruction is then triangulating noisy dense points directly, so the mesh becomes rough and spiky.
5. Project quality metadata is inconsistent, so the GUI can present downstream products as valid even when the upstream quality summary is missing or stale.

## Recommended Fix Order

1. Reuse existing algorithm-suffixed match caches during formal SfM so the workflow does not rematch already processed pairs.
2. Make MVS source selection require stronger evidence:
   - prefer shared tracks and geometric inliers over sequence-only neighbors,
   - raise source view count for aerial projects from 3 to 5-8 where memory allows,
   - reject sources from disconnected actual match components unless external-prior mode explicitly allows them.
3. Tighten depth-map validity:
   - raise confidence threshold for production dense clouds,
   - add local speckle and vertical jump filtering before fusion,
   - store confidence/support count per dense point.
4. Tighten fusion:
   - default to at least 3-view geometric agreement for production dense clouds,
   - use 2-view fallback only for preview/low-yield mode,
   - reject points with large reprojection/depth disagreement before writing PLY.
5. Update dense cloud and reconstruction quality reports from actual output metadata so GUI decisions do not rely on stale zero-valued summaries.
6. Feed the cleaned dense cloud into mesh reconstruction. Avoid meshing the raw noisy cloud directly.

