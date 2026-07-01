# Agisoft Aerial GCPs Metashape Dense Cloud Baseline

Date: 2026-06-28

## Purpose

Record a repeatable first-pass comparison between the current PlaScan dense cloud and the Metashape reference dense cloud for `agisoft_aerial_gcps`.

This is a smoke-test baseline, not a final quality acceptance result. The command uses deterministic point sampling so it can run quickly against very large binary PLY files.

## Inputs

- PlaScan candidate:
  `E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply`
- Metashape reference sample:
  `E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply`
- Output JSON:
  `E:/code/test/agisoft_aerial_gcps/reports/metashape_cloud_compare_smoke.json`

## Command

```powershell
python E:/code/plascan/testData/compare_point_cloud_to_lidar.py `
  --source E:/code/test/agisoft_aerial_gcps/mvs_output/dense_cloud.ply `
  --reference E:/code/plascan/testData/photogrammetry_benchmarks/agisoft_aerial_gcps/extracted/aerial_images_with_gcps/converted/metashape_dense_cloud/metashape_dense_cloud_leaf_local_sample_1pct_with_normals.ply `
  --output-json E:/code/test/agisoft_aerial_gcps/reports/metashape_cloud_compare_smoke.json `
  --nearest-neighbor-method kd-tree `
  --max-source-points 5000 `
  --max-reference-points 5000
```

## Result

```json
{
  "source_points": 5000,
  "reference_points": 5000,
  "distance_m": {
    "mean": 0.338641952535347,
    "rmse": 0.4038380142483831,
    "median": 0.2980609067567894,
    "p95": 0.7527004297871693,
    "max": 3.0835520557126954
  }
}
```

## Notes

- Both files are binary little-endian PLY. The comparison helper now supports binary PLY and deterministic sampling.
- The current PlaScan dense cloud contains hundreds of millions of vertices, so full in-memory comparison is not appropriate as the default path.
- A later production comparison should add tile-based coverage metrics, bidirectional distances, optional Sim3 alignment checks, and per-region error histograms.

## 50k Sample With Vertical Roughness Gate

Output JSON:
`E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_dense_sample.json`

Command:

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

Result:

```json
{
  "source_points": 50000,
  "reference_points": 50000,
  "distance_m": {
    "mean": 0.20786190653611222,
    "rmse": 0.29551684073359497,
    "median": 0.15027830103250323,
    "p95": 0.6338128908493976,
    "max": 3.6391121843456316
  },
  "vertical_error_m": {
    "mean_signed": 0.04071821455955505,
    "mean_abs": 0.15621941430091857,
    "rmse": 0.2652167174845468,
    "p50_abs": 0.09420633316040039,
    "p95_abs": 0.5911393165588379,
    "max_abs": 3.6371030807495117
  },
  "reference_coverage": {
    "covered_percent": 87.948,
    "radius_m": 0.5
  },
  "quality_gate": {
    "passed": false,
    "failure_codes": [
      "vertical_rmse_above_threshold",
      "vertical_p95_above_threshold"
    ]
  }
}
```

Interpretation:

- The current dense cloud is not simply missing coverage: sampled reference coverage is 87.948% within 0.5 m.
- The visible roughness is reflected in vertical error: RMSE is 0.265 m and absolute P95 is 0.591 m, both above the provisional production thresholds.
- The next optimization should target MVS local depth spikes, confidence-aware fusion, and terrain-mesh height aggregation rather than only increasing point count.

## 50k Sample With Tighter Coverage Radius

Output JSON:
`E:/code/plascan/docs/reports/agisoft_aerial_gcps_plascan_vs_metashape_sample.json`

Command:

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

Result:

```json
{
  "source_points": 50000,
  "reference_points": 50000,
  "distance_m": {
    "mean": 0.20786190653611222,
    "rmse": 0.29551684073359497,
    "median": 0.15027830103250323,
    "p95": 0.6338128908493976,
    "max": 3.6391121843456316
  },
  "vertical_error_m": {
    "mean_signed": 0.04071821455955505,
    "mean_abs": 0.15621941430091857,
    "rmse": 0.2652167174845468,
    "p50_abs": 0.09420633316040039,
    "p95_abs": 0.5911393165588379,
    "max_abs": 3.6371030807495117
  },
  "reference_coverage": {
    "covered_percent": 69.038,
    "radius_m": 0.25
  }
}
```

Additional interpretation:

- Reducing the coverage radius from 0.5 m to 0.25 m drops sampled reference coverage from 87.948% to 69.038%.
- The same distance and vertical-error statistics appear because the deterministic 50k samples are identical; only the coverage radius changed.
- This supports treating the current problem as both a roughness issue and a local density/coverage issue. The next A/B run should compare regenerated depth/fusion output against both 0.25 m and 0.5 m coverage gates.
