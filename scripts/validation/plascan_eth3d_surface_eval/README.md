# ETH3D surface evaluator

This package evaluates a PlaScan triangle mesh against an ETH3D
`scan_alignment.mlp` without changing either input. It applies every MLP scan
transform, voxelizes the laser points deterministically, samples the mesh with
a fixed seed, and reports bidirectional point distances plus exact triangle
topology counts.

```bash
.venv/bin/python -m scripts.validation.plascan_eth3d_surface_eval \
  --mesh /abs/path/model.ply \
  --scan-alignment /abs/path/scan_alignment.mlp \
  --output /abs/path/surface_report.json
```

The report calls reconstruction-to-scan distance `accuracy` and scan-to-
reconstruction distance `completeness`. These are deterministic sampled
distances, not the official ETH3D occlusion-aware score; the distinction is
stored in `metric_semantics`. The CLI is no-clobber. PLY and triangular OBJ
meshes are supported, and scan PLY input may be ASCII or binary little-endian.
