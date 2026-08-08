# Agisoft 100-camera model-generation optimization benchmark

Date: 2026-08-08  
Platform: Windows/MSVC, 32 logical CPU threads  
Scope: model generation only; depth estimation is excluded

## Fixed input

- Source: `testData/photogrammetry_benchmarks/agisoft_aerial_gcps`
- Prepared input: `testData/photogrammetry_benchmarks/agisoft_aerial_gcps_100/prepared/plascan`
- Selection: connected centroid-seeded nearest-neighbor growth
- Selected cameras: 100
- Symmetric graph degree: 8
- Internal adjacency edges: 376
- Image size: 1600 x 1067
- Interpolation: disabled
- Requested model workers: 30 (`logical threads - 2`)
- Bounded frame-I/O workers: 8

The MVS input was generated once with an explicit `aerial_terrain` scene
profile and then reused by every model run. It contains 99 completed frames and
one frame without a usable source view. Model loading accepted 87 frames under
the common quality policy.

## Result

| Configuration | Wall time | Frame load | TSDF build | Vertex color | Vertices | Faces |
|---|---:|---:|---:|---:|---:|---:|
| 1 worker | 8119 ms | 3449 ms | 3736 ms | 262 ms | 3 | 1 |
| automatic 30-worker budget | 3067 ms | 845 ms | 1329 ms | 223 ms | 3 | 1 |

- End-to-end speedup: **2.647x**
- Frame-loading speedup: **4.082x**
- Serial PLY SHA-256:
  `980b3a746f9d3dad4f248d1ab1ab47dbf58f91c5f79d1efe2b2c14978b63a033`
- Parallel PLY SHA-256:
  `980b3a746f9d3dad4f248d1ab1ab47dbf58f91c5f79d1efe2b2c14978b63a033`

Both runs exited successfully. Matching topology and byte-identical PLY hashes
confirm that bounded parallel loading and deterministic vertex-color
parallelism did not change the generated result.

## Interpretation and limitation

The speedup is primarily from parallel evidence-file loading and existing
parallel TSDF work receiving the automatic worker budget. Frame loading is
bounded at eight workers instead of launching one full-resolution evidence set
per logical CPU thread, which limits peak memory while keeping the NVMe queue
busy.

With interpolation disabled, the current fixed aerial depth support produces a
very small final mesh. This dataset is therefore useful for measuring the
100-frame input, validation, TSDF scheduling, and deterministic output path,
but it is not a model-quality reference and does not stress high-face-count
simplification. Hyb2 remains necessary for object-surface quality regression.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\bench\run_model_generation_benchmark.py `
  --exe build\windows-vcpkg-cuda-release\bin\mesh_reconstruct_cli.exe `
  --depth-map-dir build\benchmark_runs\agisoft_aerial_gcps_100_depth_aerial_v1\headless.files\1\reconstruction\mvs `
  --settings-json scripts\bench\agisoft_aerial_gcps_100_mesh_settings.json `
  --output-dir build\benchmark_runs\agisoft_aerial_gcps_100_mesh_aerial_ab `
  --repeat 1 `
  --parallel-workers 30
```

The machine-readable report is written to
`build/benchmark_runs/agisoft_aerial_gcps_100_mesh_aerial_ab/benchmark_results.json`.
