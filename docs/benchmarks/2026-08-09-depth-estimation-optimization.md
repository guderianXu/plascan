# Depth estimation algorithm optimization (2026-08-09)

## Scope

This benchmark isolates revision-27 MVS depth estimation on the AMD `gfx1036` OpenCL device. It keeps one physical
GPU execution lane, two host preparation slots, the full inverse-depth hypothesis count, all confidence/consistency
gates, and the existing measured-depth recovery stages. No depth interpolation or model-stage interpolation is
enabled.

The retained optimizations are:

- native-resolution coarse/middle pyramid outputs with virtual full-resolution prior propagation;
- cached current-hypothesis scores and duplicate-candidate elimination on CPU, CUDA, and OpenCL;
- one OpenCL plane homography per source/hypothesis, incremental patch projection, exact streaming top-k source
  aggregation, compact checkerboard dispatch, and reusable optional buffers;
- globally bounded OpenCL uniqueness probes aligned with CPU/CUDA at propagated-prior boundaries;
- batched reference unprojection during geometric consistency and early residual-recovery support preflight.

An experimental reduction of the local coarse depth sample count was rejected before submission: it was faster but
changed the depth tail more than the retained implementation. The final source always uses the configured full
hypothesis count.

## Fixed input and command

- Input manifest:
  `E:/code/test/hyb2/validation_revision25_aligned_v3_opencl_amd_depth_20260808/mvs_manifest.json`
- Sparse cloud:
  `E:/code/test/hyb2/hyb2.files/1/assets/aerial_triangulation/sfm_sparse/sfm_sparse.ply`
- Baseline output:
  `E:/code/test/hyb2/validation_revision27_preopt_opencl_amd_baseline_20260809`
- Final output:
  `E:/code/test/hyb2/validation_revision27_optimized_final_opencl_amd_20260809`

Both runs used:

```text
--quality highest --scene-profile orbital_object --depth-filter moderate
--device opencl --source-views 6 --threads 30 --gpu-frame-workers 2
--opencl-device-index 1
```

## Performance

| Metric | Baseline | Final source | Change |
| --- | ---: | ---: | ---: |
| Batch elapsed | 233,162.3 ms | 170,022.0 ms | -27.1%, 1.371x |
| OpenCL wall | 227,693.7 ms | 165,134.4 ms | -27.5% |
| Queue time | 219,758.0 ms | 156,011.2 ms | -29.0% |
| Kernel active | 218,954.4 ms | 155,283.0 ms | -29.1% |
| Inter-call idle | 7,935.8 ms | 9,123.2 ms | +1,187.4 ms |
| Queue occupancy | 96.5% | 94.5% | -2.0 pp |
| End-to-end kernel duty | 96.2% | 94.0% | -2.2 pp |
| OpenCL calls | 71 | 71 | unchanged |

The lower duty percentage is not a regression in throughput: the kernel work itself falls by 63.67 seconds while
the fixed host gaps become a larger fraction of the shorter run. The device remains continuously kernel-bound for
94% of the measured end-to-end OpenCL interval.

## Output comparison

All 14 frames completed in both runs. Acceptance remained exactly `2 accepted / 4 validation_only / 8 rejected`,
and fusion eligibility remained `2/14`.

| Metric | Baseline | Final source |
| --- | ---: | ---: |
| Valid depth pixels | 1,820,587 | 1,845,481 |
| Mean full-image coverage | 0.12401765 | 0.12571342 |
| Aggregate valid-mask IoU | \- | 0.98143239 |
| Common-valid relative depth error P50 | \- | 1.0779e-6 |
| Common-valid relative depth error P95 | \- | 0.00101965 |
| Common-valid relative depth error P99 | \- | 0.01282021 |

Frame 13 accounts for almost all mask divergence: it gains 24,854 valid pixels after the OpenCL uniqueness-boundary
fix. Excluding that frame, aggregate valid-mask IoU is 0.99482754. This is a correctness alignment rather than a
relaxed gate; no acceptance or fusion category changes.

The normally rebuilt final executable was also compared against the isolated no-adaptive A/B executable. All 14
depth matrices and all 14 confidence matrices were bit-for-bit identical, confirming that the reported run matches
the submitted full-sampling source behavior.
