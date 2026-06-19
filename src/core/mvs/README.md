# MVS Module

`src/core/mvs` owns PlaScan's multi-view stereo depth-map and dense-cloud path. The current pipeline is
structured around durable frame metadata, planned source views, bounded memory, and artifact-first fusion so
large projects can be resumed and diagnosed.

## Runtime State

- `MvsWorkspaceManifest` is the disk record for depth estimation. Each frame stores the reference image,
  selected source images, `source plan`, status, device, elapsed time, `depth_png`, raw depth, raw
  `confidence`, `valid mask`, and a config hash.
- `DepthMapGenerator` writes frame artifacts before reporting completion. Completed frames with a matching
  config hash can be reused; failed frames can be retried.
- GUI project metadata consumes manifest records. The workspace tree should refresh from metadata rather than
  treating directory scans as the primary state.

## Source Planning

- `MvsSourcePlanner` scores candidate source views by shared tracks, geometric inliers, triangulation angle,
  projected coverage, baseline, known overlap, and sequence distance.
- The selected source plan is saved with the depth frame record so depth generation and fusion use the same
  overlap assumptions.
- If no match or track evidence exists, planning falls back to nearby sequence views instead of defaulting to
  an unbounded all-pairs search.

## Depth Quality

- Depth post-processing keeps a preview, raw depth, raw confidence, and `valid mask` when raw artifacts are
  enabled.
- Local outlier filtering and connected-component speckle filtering remove isolated red-depth spikes while
  preserving large smooth regions.
- Advanced GUI controls map to `DepthGenConfig`: minimum consistent views, geometry consistency, maximum
  reprojection error, speckle area threshold, and fusion maximum image size.

## Memory, Cancel, And Fusion

- Long runs should use bounded resident depth frames. When the memory budget is tight, saved artifacts become
  the durable state and pixel storage can be released.
- Cancellation is cooperative: preload, source planning, hint-depth preparation, PatchMatch iteration
  boundaries, artifact saving, and fusion setup must poll the cancel flag.
- `DepthMapFusion` supports `streaming fusion` from planned source images and clears stale output/cache at the
  start of every run so a cancelled run cannot expose old dense points.
- `reconstruct_pipeline_cli --mvs-depth-only` is the safest validation mode for large aerial projects when the
  goal is to exercise depth scheduling, artifact persistence, cancellation, and manifest recovery without
  entering fusion, mesh, or terrain generation.

## Focused Tests

Useful filters:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|MvsSourcePlanner|MvsDepthPostprocess|MvsPipelineTest" --output-on-failure
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DenseCloudDialog|DepthMapPersistence|DepthFrameUtils|DenseDepth" --output-on-failure
```
