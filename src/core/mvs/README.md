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
- `MvsSceneClassifier` resolves the candidate source pool before image preload and frame-cache preparation.
  High-resolution aerial terrain uses up to eight candidates, while orbital-object capture uses a smaller
  local pool. An explicitly larger user value is retained, and the final value is capped by the number of
  available views. Logs and CLI reports record both configured and effective pool sizes.
- The selected source plan is saved with the depth frame record so depth generation and fusion use the same
  overlap assumptions.
- If no match or track evidence exists, planning falls back to nearby sequence views instead of defaulting to
  an unbounded all-pairs search.

## Depth Quality

- Depth post-processing keeps a preview, raw depth, raw confidence, and `valid mask` when raw artifacts are
  enabled.
- Local outlier filtering and connected-component speckle filtering remove isolated red-depth spikes while
  preserving large smooth regions.
- Speckle removal builds a component-removal lookup table and scans the depth image once. It therefore scales
  as `O(component count + pixel count)` instead of rescanning the image for every small component.
- Advanced GUI controls map to `DepthGenConfig`: minimum consistent views, geometry consistency, maximum
  reprojection error, speckle area threshold, and fusion maximum image size.

## Adaptive Three-Level Depth Pyramid

- The final quality profile selects the Level 1 downsample `D`: `highest/high/medium/low/lowest` map to
  `1/2/4/8/16`. `DepthPyramidPolicy` derives Level 3/2/1 as `4D/2D/D` and degrades cleanly when an image is
  too small to keep three distinct levels.
- `MvsSceneClassifier` distinguishes down-looking aerial terrain from orbital object capture by camera-center
  layout, viewing direction consistency, and sparse-cloud thickness. Users can keep automatic detection or
  explicitly select `aerial_terrain` or `orbital_object`.
- `DepthPyramidEstimator` runs coarse-to-fine PatchMatch. A parent level propagates a depth center and a
  per-pixel uncertainty radius; low-confidence or invalid regions retain a wider/global search instead of
  being locked to a bad coarse estimate.
- CPU and CUDA PatchMatch consume the same per-pixel search radius. The final frame is accepted,
  validation-only, or rejected by `DepthFrameQualityGate` before fusion.
- Pairwise epipolar rectification is used only when OpenCV reports a usable common valid canvas. Strongly
  convergent object-ring pairs can rectify completely outside the fixed image canvas; those pairs are
  rejected by `EpipolarRectifier` and automatically fall back to the original-camera plane-homography path.
- Level 1 is always persisted with its summary. When `saveIntermediatePyramidLevels` is enabled, Level 2/3
  raw depth, confidence, support count, uncertainty, valid mask, depth preview, and confidence preview are
  also written and exposed below the frame in the workspace tree.

The corresponding CLI options are:

```powershell
reconstruct_pipeline_cli.exe image_camera.lis `
  --mvs-quality high `
  --mvs-scene-profile auto `
  --mvs-depth-filter auto `
  --mvs-save-levels
```

`scripts/validation/run_depth_pyramid_regression.ps1` runs the same reconstruction CLI for the Dino and UAV9
fixtures and then invokes `model_quality_cli` when quality validation is enabled.

## Memory, Cancel, And Fusion

- Long runs should use bounded resident depth frames. When the memory budget is tight, saved artifacts become
  the durable state and pixel storage can be released.
- `DepthConsistencyCache` provides a byte-budgeted LRU for source-neighborhood depth checks. Resident and
  streaming modes use the same reprojection consistency core; low memory no longer disables this quality
  stage.
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

For the new pyramid components, the direct filter is:

```powershell
E:/code/plascan/build/windows-vcpkg-cuda-release/tests/test_mvs_pipeline.exe `
  --gtest_filter="DepthPyramidPolicyTest.*:MvsSceneClassifierTest.*:DepthPyramidPropagationTest.*:DepthPyramidEstimatorTest.*:DepthFrameQualityGateTest.*:DepthConsistencyCacheTest.*"
```

## 2026-07-14 Regression Snapshot

- Windows CUDA Release builds of `plascan_gui`, `reconstruct_pipeline_cli`, and the focused MVS tests pass.
- Dino sparse ring, `highest`, automatic scene profile: 16/16 manifest frames are `accepted`; the effective
  source pool is six.
- UAV 9-image fixture, `high`, automatic scene profile: eight frames are `accepted`, one is
  `validation_only`, and none are rejected. The aerial candidate source pool expands from the configured
  minimum of three to eight before per-frame geometry filtering. Edge frames with at most five usable source
  views use a 0.50 consistency-retention threshold; interior frames keep the stricter 0.55 threshold.
- These results validate depth scheduling and frame gating. They do not claim that the existing visual-hull
  or mesh stage meets the model-image quality gate; mesh connectivity and rendering quality remain a separate
  follow-up.
