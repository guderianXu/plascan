# MVS Module

`src/core/mvs` owns PlaScan's multi-view stereo depth-map and dense-cloud path. The current pipeline is
structured around durable frame metadata, planned source views, bounded memory, and artifact-first fusion so
large projects can be resumed and diagnosed.

## Reusable Workflow Services

- `DenseCloudRefinementService` owns chunked binary PLY refinement, multi-pass terrain spike and local-plane
  filtering, plus the in-memory fallback used by `dense_cloud_refine_cli`.
- `StreamingDepthFusionService` owns reference-window selection, small-project frame caching, consensus tuning,
  batched fusion, and the pre-reduction trigger. Callers provide only a frame loader and progress callback.
- `PointCloudArtifactIO` creates output directories and writes Binary LE PLY with an explicit normal-retention
  policy.

These services do not depend on CLI option objects and can be reused by GUI or batch workflows. Their tests
live under `src/core/mvs/tests/`.

## Runtime State

- `MvsWorkspaceManifest` is the disk record for depth estimation. Each frame stores the reference image,
  selected source images, `source plan`, status, device, elapsed time, `depth_png`, raw depth, raw
  `confidence`, `valid mask`, and a config hash.
- `DepthMapGenerator` writes frame artifacts before reporting completion. Completed frames with a matching
  config hash can be reused; failed frames can be retried.
- Orbital depth artifacts at algorithm revision 14 persist adaptive support weight, effective view count,
  and conflict ratio. Reuse rejects an incomplete revision-14 evidence set instead of silently interpreting
  missing evidence as valid geometry.
- GUI project metadata consumes manifest records. The workspace tree should refresh from metadata rather than
  treating directory scans as the primary state.

## Source Planning

- `MvsSourcePlanner` scores candidate source views by shared tracks, geometric inliers, triangulation angle,
  projected coverage, baseline, known overlap, and sequence distance. The angle and physical-front checks are
  computed through `core/camera/CameraBaseline`, so zero-baseline pairs and points behind either camera do not
  become MVS source evidence.
- `MvsSceneClassifier` resolves the candidate source pool before image preload and frame-cache preparation.
  High-resolution aerial terrain uses up to eight candidates. Dense orbital rings with at least 16 cameras
  may use six views, while sparser rings and lower-quality object capture use four; this prevents a 12-view
  ring from treating the roughly 90-degree third neighbor as mandatory confirmation. The final value is
  capped by the number of available views. Aerial planning keeps the 35-degree maximum triangulation angle.
  Orbital planning derives the required angle from the actual sorted candidate angles, with 70/90-degree
  safety caps for four/six-source jobs. Logs and CLI reports record both configured and effective pool sizes.
- The selected source plan is saved with the depth frame record so depth generation and fusion use the same
  overlap assumptions.
- When verified pair geometry is available, verified pairs are selected first and shared-track geometry may
  backfill only the remaining slots. The manifest records the requested count, shortfall, and verified,
  backfill, or sequence tier for every selected source.
- `.pimatch` stores the geometry model, inlier flags, residuals, and per-pair counts produced by the matching
  workflow. MVS reads those persisted statistics directly instead of reopening feature files or rerunning
  USAC/MAGSAC. `mvs_pair_audit_cli` exports the stored evidence and `mvs_depth_reprocess_cli` replays it without
  modifying the original workspace.
- If no match or track evidence exists, planning falls back to nearby sequence views instead of defaulting to
  an unbounded all-pairs search.
- `mvs_depth_reprocess_cli --depth-pose-candidates` runs an experimental, default-off pose audit after
  cross-view evidence is complete. It deterministically samples supported, low-conflict neighboring depth
  observations, rejects occlusions, and reports robust point-to-plane SE(3) candidates. If a PatchMatch backend
  does not expose its internal plane normals, the audit deterministically derives local surface normals from the
  completed depth and camera model; the three adaptive geometry-evidence maps remain mandatory. The anchor and
  scale remain fixed; P90 residual, motion-limit, evidence-coverage, and projection-retention gates must all pass.
  Accepted cameras are persisted only as `derived_camera_model` candidates. They never replace project
  cameras or change depth maps in the same run.

## Depth Quality

- Depth post-processing keeps a preview, raw depth, raw confidence, and `valid mask` when raw artifacts are
  enabled.
- `valid_coverage` is the canonical per-frame coverage field. Quality reporting also accepts
  `depth_quality.valid_coverage`, legacy `valid_ratio`, or a value computed from valid pixels and grid size.
  Missing coverage remains unavailable and is displayed as `—`, never as a measured zero.
- Depth artifacts are not standalone workspace resources. The photo toolbar resolves the matching frame by
  `ref_image` and overlays depth on that photo. Feature points and residual diagnostics are temporarily hidden
  while depth inspection is active, then restored from the user's existing preferences.
- Local outlier filtering and connected-component speckle filtering remove isolated red-depth spikes while
  preserving large smooth regions.
- Speckle removal builds a component-removal lookup table and scans the depth image once. It therefore scales
  as `O(component count + pixel count)` instead of rescanning the image for every small component.
- Advanced GUI controls map to `DepthGenConfig`: minimum consistent views, geometry consistency, maximum
  reprojection error, speckle area threshold, and fusion maximum image size.

## Adaptive Three-Level Depth Pyramid

- The final quality profile selects the Level 1 downsample `D`: `highest/high/medium/low/lowest` map to
  `1/2/4/8/16`. `DepthPyramidPolicy` derives Level 3/2/1 as `4D/2D/D` and degrades cleanly when an image is
  too small to keep three distinct levels. For the `high` profile, images with a short side no larger than
  1280 pixels keep a native-resolution final level (`4/2/1` for a 1024-pixel input); the memory saving from
  a half-resolution final pass is small at this size, while its lost silhouette detail is measurable.
- `MvsSceneClassifier` distinguishes down-looking aerial terrain from orbital object capture by camera-center
  layout, viewing direction consistency, and sparse-cloud thickness. Users can keep automatic detection or
  explicitly select `aerial_terrain` or `orbital_object`.
- `DepthPyramidEstimator` runs coarse-to-fine PatchMatch. A parent level propagates a depth center and a
  per-pixel uncertainty radius; low-confidence or invalid regions retain a wider/global search instead of
  being locked to a bad coarse estimate.
- A finer level that retains less than 60% of the parent coverage is treated as a quality collapse. The
  estimator keeps the parent depth already resized to the final raster size and records the selected level
  and coverage-regression reason instead of publishing a severely incomplete fine result.
- CPU and CUDA PatchMatch consume the same per-pixel search radius. The final frame is accepted,
  validation-only, or rejected by `DepthFrameQualityGate` before fusion.
- CPU and CUDA PatchMatch also consume the same reference/source valid masks. Plane-homography NCC uses only
  samples that are foreground in the reference mask and whose four source bilinear neighbors are foreground;
  a masked patch needs at least 35% valid samples (and at least four) before it can contribute photometric
  support. Calls without masks retain the legacy all-image behavior.
- `DepthCompletenessMetrics` measures coverage inside the effective project/content mask rather than against
  the whole raster. It separately records small interior holes, large interior openings, boundary-connected
  invalid regions, output-filter retention, and cross-view consistency retention.
- `valid_mask_path` is the final depth-valid mask; `support_mask_path` is the project/content support region.
  They are intentionally distinct so a missing depth sample is not silently reinterpreted as free space.
- Cross-view consistency selects its few-view policy from the actual source count and depth-filter preset.
  One source remains contradiction-only. With multiple sources, `mild`, `moderate`, and `aggressive` normally
  require respectively one, two, and three independent source confirmations; orbital `mild` frames with four
  or more sources require two. Two-source frames retain 10%, 6%, and 3% relative-depth tolerances. With three
  or more sources, aerial tolerances are 3%, 1.5%, and 0.75%, while convergent orbital capture uses the stricter
  1.25%, 0.8%, and 0.5% thresholds plus a source-to-reference round-trip check.
- The 0.80 mask-normalized coverage gate applies only to constrained project/content masks. Aerial frames using
  `full_image` retain the established edge/interior consistency thresholds instead of being downgraded merely
  because valid terrain does not cover the whole 6000×4000 raster.
- Pairwise epipolar rectification is used only when OpenCV reports a usable common valid canvas. Strongly
  convergent object-ring pairs can rectify completely outside the fixed image canvas; those pairs are
  rejected by `EpipolarRectifier` and automatically fall back to the original-camera plane-homography path.
- Level 1 is always persisted with its summary. The GUI enables `saveIntermediatePyramidLevels` by default,
  so Level 2/3 raw depth, confidence, support count, uncertainty, valid mask, depth preview, and confidence
  preview are available to the per-photo overlay. The workspace tree exposes only one non-image aggregate
  depth-map node; it never materializes individual depth frames or previews as workspace resources.

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
  boundaries, artifact saving, frame loading, and every streaming-fusion window poll the same cancel flag.
- `DepthMapFusion` supports `streaming fusion` from planned source images and clears stale output/cache at the
  start of every run so a cancelled run cannot expose old dense points.
- The GUI `Create Point Cloud` workflow accepts only the latest production SfM/BA result. It reuses a stored
  depth batch only when every expected frame exists and its algorithm revision, input signature, and
  reconstruction generation match; otherwise it regenerates depth artifacts before bounded streaming fusion.
  Input signature version 2 hashes stable image identities and camera geometry rather than archive-dependent
  paths or result bookkeeping. Legacy batches whose old path-sensitive signature changes during project
  archiving are accepted only after every stored depth camera is verified against the current project camera.
- `reconstruct_pipeline_cli --mvs-depth-only` is the safest validation mode for large aerial projects when the
  goal is to exercise depth scheduling, artifact persistence, cancellation, and manifest recovery without
  entering fusion, mesh, or terrain generation.

## Focused Tests

Useful filters:

```powershell
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "MvsWorkspaceManifest|MvsSourcePlanner|MvsDepthPostprocess|MvsPipelineTest" --output-on-failure
ctest --test-dir E:/code/plascan/build/windows-vcpkg-cuda-release -C Release -R "DenseCloudRefinement|DenseCloudQualityFilter|DepthMapPersistence|DepthFrameUtils|DenseDepth" --output-on-failure
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

## 2026-07-16 Aerial Mesh Policy

- The reconstruction CLI now carries the effective MVS scene profile into meshing. Aerial terrain uses the
  height-field path; orbital-object scenes keep Poisson reconstruction.
- For `high` and `highest` aerial runs, the height-field policy keeps at least a 320-cell resolution and skips
  a second voxel downsample. On the UAV 9-image fixture this changed median rendered coverage from 0.165 to
  0.791 and edge P90 from 94.37 px to 37.86 px. The strict model-quality gate still fails, so depth support at
  strip boundaries and residual mesh fragmentation remain follow-up work.

## 2026-07-17 Direct-Depth Regression

- Temple source planning now supplies 2--5 source views per frame instead of almost always two. One-source
  consistency remains contradiction-only; two sources require one confirmation within 10%, and three or more
  sources use the 5% threshold. The 47-degree run retained 12 accepted and four validation-only frames, with
  mean mask-normalized depth coverage `0.9380` and minimum coverage `0.8681`.
- The direct TSDF path keeps strong single observations, defaults to a 7.5-voxel truncation band, does not carve
  support-mask-exterior samples, and only fills bounded micro-holes. Small connected components below 2.5% of
  the largest component are removed. The 47-degree Temple model has one component, rendered coverage `0.8072`,
  IoU `0.7811`, edge P90 `80.82 px`, and SSIM `0.5673`. Coverage, IoU, and appearance improved substantially,
  but structural-edge quality remains follow-up work; the implementation does not widen blind hole filling.
- The fresh UAV9 CUDA regression completed with pipeline status `ok`, 9/9 depth frames, seven accepted and two
  validation-only frames. Direct parsing of all nine version-3 `SFTB` artifacts reported exactly 40000 SIFT
  keypoints per image, with no frame above the cap. The aerial profile continues to use the 35-degree angle gate.

## 2026-07-18 Dark-Background Object Completeness

- Orbital-object frames with an explicit project mask now refine only dark interior openings when the excluded
  background is dark. A protected four-pixel boundary band and a 75% retained-area floor keep the authoritative
  outer silhouette and thin columns intact; aerial and bright-background images do not use this refinement.
- On Temple, the refinement increased accepted depth frames from 12 to 15 and reduced large internal invalid
  regions from 51,796 to 15,139 pixels. The strict three-or-more-source consistency threshold remains 5%; the
  tested 7.5% relaxation admitted all frames but reduced rendered coverage and SSIM.
- Direct-depth TSDF now limits positive free-space integration to 36 voxels and requires two distinct camera
  observations by default. This prevents far background depths from erasing thin foreground columns while
  rejecting single-view interior sheets. At resolution 320 the selected Temple model has one component,
  coverage `0.8373`, IoU `0.8185`, edge P90 `81.50 px`, and SSIM `0.5573`; the 384-cell ultra preset reaches
  coverage `0.8391`, IoU `0.8204`, edge P90 `80.03 px`, and SSIM `0.5655`.
- The ultra direct-depth preset erodes only the final depth-valid boundary by two pixels before TSDF integration;
  both high and ultra require two camera observations by default, while high keeps one-pixel
  erosion. Open mesh boundaries receive one displacement-limited smoothing pass, while interior vertices and
  real large openings are left untouched. On the same Temple depth artifacts the ultra result keeps one
  component, reduces post-fill boundary edges from `139751` to `115466`, and records coverage `0.8132`, IoU
  `0.8040`, edge P90 `80.12 px`, and SSIM `0.5703`. A five-voxel truncation experiment was rejected because it
  reduced coverage to `0.7524` despite fewer boundary edges.

## 2026-07-18 Robust Mesh Appearance and Weak-Boundary Cleanup

- Direct-depth mesh vertex colors now use per-view mesh z-buffers, strict depth consistency, view-angle weights,
  robust color outlier rejection, conservative best-view fallback, and normal-aware hole propagation. Temple has
  no default-gray vertices after coloring; its render SSIM increased from `0.5703` to `0.6022` while geometry
  coverage and IoU stayed effectively unchanged.
- OBJ texture generation now projects the original MVS source photographs per face into a tiled atlas. The final
  Temple camera-atlas check mapped all `1,137,405` faces (`1,111,920` strict and `25,485` conservative fallback),
  eliminating the former global planar-UV overlap and order-dependent vertex-color bake.
- Ultra TSDF output automatically peels one open-boundary layer only where every face vertex has weak camera
  support. The Temple diagnostic removed `12,345` direct candidates and reduced boundary edges from `115,466`
  to `107,948`; coverage changed from `0.8132` to `0.8126`, IoU from `0.8040` to `0.8036`, and SSIM from
  `0.6022` to `0.6018`. High quality keeps this cleanup disabled unless explicitly requested.

## 2026-07-18 Depth-Edge and Cross-View Geometry Refinement

- Coarse-to-fine depth propagation now uses a guide-weighted median depth instead of averaging foreground and
  background samples across a discontinuity. Search-radius uncertainty is still retained, but the prior center no
  longer invents a surface between two real surfaces.
- Cross-view validation searches a 3x3 neighborhood around the subpixel projection and validates candidates with
  a source-to-reference round trip. Its pixel envelope is derived from the configured relative depth tolerance and
  the actual camera baseline, plus a 3-pixel numerical margin; a fixed 1.5-pixel gate was rejected because it
  downgraded every Temple frame. Manifests now record confirmed, occluded, contradicted, unverifiable observations
  and the final rejected-pixel count.
- With regenerated Temple depths, 15 frames remain accepted and one validation-only. At 384 cells, two-camera TSDF
  support plus weak-boundary cleanup records coverage `0.8345`, IoU `0.8230`, edge P90 `79.67 px`, and SSIM
  `0.6031`, compared with the previous ultra baseline `0.8126` / `0.8036` / `80.10 px` / `0.6018`. A one-camera
  experiment reached edge P90 `77.69 px` but raised boundary edges to `169744` and introduced interior sheets, so
  it remains rejected.
- The fresh UAV9 CUDA depth-only regression completed with status `ok`, 9/9 depth frames, eight accepted, one
  validation-only, and zero rejected. The previous baseline had seven accepted and two validation-only. All nine
  version-3 `SFTB` artifacts contain exactly 40000 SIFT keypoints, with no image above the configured cap.
- Final depth artifacts now persist `raw_geometry_support_path`. Each nonzero pixel stores one reference
  observation plus the number of source views that passed depth, neighborhood, and round-trip geometry checks;
  this is separate from the PatchMatch source-count diagnostic. Older workspaces without this artifact retain the
  normal two-camera TSDF rule.
- Ultra TSDF may recover a one-frame voxel only when its originating depth pixel has at least four geometry
  observations (reference plus three independently confirmed sources) and observation weight is at least `0.85`.
  High quality keeps this path disabled. On the regenerated Temple A/B, the conservative four-observation path
  changed coverage `0.84040` to `0.84286`, IoU `0.82478` to `0.82612`, edge P90 `80.41` to `80.34 px`, and SSIM
  `0.60371` to `0.60582`; it recovered `239167` verified single-view TSDF samples while retaining one component.
  A three-observation variant recovered more samples but increased open-boundary edges further, so it was not
  selected as the default.
- Two additional completeness experiments were rejected: geometry-filtered `validation_only` frames reduced
  coverage/SSIM to `0.84053`/`0.60289`, and restoring geometry-confirmed pixels across the two-pixel TSDF mask
  erosion increased open boundaries without improving edge P90. The remaining high P90 is therefore dominated
  by view-dependent missing surface regions rather than a uniform one-pixel edge offset.

## 2026-07-19 Post-Consistency Cross-View Repair and Quality Attribution

- Orbital-object consistency filtering now repairs a rejected reference pixel only when three distinct source
  views reproject into one depth cluster with at most 1.5% relative spread. Projection splats are limited to
  0.8 pixels and recovered depths must agree with a nearby reference surface when local evidence exists. Real
  doorway and window openings remain empty because no three-source depth cluster exists there.
- Recovered pixels are marked separately and excluded from frame admission scoring. This prevents recovery from
  promoting a baseline `validation_only` frame into fusion. Temple kept 13 accepted and three validation-only
  frames while recovering 4,193 post-consistency pixels.
- On the Temple highest-quality workflow, conservative post-filter recovery changed coverage from `0.95077` to
  `0.95115`, IoU from `0.87530` to `0.88454`, edge P90 from `12.83` to `12.23 px`, and SSIM from `0.60935` to
  `0.61084`. A two-source variant recovered 25,160 pixels but reduced coverage/SSIM and increased open
  boundaries, so it is not used.
- Ultra TSDF samples projected depth at subpixel locations with a discontinuity-aware 2x2 cluster. Neighbor
  samples farther than 2% relative depth from the nearest surface mode are not averaged. This recovered invalid
  nearest-neighbor samples while reducing the same-depth Temple boundary count by about 2.7%.
- Dino quality masks preserve large dark architectural openings and only fill small internal noise holes. The old
  mask incorrectly filled the Temple doorways, producing an artificial edge P90 near 80 px. With the corrected
  mask, the selected Ultra Temple model records coverage `0.94975`, IoU `0.88763`, edge P90 `12.48 px`, and
  SSIM `0.61545`. The strict IoU, edge, and SSIM gates remain open.
- `model_quality_report.json` includes per-view depth coverage attribution and writes
  `comparisons/<view>/missing_stage.png`. Missing foreground is separated into support-mask exclusion, invalid
  depth, insufficient geometry support, and verified depth that did not become a rendered mesh surface.

## 2026-07-20 Geometry Evidence and Orbital Free-Space Consensus

- Final depth artifacts persist the contributing source-view bit mask, inverse-depth mean/spread, and the
  cross-view repaired-pixel mask alongside geometry support. Legacy workspaces without these files still load
  with empty evidence maps. Quality diagnostics now write bidirectional edge distance, P90-tail, source-count,
  inverse-depth-spread, repair, and missing-stage images per validation view.
- Two-source depth growth and local TSDF surface-patch recovery remain opt-in. On Temple, conservative two-source
  growth found no eligible pixels, while tested TSDF patch variants increased boundary edges without materially
  lowering edge P90. Forcing a collapsed Level 1 depth result to replace its stable Level 2 parent was also
  rejected because it raised the highest-quality edge P90 to `25 px`.
- Orbital-object TSDF now treats support-mask exterior as free space only after at least five reference cameras
  agree. A single mask cannot carve geometry, and aerial or legacy workspaces keep carving disabled unless the
  setting is explicitly enabled. The threshold is configurable with
  `tsdfMinimumSupportMaskFreeSpaceViews`; `tsdfSupportMaskFreeSpaceCarving=false` remains an explicit override.
- The final Temple default regression at resolution 320 records coverage `0.95419`, IoU `0.89445`, edge P90
  `12.65 px`, and SSIM `0.61454`. The Ultra 384 all-view result records `0.95235` / `0.90611` / `12.50 px` /
  `0.61587`, keeps one main component, and passes the IoU gate. Edge P90 and SSIM remain below the strict targets.

## 2026-07-25 Project-Mask and Source-Pool Validation

- Temple regressions must forward the project exclusion masks into MVS. The content-mask fallback retained about
  43% of each image and included the dark curtain behind the object, which fused into an artificial wall. Passing
  the same project masks used by the GUI (or `--mvs-mask-dir` in the reconstruction CLI) reduced the retained
  region to about 27% and removed that wall. This artifact is a mask-input problem, not a TSDF hole-fill failure.
- With identical project masks, four and six source views produced mixed Temple results: four views slightly
  improved SSIM and one edge metric, while six improved IoU and reference-edge agreement. High-quality orbital
  reconstruction therefore uses six sources only for dense rings with at least 16 cameras. Sparser rings keep
  four nearby sources and adapt the angle gate to their measured camera spacing.

## 2026-07-27 Orbital Fusion Admission and Completeness Gate

- Direct-depth orbital fusion no longer treats a slightly low mean confidence as proof that an entire view has no
  geometric value. `validation_only` artifacts are admitted as low-weight, surface-only auxiliary observations;
  they never estimate the TSDF bounds or cast free-space votes.
- Robust frame confidence remains a continuous weight by default. Explicit hard rejection is guarded by camera
  ring coverage, so adjacent removals cannot create an angular gap larger than twice the median spacing.
- Support-mask free-space carving is off by default for orbital data. If explicitly enabled, it still requires
  multi-camera consensus and cannot erase a voxel that already carries surface evidence.
- The final mesh is checked against valid supported depth samples from every view. Per-frame, P10, median, and
  aggregate recalls are written to the model report; orbital jobs reject a result that fails the completeness
  gate instead of registering a half-model as successful.
- Frozen Dino validation fused all 16 frames (two auxiliary), with P10/median recall `0.7238`/`0.8693`.
  Temple fused all 16 frames with `0.7604`/`0.8439`. Both enforced gates and the existing geometry baselines pass.

## 2026-07-28 Hyb2 Pair Audit and Sparse-Ring Regression

- The 12-view Hyb2 audit found 13 verified pairs, no proven failed pairs, and 53 pairs with absent or insufficient
  stored match evidence. Verified-first planning fills the remaining slots from shared-track geometry instead of
  misclassifying those 53 pairs as failures.
- The high-quality replay keeps four nearby sources and a native-resolution final level for the 1024×1024 inputs.
  The four-source sparse-ring value is a scene-derived cap, not a lower-bound recommendation: generic GUI quality
  presets requesting six or more candidates cannot reintroduce the roughly 84-degree third-neighbor pair. This
  GUI/CLI convergence is recorded by depth algorithm revision 10 so revision-9 artifacts are regenerated.
  All 12 frames entered TSDF fusion. The final mesh recorded P10/median/minimum depth recall
  `0.6289`/`0.8625`/`0.5442`, compared with `0.5599`/`0.7935`/`0.4271` for the frozen old-depth mesh.
- Photo-projection validation produced median IoU `0.9246`, edge P90 `15.50 px`, and SSIM `0.4413`. The old-depth
  mesh recorded `0.9306`, `15.13 px`, and `0.3986`: silhouette agreement is close to the old baseline while
  appearance and sector completeness improve. The strict 3-pixel edge and 0.75 SSIM gates remain unmet.
- Current-depth Temple and Dino checks preserve their real openings and remain single-component. Temple recorded
  864 boundary edges and IoU `0.9077`; Dino recorded 436 boundary edges, IoU `0.9366`, and SSIM `0.8709`.
- A GUI-equivalent Hyb2 replay explicitly requested seven sources after the preset layer. Revision 10 resolved it
  to `source_pool=4 (configured=7)`. Frame 9/10 depth consistency retention reached `90.8%`/`83.0%`; their raw
  depth files and the final mesh are bit-for-bit identical to the approved four-source replay above. Consequently,
  the screenshot's six-source mesh recall `33.3%`/`35.9%` returns to `54.4%`/`61.2%`; all 12 frames enter TSDF
  and produce one component instead of failing the completeness precondition.

## 2026-07-28 Supported MC33 Surface and Sampling-Aware Orbital Defaults

- The signed-distance visibility histogram now has nine bins with an exact zero centre. Zero evidence is excluded
  from positive/negative conflict counts, and symmetric even-weight medians no longer inherit a negative tie bias.
- MC33 output keeps a face only when its source cell contains an observed non-positive corner and an observed
  positive corner. This removes the second, unsupported zero crossing formerly created at the back of the observed
  TSDF band.
- Orbital high-detail defaults cap the TSDF resolution from the median input depth-map sampling and scale the face
  budget quadratically. A 640x480 Dino/Temple workspace therefore maps a GUI request of 384 cells and 240,000 faces
  to 192 cells and 60,000 faces. `tsdfOrbitalAdaptiveResolution=false` preserves an explicit legacy request.
- Final hole filling remains visibility- and silhouette-constrained. It can close supported internal defects up to
  384 boundary edges, while large Temple column openings remain protected by size, diameter, and view evidence.
- Frozen GUI-equivalent Dino validation produced 60,826 faces, reduced boundary edges from 12,459 to 374, and
  improved minimum/P10/median depth recall to `0.7770`/`0.8226`/`0.9270`; the strict quality gate passes.
  Temple produced 61,402 faces from all 16 frames and reduced boundary edges from 6,639 to 1,032 while preserving
  the architectural openings. Its boundary ratio is `0.01114`, slightly above the generic `0.01` strict threshold,
  so it remains reported rather than being hidden by an unsafe threshold change.

## 2026-07-29 Protected Orbital Depth-Hole Recovery

- Orbital depth repair no longer rejects an entire missing-depth component only because a narrow invalid corridor
  connects it to the foreground silhouette. It preserves a four-pixel silhouette band, then considers only the
  remaining interior. Project-mask exclusions and real openings remain outside the interpolation domain.
- The interior still requires cross-view anchors, enough boundary samples, a bounded robust inverse-depth spread,
  and a component area no larger than 25% of the authoritative support region. Recovered pixels keep low confidence
  and the `crossViewRepairedMask` provenance, so they cannot silently become native observations.
- Revision 11 manifests persist source-count, depth-spread, local-depth, component-area, silhouette-protection, and
  postprocess interpolation diagnostics. A Hyb2 replay raised frame 9/10 valid-within-mask ratios from
  `85.4%/80.0%` to `98.9%/99.2%`. The resulting mesh remained one component; photo projection improved median
  coverage from `0.9633` to `0.9745` and IoU from `0.9257` to `0.9348`. The strict edge/SSIM targets remain unmet.
