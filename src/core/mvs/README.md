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
- Orbital depth artifacts at algorithm revision 15 persist adaptive support weight, effective view count,
  conflict ratio, geometry source masks, and inverse-depth moments. Before final hole handling, projected source
  depths are clustered into a dominant occlusion layer. A stable layer may refine a matching native depth,
  replace a contradicted native hypothesis when at least three sources agree, or transfer a measured layer into
  a missing supported pixel. Reuse rejects an incomplete revision-15 evidence set instead of silently
  interpreting missing evidence as valid geometry.
- Revision 24 replaces the independent OpenCL depth sweep with persistent depth/normal/score state. It initializes
  inverse-depth hypotheses, runs race-free checkerboard plane propagation and deterministic random depth/normal
  refinement, and applies uniqueness confidence only after convergence. Plane patches intersect every reference
  ray before source projection, matching CUDA plane-homography geometry. Fusion-postprocess retention remains
  distinct from cross-view consistency in quality reports. Revision-23 artifacts are not reused.
- Revision 26 makes `MvsSceneClassifier` the single source of truth for automatic aerial/orbital selection.
  Long aerial strips are recognized from camera-to-terrain plane intersection, same-side camera consistency,
  and sparse-cloud planarity rather than convergence on the global cloud centre. Camera-layout diagnostics from
  aerial triangulation remain a prior only. Revision-25 depth batches are regenerated so an earlier scene
  misclassification cannot be reused by fusion. Revision-26 GUI reuse also requires a consistent classified
  scene profile plus complete `acceptance` and `fusion_eligible` metadata for every frame.
- Revision 27 reduces depth-estimation work without enabling interpolation or weakening quality gates. Coarse and
  middle pyramid levels remain at their native working resolution, propagated priors keep the original global
  inverse-depth hypothesis count, and CPU/CUDA/OpenCL propagation reuses already-scored
  hypotheses. OpenCL additionally composes one plane homography per source and hypothesis, performs exact streaming
  top-k aggregation, dispatches only active checkerboard pixels, and applies the same globally bounded distinct-depth
  uniqueness probes as CPU/CUDA. Multi-view consistency reuses each reference
  unprojection across its ordered source batch, while residual recovery skips source reprojection when the measured
  missing-support region is already below its configured threshold. Revision-26 artifacts are regenerated.
- Revision 38 uses an exposure-robust intensity/gradient/Census photometric cost on CPU, CUDA, and OpenCL.
  It adds a stricter statistical audit tier for fifth/sixth orbital source views, geometry-calibrated confidence,
  boundary-aware postprocessing, and an optional learned-depth candidate gate. Revision-37 depth is regenerated.
- Revision 39 persists a lossless full-resolution `prepared_image`, its prepared valid mask, and the matching
  zero-distortion `prepared_camera_model` for every frame. Replay reads that raster while retaining `ref_image`
  as the original project/source-plan identity; mesh, visual-hull, color, texture, and QC consumers prefer the
  prepared raster so a distorted source image is never sampled with a zero-distortion working camera.
- Revision 40 makes Auto classification fail closed to a general/custom profile. Orbital-only recovery and
  adaptive evidence require a distinct, planar, centred, sufficiently complete camera ring with convergent
  optical axes; generic captures use the moderate filter. Cross-view source plans are frozen before any frame is
  reclassified, so processing order cannot remove sources from later frames. A frame with no measured source
  evidence omits both the all-zero source mask and its empty ordinal table, while nonzero masks without an exact
  table remain a hard error.
- Revision 41 makes general/custom frame acceptance fail closed. A low-retention Custom frame can remain only a
  validation candidate when an accurate sparse absolute-depth residual and a strong retained discrete multi-view
  core agree; it never becomes a primary fusion seed through that exception. A high-retention Custom frame is not a
  primary seed unless both geometry signals are present, and is rejected when both are absent and its post-geometry
  mean confidence is also low. Semantic project support masks remain subject to normalized coverage checks, while
  content and prepared-raster validity masks only define the technical measurement domain.
- Revision 44 separates original photometric confidence from independent cross-view geometry confidence for the
  default-off depth-layer correction experiment. Relative consistency or confidence-retention loss is explained
  only when this run actually corrected pixels and those pixels retain strong multi-source geometry, while absolute
  coverage, sparse residual, connected-component, and search-boundary gates remain mandatory. Stage snapshots save
  both confidence channels beside the combined confidence map. Residual local PatchMatch uses two overlapping
  three-view baseline groups, and reduced native grids keep the measured contour plus one inward grid shell instead
  of quantizing all boundary protection to zero. Revision-43 depth batches are regenerated.
- Downstream geometry consumers use one fail-closed frame-role contract. Only a `completed` frame with explicit
  `acceptance=accepted` and `fusion_eligible=true` is `Primary`; a completed `validation_only` frame with explicit
  eligibility metadata is `CoverageAuxiliary`; rejected, failed, incomplete, and manifestless frames are `Excluded`.
  Point-cloud fusion and visual hull use only Primary frames; the visual-hull model gate requires at least six
  Primary silhouette candidates. TSDF may use CoverageAuxiliary frames at reduced weight for local surface support,
  but auxiliaries cannot expand bounds or vote in visibility-occupancy and hole-fill topology decisions. Depth-map
  model and texture publication still require at least one Primary frame. Point-cloud reuse keeps its two-Primary
  minimum, while a depth TSDF model requires at least three usable Primary-or-Auxiliary frames in total.
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
- `mvs_depth_reprocess_cli --source-max-angle-deg N` is a default-off experiment control (`0` disables it) scoped
  to the PatchMatch source plan; it does not change the separate orbital geometry-repair source expansion.
  The measured value is the median triangulation angle of at most 2048 shared sparse points that project in front
  of both cameras, not the angle between camera optical axes. A positive value is combined as
  `min(scene-derived maximum, N)`, so it can only remove wider-baseline
  candidates and can never relax the scene policy. While the cap is enabled, unmeasured sequence fallback is
  disabled rather than silently bypassing the angle boundary; any resulting source shortfall remains explicit.
  The config hash, each fresh depth artifact's
  `source_angle_diagnostics`, and `mvs_replay_report.json` retain the configured/effective limits, selected maximum,
  angle-rejection count, and the explicit `patchmatch_source_plan` scope for A/B audits. A selected maximum of zero
  means that the plan contains no selected source with a positive measured triangulation angle.
  A cap-enabled frame with any source shortfall cannot become a primary fusion seed or be counted as a cap-induced
  quality gain. A non-rejected decision is capped at validation-only and records
  `source_angle_cap_source_shortfall`; an already rejected decision remains unchanged, while the source-count and
  angle-rejection diagnostics still expose the shortfall boundary.
- Source-pool/ranking replay is a separate default-off expert experiment. Complete-pool mode improved independent
  depth metrics on both Office and Courtyard, but Courtyard still exposed an admission-role regression, so it remains
  opt-in and is not guarded by `source_quality_score` (the regressed frames did not have lower source quality). A uses the legacy early-stop pool
  (`--source-complete-visibility-pool` absent) and strength zero. B enables
  `--source-complete-visibility-pool` with strength zero, so the planner evaluates the complete candidate set
  represented by the existing visibility graph and still ranks with the unmodified legacy score. C uses the same
  pool with the internal-diagnostic-only `--source-angle-soft-ranking-strength 1`; within each authoritative verified, ordinary, bounded-failed,
  or strict-failed selection stage it ranks by
  `adjusted = legacy * exp(-strength * t)`, where the penalty is zero through the soft maximum and
  `t = (angle - softMax) / (max - softMax)` above it. Strength is limited to `[0,4]`. A positive strength requires
  the complete-pool switch and an exactly zero hard angle cap; invalid or non-finite values fail closed.
  Soft ranking is intentionally absent from ordinary reconstruction and GUI controls: its Office result was mixed and
  its Courtyard treatment was byte-identical to the strength-zero complete-pool arm for all 38 frames.
- Complete-pool mode does not add camera pairs to the visibility graph. With at most 32 views the graph contains
  every co-visible pair plus explicitly required verified pairs (`all_co_visible_or_required_pairs`); larger jobs
  retain the deterministic bounded graph. Frames outside that graph are not counted as visibility candidates.
  Orbital adaptive maximum-angle inference always uses only candidates evaluated by the legacy early-stop pass,
  so enabling B/C cannot change angle eligibility through feedback from newly evaluated candidates.
- The adjusted score is ranking-only: eligibility, hard angle rejection, quality thresholds,
  `source_quality_score`, same-view duplicate resolution, tier authority, fallback policy, and requested source
  count all remain legacy decisions. Verified-first C constructs independent complete control and treatment plans,
  each with its own tier backfill state. If their counts differ or the control has a shortfall, the control plan is
  retained and diagnostics report `applied=false`; treatment never creates a silent source shortfall.
- B/C depth artifacts record the machine-readable comparison at
  `source_angle_diagnostics.soft_ranking` (also mirrored inside `depth_quality`). It includes visibility-graph scope,
  legacy/complete evaluated counts, separate control/treatment candidate universes and qualified counts, complete
  selections, and count/set/order invariants. Each candidate record carries its actual selection-stage soft/effective
  maximum, legacy/adjusted score and rank, plus an identity-level `selected_by_plan` flag; bounded and strict records
  for the same view therefore never claim selection on behalf of the other tier. A adds no source-ranking config,
  plan, replay-report, or diagnostic keys, preserving the default canonical config/hash and source-plan schema.
- Selected-frame stage snapshots are a separate, default-off diagnostic. Replay can atomically persist depth,
  confidence, and valid-mask triplets after PatchMatch, after cross-view consistency, after confidence
  postprocessing, and at final admission. Cross-view and later snapshots may also carry the observe-only
  `depth_layer_reliability` class map (`reliable`, `ambiguous_low_texture`, or `rejected_layer`). The bounded
  snapshot manifest is explicitly non-authoritative and is
  excluded from the algorithm config hash; snapshot failure never changes the production depth result. The strict
  ETH3D evaluator binds every stage camera and pose back to the authoritative workspace frame before scoring it.
- `enableDepthLayerReliabilityAnchorGate` is a default-off experiment. It does not delete or invalidate depth;
  it only prevents low-texture ambiguous or rejected native depth from becoming an anchored-hole interpolation
  boundary. A missing or malformed reliability map fails closed for native anchors, while projected strong anchors
  remain available. Enabling it changes the depth config hash and reports admitted/rejected anchor counts.
- `enableDepthLayerReliabilityGuidedCorrection` is a separate default-off experiment. The reliability classifier
  uses a robust quadratic inverse-depth surface, but the fitted surface is diagnostic only and is never copied into
  the product. Candidate depths are restricted to the native estimate and directly projected measured source depths.
  Each candidate is scored with confidence-weighted robust relative-depth residuals, reprojection-footprint error,
  asymmetric occlusion handling, at least three distinct sources, and at least two baseline sectors.
  `RejectedLayer` may switch only when the measured candidate has a strict cost advantage;
  `AmbiguousLowTexture` may only make a continuous correction of at most one percent and never jump directly to a
  different layer. `sourceQualityScore` remains diagnostic and is not a new admission threshold. Missing or malformed
  reliability evidence disables the extra path, and enabling it changes the depth config hash. For Custom/general scenes the
  treatment now builds the same measured source-depth layers before confidence postprocessing, but it is restricted
  to weak native reliability classes: reliable pixels and missing pixels remain byte-exact and the Orbital-only
  depth-transfer behavior is not enabled. The Custom path first preserves the legacy consistency mask, then overlays
  only pixels whose depth was actually refined or switched by a stable independent-source cluster; merely evaluating
  an ambiguous pixel cannot retain it or reduce its confidence.
- Guided Custom processing also performs a bounded second hypothesis search only inside connected ambiguous/rejected
  regions that pass the minimum component size. It reuses the frame's recorded CPU/CUDA/OpenCL PatchMatch backend and
  the frozen source plan, then applies the same backend-independent measured-geometry scorer before accepting a
  replacement. Accepted pixels inherit confidence from their actual candidate/source evidence and persist the exact
  supporting-source mask and inverse-depth moments; opening the experiment never grants a confidence or admission
  bonus. If either local hypothesis, three-source/two-sector support, or the robust cost advantage is missing, the
  original consistency result is retained unchanged.
- The cross-view snapshot may additionally contain a nine-channel `geometry_rerank` matrix ordered as
  native cost, candidate cost, cost advantage, effective source weight, relative correction, weakest source
  confidence, source count, baseline-sector count, and decision action. It is budgeted and non-authoritative like the
  other snapshot payloads and cannot affect cache reuse or publication. Later stages keep their independent
  depth/confidence/mask triplets without duplicating the unchanged nine-channel evidence payload.
- The selected source plan is saved with the depth frame record so depth generation and fusion use the same
  overlap assumptions.
- When verified pair geometry is available, verified pairs are selected first and shared-track geometry may
  backfill only the remaining slots. The manifest records the requested count, shortfall, and verified,
  backfill, or sequence tier for every selected source.
- A failed production pair may fill source slots five and six only through `strict_pair_audit_backfill`: at least
  24 inliers, 32 matches, 40 shared tracks, 0.30 coverage, a 90% Wilson lower bound of 0.65, and at most 55 degrees.
  This tier is separately recorded and never replaces the verified-pair majority.
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
- `valid_coverage` is the canonical per-frame coverage field. Quality reporting can derive it from
  valid-pixel count and grid size when those canonical measurements are present. Missing coverage remains
  unavailable and is displayed as `—`, never as a measured zero.
- Depth artifacts are not standalone workspace resources. The photo toolbar resolves the matching frame by
  `ref_image` and overlays depth on that photo. Feature points and residual diagnostics are temporarily hidden
  while depth inspection is active, then restored from the user's existing preferences.
- Local outlier filtering and connected-component speckle filtering remove isolated red-depth spikes while
  preserving large smooth regions. A depth discontinuity with neighbours on its own layer is retained, and
  low-confidence silhouette pixels receive a relaxed threshold only when independent cross-view geometry agrees.
- `scripts/validation/compare_depth_to_reference_mesh.py` projects a registered high-precision mesh into every
  camera, reports per-pixel relative depth error, confidence reliability bins, ECE, Pearson/Spearman correlation,
  and can save compressed pixel arrays for offline threshold calibration.
- Frame admission uses conservative internal confidence. After geometry filtering, PatchMatch-derived confidence
  is monotonically calibrated for persisted artifacts and fusion weights; already probabilistic learned candidates
  keep their model confidence unchanged.
- Optional learned MVS candidates use `learned_depth_<frame>.bin` and `_conf.bin` artifacts. They are default-off
  and are merged only after independent camera-geometry support, inverse-depth spread, and depth-difference gates;
  learned values never contribute to the evidence used to accept themselves.
- Dominant-layer selection records refined, switched, transferred, ambiguous, and unresolved pixel counts in
  `dominant_depth_layer_selection`. Later hole repair preserves this selected-layer mask, so a measured
  cross-view transfer is not accidentally reclassified as unconstrained spatial interpolation.
- Speckle removal builds a component-removal lookup table and scans the depth image once. It therefore scales
  as `O(component count + pixel count)` instead of rescanning the image for every small component.
- Advanced GUI controls map to `DepthGenConfig`: minimum consistent views, geometry consistency, maximum
  reprojection error, speckle area threshold, and fusion maximum image size.

## Compute Backends And Scheduling

- The public estimator, CPU implementation, CUDA implementation, and no-CUDA stubs are separate translation
  units. A build configured with `PLASCAN_ENABLE_CUDA=OFF` compiles and runs the real CPU estimator even on a
  workstation where the CUDA toolkit is installed.
- CUDA workspaces, execution locks, upload streams, and gray-image cache keys are isolated by device index.
  One device still serializes its constant-memory camera updates, while separate devices may execute frames
  concurrently.
- `DepthComputeScheduler` owns one priority queue and one worker model shared by CPU, CUDA, and OpenCL. Auto probes
  both CUDA and OpenCL, then uses every successfully leased physical accelerator across the two backend families.
  A physical PCI identity is admitted only once. When an OpenCL driver cannot expose the standard PCI identity,
  the device name is matched to the CUDA inventory; an unresolved NVIDIA OpenCL interface is conservatively skipped
  while CUDA is active. The same NVIDIA GPU therefore cannot acquire two leases or two execution lanes through
  different APIs. Explicit CUDA or OpenCL requests remain strict and never silently substitute another backend.
  Explicit OpenCL mode may use NVIDIA OpenCL when requested; because CUDA is not admitted into an explicit OpenCL
  worker pool, this cannot create duplicate execution lanes for the same physical GPU.
- Heterogeneous scheduling is frame-level: one depth map remains on one device, while different reference frames
  may run concurrently on CUDA discrete GPUs and OpenCL integrated GPUs. Every selected physical accelerator is
  represented before host preparation lanes are duplicated. CUDA and discrete OpenCL devices retain one kernel
  execution lane; a unified-memory OpenCL GPU uses at most two persistent lanes so an independent frame can cover
  long Windows driver submission gaps. Progress reports physical GPU count separately from active host slots.
- Auto reserves an initial calibration frame for each participating accelerator and tracks an exponential moving
  average of its frame time. Faster devices naturally claim more work. Near the queue tail, a slower device stops
  claiming frames when its projected completion time, including in-flight work, is no better than the fastest
  calibrated device clearing its in-flight work and every remaining frame that backend can actually claim. A
  cross-backend retry excluded from the fastest backend therefore keeps an eligible slower backend awake. This
  prevents both integrated-GPU stragglers and an ineligible-fastest-worker retry deadlock. If no accelerator can be
  prepared and leased, Auto uses native CPU; CPU is not mixed into an active CUDA/OpenCL batch.
- A first frame failure in heterogeneous Auto is returned once to a different backend. The failing physical worker
  stops taking ordinary frames while a healthy alternative exists, and a device paused by the tail-profitability
  gate can be reactivated for the retry. A second failure is final; single-backend and explicit jobs keep their
  strict failure behavior, so retry cannot become an unbounded or silent fallback loop.
- Streaming hybrid jobs use two artifact-saving workers behind task-count and actual resident-byte limits. The
  queue accounts for producers waiting to enter plus queued and active `cv::Mat` allocations, keeps the existing
  stage barriers and manifest lock, and falls back to one saver for cached or single-GPU jobs. Aggregate wall time,
  worker busy time, producer wait, peak resident tasks/bytes, and failure counts expose whether storage still
  starves GPU submission. Producer accounting transfers only after queue insertion commits; compute and saver
  thread exceptions are contained and release in-flight state, and the final filtered-save barrier observes
  cancellation instead of continuing into fusion. The same cancellation token is wired into the BFS fusion loops.
- An absent CPU-thread setting resolves to the machine's logical thread count minus two. Explicit CLI budgets are
  honored up to the hardware limit instead of being capped at seven. Frame compute divides that total by the actual
  CUDA/OpenCL host-slot count and distributes the integer remainder; each slot's OpenMP post-processing observes
  that share, while preload, visibility, standalone consistency, and fusion stages retain the full budget.
- A heterogeneous batch keeps the stable `auto` token in its workspace hash and records `CUDA:N` or `OpenCL:N` for
  each frame. GUI project metadata reports the combined batch as `hybrid`; Auto may reuse either a compatible
  uniform batch or a compatible hybrid batch, while an explicit backend may reuse only the same uniform family.
- The OpenCL C 1.2 backend runs inverse-depth initialization, stateful plane PatchMatch, multi-source robust
  intensity/gradient/Census cost,
  mask-aware sampling, depth hints, coarse-to-fine refinement, and confidence filtering. Discrete OpenCL GPUs have
  one command-queue/kernel lane; unified-memory GPUs have two bounded lanes. With the default two-stage host
  pipeline, a second worker prepares the next frame and can submit it to the second integrated-GPU lane while the
  first queue is stalled or busy. Scaled float images and all OpenCL buffers are bounded/reused, and reference
  patches are tiled in work-group local memory. Every quality
  profile retains its configured full inverse-depth hypothesis count, including inside propagated priors, followed
  by 13 refinement samples; the program does not use relaxed-math compilation. CPU
  packing and post-processing remain outside lane ownership. CPU execution
  remains the native C++/OpenMP implementation rather than using a CPU OpenCL device.
- Initialization, checkerboard propagation/refinement, final uniqueness scoring, and asynchronous readback are
  submitted as one in-order event chain. Per-device logs report active kernel time, total chain time, command
  count, and kernel duty ratio. Batch logs additionally report the queue occupancy, inter-call idle time,
  in-queue non-kernel time, and end-to-end kernel duty across all calls, which is the relevant measurement when an
  operating-system GPU graph appears discontinuous. Integrated GPUs use persistent host-allocatable input/output buffers; intermediate
  depth, normal, and score state is never read back between passes.
- The Hyb2 14-frame orbital replay on AMD `gfx1036` accepted 13/14 fusion frames without relaxing the 90% fusion
  retention gate. Mean confidence was 0.8503, mean valid-within-mask ratio was 0.9615, and mean fusion retention
  was 0.9216. Profiled kernel duty stayed near 100%; the summed per-frame depth time was 212.9 seconds. The one
  validation-only frame measured 89.90% retention, so the gate was kept unchanged and no extra interpolation was
  enabled to hide the shortfall.
- `mvs_depth_reprocess_cli --opencl-device-index N` pins replay to one enumerated OpenCL GPU. This is intended for
  repeatable vendor/device comparisons and prevents an Auto heterogeneous job from hiding per-device quality or
  timing differences. The task-lifetime GPU lease still rejects a second process targeting the same device.
- A task-lifetime lease keyed by physical PCI identity prevents a second PlaScan GUI/CLI process from using the
  same GPU during depth estimation. Standard OpenCL PCI identity is preferred; CUDA-name matching closes the common
  cross-API fallback gap. OpenCL failures are reported directly instead of silently running a GPU-tagged frame on
  the CPU.
- Image preload also caches the normalized/undistorted MVS image and camera. Frames reuse this prepared input
  instead of repeating camera normalization and distortion remapping every time the same image is a reference
  or source view; undistorted storage is shared with the gray cache when no distortion is present.
- Per-frame CUDA and OpenCL logs report preparation, device-slot wait, device execution/readback, post-processing,
  and total time. OpenCL additionally separates command queue turnaround from profiled kernel execution. These
  fields distinguish expected frame-pipeline overlap from a driver/queue or device-side utilization gap.
- MVS has no Vulkan Compute backend or placeholder interface. Vulkan remains a GUI rendering dependency only.

The implementation plan and current boundary are documented in
`docs/plans/MVS_HETEROGENEOUS_COMPUTE_PLAN.md`.

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
  estimator normally keeps the parent depth resized to the final raster size and records the selected level
  and coverage-regression reason instead of publishing a severely incomplete fine result.
- `--mvs-native-depth-grid` (or replay `--native-depth-grid`) is an experimental, default-off path for
  large general/custom scenes. For a non-rectified frame it keeps the final PatchMatch grid, including a
  coarse parent selected after fine-level collapse, instead of nearest-neighbor upscaling it to the prepared
  raster before consistency. Orbital, aerial, rectified, and unresolved-profile paths fail closed to the
  legacy full-size contract. The stored depth camera is scaled to the exact grid dimensions with the
  half-pixel convention, while the full-resolution prepared raster/camera triplet remains available for
  replay, fusion color sampling, meshing, and texturing. Pixel-domain settings continue to mean prepared
  full-raster pixels: boundary morphology, local-outlier kernels and same-layer radii, speckle/small-hole
  areas, sparse-residual neighborhoods, consistency search/round-trip limits, and fusion reprojection
  limits are quantized onto the actual depth grid. Fusion applies view-count/streaming overrides in the
  prepared-raster domain first, then independently scales the resulting threshold and local-gradient radius
  for each target frame; the manifest labels the recorded fusion values as pre-override bases. Thus a 3x3
  filter becomes identity when its one-pixel radius is subpixel on a ds4 grid instead of silently becoming
  a 12-full-pixel footprint. Every frame
  records `effective_native_final_depth_grid`, raster/grid dimensions, exact x/y, linear and area scales,
  plus configured/effective parameter values in `pixel_domain_diagnostics`. The request participates in
  the workspace hash. Revision 44 evaluates a zero-radius reduced-grid consistency lookup over the bounded
  nearest subpixel footprint and preserves the measured contour plus one inward protection shell; other
  configured subpixel thresholds remain inactive rather than being silently widened.
- CPU, CUDA, and OpenCL PatchMatch consume the same per-pixel search radius. The final frame is accepted,
  validation-only, or rejected by `DepthFrameQualityGate` before fusion.
- Revision 45 keeps a compact per-pixel photometric source bitset and uses it as a bounded spatial prior;
  the prior may change which valid sources are averaged but cannot increase their measured NCC. CPU/CUDA/OpenCL
  propagation tests near and odd-distance far neighbours plus a local depth-gradient normal. When all first-pass
  frames are resident, a second narrow PatchMatch pass reads frozen source depth maps and adds round-trip
  reprojection error only to the hypothesis objective. Its output confidence remains photometric, while the later
  cross-view stage continues to produce independent geometric confidence and geometry-source masks. The bitset is
  persisted separately as `raw_photometric_source_mask_path`; revision-45 cache reuse requires that artifact.
  OpenCL source-depth guidance fails explicitly at the estimator boundary; resident batch processing records a CPU
  reference second pass, while streaming or memory-pressure runs log a controlled skip.
- CPU, CUDA, and OpenCL PatchMatch also consume the same reference/source valid masks. Plane-homography NCC uses only
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
- Level 1 is always persisted with its summary. Level 2/3 raw depth, confidence, support count, uncertainty,
  valid mask, depth preview, and confidence preview are retained only when
  `saveIntermediatePyramidLevels` is explicitly enabled for diagnostics. Production GUI runs leave it off to
  avoid retaining debug rasters for every frame. The workspace tree exposes only one non-image aggregate
  depth-map node; it never materializes individual depth frames or previews as workspace resources.

The corresponding CLI options are:

```powershell
reconstruct_pipeline_cli.exe image_camera.lis `
  --mvs-quality high `
  --mvs-scene-profile auto `
  --mvs-depth-filter auto `
  --mvs-save-levels
```

Add `--mvs-native-depth-grid` only for an explicit general-scene A/B run; omitting it preserves the
full-size final-depth contract.

`scripts/validation/run_depth_pyramid_regression.ps1` runs the same reconstruction CLI for the Dino and UAV9
fixtures and then invokes `model_quality_cli` when quality validation is enabled.

## Memory, Cancel, And Fusion

- Long runs should use bounded resident depth frames. When the memory budget is tight, saved artifacts become
  the durable state and pixel storage can be released.
- `DepthMemoryPolicy` estimates the full consistency peak, including resident frame data, immutable snapshots,
  retained geometry evidence, projected source rasters, repair scratch space, and allocator overhead. It keeps
  at least 20% of physical RAM (or two transient-frame working sets) free and rechecks the budget immediately
  before consistency starts, preventing page-file thrashing from appearing as an idle hang.
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
  The input signature hashes stable image identities and camera geometry rather than archive-dependent paths
  or result bookkeeping. A batch with a different signature or algorithm revision is rejected and regenerated.
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
