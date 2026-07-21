# Mask-aware depth overlay and direct depth-surface design

## Objective

Improve PlaScan arbitrary-3D reconstruction and depth inspection through one shared persisted depth dataset:

1. Inspect depth only as an overlay on the corresponding source photo; do not expose depth previews as standalone workspace resources.
2. Make project masks, confidence, pyramid quality, and cross-view consistency authoritative inputs to production geometry.
3. Generate the default arbitrary-3D model directly from depth frames with confidence-weighted TSDF fusion and Marching Cubes, without creating a dense point cloud as an intermediate product.
4. Keep GUI, CLI, resume, quality reporting, and regression tools on the same depth metadata and raw artifacts.

The first acceptance datasets are the 16-image Middlebury temple project and the existing 9-image aerial subset.

## Confirmed diagnosis

The current temple result has several independent problems:

- The model workflow asks for a surface reconstruction but `buildMeshFromDepthMaps()` accepts a visual hull first and returns it, so the selected production mode is bypassed.
- Sending the current fused cloud to Poisson produces fragmented geometry. Poisson amplifies conflicting depth sheets but is not the first cause.
- The configured MVS source pool is six views, while current frames commonly use only two or three verified sources. This weakens occlusion and cross-view checks.
- The depth overlay is already partially implemented, but the feature overlay has Z value 1000 while the depth pixmap has Z value 10. Enabling depth inspection therefore leaves blue feature crosses above the colorized depth.
- The Temple archive contains 16 completed `depth_map_results`, but its reconstruction report predates MVS. The streamed fusion completion path does not refresh that report.
- Depth artifacts store coverage as `depth_quality.valid_coverage`, while the reconstruction report reads a top-level `valid_ratio`. Even a refreshed report can therefore show a false `0.0%`.

These are workflow, rendering-state, and metadata-schema defects. They must be corrected separately from later numerical depth-quality tuning.

## User experience

### Photo depth inspection

The existing checkable **显示深度图** action remains the only interactive depth viewer. It is off by default. When enabled, changing photos keeps the mode enabled and resolves only the depth record whose normalized `ref_image` exactly matches the active photo.

The adjacent menu exposes:

- 最终结果
- 级别 1
- 级别 2
- 级别 3
- 显示强度

The final result means the accepted production depth recorded by `selected_level`; it is not an independently guessed file. Level actions display only their own persisted artifacts. `显示强度` converts the retained source-image region to grayscale before applying depth colors; it does not substitute confidence for depth.

If the active photo has no matching artifact, the action is unavailable for that photo and the original photo remains visible. There is no fallback to another photo or another level.

### Clean overlay state

Depth inspection temporarily suppresses feature points, match residuals, and other automatic feature diagnostics that would obscure depth colors. Their user-selected toggle states are preserved and restored when depth inspection is disabled. An asynchronous feature callback must not re-add those items while depth inspection is active.

The authoritative mask boundary remains visible because it explains depth support and edge clipping. User annotations and active editing handles keep their existing behavior.

Depth is drawn over the current photo in the same pixel coordinate system. Invalid depth and excluded pixels are transparent. Robust P2-to-P98 normalization is computed from finite, positive, valid pixels only. Rotation, zoom, pan, and photo switching must not break alignment.

### Workspace behavior

`DataTreeWidget` no longer creates a depth-map group or individual depth-preview items for either new or existing projects. The dashboard may show depth progress and aggregate quality, but not clickable standalone depth resources.

This is a presentation-only change. `depth_map_results`, raw depth, confidence, masks, pyramid artifacts, previews, invalidation metadata, and cleanup ownership remain persisted. Preview PNG files remain available to logs, tests, and regression tools but are not treated as GUI documents.

## Overlay component responsibilities

### DepthOverlayController

Retain the existing controller and complete its responsibilities:

- Resolve records by normalized exact `ref_image` match.
- Resolve final or Level 1/2/3 artifacts from metadata without directory scanning as primary state.
- Load raw depth and valid mask asynchronously.
- Produce a transparent colorized `QImage` with robust normalization.
- Optionally produce the masked grayscale source image for intensity mode.
- Cache by project, image, level, intensity mode, artifact modification time, and display parameters.
- Reject stale results after project, photo, artifact, or mode changes.
- Emit explicit unavailable and load-error states without replacing the base photo.

### CanvasWidget and LayerRenderer

`CanvasWidget` owns depth-inspection state and the temporary suppression state for conflicting diagnostics. `LayerRenderer` continues to own one dedicated depth pixmap item. Enabling or clearing it must not destroy feature data or mutate persistent feature-visibility preferences.

### MainMenu, MainWindow, and DataTreeWidget

`MainMenu` and `MainWindow` keep the existing action wiring, synchronize availability with the active image, and preserve the enabled mode during photo changes. `DataTreeWidget` ignores `depth_map_results` only when materializing resource nodes; metadata normalization and generated-data cleanup continue to recognize them.

## Mask-aware MVS inputs

Every MVS reference view may carry an optional project photo mask. Project masks use `255 = excluded` and `0 = retained`; MVS valid masks use the opposite convention. Convert explicitly and resize only with nearest-neighbor interpolation.

For every pyramid level:

1. Use the authoritative project mask when available; otherwise record fallback to the automatic content mask.
2. Prevent excluded reference pixels from entering propagation, random search, confidence, support, or uncertainty.
3. Apply the same valid region to depth and confidence artifacts.
4. Re-apply the full-resolution authoritative mask after unrectification and before persistence.
5. Preserve intentional holes. Cleanup may remove isolated retained islands but must not close openings such as the temple doorway.

The source-view planner should attempt the configured source count. Verified geometry remains preferred, but a frame that receives fewer sources must record the shortage and source provenance so quality gates can distinguish a genuinely well-supported frame from a weak one.

## Depth pyramid and frame acceptance

- **Level 3:** establishes global depth range and coarse structure.
- **Level 2:** establishes the main surface with multi-view reprojection checks.
- **Level 1:** refines edges and texture transitions around the inherited prior.
- **Final accepted depth:** comes from the finest successful level; a coarser fallback is allowed only when explicitly recorded.

Before a frame is eligible for production surface fusion, require:

- finite positive depth;
- authoritative valid mask;
- finite confidence;
- sufficient source-view support for the scene profile;
- bounded cross-view depth or reprojection disagreement;
- an accepted or explicitly degraded quality decision.

Frame rejection and degraded acceptance must be visible in metadata and model diagnostics. A rejected frame may still be inspected in the GUI but must not silently contribute to the production model.

## Direct TSDF surface generation

### Inputs and bounds

The default arbitrary-3D model input is a collection of accepted depth frames containing:

- reference camera and image dimensions;
- raw metric depth;
- confidence;
- depth-valid mask, which marks pixels that have a finite accepted depth;
- authoritative support mask, which marks the retained project/content silhouette independently of depth validity;
- selected source views and quality metadata;
- source image for optional vertex color sampling.

Persisted depth values keep the existing `Camera::projectWorldPointWithDepth()` / `Camera::unprojectPixel()` convention, including `depthAxisFlipped()`. They are camera-axis depth, not Euclidean ray length. Bounds estimation, voxel projection, cross-view comparison, and synthetic tests must use the same camera API instead of re-deriving a second convention.

Volume bounds are estimated from robustly sampled, unprojected valid depth observations and expanded by a small truncation margin. A minimum number of accepted frames and spatial samples is required. Sparse tie points may be used only for diagnostic bound validation; they are not surface samples and cannot fill missing geometry.

### Confidence-weighted projective fusion

Use a CPU/OpenMP projective TSDF implementation first. For each voxel and eligible frame:

1. Project the voxel into the reference camera.
2. Reject pixels outside the image, authoritative mask, valid depth, or confidence threshold.
3. Compare voxel camera-space depth with the observed depth and compute a truncated signed distance.
4. Weight the update by depth confidence, source support, cross-view consistency, and viewing angle.
5. Integrate color separately from geometry so missing texture never changes occupancy.

The production default requires at least three accepted input frames. A surface voxel must receive consistent weighted observations from at least two distinct cameras before extraction; these defaults remain configurable by scene profile and are recorded in the model report.

The two masks must never be conflated. A zero in the depth-valid mask can mean missing or low-confidence depth and therefore causes no TSDF update. Only a zero in the separately persisted authoritative support mask is an exterior observation that may provide an empty-space constraint inside the reconstruction bounds. These exterior rays must carve unsupported volume without closing retained holes. Legacy frames without a separate support mask receive no silhouette carving rather than treating all invalid depth as free space. Conflicting front/back observations are resolved through signed-distance agreement and minimum accumulated weight, not by averaging exported points.

### Resolution and memory

`meshResolution = 320` means 320 voxels on the longest volume axis, with the other axes derived from physical aspect ratio. Before allocating the TSDF, weight, and optional color volumes, compute and report the required memory. If memory is insufficient, fail with the requested dimensions and byte estimate; do not silently lower resolution or change reconstruction mode.

The first implementation is deterministic CPU/OpenMP. Its interface must keep volume allocation and integration separable so CUDA acceleration can be added later without changing project metadata or model semantics.

### Surface extraction and cleanup

Extract the zero iso-surface with Marching Cubes only where accumulated weight passes the production threshold. Then:

- remove non-finite and degenerate vertices/faces;
- remove small disconnected components using absolute and largest-component-relative thresholds;
- preserve intentional openings rather than applying blanket hole filling;
- orient and normalize finite vertex normals;
- assign colors from confidence-weighted photo samples when requested.

The output model records `reconstruction_mode = depth_tsdf`, input/accepted frame counts, volume bounds and dimensions, voxel/truncation sizes, integration rejection counts, occupied voxel count, extracted vertices/faces, component statistics, and color coverage.

### Explicit model modes

`depth_tsdf` is the default arbitrary-3D path. It does not emit or consume a dense point cloud. Visual hull and Poisson remain explicit legacy or diagnostic modes only. Failure of TSDF validation, allocation, integration, or extraction returns an actionable error; it must never silently fall back to visual hull, Poisson, sparse points, or a coarser depth level.

## Metadata and reconstruction quality

Depth coverage uses `valid_coverage` as the canonical name. New depth records expose a top-level value and retain the detailed `depth_quality.valid_coverage`. Readers accept, in order:

1. top-level `valid_coverage`;
2. nested `depth_quality.valid_coverage`;
3. legacy top-level `valid_ratio`;
4. a computed ratio from `valid_pixel_count / (grid_width * grid_height)` when dimensions are valid.

No available measurement is represented as unavailable, not numerical zero. The dashboard displays `—` and emits no low-coverage warning until at least one completed frame has a valid measurement.

The reconstruction quality report is refreshed after the final depth artifact of a successful batch is committed, before any optional fusion or model transition. It is refreshed again after model metadata is registered. Streamed fusion and direct-TSDF completion paths use the same refresh contract.

The report records depth frame count, completed/accepted/rejected counts, mean valid coverage, mean confidence, mean source count, accepted fallback levels, and model statistics. Registration counts and unregistered image lists must be derived from one consistent source.

## Concurrency and state changes

Overlay loading uses the existing GUI worker/QFuture pattern with `QPointer` and request-generation guards. It never blocks the GUI thread.

TSDF construction runs outside the GUI thread, reports bounded progress, observes cancellation between integration batches, and publishes model metadata only after the model file is completely written. Cancellation leaves persisted depth artifacts intact and does not register a partial model.

Regenerating or invalidating depth clears overlay cache entries and invalidates dependent TSDF model records through existing project generation identifiers. Hiding depth nodes must not change cleanup or dependency behavior.

## Error handling

- Missing matching depth record: keep the original photo and mark overlay unavailable.
- Missing requested raw level: report that level unavailable; never substitute another level.
- Corrupt depth, confidence, mask, or camera dimensions: include exact paths and expected/actual dimensions.
- Stale asynchronous overlay result: discard without changing the current view.
- Insufficient accepted TSDF frames or samples: report frame-level rejection counts.
- Invalid bounds or non-finite camera transform: identify the frame and stop before allocation.
- Insufficient memory: report volume dimensions, requested resolution, and byte estimate.
- Empty or fragmented extracted surface: report occupied voxel, weight, vertex, face, and component counts; do not invoke another model mode.

## Verification

### Unit and contract tests

- Exact depth-record matching with no cross-photo fallback.
- Final and Level 1/2/3 selection from metadata.
- Stale-result rejection and restoration of temporarily suppressed feature layers.
- No standalone depth nodes while metadata remains available to overlay and cleanup.
- Correct project-mask conversion and hole preservation at every level.
- Canonical and legacy coverage schemas produce the same report value; missing coverage remains unavailable.
- Successful depth completion refreshes quality before the model transition.
- Synthetic TSDF planes and closed shapes produce correct zero surfaces and finite normals.
- Conflicting depth sheets, exterior mask rays, weak support, and disconnected islands are rejected while intentional openings remain open.
- Insufficient memory and insufficient input fail without fallback.
- CLI model contracts select `depth_tsdf` without requiring a dense-cloud path.

All MVS-specific tests remain under `src/core/mvs/tests`.

### Build and focused tests

Use only `E:\code\plascan\build\windows-vcpkg-cuda-release`. Build and run at least:

- `test_aerial_triangulation_workflow`
- `test_mvs_depth_pyramid`
- `test_mvs_types`
- `test_mvs_pipeline`
- `test_mvs_rectifier_unit`
- `test_cli_contracts`
- affected GUI/project-quality and mesh tests

### End-to-end regression

- Temple: inspect representative final/L1/L2/L3 overlays, confirm clean overlay state, generate a `depth_tsdf` model, and run the image/model quality report.
- Temple initial acceptance floor: coverage at least 0.75, IoU at least 0.70, edge P90 below 65 pixels, SSIM at least 0.20, largest connected component ratio at least 0.90, and recognizable roof, columns, doorway, and base without broad layered ridges.
- Nine-image UAV subset: rerun the full workflow, confirm each image has no more than 40,000 SIFT keypoints, and verify terrain depth/model quality does not regress.
- Reports must identify the actual `depth_tsdf` mode and must not claim a silent visual-hull or Poisson fallback.

## Scope boundaries

This work does not add a standalone depth workspace, GPU shader colorization, confidence visualization, texture-atlas generation, or CUDA TSDF integration. Dense-cloud generation remains available as a separately requested product, but it is not part of the default depth-to-model path.
