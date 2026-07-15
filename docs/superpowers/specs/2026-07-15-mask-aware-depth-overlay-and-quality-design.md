# Mask-aware depth overlay and reconstruction quality design

## Objective

Improve PlaScan depth-map inspection and arbitrary-3D reconstruction quality in two connected steps:

1. Add a Metashape-style depth overlay to the active photo view so users can inspect each depth result against the source texture and mask.
2. Make project masks and the three-level depth pyramid authoritative inputs to MVS and mesh generation, reducing fragmented sheets, filled openings, edge erosion, and over-smoothed geometry.

The first target datasets are the 16-image Middlebury temple project and the existing 9-image aerial subset. The change must not introduce a second GUI-only depth implementation; GUI and CLI must consume the same persisted depth artifacts and quality metadata.

## Current diagnosis

The PlaScan temple mesh preserves only a thick outer shell. Columns and openings are split into conflicting depth sheets, boundary regions are removed, and broad horizontal ridges remain. Poisson reconstruction amplifies these defects but is not their first cause.

The current MVS path builds an intensity-derived content mask and applies it after depth estimation. Project photo masks are available through `ProjectIO`, but are not explicit per-view inputs to `DepthMapGenerator`. The automatic content mask also uses the opposite semantic convention from project masks:

- Project photo mask: `255 = excluded`, `0 = retained foreground`.
- MVS valid mask: `255 = valid`, `0 = invalid`.

The project already persists final and intermediate pyramid artifacts (`raw_depth_path`, `raw_confidence_path`, `valid_mask_path`, previews, and `pyramid_levels`). These records are sufficient for a metadata-driven overlay without scanning output directories.

## User experience

### Toolbar

Add a checkable **显示深度图** shortcut to the image-view toolbar. It is off by default. Once enabled, changing photos keeps the mode enabled and automatically loads the corresponding depth result.

The adjacent menu contains:

- 所有级别
- 级别 1
- 级别 2
- 级别 3
- 显示强度

`所有级别` displays the final accepted depth result. Level 1/2/3 display their persisted artifacts. `显示强度` converts the retained source-image area to grayscale before applying the depth colors; it does not display confidence. Confidence remains a separate future mode.

The shortcut is disabled when the active photo has no matching depth record. It must never fall back to another photo's result. Missing or stale artifacts produce a concise log entry and leave the original image visible.

### Rendering

Depth is drawn over the current photo, not in a separate tab. Invalid depth and excluded mask pixels are fully transparent. The depth overlay sits above the base image and below feature points, residuals, markers, and editing handles.

Depth color normalization uses robust percentiles from valid finite depth values (P2 to P98). This prevents a few outliers from flattening visible depth contrast. The overlay preserves the source image's pixel coordinate system, rotation, zoom, and pan.

## Components

### DepthOverlayController

Add a GUI controller responsible for:

- Resolving a depth record by normalized `ref_image` path.
- Selecting final or Level 1/2/3 artifact metadata.
- Asynchronously loading raw depth and valid mask.
- Producing a transparent colorized `QImage` using robust normalization.
- Optionally creating a masked grayscale base image for intensity mode.
- Caching rendered overlays by image path, depth level, intensity mode, artifact modification time, and display parameters.
- Discarding stale asynchronous results when the active image or project changes.

The controller reads project metadata only. Directory scanning is not a primary state source.

### CanvasWidget and LayerRenderer

`CanvasWidget` owns display state and delegates artifact loading to `DepthOverlayController`. It exposes slots for enabled state, selected level, and intensity mode, and emits availability/state signals for toolbar synchronization.

`LayerRenderer` owns one dedicated depth pixmap item. It supports replacing, hiding, and clearing that item without disturbing the base image or feature layers.

### MainMenu and MainWindow

`MainMenu` creates the checkable shortcut and exclusive level actions. `MainWindow` connects them to `CanvasWidget`, updates enabled state from the current image and project metadata, and preserves the user's overlay mode while switching photos.

## Mask-aware MVS

Extend the MVS view/config data so every reference view can carry an optional project photo mask. Before PatchMatch, convert project mask semantics to the internal valid-mask convention. Resize only with nearest-neighbor interpolation.

For every pyramid level:

1. Build the level mask from the authoritative project mask when available.
2. Otherwise use the existing intensity-derived content mask.
3. Prevent propagation and random search from accepting excluded reference pixels.
4. Apply the same valid region to depth, confidence, support count, and uncertainty.
5. Re-apply the full-resolution authoritative mask after unrectification and before persistence.

Mask edges must not be morphologically expanded into excluded background. Any optional cleanup may remove isolated retained islands but must not close intentional openings such as the temple doorway.

## Three-level depth responsibilities

- **Level 3, global structure:** coarse search and stable depth range. Its result is a prior, not direct production geometry.
- **Level 2, geometric body:** multi-view reprojection consistency and minimum support produce stable main surfaces.
- **Level 1, edge and texture refinement:** narrow search around the previous prior, guided by source-image edges and constrained by the authoritative mask.

The final accepted depth must come from the finest successful level. A coarser fallback is allowed only when recorded explicitly in metadata and quality reports.

## Fusion and mesh gating

Before a pixel contributes to production geometry, require:

- finite positive depth;
- valid authoritative mask;
- minimum source-view support;
- bounded reprojection/depth disagreement;
- locally consistent normal or depth gradient.

Reject isolated sheets and mutually conflicting front/back surfaces before normal estimation. Arbitrary-3D mesh generation consumes only the gated fused depth points with finite oriented normals. Sparse tie points and coarse pyramid levels do not silently enter a production mesh. If insufficient oriented points remain, return a diagnostic containing per-frame rejection counts instead of only a Poisson failure.

## Metadata and quality reporting

Each depth result records:

- reference image and selected source images;
- authoritative mask source and coverage;
- each pyramid level's paths, dimensions, valid coverage, mean confidence, support statistics, and elapsed time;
- final accepted level and any fallback reason;
- depth discontinuity ratio and rejection counts.

Mesh reports include input frame count, accepted points, rejected points by reason, finite oriented normal count, and reconstruction mode.

## Concurrency and memory

Overlay loading uses the existing GUI task-runner/QFuture pattern with `QPointer` guards. It never blocks the GUI thread. The LRU cache is bounded by byte size and invalidated by artifact modification time.

MVS remains CUDA-accelerated where supported. Mask decoding and overlay colorization use CPU workers because disk IO and host-to-device transfer would dominate this small interactive task.

## Error handling

- Missing depth metadata: disable the shortcut for the active photo.
- Missing raw depth but valid preview: report that level as unavailable rather than silently using a different level.
- Corrupt raw artifact or size mismatch: log exact paths and expected/actual dimensions; keep the original image visible.
- Missing project mask: record fallback to automatic content mask.
- Stale asynchronous result: discard without changing the current view.

## Verification

### Unit and contract tests

- Resolve depth records by normalized reference-image path without cross-image fallback.
- Select final and Level 1/2/3 artifacts from metadata.
- Convert project mask semantics correctly and preserve holes.
- Normalize depth using only finite valid pixels.
- Produce transparent excluded/invalid pixels and stable overlay dimensions.
- Keep overlay Z-order below feature and marker layers.
- Reject stale asynchronous results.

### Integration tests

- Temple: generate all three depth levels, verify mask-constrained coverage, inspect representative front/side views, and build a mesh with finite oriented normals.
- Nine-image aerial subset: verify that mask fallback does not remove valid ground and that depth/mesh behavior does not regress.
- GUI: enable the overlay, switch photos, switch levels and intensity mode, zoom/rotate, regenerate depth, and confirm non-blocking refresh.
- CLI/GUI parity: both paths consume the same depth records and produce the same mesh input statistics.

### Success criteria

- Temple depth boundaries follow the project mask and preserve intentional openings.
- Level 1 contains visibly finer geometric transitions than Levels 2 and 3.
- The final temple mesh retains columns, doorway, roof, and base as connected recognizable structures without the current broad layered ridges.
- Overlay switching remains responsive and never displays a depth map from another image.

## Scope boundaries

This work does not add GPU shader rendering, confidence visualization, texture-atlas generation, or manual mask editing. Those may be added after depth correctness and overlay diagnostics are validated.
